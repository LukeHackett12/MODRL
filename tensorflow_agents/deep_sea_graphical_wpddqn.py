from os import times

from scipy.interpolate.interpolate import interp1d
from replay_buffer import PrioritizedReplayBuffer
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pylab as pl

import numpy as np
from typing import NamedTuple
from collections import deque
import random
import sys
from copy import deepcopy
from enum import Enum

from numpy.core.numeric import NaN
from tensorflow.python.ops.gen_array_ops import shape
from custom_envs.deep_sea_treasure.engine import DeepSeaTreasure

import tensorflow as tf
from tensorflow import keras, Tensor
from keras import backend as K

GROUP_NUM = 5

class Transition(NamedTuple):
    currStates: Tensor
    actions: Tensor
    policy: Tensor
    rewards: Tensor
    nextStates: Tensor
    dones: Tensor


class DQNAgent(object):
    def __init__(self, stateShape, actionSpace, numPicks, memorySize, numRewards, sync=100000, burnin=10000, alpha=0.0001, epsilon=1, epsilon_decay=0.99999975, epsilon_min=0.01, gamma=0.9):
        self.numPicks = numPicks
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0

        self.sync = sync
        self.burnin = burnin
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.walpha = 0.01

        self.train_network = self.createNetwork(stateShape, len(actionSpace), self.alpha)
        self.target_network = self.createNetwork(stateShape, len(actionSpace), self.alpha)

        # Store network weights for each policy
        self.policy_train_weights = [deepcopy(self.train_network.get_weights())] * self.numRewards
        self.policy_target_weights = [deepcopy(self.train_network.get_weights())] * self.numRewards

        #Individual replay buffers for policies and for w net
        self.replayMemory = [PrioritizedReplayBuffer(memorySize, 0.6)] * self.numRewards

        # Create and store network weights for W-values
        self.w_train_network = self.createNetwork(stateShape, numRewards, self.alpha)
        self.w_target_network = self.createNetwork(stateShape, numRewards, self.alpha)
        self.wnet_train_weights = [deepcopy(self.w_train_network.get_weights())] * self.numRewards
        self.wnet_target_weights = [deepcopy(self.w_train_network.get_weights())] * self.numRewards


    def createNetwork(self, n_input, n_output, learningRate):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(24, activation='relu', input_shape=n_input))
        model.add(keras.layers.Dense(48, activation='relu'))
        model.add(keras.layers.Dense(n_output, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learningRate))
        return model

    def trainDQN(self):
        if len(self.replayMemory[0]) <= self.numPicks:
            return [(0, 0)] * self.numRewards

        agentsLoss = []
        beta = 0.4 + self.step * (1.0 - 0.4) / 100000
        h = tf.keras.losses.Huber()

        for i in range(self.numRewards):
            samples = self.replayMemory[i].sample(self.numPicks, beta)
            currStates, actions, policies, rewards, nextStates, dones, weights, indices = samples

            currStates = np.squeeze(np.array(currStates), 1)
            nextStates = np.squeeze(np.array(nextStates), 1)

            rewards = np.array(rewards).reshape(self.numPicks,).astype(float)
            actions = np.array(actions).reshape(self.numPicks,).astype(int)
            policies = np.array(policies).reshape(self.numPicks,).astype(int)

            dones = np.array(dones).astype(bool)
            notDones = (~dones).astype(float)
            dones = dones.astype(float)

            self.train_network.set_weights(self.policy_train_weights[i])
            Q_currents_all = self.train_network(currStates, training=False).numpy()

            self.target_network.set_weights(self.policy_target_weights[i])
            Q_futures_all = self.target_network(nextStates, training=False).numpy().max(axis=1)

            Q_currents = np.copy(Q_currents_all)
            Q_futures = np.copy(Q_futures_all)

            # Q-Learning
            Q_currents[np.arange(self.numPicks), actions] = rewards * dones + (rewards + Q_futures * self.gamma)*notDones
            lossQ = self.train_network.train_on_batch(currStates, Q_currents)
            self.policy_train_weights[i] = deepcopy(self.train_network.get_weights())

            #PER Update
            loss = h(Q_currents_all[np.arange(self.numPicks), actions], Q_currents[np.arange(self.numPicks), actions])
            prios = (np.abs(loss)*weights) + 1e-5
            self.replayMemory[i].update_priorities(indices, prios)

            lossW = 0

            # Leave in exploration actions for now, can remove with "policy[p] != -1"
            inverted_policy_mask = np.array([p for p in range(self.numPicks) if policies[p] != i])
            if len(inverted_policy_mask) > 0:
                # W-Learning
                self.w_train_network.set_weights(self.wnet_train_weights[i])
                self.w_target_network.set_weights(self.wnet_target_weights[i])
                currStatesNP = currStates[inverted_policy_mask]
                policiesNP = policies[inverted_policy_mask]
                rewardNP = rewards[inverted_policy_mask]
                donesNP = dones[inverted_policy_mask]
                notDonesNP = notDones[inverted_policy_mask]

                Q_currents_np = Q_currents_all[inverted_policy_mask].max(axis=1)
                Q_futures_np = Q_futures_all[inverted_policy_mask]

                w_train = self.w_train_network(currStatesNP, training=False).numpy()
                w_target = self.w_target_network(currStatesNP, training=False).numpy()

                # maybe (Q_currents_not_policy - ((rewardNP * dones) + (self.gamma * Q_futures_not_policy) * notDonesNP)) * walpha^delay ?
                w_train[np.arange(len(inverted_policy_mask)), policiesNP] = ((1-self.walpha) * w_target) + \
                    ((self.walpha**self.delay) * (Q_currents_np - ((rewardNP * donesNP) + (self.gamma * Q_futures_np) * notDonesNP)))
                lossW = self.w_train_network.train_on_batch(currStatesNP, w_train)
                self.wnet_train_weights[i] = self.w_train_network.get_weights()

            agentsLoss.append((lossQ, lossW))

        self.replayMemory.update_priorities(indices, prios)
        return agentsLoss

    def selectAction(self, state):
        self.step += 1

        if self.step % self.sync == 0:
            self.policy_target_weights = deepcopy(self.policy_train_weights)
            self.w_target_weights = deepcopy(self.w_train_weights)
            self.epsilon = max(self.epsilon-self.epsilon_decay, self.epsilon_min)

        emptyPolicies = [0] * self.numRewards
        policy, qs, ws = (-1, emptyPolicies, emptyPolicies)
        random = True
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            ws = []
            if np.random.rand(1) < self.epsilon:
                policy = np.random.randint(0, self.numRewards)
            else:
                for i in range(self.numRewards):
                    self.w_train_network.set_weights(self.wnet_weights[i])
                    w_val = self.w_train_network(state, training=False).numpy()[0]
                    ws.append(w_val[np.argmax(w_val)])

                random = False
                policy = np.argmax(ws)

            self.train_network.set_weights(self.policy_train_weights[policy])

            pred = np.squeeze(self.train_network(state, training=False).numpy(), axis=0)
            action = np.argmax(pred)
            qs = pred
            random = False

        return action, policy, qs, ws, random

    def addMemory(self, state, action, policy, reward, nextState, done):
        for i in range(self.numRewards):
            self.replayMemory[i].add(state, action, policy, reward[i], nextState, done)

    def save(self):
        save_path = (
            f"./deep_sea_wnet{int(self.step)}.chkpt"
        )
        print(f"DeepSeaWnet saved to {save_path} done!")


class MultiObjectiveDeepSeaW(object):
    def __init__(self, episodes):
        self.current_episode = 0
        self.episodes = episodes

        self.episode_score = []
        self.episode_qs = []
        self.episode_height = []
        self.episode_loss = []
        self.episode_ws = []
        self.episode_policies = []

        self.fig, self.ax = plt.subplots(2, 3)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.show(block=False)

        self.numRewards = 3

        self.env = DeepSeaTreasure(width=5, speed=1e8, graphical_state=True,
                               render=False, is_debug=True, random_starts=True, frame_stack=2)
        self.agent = DQNAgent(stateShape=(2,), actionSpace=(84,84,2), numPicks=32, memorySize=10000, numRewards=self.numRewards)

    def train(self):
        for _ in range(self.episodes):
            self.episode()
            self.current_episode += 1

        self.env.close()

    def episode(self):
        done = False
        rewardsSum = 0
        lossSums = [0] * (self.numRewards)
        policies = [0] * (self.numRewards)

        qSums = [0] * (self.numRewards)
        wSums = [0] * (self.numRewards)
        actions = 1

        state = self.env.reset()

        while not done:
            action, policy, qs, ws, random = self.agent.selectAction(state)
            if not random:
                policies[policy] += 1
                qSums = [qSums[i] + qs[i] for i in range(len(policies))]
                wSums = [wSums[i] + ws[i] for i in range(len(policies))]
                actions += 1

            obs, reward, done, _ = self.env.step_all(action)

            nextState = obs
            rewardsSum = np.add(rewardsSum, sum(reward))

            self.agent.addMemory(state, action, policy, reward, nextState, done)
            loss = self.agent.trainDQN()
            state = nextState
            lossSums = [lossSums[i] + loss[i][0] for i in range(len(policies))]

        print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(
            self.agent.epsilon, rewardsSum, lossSums, self.current_episode))

        self.episode_score.append(rewardsSum)
        self.episode_loss.append(lossSums)
        self.episode_policies.append(policies)
        self.episode_qs.append([qSum/actions for qSum in qSums])
        self.episode_ws.append([wSum/actions for wSum in wSums])
        self.plot()

        print("Report: \nrewardSum:{}\nloss:{}\npolicies:{}\nqAverage:{}\nws:{}".format(self.episode_score[-1],
                                                                                                  self.episode_loss[-1],
                                                                                                  self.episode_policies[-1],
                                                                                                  self.episode_qs[-1],
                                                                                                  self.episode_ws[-1]))

    def plot(self):
        spline_x = np.linspace(0, self.current_episode, num=self.current_episode)

        ep_scores = np.array(self.episode_score)
        ep_groups = [ep_scores[i * GROUP_NUM:(i + 1) * GROUP_NUM] for i in range((len(ep_scores) + GROUP_NUM - 1) // GROUP_NUM)]
        # Pad for weird numpy error for now
        ep_groups[-1] = np.append(ep_groups[-1], [np.mean(ep_groups[-1])] * (GROUP_NUM - len(ep_groups[-1])))
        x_groups = [i*GROUP_NUM for i in range(len(ep_groups))]

        self.ax[0].clear()
        if len(x_groups) > 5:
            ep_avgs = np.mean(ep_groups, 1)
            avg_spl = interp1d(x_groups, ep_avgs, kind='cubic', fill_value="extrapolate")
            ep_std = np.std(ep_groups, 1)
            std_spl = interp1d(x_groups, ep_std, kind='cubic', fill_value="extrapolate")
            self.ax[0].plot(spline_x, avg_spl(spline_x), lw=0.7, c="blue")
            self.ax[0].fill_between(spline_x, avg_spl(spline_x)-std_spl(spline_x), avg_spl(spline_x)+std_spl(spline_x), alpha=0.5, facecolor="red", interpolate=True)

        self.ax[0].title.set_text('Training Score')
        self.ax[0].set_xlabel('Episode')
        self.ax[0].set_ylabel('Score')

        policies = np.transpose(self.episode_policies)
        colors = pl.cm.jet(np.linspace(0, 1, len(policies)*2))

        self.ax[1].clear()
        self.ax[1].title.set_text('Policy Choices')
        for i, policy in enumerate(policies):
            if len(x_groups) > 5:
                ep_groups = [policy[i * GROUP_NUM:(i + 1) * GROUP_NUM] for i in range((len(policy) + GROUP_NUM - 1) // GROUP_NUM)]
                # Pad for weird numpy error for now
                ep_groups[-1] = np.append(ep_groups[-1], [np.mean(ep_groups[-1])] * (GROUP_NUM - len(ep_groups[-1])))
                x_groups = [i*GROUP_NUM for i in range(len(ep_groups))]

                ep_avgs = np.mean(ep_groups, 1)
                avg_spl = interp1d(x_groups, ep_avgs, kind='cubic', fill_value="extrapolate")
                ep_std = np.std(ep_groups, 1)
                std_spl = interp1d(x_groups, ep_std, kind='cubic', fill_value="extrapolate")
                self.ax[1].plot(spline_x, avg_spl(spline_x), lw=0.7, c=colors[i], label="{} policy".format(PolEnum(i).name))
                self.ax[1].fill_between(spline_x, avg_spl(spline_x)-std_spl(spline_x), avg_spl(spline_x)+std_spl(spline_x), alpha=0.5, facecolor=colors[-1-i], interpolate=True)

        self.ax[1].legend()

        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(.001)



if __name__ == '__main__':
    agent = MultiObjectiveDeepSeaW(3000)
    agent.train()
