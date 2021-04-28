from os import times

from scipy.interpolate.interpolate import interp1d
from replay_buffer_policy import PrioritizedReplayBuffer
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pylab as pl

import gc
import psutil

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

GROUP_NUM = 20

class Transition(NamedTuple):
    currStates: Tensor
    actions: Tensor
    policy: Tensor
    rewards: Tensor
    nextStates: Tensor
    dones: Tensor

class PolEnum(Enum):
    Score = 0
    Time = 1
    Random = 2

class DQNAgent(object):
    def __init__(self, stateShape, actionSpace, numPicks, memorySize, numRewards, sync=50, burnin=500, alpha=0.0001, epsilon=1, epsilon_decay=0.99985, epsilon_min=0.01, gamma=0.9, optim=keras.optimizers.Adam(lr=0.0001)):
        self.numPicks = numPicks
        self.stateShape = stateShape
        self.actionSpace = actionSpace
        self.numRewards = numRewards

        self.step = 0

        self.sync = sync
        self.burnin = burnin
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.walpha = 0.001
        self.delay = 1.1

        self.optim = optim

        self.train_network = self.createNetwork(stateShape, len(actionSpace), self.alpha)
        self.target_network = self.createNetwork(stateShape, len(actionSpace), self.alpha)

        # Store network weights for each policy
        self.policy_train_weights = [deepcopy(self.train_network.get_weights())] * self.numRewards
        self.policy_target_weights = [deepcopy(self.train_network.get_weights())] * self.numRewards

        #Individual replay buffers for policies and for w net
        self.replayMemory = [PrioritizedReplayBuffer(memorySize, 0.6)] * self.numRewards

        # Create and store network weights for W-values
        self.w_train_network = self.createNetwork(stateShape, numRewards, self.alpha)
        self.wnet_train_weights = [deepcopy(self.w_train_network.get_weights())] * self.numRewards


    def createNetwork(self, n_input, n_output, learningRate):
        model = keras.models.Sequential()
        model.add(keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=n_input))
        model.add(keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='linear'))
        model.add(keras.layers.Dense(n_output, activation='linear'))

        model.compile(loss=keras.losses.Huber(), optimizer=self.optim)

        return model

    def trainDQN(self):
        if len(self.replayMemory[0]) <= self.numPicks:
            return [(0, 0)] * self.numRewards

        agentsLoss = []
        beta = 0.4 + self.step * (1.0 - 0.4) / 150000

        for i in range(self.numRewards):
            samples = self.replayMemory[i].sample(self.numPicks, beta)
            currStates, actions, policies, rewards, nextStates, dones, weights, indices = samples

            currStates = np.array(currStates).transpose(0,2,3,1)
            nextStates = np.array(nextStates).transpose(0,2,3,1)

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
            prios = (np.abs(lossQ)*weights) + 1e-5
            self.replayMemory[i].update_priorities(indices, prios)

            lossW = 0

            # Leave in exploration actions for now, can remove with "policy[p] != -1"
            inverted_policy_mask = np.array([p for p in range(self.numPicks) if policies[p] != i])
            if len(inverted_policy_mask) > 0:
                # W-Learning
                self.w_train_network.set_weights(self.wnet_train_weights[i])
                
                currStatesNP = currStates[inverted_policy_mask]
                policiesNP = policies[inverted_policy_mask]
                rewardNP = rewards[inverted_policy_mask]
                donesNP = dones[inverted_policy_mask]
                notDonesNP = notDones[inverted_policy_mask]

                Q_currents_np = Q_currents_all[inverted_policy_mask].max(axis=1)
                Q_futures_np = Q_futures_all[inverted_policy_mask]

                w_train = self.w_train_network(currStatesNP, training=False).numpy()

                # maybe (Q_currents_not_policy - ((rewardNP * dones) + (self.gamma * Q_futures_not_policy) * notDonesNP)) * walpha^delay ?
                w_train[np.arange(len(inverted_policy_mask)), policiesNP] = ((1-self.walpha) * w_train[np.arange(len(inverted_policy_mask)), policiesNP]) + \
                    ((self.walpha**self.delay) * (Q_currents_np - ((rewardNP * donesNP) + (self.gamma * Q_futures_np) * notDonesNP)))
                lossW = self.w_train_network.train_on_batch(currStatesNP, w_train)
                self.wnet_train_weights[i] = self.w_train_network.get_weights()

            agentsLoss.append((lossQ, lossW))

        return agentsLoss

    def selectAction(self, state):
        self.step += 1
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        
        state = np.expand_dims(np.array(state),0).transpose(0,2,3,1)

        if self.step % self.sync == 0:
            self.policy_target_weights = deepcopy(self.policy_train_weights)

        emptyPolicies = [0] * self.numRewards
        policy, qs, ws = (-1, -1, emptyPolicies)
        random = True
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            ws = []
            if np.random.rand(1) < self.epsilon:
                policy = np.random.randint(0, self.numRewards)
            else:
                for i in range(self.numRewards):
                    self.w_train_network.set_weights(self.wnet_train_weights[i])
                    w_val = self.w_train_network(state, training=False).numpy()[0]
                    ws.append(w_val[np.argmax(w_val)])
                    random = False

                policy = np.argmax(ws)

            self.train_network.set_weights(self.policy_train_weights[policy])
            pred = np.squeeze(self.train_network(state, training=False).numpy(), axis=0)
            action = np.argmax(pred)
            qs = np.max(pred)

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

        self.fig, self.ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10,4))
        self.fig.tight_layout()
        self.fig.canvas.draw()

        self.numRewards = 2

        self.env = DeepSeaTreasure(width=5, speed=1e8, graphical_state=True,
                               render=False, is_debug=True, frame_stack=2)

    def train(self):
        self.agent = DQNAgent(stateShape=(84, 84, 2), actionSpace=self.env.get_action_space(), numPicks=32, memorySize=10000, numRewards=self.numRewards, optim=keras.optimizers.Adam(lr=0.0001))
        
        for _ in range(self.episodes):
            self.episode()
            self.current_episode += 1

        self.plot_compare()

    def episode(self):
        done = False
        rewardsSum = 0
        lossSums = [0] * (self.numRewards)
        policies = [0] * (self.numRewards + 1)

        qSums = [0] * (self.numRewards)
        wSums = [0] * (self.numRewards)
        actions = 1

        state = self.env.reset()

        while not done:
            action, policy, qs, ws, random = self.agent.selectAction(state)
            policies[policy] += 1
            if not random:
                qSums[policy]+= qs
                wSums = [wSums[i] + ws[i] for i in range(len(wSums))]
                actions += 1

            obs, reward, done, _ = self.env.step_all(action)

            nextState = obs
            rewardsSum = np.add(rewardsSum, sum(reward))

            self.agent.addMemory(state, action, policy, reward, nextState, done)
            loss = self.agent.trainDQN()
            state = nextState
            lossSums = [lossSums[i] + loss[i][0] for i in range(len(lossSums))]

        print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(
            self.agent.epsilon, rewardsSum, lossSums, self.current_episode))

        self.episode_score.append(rewardsSum)
        self.episode_loss.append(lossSums)
        self.episode_policies.append(policies)
        self.episode_qs.append([qSum/actions for qSum in qSums])
        self.episode_ws.append([wSum/actions for wSum in wSums])

        print("Report: \nrewardSum:{}\nloss:{}\npolicies:{}\nqAverage:{}\nws:{}".format(self.episode_score[-1],
                                                                                                  self.episode_loss[-1],
                                                                                                  self.episode_policies[-1],
                                                                                                  self.episode_qs[-1],
                                                                                                  self.episode_ws[-1]))
        print("memory len:" + str(len(self.agent.replayMemory[0])))
        print("memory used:" + str(psutil.virtual_memory().used // 1e6))
        tf.keras.backend.clear_session()
        gc.collect()

    def plot_compare(self):
        spline_x = np.linspace(0, self.current_episode, num=self.current_episode)

        ep_adam_scores = np.array(self.episode_score)
        ep_adam_groups = [ep_adam_scores[i * GROUP_NUM:(i + 1) * GROUP_NUM] for i in range((len(ep_adam_scores) + GROUP_NUM - 1) // GROUP_NUM)]
        # Pad for weird numpy error for now
        ep_adam_groups[-1] = np.append(ep_adam_groups[-1], [np.mean(ep_adam_groups[-1])] * (GROUP_NUM - len(ep_adam_groups[-1])))
        x_groups = [i*GROUP_NUM for i in range(len(ep_adam_groups))]

        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[2].clear()

        ep_adam_avgs = np.mean(ep_adam_groups, 1)
        avg_adam_spl = interp1d(x_groups, ep_adam_avgs, kind='cubic', fill_value="extrapolate")
        ep_adam_std = np.std(ep_adam_groups, 1)
        std_adam_spl = interp1d(x_groups, ep_adam_std, kind='cubic', fill_value="extrapolate")

        self.ax[0].plot(spline_x,ep_adam_scores, lw=0.7, c="blue")

        self.ax[1].plot(x_groups,ep_adam_avgs, lw=0.7, c="blue")
        self.ax[1].fill_between(x_groups, ep_adam_avgs-ep_adam_std, ep_adam_avgs+ep_adam_std, alpha=0.5, facecolor="red", interpolate=True)

        self.ax[2].plot(spline_x, avg_adam_spl(spline_x), lw=0.7, c="blue")
        self.ax[2].fill_between(spline_x, avg_adam_spl(spline_x)-std_adam_spl(spline_x), avg_adam_spl(spline_x)+std_adam_spl(spline_x), alpha=0.5, facecolor="red", interpolate=True)

        self.fig.text(0.5, 0.04, 'Episode', ha='center')
        self.fig.text(0.04, 0.5, 'Score', va='center', rotation='vertical')
        self.fig.suptitle('Episode Score smoothing')
        plt.show(block=True)


if __name__ == '__main__':
    agent = MultiObjectiveDeepSeaW(3000)
    agent.train()
