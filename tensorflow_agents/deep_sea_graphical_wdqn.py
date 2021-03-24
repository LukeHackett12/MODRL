from os import times
from tensorflow_agents.deep_sea_baseline_ddqn import GROUP_NUM
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pylab as pl

import cv2

import numpy as np
from typing import NamedTuple
from collections import deque
import random
import sys
from copy import deepcopy
from enum import Enum

from scipy.interpolate import interp1d
from numpy.core.numeric import NaN
from custom_envs.deep_sea_treasure.engine import DeepSeaTreasure

import tensorflow as tf
from tensorflow import keras, Tensor
from keras import backend as K

GROUP_NUM = 10

class PolEnum(Enum):
    Score = 0
    Time = 1
    Random = 2

class Transition(NamedTuple):
    currStates: Tensor
    actions: Tensor
    policy: Tensor
    rewards: Tensor
    nextStates: Tensor
    dones: Tensor


class DQNAgent(object):
    def __init__(self, stateShape, actionSpace, numPicks, memorySize, numRewards, sync=100, burnin=500, alpha=0.00025, epsilon=1, epsilon_decay=0.99975, epsilon_min=0.01, gamma=0.9):
        self.numPicks = numPicks
        self.replayMemory = deque(maxlen=memorySize)
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
        self.delay = 1

        self.numRewards = numRewards

        self.train_network = self.createNetwork(stateShape, len(actionSpace), self.alpha)
        self.target_network = self.createNetwork(stateShape, len(actionSpace), self.alpha)

        # Store network weights for each policy
        self.policy_train_weights = [deepcopy(self.train_network.get_weights())] * self.numRewards
        self.policy_target_weights = [deepcopy(self.train_network.get_weights())] * self.numRewards

        # Create and store network weights for W-values
        self.w_network = self.createNetwork(stateShape, numRewards, self.alpha)
        self.wnet_weights = [deepcopy(self.w_network.get_weights())] * self.numRewards

    def createNetwork(self, n_input, n_output, learningRate):
        model = keras.models.Sequential()

        model.add(keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=n_input))
        model.add(keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='linear'))
        model.add(keras.layers.Dense(n_output, activation='linear'))

        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=learningRate))
        print(model.summary())
        return model

    def trainDQN(self):
        if len(self.replayMemory) <= self.numPicks:
            return [(0, 0)] * self.numRewards

        samples = random.sample(self.replayMemory, self.numPicks)
        batch = Transition(*zip(*samples))
        currStates, actions, policies, rewards, nextStates, dones = batch

        currStates = np.array(currStates)
        nextStates = np.array(nextStates)

        rewards = np.array(rewards).transpose().astype(float)
        actions = np.array(actions).reshape(self.numPicks,).astype(int)
        policies = np.array(policies).reshape(self.numPicks,).astype(int)

        dones = np.array(dones).astype(bool)
        notDones = (~dones).astype(float)
        dones = dones.astype(float)

        agentsLoss = []

        for i, reward in enumerate(rewards):
            self.train_network.set_weights(self.policy_train_weights[i])
            self.target_network.set_weights(self.policy_target_weights[i])

            Q_currents_all = self.train_network(currStates, training=False).numpy()
            Q_futures_all = self.target_network(nextStates, training=False).numpy().max(axis=1)

            Q_currents = np.copy(Q_currents_all)
            Q_futures = np.copy(Q_futures_all)

            reward = np.array(reward).reshape(self.numPicks,).astype(float)
            # Q-Learning
            Q_currents[np.arange(self.numPicks), actions] = reward * dones + (reward + Q_futures * self.gamma)*notDones
            lossQ = self.train_network.train_on_batch(currStates, Q_currents)
            self.policy_train_weights[i] = deepcopy(self.train_network.get_weights())

            lossW = 0

            # Leave in exploration actions for now, can remove with "policy[p] != -1"
            inverted_policy_mask = np.array([p for p in range(self.numPicks) if policies[p] != i])
            if len(inverted_policy_mask) > 0 and len(self.replayMemory) > self.burnin:
                # W-Learning
                self.w_network.set_weights(self.wnet_weights[i])
                currStatesNP = currStates[inverted_policy_mask]
                policiesNP = policies[inverted_policy_mask]
                rewardNP = reward[inverted_policy_mask]
                donesNP = dones[inverted_policy_mask]
                notDonesNP = notDones[inverted_policy_mask]

                Q_currents_np = Q_currents_all[inverted_policy_mask].max(axis=1)
                Q_futures_np = Q_futures_all[inverted_policy_mask]

                w_targets = self.w_network(currStatesNP, training=False).numpy()

                # maybe (Q_currents_not_policy - ((rewardNP * dones) + (self.gamma * Q_futures_not_policy) * notDonesNP)) * walpha^delay ?
                w_targets[np.arange(len(inverted_policy_mask)), policiesNP] = ((1-self.walpha) * w_targets[np.arange(len(inverted_policy_mask)), policiesNP]) + \
                    ((self.walpha**self.delay) * (Q_currents_np - ((rewardNP * donesNP) + (self.gamma * Q_futures_np) * notDonesNP)))
                lossW = self.w_network.train_on_batch(currStatesNP, w_targets)
                self.wnet_weights[i] = self.w_network.get_weights()

            agentsLoss.append((lossQ, lossW))

        return agentsLoss

    def selectAction(self, state):
        self.step += 1

        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        if self.step % self.sync == 0:
            self.policy_target_weights = deepcopy(self.policy_train_weights)

        emptyPolicies = [0] * self.numRewards
        policy, qs, ws = (-1, emptyPolicies, emptyPolicies)
        random = True
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, len(self.actionSpace))
        else:
            ws = []
            actions = []
            preds = []

            for i in range(self.numRewards):
                self.train_network.set_weights(self.policy_train_weights[i])

                pred = np.squeeze(self.train_network(np.expand_dims(state, 0), training=False).numpy(), axis=0)
                preds.append(pred)

                action = np.argmax(pred)
                actions.append(action)

            if np.random.rand(1) < self.epsilon:
                policy = np.random.randint(0, self.numRewards)
            else:
                for i in range(self.numRewards):
                    self.w_network.set_weights(self.wnet_weights[i])
                    w_val = self.w_network(np.expand_dims(state, 0), training=False).numpy()[0]
                    ws.append(w_val[np.argmax(w_val)])

                random = False
                policy = np.argmax(ws)

            action = actions[policy]
            qs = preds[policy]

        return action, policy, qs, ws, random

    def addMemory(self, memory):
        self.replayMemory.append(memory)

    def save(self):
        save_path = (
            f"./mountain_car_wnet_{int(self.step)}.chkpt"
        )
        print(f"MountainNet saved to {save_path} done!")


class DeepSeaGraphicalWAgent(object):
    def __init__(self, episodes):
        self.current_episode = 0
        self.episodes = episodes

        self.episode_score = []
        self.episode_qs = []
        self.episode_height = []
        self.episode_loss = []
        self.episode_ws = []
        self.episode_policies = []

        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.canvas.draw()
        plt.show(block=False)

        self.numRewards = 2

        self.env = DeepSeaTreasure(width=5, speed=10000, graphical_state=True, render=True, is_debug=False)
        self.agent = DQNAgent(stateShape=(64, 64, 1), actionSpace=self.env.get_action_space(), numPicks=32, memorySize=10000, numRewards=self.numRewards)

    def train(self):
        for _ in range(self.episodes):
            self.episode()
            self.current_episode += 1

        plt.show(block=True)
        self.env.close()

    def episode(self):
        done = False
        rewardsSum = 0

        lossSums = [0] * (self.numRewards)
        policies = [0] * (self.numRewards)
        qSums = [0] * (self.numRewards)
        wSums = [0] * (self.numRewards)
        actions = 1

        state = self.process_state(self.env.reset())
        maxHeight = -1

        while not done:
            action, policy, qs, ws, random = self.agent.selectAction(state)
            if not random:
                policies[policy] += 1
                qSums = [qSums[i] + qs[i] for i in range(len(policies))]
                wSums = [wSums[i] + ws[i] for i in range(len(policies))]
                actions += 1

            obs, reward, done, _ = self.env.step_all(action)

            nextState = state - self.process_state(obs)
            rewardsSum = np.add(rewardsSum, sum(reward))

            self.agent.addMemory((state, action, policy, reward, nextState, done))
            state = nextState

            loss = self.agent.trainDQN()
            lossSums = [lossSums[i] + loss[i][0] for i in range(len(policies))]

        print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(
            self.agent.epsilon, rewardsSum, lossSums, self.current_episode))

        self.episode_score.append(rewardsSum)
        self.episode_height.append(maxHeight)
        self.episode_loss.append(lossSums)
        self.episode_policies.append(policies)
        self.episode_qs.append([qSum/actions for qSum in qSums])
        self.episode_ws.append([wSum/actions for wSum in wSums])
        self.plot()

        print("Report: \nrewardSum:{}\nheight:{}\nloss:{}\npolicies:{}\nqAverage:{}\nws:{}".format(self.episode_score[-1],
                                                                                                   self.episode_height[-1],
                                                                                                   self.episode_loss[-1],
                                                                                                   self.episode_policies[-1],
                                                                                                   self.episode_qs[-1],
                                                                                                   self.episode_ws[-1]))

    def process_state(self, state):
        state = cv2.resize(state.astype('float32'), (64, 64), interpolation=cv2.INTER_AREA)
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(state, 2)

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
    agent = DeepSeaGraphicalWAgent(1000)
    agent.train()
