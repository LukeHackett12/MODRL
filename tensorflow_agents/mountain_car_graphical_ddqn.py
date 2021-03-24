from custom_envs.mountain_car.engine import MountainCar
from os import times

from replay_buffer import PrioritizedReplayBuffer
from tensorflow_agents.deep_sea_baseline_ddqn import GROUP_NUM
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

from scipy.interpolate import interp1d
from numpy.core.numeric import NaN

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
    rewards: Tensor
    nextStates: Tensor
    dones: Tensor


class DQNAgent(object):
    def __init__(self, stateShape, actionSpace, numPicks, memorySize, sync=100, burnin=1000, alpha=0.00025, epsilon=1, epsilon_decay=0.05, epsilon_min=0.01, gamma=0.99):
        self.numPicks = numPicks
        self.replayMemory = PrioritizedReplayBuffer(memorySize, 0.6)
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

        self.trainNetwork = self.createNetwork(
            stateShape, len(actionSpace), self.alpha)
        self.targetNetwork = self.createNetwork(
            stateShape, len(actionSpace), self.alpha)
        self.targetNetwork.set_weights(
            self.trainNetwork.get_weights())

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
        if self.step <= self.numPicks or len(self.replayMemory) <= self.burnin:
            return 0

        beta = 0.4 + self.step * (1.0 - 0.4) / 400000
        batch = self.replayMemory.sample(self.numPicks, beta)
        currStates, actions, rewards, nextStates, dones, weights, indices = batch

        currStates = np.array(currStates).transpose(0, 2, 3, 1)
        Q_currents = self.trainNetwork(currStates, training=False).numpy()

        nextStates = np.array(nextStates).transpose(0, 2, 3, 1)
        Q_futures = self.targetNetwork(nextStates, training=False).numpy().max(axis=1)

        rewards = np.array(rewards).reshape(self.numPicks,).astype(float)
        actions = np.array(actions).reshape(self.numPicks,).astype(int)

        dones = np.squeeze(np.array(dones)).astype(bool)
        notDones = (~dones).astype(float)
        dones = dones.astype(float)

        Q_currents_cp = deepcopy(Q_currents)
        Q_currents_cp[np.arange(self.numPicks), actions] = rewards + Q_futures * self.gamma * notDones

        loss = self.trainNetwork.train_on_batch(currStates, Q_currents_cp)
        prios = (np.abs(loss)*weights) + 1e-5
        self.replayMemory.update_priorities(indices, prios)

        return loss

    def selectAction(self, state):
        self.step += 1

        q = -100000
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            preds = np.squeeze(self.trainNetwork(
                np.expand_dims(np.array(state).transpose(1, 2, 0), 0), training=False).numpy(), axis=0)
            action = np.argmax(preds)
            q = preds[action]
        return action, q

    def addMemory(self, state, action, reward, nextState, done):
        self.replayMemory.add(state, action, reward, nextState, done)

    def save(self):
        save_path = (
            f"./mountain_car_wnet_{int(self.step)}.chkpt"
        )
        print(f"MountainNet saved to {save_path} done!")


class MountainCarGraphicalDDQN(object):
    def __init__(self, episodes):
        self.current_episode = 0
        self.episodes = episodes

        self.episode_score = []
        self.episode_qs = []
        self.episode_height = []
        self.episode_loss = []
        self.episode_policies = []

        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.canvas.draw()
        plt.show(block=False)

        self.env = MountainCar(speed=10000, graphical_state=True, render=False, is_debug=False, frame_stack=4)
        self.agent = DQNAgent(stateShape=(84, 84, 4), actionSpace=self.env.get_action_space(), numPicks=32, memorySize=100000)

    def train(self):
        for _ in range(self.episodes):
            self.episode()
            self.current_episode += 1

        plt.show(block=True)
        self.env.close()

    def episode(self):
        done = False
        rewardsSum = 0

        lossSum = 0
        qSums = 0
        actions = 1

        state = self.env.reset()
        maxHeight = -1

        while not done:
            action, qs = self.agent.selectAction(state)
            if qs != -100000:
                qSums += qs
                actions += 1

            obs, reward, done, height = self.env.step_all(action)
            maxHeight = max(height, maxHeight)
            reward[0] += height
            if height >= 0.5:
                for i in range(len(reward)):
                    reward[i] += 10

            nextState = obs
            rewardsSum = np.add(rewardsSum, sum(reward))

            self.agent.addMemory(state, action, reward[0], nextState, done)
            state = nextState

            loss = self.agent.trainDQN()
            lossSum += loss

        if self.current_episode % 10 == 0:
            self.agent.epsilon = max(self.agent.epsilon-self.agent.epsilon_decay, self.agent.epsilon_min)
        if self.current_episode % self.agent.sync == 0:
            self.agent.targetNetwork.set_weights(self.agent.trainNetwork.get_weights())

        print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(
            self.agent.epsilon, rewardsSum, lossSum, self.current_episode))

        self.episode_score.append(rewardsSum)
        self.episode_height.append(maxHeight)
        self.episode_loss.append(lossSum)
        self.episode_qs.append([qSums/actions])
        self.plot()

        print("Report: \nrewardSum:{}\nheight:{}\nloss:{}\nqAverage:{}".format(self.episode_score[-1],
                                                                               self.episode_height[-1],
                                                                               self.episode_loss[-1],
                                                                               self.episode_qs[-1]))

    def plot(self):
        spline_x = np.linspace(0, self.current_episode, num=self.current_episode)

        ep_scores = np.array(self.episode_score)
        ep_groups = [ep_scores[i * GROUP_NUM:(i + 1) * GROUP_NUM] for i in range((len(ep_scores) + GROUP_NUM - 1) // GROUP_NUM)]
        # Pad for weird numpy error for now
        ep_groups[-1] = np.append(ep_groups[-1], [np.mean(ep_groups[-1])] * (GROUP_NUM - len(ep_groups[-1])))
        x_groups = [i*GROUP_NUM for i in range(len(ep_groups))]

        self.ax[0].clear()
        self.ax[0].title.set_text('Training Score')
        self.ax[0].set_xlabel('Episode')
        self.ax[0].set_ylabel('Score')
        if len(x_groups) > 5:
            ep_avgs = np.mean(ep_groups, 1)
            avg_spl = interp1d(x_groups, ep_avgs, kind='cubic', fill_value="extrapolate")
            ep_std = np.std(ep_groups, 1)
            std_spl = interp1d(x_groups, ep_std, kind='cubic', fill_value="extrapolate")
            self.ax[0].plot(spline_x, avg_spl(spline_x), lw=0.7, c="blue")
            self.ax[0].fill_between(spline_x, avg_spl(spline_x)-std_spl(spline_x), avg_spl(spline_x)+std_spl(spline_x), alpha=0.5, facecolor="red", interpolate=True)

        self.ax[1].clear()
        self.ax[1].title.set_text('Training Height')
        self.ax[1].set_xlabel('Episode')
        self.ax[1].set_ylabel('Height')
        if len(x_groups) > 5:
            ep_heights = np.array(self.episode_height)
            ep_groups = [ep_heights[i * GROUP_NUM:(i + 1) * GROUP_NUM] for i in range((len(ep_heights) + GROUP_NUM - 1) // GROUP_NUM)]
            # Pad for weird numpy error for now
            ep_groups[-1] = np.append(ep_groups[-1], [np.mean(ep_groups[-1])] * (GROUP_NUM - len(ep_groups[-1])))
            x_groups = [i*GROUP_NUM for i in range(len(ep_groups))]

            ep_avgs = np.mean(ep_groups, 1)
            avg_spl = interp1d(x_groups, ep_avgs, kind='cubic', fill_value="extrapolate")
            ep_std = np.std(ep_groups, 1)
            std_spl = interp1d(x_groups, ep_std, kind='cubic', fill_value="extrapolate")
            self.ax[1].plot(spline_x, avg_spl(spline_x), lw=0.7, c="blue")
            self.ax[1].fill_between(spline_x, avg_spl(spline_x)-std_spl(spline_x), avg_spl(spline_x)+std_spl(spline_x), alpha=0.5, facecolor="red", interpolate=True)

        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(.001)


if __name__ == '__main__':
    agent = MountainCarGraphicalDDQN(2000)
    agent.train()
