from os import times

from gym.wrappers import frame_stack
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
    rewards: Tensor
    nextStates: Tensor
    dones: Tensor


class DQNAgent(object):
    def __init__(self, stateShape, actionSpace, numPicks, memorySize, numRewards, sync=100, burnin=500, alpha=0.0001, epsilon=1, epsilon_decay=0.99975, epsilon_min=0.01, gamma=0.99):
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

        self.trainNetwork = self.createNetwork(
            stateShape, len(actionSpace), self.alpha)

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

        samples = random.sample(self.replayMemory, self.numPicks)#self.replayMemory.sample(self.numPicks, self.beta)  
        batch = Transition(*zip(*samples))
        currStates, actions, rewards, nextStates, dones = batch

        currStates = np.array(currStates).transpose(0,2,3,1)
        Q_currents = self.trainNetwork(currStates, training=False).numpy()

        nextStates = np.array(nextStates).transpose(0,2,3,1)
        Q_futures = self.trainNetwork(nextStates, training=False).numpy().max(axis=1)

        rewards = np.array(rewards).reshape(self.numPicks,).astype(float)
        actions = np.array(actions).reshape(self.numPicks,).astype(int)

        dones = np.squeeze(np.array(dones)).astype(bool)
        notDones = (~dones).astype(float)
        dones = dones.astype(float)

        Q_currents[np.arange(self.numPicks), actions] = rewards * dones + (rewards + Q_futures * self.gamma)*notDones
        loss = self.trainNetwork.train_on_batch(currStates, Q_currents)
        return loss

    def selectAction(self, state):
        self.step += 1
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

        q = -100000
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            preds = np.squeeze(self.trainNetwork(
                np.expand_dims(np.array(state).transpose(1,2,0), 0), training=False).numpy(), axis=0)
            action = np.argmax(preds)
            q = preds[action]
        return action, q

    def addMemory(self, memory):
        self.replayMemory.append(memory)

    def save(self):
        save_path = (
            f"./mountain_car_wnet_{int(self.step)}.chkpt"
        )
        print(f"MountainNet saved to {save_path} done!")


class DeepSeaTreasureGraphicalDQN(object):
    def __init__(self, episodes):
        self.current_episode = 0
        self.episodes = episodes

        self.episode_score = []
        self.episode_qs = []
        self.episode_height = []
        self.episode_loss = []
        self.episode_policies = []

        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.fig.canvas.draw()
        plt.show(block=False)

        self.numRewards = 2

        self.env = DeepSeaTreasure(width=5, speed=10000, graphical_state=True, render=True, is_debug=False, frame_stack=2)
        self.agent = DQNAgent(stateShape=(84, 84, 2), actionSpace=self.env.get_action_space(), numPicks=32, memorySize=10000, numRewards=self.numRewards)

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
        qSums = [0] * (self.numRewards)
        actions = 1

        state = self.env.reset()
        maxHeight = -1

        while not done:
            action, qs = self.agent.selectAction(state)
            if qs != -100000:
                qSums += qs
                actions += 1

            obs, reward, done, _ = self.env.step_all(action)

            nextState = obs
            rewardsSum = np.add(rewardsSum, sum(reward))

            self.agent.addMemory((state, action, (reward[0]+reward[1]), nextState, done))
            state = nextState

            loss = self.agent.trainDQN()
            lossSum += loss

        print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(
            self.agent.epsilon, rewardsSum, lossSum, self.current_episode))

        self.episode_score.append(rewardsSum)
        self.episode_height.append(maxHeight)
        self.episode_loss.append(lossSum)
        self.episode_qs.append([qSum/actions for qSum in qSums])
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

        self.ax.clear()
        if len(x_groups) > 5:
            ep_avgs = np.mean(ep_groups, 1)
            avg_spl = interp1d(x_groups, ep_avgs, kind='cubic', fill_value="extrapolate")
            ep_std = np.std(ep_groups, 1)
            std_spl = interp1d(x_groups, ep_std, kind='cubic', fill_value="extrapolate")
            self.ax.plot(spline_x, avg_spl(spline_x), lw=0.7, c="blue")
            self.ax.fill_between(spline_x, avg_spl(spline_x)-std_spl(spline_x), avg_spl(spline_x)+std_spl(spline_x), alpha=0.5, facecolor="red", interpolate=True)

        self.ax.title.set_text('Training Score')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Score')
        
        '''
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
        '''
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(.001)


if __name__ == '__main__':
    agent = DeepSeaTreasureGraphicalDDQN(1000)
    agent.train()
