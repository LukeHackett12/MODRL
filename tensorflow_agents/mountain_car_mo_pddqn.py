from copy import deepcopy
from tensorflow_agents.deep_sea_baseline_dqn import GROUP_NUM
import matplotlib.pyplot as plt
import matplotlib

from scipy.interpolate import interp1d
import numpy as np
from typing import NamedTuple
from collections import deque
import random
from custom_envs.mountain_car.engine import MountainCar

import tensorflow as tf

from tensorflow import keras, Tensor
from keras import backend as K

from replay_buffer import PrioritizedReplayBuffer

GROUP_NUM = 10


class Transition(NamedTuple):
    currStates: Tensor
    actions: Tensor
    rewards: Tensor
    nextStates: Tensor
    dones: Tensor


class DQNAgent(object):
    def __init__(self, stateShape, actionSpace, numPicks, memorySize, burnin=1000):
        self.numPicks = numPicks
        self.memorySize = memorySize
        self.replayMemory = PrioritizedReplayBuffer(memorySize, 0.6)
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0
        self.sync = 200
        self.burnin = burnin

        self.alpha = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.5
        self.epsilon_min = 0.01
        self.eps_threshold = 0

        self.gamma = 0.99

        self.trainNetwork = self.createNetwork(
            stateShape, len(actionSpace), self.alpha)
        self.targetNetwork = self.createNetwork(
            stateShape, len(actionSpace), self.alpha)
        self.targetNetwork.set_weights(
            self.trainNetwork.get_weights())

    def createNetwork(self, n_input, n_output, learningRate):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(
            24, activation='relu', input_shape=n_input))
        model.add(keras.layers.Dense(48, activation='relu'))
        model.add(keras.layers.Dense(n_output, activation='linear'))
        model.compile(
            loss='mse', optimizer=keras.optimizers.Adam(lr=learningRate))
        print(model.summary())
        return model

    def trainDQN(self):
        if len(self.replayMemory) <= self.numPicks or len(self.replayMemory) < self.burnin:
            return 0

        beta = 0.4 + self.step * (1.0 - 0.4) / 300
        samples = self.replayMemory.sample(self.numPicks, beta)
        #batch = Transition(*zip(*samples))
        currStates, actions, rewards, nextStates, dones, weights, indices = samples

        currStates = np.squeeze(np.array(currStates), 1)
        Q_currents = self.trainNetwork(currStates, training=False).numpy()

        nextStates = np.squeeze(np.array(nextStates), 1)
        Q_futures = self.targetNetwork(nextStates, training=False).numpy().max(axis=1)

        rewards = np.array(rewards).reshape(self.numPicks,).astype(float)
        actions = np.array(actions).reshape(self.numPicks,).astype(int)

        dones = np.array(dones).astype(bool)
        notDones = (~dones).astype(float)
        dones = dones.astype(float)

        Q_currents_cp = deepcopy(Q_currents)
        Q_currents_cp[np.arange(self.numPicks), actions] = rewards * dones + (rewards + Q_futures * self.gamma)*notDones

        loss = tf.multiply(tf.pow(tf.subtract(Q_currents[np.arange(self.numPicks), actions], Q_currents_cp[np.arange(self.numPicks), actions]), 2), weights).numpy()
        prios = loss + 1e-5
        self.replayMemory.update_priorities(indices, prios)

        loss = self.trainNetwork.train_on_batch(currStates, Q_currents)
        return loss

    def selectAction(self, state):
        self.step += 1

        if self.step % self.sync == 0:
            self.targetNetwork.set_weights(
                self.trainNetwork.get_weights())

        q = -100000
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            preds = np.squeeze(self.trainNetwork(
                state, training=False).numpy(), axis=0)
            action = np.argmax(preds)
            q = preds[action]
        return action, q

    def addMemory(self, state, action, reward, nextState, done):
        self.replayMemory.add(state, action, reward, nextState, done)

    def save(self):
        save_path = (
            f"./mountain_car_tfngmo_{int(self.step)}.chkpt"
        )
        '''self.trainNetwork.save(
            save_path
        )'''
        print(f"MountainNet saved to {save_path} done!")


class MultiObjectiveMountainCarPDDQN(object):
    def __init__(self, episodes):
        self.current_episode = 0
        self.episodes = episodes

        self.episode_score = []
        self.episode_qs = []
        self.episode_height = []
        self.episode_loss = []

        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.canvas.draw()
        plt.show(block=False)

        self.env = MountainCar(speed=1e8, graphical_state=False,
                               render=True, is_debug=True, random_starts=True)
        self.agent = DQNAgent(stateShape=(
            2,), actionSpace=self.env.get_action_space(), numPicks=32, memorySize=10000)

    def train(self):
        for _ in range(self.episodes):
            self.episode()
            self.plot()
            self.current_episode += 1
        plt.show(block=True)

    def episode(self):
        done = False
        rewardsSum = 0
        qSum = 0
        qActions = 1
        lossSum = 0

        state = self.env.reset().reshape(1, 2)
        maxHeight = -10000
        win = False

        while not done:
            action, q = self.agent.selectAction(state)
            if q != -100000:
                qSum += q
                qActions += 1

            obs, reward, done, _ = self.env.step_all(action)
            # env.render()

            reward = reward[0]

            maxHeight = max(obs[0], maxHeight)
            if obs[0] >= 0.5:
                win = True
                reward += 10

            nextState = obs.reshape(1, 2)
            rewardsSum = np.add(rewardsSum, reward)

            loss = self.agent.trainDQN()
            self.agent.addMemory(state, action, reward, nextState, done)
            state = nextState
            lossSum += loss

        if win:
            self.agent.save()

        self.agent.epsilon = max(self.agent.epsilon-self.agent.epsilon_decay, self.agent.epsilon_min)
        print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(
            self.agent.epsilon, rewardsSum, lossSum, self.current_episode))

        self.episode_score.append(rewardsSum)
        self.episode_qs.append(qSum/qActions)
        self.episode_height.append(maxHeight)
        self.episode_loss.append(lossSum)

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

        ep_heights = np.array(self.episode_height)
        ep_groups = [ep_heights[i * GROUP_NUM:(i + 1) * GROUP_NUM] for i in range((len(ep_heights) + GROUP_NUM - 1) // GROUP_NUM)]
        # Pad for weird numpy error for now
        ep_groups[-1] = np.append(ep_groups[-1], [np.mean(ep_groups[-1])] * (GROUP_NUM - len(ep_groups[-1])))
        x_groups = [i*GROUP_NUM for i in range(len(ep_groups))]

        self.ax[1].clear()
        if len(x_groups) > 5:
            ep_avgs = np.mean(ep_groups, 1)
            avg_spl = interp1d(x_groups, ep_avgs, kind='cubic', fill_value="extrapolate")
            ep_std = np.std(ep_groups, 1)
            std_spl = interp1d(x_groups, ep_std, kind='cubic', fill_value="extrapolate")
            self.ax[1].plot(spline_x, avg_spl(spline_x), lw=0.7, c="blue")
            self.ax[1].fill_between(spline_x, avg_spl(spline_x)-std_spl(spline_x), avg_spl(spline_x)+std_spl(spline_x), alpha=0.5, facecolor="red", interpolate=True)

        self.ax[1].title.set_text('Training Height')
        self.ax[1].set_xlabel('Episode')
        self.ax[1].set_ylabel('Height')

        plt.show(block=False)
        plt.pause(.001)
