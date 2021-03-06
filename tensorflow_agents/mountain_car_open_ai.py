# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import NamedTuple
from collections import namedtuple, deque
import random
from dqn_utils.dqn_models import Experience
from PIL import Image
import gym

import tensorflow as tf
from tensorflow import keras, Tensor


class Transition(NamedTuple):
    currStates: Tensor
    actions: Tensor
    rewards: Tensor
    nextStates: Tensor
    dones: Tensor


class DQNAgent:
    def __init__(self, stateShape, actionSpace, numPicks, memorySize):
        self.numPicks = numPicks
        self.replayMemory = deque(maxlen=memorySize)
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0
        self.sync = 200

        self.alpha = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.01
        self.gamma = 0.99

        self.trainNetwork = self.createNetwork(
            stateShape, actionSpace.n, self.alpha)
        self.targetNetwork = self.createNetwork(
            stateShape, actionSpace.n, self.alpha)

        self.targetNetwork.set_weights(self.trainNetwork.get_weights())

    def createNetwork(self, n_input, n_output, learningRate):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(
            24, activation='relu', input_shape=n_input))
        model.add(keras.layers.Dense(48, activation='relu'))
        model.add(keras.layers.Dense(n_output, activation='linear'))
        model.compile(
            loss='mse', optimizer=keras.optimizers.Adam(lr=learningRate))
        return model

    def trainDQN(self):
        if len(self.replayMemory) <= self.numPicks:
            return 0

        samples = random.sample(self.replayMemory, self.numPicks)
        batch = Transition(*zip(*samples))
        currStates, actions, rewards, nextStates, dones = batch

        currStates = np.squeeze(np.array(currStates), 1)
        Q_currents = self.trainNetwork.predict(currStates)

        nextStates = np.squeeze(np.array(nextStates), 1)
        Q_futures = self.targetNetwork.predict(nextStates).max(axis=1)

        rewards = np.array(rewards).reshape(self.numPicks,).astype(float)
        actions = np.array(actions).reshape(self.numPicks,).astype(int)

        dones = np.array(dones).astype(bool)
        notDones = (~dones).astype(float)
        dones = dones.astype(float)

        Q_currents[np.arange(self.numPicks), actions] = rewards * \
            dones + (rewards + Q_futures * self.gamma)*notDones

        hist = self.trainNetwork.fit(tf.convert_to_tensor(
            value=currStates), tf.convert_to_tensor(value=Q_currents), epochs=1, verbose=0)
        return hist.history['loss'][0]

    def selectAction(self, state):
        self.step += 1
        self.epsilon = max(self.epsilon-self.epsilon_decay, self.epsilon_min)

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

    def addMemory(self, memory):
        self.replayMemory.append(memory)

    def save(self, ep):
        save_path = (
            f"./mountain_car_{int(ep)}.chkpt"
        )
        self.trainNetwork.save(
            save_path
        )
        print(f"MountainNet saved to {save_path} done!")


class OpenAIMountainCar(object):
    def __init__(self, episodes):
        self.current_episode = 0
        self.episodes = episodes

        self.episode_score = []
        self.episode_qs = []
        self.episode_height = []
        self.episode_loss = []

        self.fig, self.ax = plt.subplots(2, 2)
        self.fig.canvas.draw()
        plt.show(block=False)

        self.env = gym.make('MountainCar-v0')
        self.agent = DQNAgent(stateShape=self.env.observation_space.shape,
                              actionSpace=self.env.action_space, numPicks=32, memorySize=20000)

    def train(self):
        for _ in range(self.episodes):
            self.episode()
            self.plot()
            self.current_episode += 1

        self.env.close()

    def episode(self):
        done = False
        rewardsSum = 0
        qSum = 0
        qActions = 1
        lossSum = 0

        state = self.env.reset().reshape(1, 2)
        maxHeight = -10000

        while not done:
            action, q = self.agent.selectAction(state)
            if q != -100000:
                qSum += q
                qActions += 1

            obs, reward, done, _ = self.env.step(action)
            self.env.render()


            maxHeight = max(obs[0], maxHeight)
            if obs[0] >= 0.5:
                reward += 10

            nextState = obs.reshape(1, 2)
            rewardsSum = np.add(rewardsSum, reward)

            loss = self.agent.trainDQN()
            self.agent.addMemory((state, action, reward, nextState, done))
            state = nextState
            lossSum += loss

        if rewardsSum != -200:
            self.agent.save(self.current_episode)

        print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(
            self.agent.epsilon, rewardsSum, lossSum, self.current_episode))

        self.episode_score.append(rewardsSum)
        self.episode_qs.append(qSum/qActions)
        self.episode_height.append(maxHeight)
        self.episode_loss.append(lossSum)

    def plot(self):
        self.ax[0][0].title.set_text('Training Score')
        self.ax[0][0].set_xlabel('Episode')
        self.ax[0][0].set_ylabel('Score')
        self.ax[0][0].plot(self.episode_score, 'b')

        self.ax[0][1].title.set_text('Training Height')
        self.ax[0][1].set_xlabel('Episode')
        self.ax[0][1].set_ylabel('Height')
        self.ax[0][1].plot(self.episode_height, 'g')

        self.ax[1][0].title.set_text('Training Loss')
        self.ax[1][0].set_xlabel('Episode')
        self.ax[1][0].set_ylabel('Loss')
        self.ax[1][0].plot(self.episode_loss, 'r')

        self.ax[1][1].title.set_text('Training Q Vals')
        self.ax[1][1].set_xlabel('Episode')
        self.ax[1][1].set_ylabel('Qs')
        self.ax[1][1].plot(self.episode_qs, 'c')
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(.001)
