# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import keyboard
from typing import NamedTuple
from collections import namedtuple, deque
import random
import enum
from PIL import Image
from torchvision import transforms as T
import torch
from dqn_utils.dqn_models import Experience
from custom_envs.mountain_car.engine import MountainCar

import tensorflow as tf
from tensorflow import keras, Tensor
from keras import backend as K
K.set_image_dim_ordering('th')


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class Transition(NamedTuple):
    currStates: Tensor
    actions: Tensor
    rewards: Tensor
    nextStates: Tensor
    dones: Tensor


class DQN():
    def __init__(self, n_input, n_output, learningRate):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(32, input_shape=np.array(
            n_input), kernel_size=8, strides=4, activation='relu'))
        self.model.add(keras.layers.Conv2D(
            32, kernel_size=4, strides=2, activation='relu'))
        self.model.add(keras.layers.Conv2D(
            64, kernel_size=3, strides=1, activation='relu'))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(512, activation='relu'))
        self.model.add(keras.layers.Dense(n_output, activation='linear'))
        self.model.compile(
            loss='mse', optimizer=keras.optimizers.Adam(lr=learningRate))


class DQNAgent:
    def __init__(self, stateShape, actionSpace, numPicks, memorySize):
        self.numPicks = numPicks
        self.memorySize = memorySize
        self.replayMemory = deque(maxlen=memorySize)
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0
        self.sync = 100

        self.alpha = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.001
        self.epsilon_min = 0.01
        self.eps_threshold = 0

        self.gamma = 0.99
        self.tau = 0.01

        self.trainNetwork = DQN(stateShape, len(actionSpace), self.alpha)
        self.targetNetwork = DQN(stateShape, len(actionSpace), self.alpha)
        self.targetNetwork.model.set_weights(
            self.trainNetwork.model.get_weights())

    def trainDQN(self):
        if len(self.replayMemory) <= self.numPicks:
            return 0

        samples = random.sample(self.replayMemory, self.numPicks)
        batch = Transition(*zip(*samples))
        currStates, actions, rewards, nextStates, dones = batch

        currStates = np.squeeze(np.array(currStates), 1)
        Q_currents = self.trainNetwork.model.predict(currStates)

        nextStates = np.squeeze(np.array(nextStates), 1)
        Q_futures = self.targetNetwork.model.predict(nextStates).max(axis=1)

        rewards = np.array(rewards).reshape(self.numPicks,).astype(float)
        actions = np.array(actions).reshape(self.numPicks,).astype(int)

        dones = np.array(dones).astype(bool)
        notDones = (~dones).astype(float)
        dones = dones.astype(float)

        Q_currents[np.arange(self.numPicks), actions] = rewards * \
            dones + (rewards + Q_futures * self.gamma)*notDones

        hist = self.trainNetwork.model.fit(tf.convert_to_tensor(
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
            preds = np.squeeze(self.trainNetwork.model(
                state, training=False).numpy(), axis=0)
            action = np.argmax(preds)
            q = preds[action]
        return action, q

    def addMemory(self, memory):
        self.replayMemory.append(memory)

    def save(self):
        save_path = (
            f"./mountain_car_{int(self.step)}.chkpt"
        )
        self.trainNetwork.model.save(
            save_path
        )
        print(f"MountainNet saved to {save_path} done!")


class MultiObjectiveMountainCarGraphical(object):
    def __init__(self, episodes):
        self.current_episode = 0
        self.episodes = episodes

        self.episode_score = []
        self.episode_qs = []
        self.episode_height = []
        self.episode_loss = []

        self.fig, self.ax = plt.subplots(2, 2)
        self.fig.canvas.draw()
        self.plt.show(block=False)

        self.env = MountainCar(
            speed=1e8, graphical_state=True, render=True, is_debug=True)
        self.agent = DQNAgent(stateShape=(40, 40, 3),
                              actionSpace=self.env.get_action_space(), numPicks=32, memorySize=10000)

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

        currState = self.process_screen(self.env.reset())
        lastState = currState
        state = currState - lastState
        state = tf.expand_dims(tf.constant(state), 0)

        maxHeight = -10000

        while not done:
            action, q = self.agent.selectAction(state)
            if q != -100000:
                qSum += q
                qActions += 1

            obs, reward, done, info = self.env.step_all(action)

            reward = reward[0]

            maxHeight = max(info[0], maxHeight)
            if info[0] >= 0.5:
                reward += 10

            lastState = currState
            currState = self.process_screen(obs)

            nextState = currState - lastState
            nextState = tf.expand_dims(tf.constant(nextState), 0)
            rewardsSum = np.add(rewardsSum, reward)

            loss = self.agent.trainDQN()
            self.agent.addMemory((state, action, reward, nextState, done))
            state = nextState
            lossSum += loss

        if rewardsSum != -202:
            self.agent.save()

        print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(
            self.agent.epsilon, rewardsSum, lossSum, self.current_episode))

        self.episode_score.append(rewardsSum)
        self.episode_qs.append(qSum/qActions)
        self.episode_height.append(maxHeight)
        self.episode_loss.append(lossSum)

    def process_screen(observation):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = observation
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((40, 40), interpolation=Image.CUBIC),
                            T.ToTensor()])

        return resize(screen).numpy().transpose((2, 1, 0))

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
