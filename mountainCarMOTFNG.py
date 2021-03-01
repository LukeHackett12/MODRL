# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import NamedTuple
from collections import deque
import random
from custom_envs.mountain_car.engine import MountainCar

import tensorflow as tf
from tensorflow import keras, Tensor
from keras import backend as K
tf.enable_eager_execution()

class Transition(NamedTuple):
    currStates: Tensor
    actions: Tensor
    rewards: Tensor
    nextStates: Tensor
    dones: Tensor

class DQNAgent:
    def __init__(self, stateShape, actionSpace, numPicks, memorySize):
        self.numPicks = numPicks
        self.memorySize = memorySize
        self.replayMemory = deque(maxlen=memorySize)
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0
        self.sync = 1

        self.alpha = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.01
        self.eps_threshold = 0

        self.gamma = 0.99
        self.tau = 0.01

        self.trainNetwork = self.createNetwork(stateShape, len(actionSpace), self.alpha)
        self.targetNetwork = self.createNetwork(stateShape, len(actionSpace), self.alpha)
        self.targetNetwork.set_weights(
            self.trainNetwork.get_weights())

    def createNetwork(self, n_input, n_output, learningRate):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(24, activation='relu', input_shape=n_input))
        model.add(keras.layers.Dense(48, activation='relu'))
        model.add(keras.layers.Dense(n_output,activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learningRate))
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
        Q_futures = self.targetNetwork.predict(nextStates).max(axis = 1)

        rewards = np.array(rewards).reshape(self.numPicks,).astype(float)
        actions = np.array(actions).reshape(self.numPicks,).astype(int)

        dones = np.array(dones).astype(bool)
        notDones = (~dones).astype(float)
        dones = dones.astype(float)

        Q_currents[np.arange(self.numPicks), actions] = rewards*dones + (rewards + Q_futures * self.gamma)*notDones

        hist = self.trainNetwork.fit(tf.convert_to_tensor(currStates), tf.convert_to_tensor(Q_currents), epochs=1, verbose=0)  
        return hist.history['loss'][0]

    def selectAction(self, state):
        self.step += 1
        self.epsilon = max(self.epsilon, self.epsilon_min)

        q = -100000
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            preds = np.squeeze(self.trainNetwork(state, training=False).numpy(), axis=0)
            action = np.argmax(preds)
            q = preds[action]
        return action, q

    def addMemory(self, memory):
        self.replayMemory.append(memory)

    def save(self):
        save_path = (
            f"./mountain_car_tfngmo_{int(self.step)}.chkpt"
        )
        agent.trainNetwork.save(
            save_path
        )
        print(f"MountainNet saved to {save_path} done!")


episode_score = []
episode_qs = []
episode_height = []
episode_loss = []
episode_decay = []

fig, ax = plt.subplots(2, 2)
fig.canvas.draw()
plt.show(block=False)

def plot_episode():
    ax[0][0].title.set_text('Training Score')
    ax[0][0].set_xlabel('Episode')
    ax[0][0].set_ylabel('Score')
    ax[0][0].plot(episode_score, 'b')

    ax[0][1].title.set_text('Training Height')
    ax[0][1].set_xlabel('Episode')
    ax[0][1].set_ylabel('Height')
    ax[0][1].plot(episode_height, 'g')

    ax[1][0].title.set_text('Training Loss')
    ax[1][0].set_xlabel('Episode')
    ax[1][0].set_ylabel('Loss')
    ax[1][0].plot(episode_loss, 'r')

    ax[1][1].title.set_text('Training Q Vals')
    ax[1][1].set_xlabel('Episode')
    ax[1][1].set_ylabel('Qs')
    ax[1][1].plot(episode_qs, 'c')
    fig.canvas.draw()
    plt.show(block=False)
    plt.pause(.001)

env = MountainCar(speed=1e8, graphical_state=False, render=False, is_debug=True, random_starts=True)
agent = DQNAgent(stateShape=(2,), actionSpace=env.get_action_space(), numPicks=32, memorySize=10000)

def episode():
    done = False
    rewardsSum = 0
    qSum = 0
    qActions = 1
    lossSum = 0

    state = env.reset().reshape(1,2)
    maxHeight = -10000

    while not done:
        action, q = agent.selectAction(state)
        if q != -100000:
            qSum += q
            qActions += 1

        obs, reward, done, _ = env.step_all(action)
        #env.render()

        reward = reward[0]

        maxHeight = max(obs[0], maxHeight)
        if obs[0] >= 0.5:
            reward += 10

        nextState = obs.reshape(1,2)
        rewardsSum = np.add(rewardsSum, reward)

        loss = agent.trainDQN()
        agent.addMemory((state, action, reward, nextState, done))
        state = nextState
        lossSum += loss

    episode_score.append(rewardsSum)
    episode_qs.append(qSum/qActions)
    episode_height.append(maxHeight)
    episode_loss.append(lossSum)
    episode_decay.append(agent.epsilon)
    plot_episode()

    agent.epsilon -= agent.epsilon_decay

    if ep % agent.sync == 0:
        agent.targetNetwork.set_weights(
            agent.trainNetwork.get_weights())

    if rewardsSum != -202:
        agent.save()

    print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(
        agent.epsilon, rewardsSum, lossSum, ep))


ep = 1
while ep < 10000:
    episode()
    ep += 1

env.close()
