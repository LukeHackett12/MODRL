from os import times

from gym.wrappers import frame_stack
from tensorflow_agents.deep_sea_baseline_ddqn import GROUP_NUM
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pylab as pl
import pickle
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

from replay_buffer import PrioritizedReplayBuffer

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
    def __init__(
        self,
        stateShape,
        actionSpace,
        numPicks,
        memorySize,
        numRewards,
        sync=50,
        burnin=0,#500,
        alpha=0.0001,
        epsilon=1,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        gamma=0.99,
    ):
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

        self.numRewards = numRewards

        self.trainNetwork = self.createNetwork(stateShape, len(actionSpace), self.alpha)
        self.targetNetwork = self.createNetwork(
            stateShape, len(actionSpace), self.alpha
        )
        self.targetNetwork.set_weights(self.trainNetwork.get_weights())

    def createNetwork(self, n_input, n_output, learningRate):
        model = keras.models.Sequential()

        model.add(
            keras.layers.experimental.preprocessing.Rescaling(
                1.0 / 255, input_shape=n_input
            )
        )
        model.add(keras.layers.Conv2D(32, kernel_size=8, strides=4, activation="relu"))
        model.add(keras.layers.Conv2D(64, kernel_size=4, strides=2, activation="relu"))
        model.add(keras.layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation="linear"))
        model.add(keras.layers.Dense(n_output, activation="linear"))

        model.compile(
            loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=learningRate)
        )
        print(model.summary())
        return model

    def trainDQN(self):
        if self.step <= self.numPicks or len(self.replayMemory) <= self.burnin:
            return 0

        self.beta = 0.4 + self.step * (1.0 - 0.4) / 30000
        samples = self.replayMemory.sample(self.numPicks, self.beta)
        currStates, actions, rewards, nextStates, dones, weights, indices = samples

        currStates = np.array(currStates).transpose(0, 2, 3, 1)
        Q_currents = self.trainNetwork(currStates, training=False).numpy()

        nextStates = np.array(nextStates).transpose(0, 2, 3, 1)
        Q_futures = self.targetNetwork(nextStates, training=False).numpy().max(axis=1)

        rewards = (
            np.array(rewards)
            .reshape(
                self.numPicks,
            )
            .astype(float)
        )
        actions = (
            np.array(actions)
            .reshape(
                self.numPicks,
            )
            .astype(int)
        )

        dones = np.squeeze(np.array(dones)).astype(bool)
        notDones = (~dones).astype(float)
        dones = dones.astype(float)

        Q_currents_cp = deepcopy(Q_currents)
        Q_currents_cp[np.arange(self.numPicks), actions] = (
            rewards + Q_futures * self.gamma * notDones
        )

        h = tf.keras.losses.Huber()
        loss = h(
            Q_currents[np.arange(self.numPicks), actions],
            Q_currents_cp[np.arange(self.numPicks), actions],
        )
        prios = (np.abs(loss) * weights) + 1e-5
        self.replayMemory.update_priorities(indices, prios)

        loss = self.trainNetwork.train_on_batch(currStates, Q_currents_cp)
        return loss

    def selectAction(self, state):
        self.step += 1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        q = -100000
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            preds = np.squeeze(
                self.trainNetwork(
                    np.expand_dims(np.array(state).transpose(1, 2, 0), 0),
                    training=False,
                ).numpy(),
                axis=0,
            )
            action = np.argmax(preds)
            q = preds[action]
        return action, q

    def addMemory(self, state, action, reward, nextState, done):
        self.replayMemory.add(state, action, reward, nextState, done)

    def save(self):
        save_path = f"./dst_net_{int(self.step)}.chkpt"
        train_w = self.trainNetwork.get_weights()
        target_w = self.trainNetwork.get_weights()

        with open(save_path, "wb") as f:
            pickle.dump([train_w, target_w], f)

        print(f"DSTNet saved to {save_path} done!")

    def load(self):
        save_path = "./dst_net_mixed.chkpt"
        with open(save_path, "rb") as f:
            weights = pickle.load(f)

        self.trainNetwork.set_weights(weights[0])
        self.trainNetwork.set_weights(weights[1])


class DeepSeaTreasureGraphicalPDDQN(object):
    def __init__(self, episodes):
        self.current_episode = 0
        self.episodes = episodes

        self.episode_score = []
        self.episode_qs = []
        self.episode_height = []
        self.episode_loss = []
        self.episode_policies = []

        self.fig, self.ax = plt.subplots(figsize=(6, 4))

        self.numRewards = 2

        self.env = DeepSeaTreasure(
            width=5,
            speed=10000000,
            graphical_state=True,
            render=False,
            is_debug=True,
            frame_stack=2,
            reshape_reward_weights=[[1, 1]],
            seed=1234
        )
        self.agent = DQNAgent(
            stateShape=(84, 84, 2),
            actionSpace=self.env.get_action_space(),
            numPicks=32,
            memorySize=10000,
            numRewards=self.numRewards,
        )
        self.agent.load()

    def train(self):
        for _ in range(self.episodes):
            self.episode()
            self.current_episode += 1

        self.plot()
        self.agent.save()

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

            obs, reward, done, _ = self.env.step_all(action)

            nextState = obs
            rewardsSum = np.add(rewardsSum, sum(reward))

            self.agent.addMemory(
                state, action, (reward[0] + reward[1]), nextState, done
            )
            state = nextState

            loss = self.agent.trainDQN()
            lossSum += loss

        if self.current_episode % self.agent.sync == 0:
            self.agent.targetNetwork.set_weights(self.agent.trainNetwork.get_weights())

        print(
            "now epsilon is {}, the reward is {} with loss {} in episode {}".format(
                self.agent.epsilon, rewardsSum, lossSum, self.current_episode
            )
        )

        self.episode_score.append(rewardsSum)
        self.episode_height.append(maxHeight)
        self.episode_loss.append(lossSum)
        self.episode_qs.append(qSums / actions)

        print(
            "Report: \nrewardSum:{}\nheight:{}\nloss:{}\nqAverage:{}".format(
                self.episode_score[-1],
                self.episode_height[-1],
                self.episode_loss[-1],
                self.episode_qs[-1],
            )
        )

    def plot(self):
        spline_x = np.linspace(0, self.current_episode, num=self.current_episode)

        ep_scores = np.array(self.episode_score)
        ep_groups = [
            ep_scores[i * GROUP_NUM : (i + 1) * GROUP_NUM]
            for i in range((len(ep_scores) + GROUP_NUM - 1) // GROUP_NUM)
        ]
        # Pad for weird numpy error for now
        ep_groups[-1] = np.append(
            ep_groups[-1], [np.mean(ep_groups[-1])] * (GROUP_NUM - len(ep_groups[-1]))
        )
        x_groups = [i * GROUP_NUM for i in range(len(ep_groups))]

        self.ax.clear()
        if len(x_groups) > 5:
            ep_avgs = np.mean(ep_groups, 1)
            avg_spl = interp1d(
                x_groups, ep_avgs, kind="cubic", fill_value="extrapolate"
            )
            ep_std = np.std(ep_groups, 1)
            std_spl = interp1d(x_groups, ep_std, kind="cubic", fill_value="extrapolate")
            self.ax.plot(spline_x, avg_spl(spline_x), lw=0.7, c="blue")
            self.ax.fill_between(
                spline_x,
                avg_spl(spline_x) - std_spl(spline_x),
                avg_spl(spline_x) + std_spl(spline_x),
                alpha=0.5,
                facecolor="red",
                interpolate=True,
            )

        self.ax.title.set_text("Training Score")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Score")
        self.fig.canvas.draw()
        plt.savefig("dst_pddqn_retrain.png")


if __name__ == "__main__":
    agent = DeepSeaTreasureGraphicalPDDQN(2500)
    agent.train()
