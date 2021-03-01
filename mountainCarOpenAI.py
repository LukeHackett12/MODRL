# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from PIL import Image
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import enum
import random

import math
from collections import namedtuple, deque
from typing import NamedTuple

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
BoolTensor = torch.cuda.ByteTensor if use_cuda else torch.BoolTensor

class Transition(NamedTuple):
    currStates: FloatTensor
    actions: int
    rewards: FloatTensor
    nextStates: FloatTensor
    dones: BoolTensor


'''

        model.add(keras.layers.Dense(
            24, activation='relu', input_shape=state_shape))
        model.add(keras.layers.Dense(48, activation='relu'))
        model.add(keras.layers.Dense(self.env.action_space.n,activation='linear'))
'''

class DQN(nn.Module):
  def __init__(self, n_input, n_output):
      super(DQN, self).__init__()

      self.hidden_layer_1 = nn.Linear(n_input, 100)
      self.hidden_layer_2 = nn.Linear(12, 24)
      self.output_layer = nn.Linear(100, n_output)

  def forward(self, x):
      out = F.relu(self.hidden_layer_1(x))
      out = self.output_layer(out)
      return out

class DQNAgent:
    def __init__(self, stateShape, actionSpace, numPicks, memorySize):
        self.numPicks = numPicks
        self.memorySize = memorySize
        self.replayMemory = deque(maxlen=numPicks)
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0

        self.sync = 1
        self.alpha = 1e-3
        self.epsilon = 1
        self.epsilon_decay = 0.02
        self.epsilon_min=0.01
        self.gamma = 0.99

        self.trainNetwork = DQN(2, actionSpace.n).to(device)
        self.targetNetwork = DQN(2, actionSpace.n).to(device)
        
        self.targetNetwork.load_state_dict(self.trainNetwork.state_dict())

        self.trainNetwork.train()
        self.targetNetwork.eval()
        
        for p in self.trainNetwork.parameters():
            p.requires_grad = True

        self.optimizer = optim.Adam(self.trainNetwork.parameters(), self.alpha)

    def trainDQN(self):
        if self.step <= self.numPicks:
            return 1
        
        samples = random.sample(self.replayMemory, self.numPicks)
        batch = zip(*samples)
        currState,action,reward,nextState,done = batch

        reward = torch.cat(reward).to(device)
        done = torch.cat(done).to(device)
        currState = torch.cat(currState).to(device)
        nextState = torch.cat(nextState).to(device)
        action = torch.cat(action).to(device)

        Q_current = self.trainNetwork(currState).gather(1, action).squeeze(1)
        Q_futures = self.targetNetwork(nextState).detach().max(1)[0]

        Q_next = ((reward*done) + (reward + Q_futures * self.gamma)*~done).to(device)

        loss = F.mse_loss(Q_current, Q_next)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainNetwork.parameters(), 1)
        self.optimizer.step()

        return loss.cpu().detach().numpy().max()

    def selectAction(self, currState):
      self.step += 1
      
      q = -100000
      if np.random.rand(1) < 0:
          action = self.actionSpace.sample()
      else:
          currState = currState.to(device)
          with torch.no_grad():
              self.trainNetwork.eval()
              preds = self.trainNetwork(currState).squeeze(0).detach().cpu().numpy()
              action = np.argmax(preds)
              q = preds[action]
              self.trainNetwork.train()
      return LongTensor([[action]]), q

    def addMemory(self, memory):
        self.replayMemory.append(memory)

    def save(self):
        save_path = (
            f"./mountain_car_openai_torch_{int(self.step)}.chkpt"
        )
        torch.save(
            dict(model=self.trainNetwork.state_dict(), epsilon=self.epsilon),
            save_path,
        )
        print(f"mountain_car saved to {save_path} done!")

episode_score = []
episode_qs = []
episode_height = []
episode_loss = []
episode_decay = []

fig, ax = plt.subplots(2, 2)
fig.canvas.draw()
plt.show(block=False)

def plot_episode():
    heights_t = torch.tensor(episode_score, dtype=torch.float)
    ax[0][0].title.set_text('Training Score')
    ax[0][0].set_xlabel('Episode')
    ax[0][0].set_ylabel('Score')
    ax[0][0].plot(heights_t.numpy(), 'b')

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

env = gym.make('MountainCar-v0')
agent = DQNAgent(stateShape=env.observation_space.shape[0],
                   actionSpace=env.action_space, numPicks=256, memorySize=8000)

def episode():
    done = False
    rewardsSum = 0
    qSum = 0
    qActions = 1
    lossSum = 0

    state = env.reset()
    stateT = FloatTensor([state]).to(device)
    maxHeight = -100

    while not done:
        action, q = agent.selectAction(stateT)
        if q != -100000:
            qSum += q
            qActions += 1

        obs, reward, done, _ = env.step(action.numpy()[0, 0])

        maxHeight = max(obs[0], maxHeight)
        if obs[0] >= 0.5:
            reward += 10

        env.render()

        nextState = FloatTensor([obs]).to(device)
        rewardT =FloatTensor([reward]).to(device)
        doneT = BoolTensor([done]).to(device)

        rewardsSum = np.add(rewardsSum, reward)

        loss = agent.trainDQN()
        lossSum += loss
        agent.addMemory((stateT, action, rewardT, nextState, doneT))
        stateT = nextState

    episode_score.append(rewardsSum)
    episode_qs.append(qSum/qActions)
    episode_height.append(maxHeight)
    episode_loss.append(lossSum)
    episode_decay.append(agent.epsilon)
    plot_episode()

    if ep % agent.sync == 0:
        agent.targetNetwork.load_state_dict(agent.trainNetwork.state_dict())

    if ep % agent.sync == 0:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon - agent.epsilon_decay)

    if rewardsSum != -200 and ep > 300:
        agent.save()

    print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(agent.epsilon, rewardsSum, lossSum, ep)) 

ep = 1
while ep < 10000:
    episode()
    ep += 1

env.close()
