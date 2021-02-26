# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from PIL import Image
import gym

import torch
from torch import FloatTensor, LongTensor, BoolTensor
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


class StatusEnum(enum.Enum):
    small = 1
    tall = 2
    fireball = 3


class Transition(NamedTuple):
    currStates: FloatTensor
    actions: LongTensor
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

      self.hidden_layer_1 = nn.Linear(n_input, 24, bias=False)
      nn.init.kaiming_uniform_(self.hidden_layer_1.weight, nonlinearity='relu')
      self.hidden_layer_2 = nn.Linear(24, 48, bias=False)
      nn.init.kaiming_uniform_(self.hidden_layer_2.weight, nonlinearity='relu')
      self.output_layer = nn.Linear(48, n_output, bias=False)

  def forward(self, x):
      out = F.relu(self.hidden_layer_1(x))
      out = F.relu(self.hidden_layer_2(out))
      out = self.output_layer(out)
      return F.softmax(out, dim=1)

class DQNAgent:
    def __init__(self, stateShape, actionSpace, numPicks, memorySize):
        self.numPicks = numPicks
        self.memorySize = memorySize
        self.replayMemory = deque(maxlen=numPicks)
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0

        self.sync = 1
        self.alpha = 0.0001
        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min=0.001
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
        batch = Transition(*zip(*samples))
        currState,action,reward,nextState,done = batch

        reward = torch.flatten(FloatTensor(reward)).to(device)
        done = torch.flatten(BoolTensor(done)).to(device)

        currState = torch.cat(currState).to(device)
        action = torch.cat(action).to(device)

        Q_current = self.trainNetwork(currState).squeeze(1).gather(1, action.unsqueeze(1))
        
        nextStateNonFinal = torch.cat([s for s in nextState
                                                if s is not None])
        nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None,
                                          nextState)), dtype=torch.bool).to(device)

        Q_futures = torch.zeros(self.numPicks, device=device)
        Q_futures[nonFinalMask] = self.targetNetwork(nextStateNonFinal).max(1)[0]

        Q_next = (reward + Q_futures * self.gamma).unsqueeze(1).to(device)

        self.optimizer.zero_grad()
        loss_fn = nn.MSELoss()
        loss = loss_fn(Q_current, Q_next)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainNetwork.parameters(), 1)
        self.optimizer.step()

        return loss.cpu().detach().numpy().max()

    def selectAction(self, currState):
      self.step += 1
      
      q = -100000
      if np.random.rand(1) < self.epsilon:
          action = self.actionSpace.sample()
      else:
          currState = currState.to(device)
          with torch.no_grad():
              self.trainNetwork.eval()
              preds = self.trainNetwork(currState).squeeze(0).detach().cpu().numpy()
              action = np.argmax(preds)
              q = preds[action]
              self.trainNetwork.train()
      return action, q

    def addMemory(self, memory, loss):
        self.replayMemory.append(memory)

    def save(self):
        save_path = (
            f"./mountain_car_{int(self.step)}.chkpt"
        )
        torch.save(
            dict(model=self.trainNetwork.state_dict(), epsilon=self.epsilon),
            save_path,
        )
        print(f"MarioNet saved to {save_path} done!")

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
                   actionSpace=env.action_space, numPicks=32, memorySize=10000)

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

        obs, reward, done, _ = env.step(action)

        maxHeight = max(obs[0], maxHeight)
        if obs[0] >= 0.5:
            reward += 10

        env.render()

        if not done:
            nextState = FloatTensor([obs]).to(device)
        else:
            nextState = None

        actionT = LongTensor([action]).to(device)
        rewardT =FloatTensor([reward]).to(device)
        doneT = BoolTensor([done]).to(device)

        rewardsSum = np.add(rewardsSum, reward)

        loss = agent.trainDQN()
        lossSum += loss
        agent.addMemory((stateT, actionT, rewardT, nextState, doneT), loss)
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

    if ep % 150 == 0:
        agent.save()

    print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(agent.epsilon, rewardsSum, lossSum, ep)) 

ep = 1
while ep < 10000:
    episode()
    ep += 1

env.close()
