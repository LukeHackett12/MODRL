#from nes_py.wrappers import JoypadSpace
#import gym_super_mario_bros
#from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from custom_envs.mountain_car.engine import MountainCar

import torch
from torch import FloatTensor, LongTensor, BoolTensor
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T

import enum
import random
from collections import namedtuple, deque
from typing import NamedTuple

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

from PIL import Image
from dqn_utils.dqn_models import Experience

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


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self._input_shape = input_shape
        self._num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x).view(x.size()[0], -1)
        return self.fc(x)
    
    @property
    def feature_size(self):
        x = self.features(torch.zeros(1, *self._input_shape))
        return x.view(1, -1).size(1)

class DQNAgent:
    def __init__(self, stateShape, actionSpace, numPicks, memorySize):
        self.numPicks = numPicks
        self.memorySize = memorySize
        self.replayMemory = Experience(memorySize, numPicks, 0.9)
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0
        self.sync = 10

        self.alpha = 0.00025
        self.epsilon = 0.99
        self.epsilon_decay = 0.9999975
        self.epsilon_min=0.05
        self.eps_threshold = 0

        self.gamma = 0.999
        self.tau = 1e-3

        self.trainNetwork = DQN(stateShape, len(actionSpace)).to(device)
        self.targetNetwork = DQN(stateShape, len(actionSpace)).to(device)
        
        self.targetNetwork.load_state_dict(self.trainNetwork.state_dict())
        self.targetNetwork.eval()
        '''
        for p in self.trainNetwork.parameters():
            p.requires_grad = True
        for p in self.targetNetwork.parameters():
            p.requires_grad = False
        '''
        self.optimizer = optim.RMSprop(self.trainNetwork.parameters())

    def trainDQN(self):
        if self.step <= self.numPicks:
            return 1
        
        samples = self.replayMemory.select(0.9)
        batch = Transition(*zip(*samples[0]))
        currState,action,reward,nextState,done = batch

        reward = torch.flatten(FloatTensor(reward)).to(device)
        done = torch.flatten(BoolTensor(done)).to(device)

        currState = torch.cat(currState)
        action = torch.cat(action)

        Q_current = self.trainNetwork(currState).gather(1, action.unsqueeze(1))
        
        nextStateNonFinal = torch.cat([s for s in nextState
                                                if s is not None])
        nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None,
                                          nextState)), dtype=torch.bool).to(device)

        Q_futures = torch.zeros(self.numPicks, device=device)
        Q_futures[nonFinalMask] = self.targetNetwork(nextStateNonFinal.squeeze(1)).max(1)[0].detach()

        Q_next = (reward + (reward + Q_futures * self.gamma)*~done).unsqueeze(1)

        self.optimizer.zero_grad()
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(Q_current, Q_next)
        for param in self.trainNetwork.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)
        loss.backward()
        self.optimizer.step()

        '''self.epsilon *= self.epsilon_decay'''
        self.soft_update()
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return 1+loss.cpu().detach().numpy().max()

    def selectAction(self, currState):
      self.step += 1
      '''self.eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
        math.exp(-1. * self.step / self.epsilon_decay)'''
      self.epsilon *= self.epsilon_decay

      if np.random.rand(1) < self.epsilon:
          action = random.sample(self.actionSpace, 1)[0]
      else:
          currState = currState.unsqueeze(0).to(device)
          with torch.no_grad():
              action = np.argmax(self.trainNetwork(currState.squeeze(1)).detach().cpu().numpy())
      return action

    def addMemory(self, memory, loss):
        self.replayMemory.add(memory, loss)

    def soft_update(self):
        for target_param, train_param in zip(self.targetNetwork.parameters(), self.trainNetwork.parameters()):
            target_param.data.copy_(self.tau*train_param.data + (1.0-self.tau)*target_param.data)

    def save(self):
        save_path = (
            f"./mountain_car_{int(self.step)}.chkpt"
        )
        torch.save(
            dict(model=self.trainNetwork.state_dict(), epsilon=self.epsilon),
            save_path,
        )
        print(f"MarioNet saved to {save_path} done!")

resize = T.Compose([T.ToPILImage(),
                    T.Resize(84, interpolation=Image.CUBIC),
                    T.ToTensor()])

def process_screen(observation):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = observation.transpose((2, 1, 0))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

episode_score = []

def plot_heights():
    plt.figure(2)
    plt.clf()
    heights_t = torch.tensor(episode_score, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(heights_t.numpy())
    # Take 100 episode averages and plot them too
    if len(heights_t) >= 100:
        means = heights_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

env = MountainCar(speed=1e8, graphical_state=True, render=True, is_debug=True)
agent = DQNAgent(stateShape=(3, 84, 84),
                   actionSpace=env.get_action_space(), numPicks=128, memorySize=10000)

def episode():
    done = False
    rewardSum = 0
    lossSum = 0

    env.reset()
    currState = process_screen(env.get_state())
    lastState = currState
    state = currState - lastState

    maxScore = -100000

    while not done:
        stateT = Variable(state)

        action = agent.selectAction(stateT)
        obs, reward, done, score = env.step_all(action)
        maxScore =  max(maxScore, score)
       
        reward = np.sum(reward)
        lastState = currState
        currState = process_screen(obs)

        if not done:
            nextState = Variable(currState - lastState).to(device)
        else:
            nextState = None

        #reward += obs[0]

        actionT = Variable(LongTensor([action])).to(device)
        rewardT = Variable(FloatTensor([reward])).to(device)
        doneT = Variable(BoolTensor([done])).to(device)

        rewardSum += reward

        loss = agent.trainDQN()
        agent.addMemory((stateT, actionT, rewardT, nextState, doneT), loss)
        lossSum += loss

        state = nextState

        if ep % 100 == 0:
            env.render()

    episode_score.append(maxScore)
    plot_heights()

    if ep % 150 == 0:
        agent.save()

    print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(agent.epsilon, rewardSum, lossSum, ep)) 

ep = 1
while ep < 10000:
    episode()
    ep += 1

env.close()