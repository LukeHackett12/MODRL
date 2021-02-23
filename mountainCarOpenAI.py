# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from dqn_utils.dqn_models import Experience
from PIL import Image
import gym

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
      self.hidden_layer_2 = nn.Linear(24, 48, bias=False)
      self.output_layer = nn.Linear(48, n_output, bias=False)

  def forward(self, x):
      out = nn.functional.relu(self.hidden_layer_1(x))
      out = nn.functional.relu(self.hidden_layer_2(out))
      out = self.output_layer(out)
      return out

class DQNAgent:
    def __init__(self, stateShape, actionSpace, numPicks, memorySize):
        self.numPicks = numPicks
        self.memorySize = memorySize
        self.replayMemory = deque(maxlen=memorySize)
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0
        self.sync = 10

        self.alpha = 0.001
        self.epsilon = 0.99
        self.epsilon_decay = 0.9999
        self.epsilon_min=0.01
        self.eps_threshold = 0

        self.gamma = 0.99
        self.tau = 0.01

        self.trainNetwork = DQN(2, actionSpace.n).to(device)
        self.targetNetwork = DQN(2, actionSpace.n).to(device)
        
        self.targetNetwork.load_state_dict(self.trainNetwork.state_dict())
        self.targetNetwork.eval()
        
        for p in self.trainNetwork.parameters():
            p.requires_grad = True
        '''
        for p in self.targetNetwork.parameters():
            p.requires_grad = False
        '''
        self.optimizer = optim.Adam(self.trainNetwork.parameters(), self.alpha)

    def trainDQN(self):
        if self.step <= self.numPicks:
            return 1
        
        samples = random.sample(self.replayMemory, self.numPicks)
        batch = Transition(*zip(*samples))
        currState,action,reward,nextState,done = batch

        '''
        states => actions
        need ideal actions => Q LERN
        '''
        '''currState, nextState, action, reward, done = map(
            torch.stack, zip(*samples))
        '''
        reward = torch.flatten(FloatTensor(reward)).to(device)
        done = torch.flatten(BoolTensor(done)).to(device)

        currState = torch.cat(currState).unsqueeze(1).to(device)
        action = torch.cat(action).to(device)

        Q_current = self.trainNetwork(currState).squeeze(1).gather(1, action.unsqueeze(1))
        
        nextStateNonFinal = torch.cat([s for s in nextState
                                                if s is not None])
        nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None,
                                          nextState)), dtype=torch.bool).to(device)

        Q_futures = torch.zeros(self.numPicks, device=device)
        Q_futures[nonFinalMask] = self.targetNetwork(nextStateNonFinal.unsqueeze(1)).max(2)[0].squeeze(1).detach()
       # Q_futures_best = torch.argmax(Q_futures, axis=1)
        # Q_next = self.trainNetwork(nextState)
        '''(reward + (self.gamma * Q_futures)).unsqueeze(1).float() '''

        Q_next = ((reward * done) + (reward + Q_futures * self.gamma)*~done).unsqueeze(1).to(device)

        self.optimizer.zero_grad()
        loss_fn = nn.MSELoss()
        loss = loss_fn(Q_current, Q_next)
        
        #nn.utils.clip_grad_value_(self.trainNetwork.parameters(), clip_value=10.0)
        '''
        for param in self.trainNetwork.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-10,10)
        '''
        loss.backward()
        self.optimizer.step()

        '''self.epsilon *= self.epsilon_decay'''
        self.soft_update()
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss.cpu().detach().numpy().max()

    def selectAction(self, currState):
      self.step += 1
      '''self.eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
        math.exp(-1. * self.step / self.epsilon_decay)'''
      self.epsilon *= self.epsilon_decay

      if np.random.rand(1) < self.epsilon:
          action = self.actionSpace.sample()
      else:
          currState = currState.unsqueeze(0).to(device)
          with torch.no_grad():
              action = np.argmax(self.trainNetwork(currState).detach().cpu().numpy())
      return action

    def addMemory(self, memory, loss):
        self.replayMemory.append(memory)

    def soft_update(self):
        '''
        for target_param, train_param in zip(self.targetNetwork.parameters(), self.trainNetwork.parameters()):
            target_param.data.copy_(self.tau*train_param.data + (1.0-self.tau)*target_param.data)'''

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
episode_loss = []
episode_decay = []

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.canvas.draw()
plt.show(block=False)

def plot_episode():    
    heights_t = torch.tensor(episode_score, dtype=torch.float)
    ax1.title.set_text('Training Score...')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.plot(heights_t.numpy(), 'b')
    # Take 100 episode averages and plot them too
    if len(heights_t) >= 100:
        means = heights_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax1.plot(means.numpy(), 'c')

    ax2.title.set_text('Training Loss...')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.plot(episode_loss, 'r')

    ax3.title.set_text('Exploration Decay...')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Decay')
    ax3.plot(episode_decay, 'r')
    fig.canvas.draw()
    plt.show(block=False)
    plt.pause(.001)


env = gym.make('MountainCar-v0')
agent = DQNAgent(stateShape=env.observation_space.shape[0],
                   actionSpace=env.action_space, numPicks=128, memorySize=10000)

def episode():
    done = False
    rewardsSum = 0
    lossSum = 0

    state = env.reset()
    stateT = Variable(FloatTensor([state])).to(device)

    while not done:

        action = agent.selectAction(stateT)
        obs, reward, done, _ = env.step(action)
        env.render()

        if not done:
            nextState = Variable(FloatTensor([obs])).to(device)
        else:
            nextState = None

        # reward += obs[0]

        actionT = Variable(LongTensor([action])).to(device)
        rewardT = Variable(FloatTensor([reward])).to(device)
        doneT = Variable(BoolTensor([done])).to(device)

        rewardsSum = np.add(rewardsSum, reward)
        state = nextState

        loss = agent.trainDQN()
        agent.addMemory((stateT, actionT, rewardT, nextState, doneT), loss)
        lossSum += loss

    episode_score.append(rewardsSum)
    episode_loss.append(lossSum)
    episode_decay.append(agent.epsilon)
    plot_episode()

    if ep % 10 == 0:
      agent.targetNetwork.load_state_dict(agent.trainNetwork.state_dict())

    if ep % 150 == 0:
        agent.save()

    print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(agent.epsilon, rewardsSum, lossSum, ep)) 

ep = 1
while ep < 10000:
    episode()
    ep += 1

env.close()
