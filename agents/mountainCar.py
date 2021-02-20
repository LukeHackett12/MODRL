#from nes_py.wrappers import JoypadSpace
#import gym_super_mario_bros
#from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from custom_envs.mountain_car.engine import MountainCar

import numpy as np

import torch
from torch import FloatTensor, LongTensor, BoolTensor
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T

import enum
from collections import namedtuple, deque
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

from PIL import Image
from agents.dqn_models import Experience

device = torch.device("cpu")

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
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class MarioAgent:
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
        self.epsilon_decay = 0.99975
        self.epsilon_min=0.05
        self.eps_threshold = 0

        self.gamma = 0.999
        self.tau = 1e-3

        self.trainNetwork = DQN(stateShape[1],stateShape[2], actionSpace.n).to(device)
        self.targetNetwork = DQN(stateShape[1],stateShape[2], actionSpace.n).to(device)
        
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

        '''
        states => actions
        need ideal actions => Q LERN
        '''
        '''currState, nextState, action, reward, done = map(
            torch.stack, zip(*samples))
        '''
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
       # Q_futures_best = torch.argmax(Q_futures, axis=1)
        #Q_next = self.trainNetwork(nextState)
        '''(reward + (self.gamma * Q_futures)).unsqueeze(1).float() '''

        Q_next = (reward * done + (reward + Q_futures * self.gamma)*~done).unsqueeze(1)

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
          action = self.actionSpace.sample()
      else:
          currState = currState.unsqueeze(0).to(device)
          with torch.no_grad():
              action = np.argmax(self.trainNetwork(currState.squeeze(1)).detach().cpu().numpy())
      return action

    def addMemory(self, memory, loss):
        self.replayMemory.add(memory, loss)

    def soft_update(self):
        for target_param, train_param in zip(self.targetNetwork.net.parameters(), self.trainNetwork.net.parameters()):
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
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

episode_heights = []

def plot_heights():
    plt.figure(2)
    plt.clf()
    heights_t = torch.tensor(episode_heights, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Height')
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

env = MountainCar(seed=100)
env.reset()
print(np.shape(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy()))
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()


agent = MarioAgent(stateShape=(3, 40, 60),
                   actionSpace=env.action_space, numPicks=128, memorySize=10000)

def episode():
    done = False
    env.reset()
    rewardSum = 0
    lossSum = 0

    currState = get_screen()
    lastState = currState
    state = currState - lastState

    maxPos = -100

    while not done:
        stateT = Variable(state)

        action = agent.selectAction(stateT)
        obs, reward, done, _ = env.step(action)
        maxPos =  max(maxPos, obs[0])
       
        lastState = currState
        currState = get_screen()

        if not done:
            nextState = Variable(currState - lastState).to(device)
        else:
            nextState = None

        #reward += obs[0]

        actionT = Variable(LongTensor([action])).to(device)
        rewardT = Variable(FloatTensor([reward])).to(device)
        doneT = Variable(BoolTensor([done])).to(device)

        rewardSum += reward
        state = nextState

        loss = agent.trainDQN()
        agent.addMemory((stateT, actionT, rewardT, nextState, doneT), loss)
        lossSum += loss

        if ep % 100 == 0:
            env.render()

    episode_heights.append(maxPos)
    plot_heights()

    if ep % 150 == 0:
        agent.save()

    print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(agent.epsilon, rewardSum, lossSum, ep)) 

ep = 1
while ep < 10000:
    episode()
    ep += 1

env.close()