# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
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
import math
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
    def __init__(self, stateShape, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(stateShape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
 
        self.lin = nn.Linear(64*4, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.lin(x))
        x = self.head(x)
        return F.softmax(x, dim=1)

class DQNAgent:
    def __init__(self, stateShape, actionSpace, numPicks, memorySize):
        self.numPicks = numPicks
        self.memorySize = memorySize
        self.replayMemory = deque(maxlen=memorySize)
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0
        self.sync = 10

        self.alpha = 0.000025
        self.epsilon = 1
        self.epsilon_decay = 10**5
        self.epsilon_min = 0.02
        self.eps_threshold = 0
        self.updateAfter = 200
        self.gamma = 0.99
        self.tau = 1e-3

        self.trainNetwork = DQN(stateShape, actionSpace.n).to(device)
        self.targetNetwork = DQN(stateShape, actionSpace.n).to(device)

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
        if self.step <= self.numPicks and self.step <= self.updateAfter:
            return 0

        samples = random.sample(self.replayMemory, self.numPicks)
        batch = Transition(*zip(*samples))
        currState, action, reward, nextState, done = batch

        '''
        states => actions
        need ideal actions => Q LERN
        '''
        '''currState, nextState, action, reward, done = map(
            torch.stack, zip(*samples))
        '''
        reward = torch.flatten(FloatTensor(reward)).to(device)
        done = torch.flatten(BoolTensor(done)).to(device)

        currState = torch.cat(currState).to(device)


        actionArray = np.zeros((len(action), self.actionSpace.n))
        actionArray[np.arange(len(action)), action] = 1
        actionArray = torch.from_numpy(actionArray).to(device)

        Q_current = (self.trainNetwork(currState) * actionArray).sum(dim=1).float().to(device)

        nextStateNonFinal = torch.cat([s for s in nextState
                                       if s is not None])
        nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None,
                                              nextState)), dtype=torch.bool).to(device)

        Q_futures = torch.zeros(self.numPicks, device=device)
        Q_futures[nonFinalMask] = self.targetNetwork(
            nextStateNonFinal).max(1)[0].detach()
       # Q_futures_best = torch.argmax(Q_futures, axis=1)
        # Q_next = self.trainNetwork(nextState)
        '''(reward + (self.gamma * Q_futures)).unsqueeze(1).float() '''

        Q_next = (reward + Q_futures * self.gamma * (~done)).float().to(device)

        self.optimizer.zero_grad()
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(Q_current, Q_next)
        loss.backward()
        for param in self.trainNetwork.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.soft_update()
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss.cpu().detach().numpy().max()

    def selectAction(self, currState):
        self.step += 1
        self.eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
            math.exp(-1. * self.step / self.epsilon_decay)
        '''self.epsilon *= self.epsilon_decay'''

        q = -1

        if np.random.rand(1) < self.eps_threshold:
            action = self.actionSpace.sample()
        else:
            with torch.no_grad():
                preds = self.trainNetwork(currState).squeeze(0).detach().cpu().numpy()
                action = np.argmax(preds)
                q = preds[action]

        return action, q

    def addMemory(self, memory, loss):
        self.replayMemory.append(memory)

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
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# Borrowed from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#input-extraction


def get_car_location(env, screen_width):
    xmin = env.env.min_position
    xmax = env.env.max_position
    world_width = xmax - xmin
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CAR

# Borrowed from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#input-extraction


def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array')
    # Cart is in the lower half, so strip off the top and bottom of the screen
    screen_height, screen_width, _ = screen.shape
    # screen = screen[int(screen_height * 0.8), :]
    view_width = int(screen_width)
    car_location = get_car_location(env, screen_width)
    if car_location < view_width // 2:
        slice_range = slice(view_width)
    elif car_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(car_location - view_width // 2,
                            car_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, slice_range, :]
    return screen


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
env.reset()
init_screen = resize(get_screen(env))
channels, screen_height, screen_width = init_screen.shape
agent = DQNAgent(stateShape=[channels, screen_height, screen_width],
                 actionSpace=env.action_space, numPicks=32, memorySize=100000)


def episode():
    done = False
    rewardsSum = 0
    qSum = 0
    lossSum = 0

    env.reset()

    last_screen = resize(get_screen(env)).unsqueeze(0).to(device)
    current_screen = resize(get_screen(env)).unsqueeze(0).to(device)
    state = current_screen - last_screen
    maxHeight = -100

    while not done:

        action, q = agent.selectAction(state)
        if q > -1:
            qSum += q

        obs, reward, done, _ = env.step(action)

        if obs[0] >= 0.5:
            reward += 10

        maxHeight = max(obs[0], maxHeight)

        env.render()

        last_screen = current_screen
        current_screen = resize(get_screen(env)).unsqueeze(0).to(device)
        if not done:
            nextState = current_screen - last_screen
        else:
            nextState = None

        rewardT = Variable(FloatTensor([reward])).to(device)
        doneT = Variable(BoolTensor([done])).to(device)

        rewardsSum = np.add(rewardsSum, reward)
        
        loss = agent.trainDQN()
        lossSum += loss
        agent.addMemory((state, action, rewardT, nextState, doneT), loss)

        state = nextState

    avgQ = qSum/200

    episode_score.append(rewardsSum)
    episode_qs.append(avgQ)
    episode_height.append(maxHeight)
    episode_loss.append(lossSum)
    episode_decay.append(agent.eps_threshold)
    plot_episode()

    #if ep % agent.sync == 0:
    #   agent.targetNetwork.load_state_dict(agent.trainNetwork.state_dict())

    if ep % 150 == 0:
        agent.save()

    print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(
        agent.epsilon, rewardsSum, lossSum, ep))


ep = 1
while ep < 10000:
    episode()
    ep += 1

env.close()
