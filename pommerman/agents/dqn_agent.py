from . import BaseAgent

import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        '''
            three conv2d layers and three fully connected layers
            input is observations(state), output values of actions of the current state
        
            the board is processed by conv layers and then flattened out 
                first channel, board
                second channel, bomb blast strength
                third channel, bomb life

            and concatenated with other informations: 
                ammo(1), position(2), blast strength(1), can kick(1), alive(4)

            output an array of values of six discrete actions to take
        '''

        # dimension 1x11x11
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(32)

        # flattented out dimension + other info dimension
        self.fc1 = nn.Linear(32 * 3 * 3 + 9, 128)
        # middle layer dimension
        self.fc2 = nn.Linear(128, 64)
        # output dimension is action space dimension
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        # board and other data, dim=(batch_size, 3, board_size, board_size)
        x1 = x[:, :3]
        # extra info off the board, dim=(batch_size, 1, board_size, 9)
        x2 = x[:, -1, 0, :9]

        # feed the board rep into conv layers
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))

        # put two parts of the data together
        x1 = x1.view(x1.shape[0], -1)
        x = torch.cat((x1, x2), dim=1)

        # feed concat data into fc layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class DQNAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)
        print('Initialized dqn agent.')

        # store current and previous observation
        self.cur_obs = None
        self.prev_obs = None

        self.cur_state = None
        self.prev_state = None

        self.cur_action = None
        self.prev_action = None

        # constants
        self.EXTRA_BOMB_REWARD = 2
        self.RANGE_UP_REWARD = 2
        self.KICK_ENABLED_REWARD = 2
        self.STAY_PUT_REWARD = -0.5

        self.WIN_LOSE_MULTIPLIER = 5

        self.BATCH_SIZE = 16
        self.GAMMA = 0.99
        self.EPS = 0.9
        self.TARGET_UPDATE = 10

        self.counter = 0
        self.episode_loss = 0

        # dqn networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

    def act(self, obs, action_space):

        # update current and previous informations
        self.prev_obs = self.cur_obs
        self.cur_obs = obs

        self.prev_state = self.cur_state
        self.cur_state = self.get_state_from_obs(obs)

        self.prev_action = self.cur_action
        self.cur_action = self.select_action(self.cur_state)

        # get self id, teammate id and enemy ids
        teammate = obs['teammate'].value
        me = (teammate - 8) % 4 + 10
        enemies = [(me - 9) % 4 + 10, (teammate - 9) % 4 + 10]

        # store the transition
        if self.prev_action is not None:
            # since we can't know what next state will be, we store the following tuple instead
            reward = torch.FloatTensor([self.get_reward()], device=self.device)
            self.memory.push(self.prev_state, self.prev_action, self.cur_state, reward)

        self.optimize_model()

        return self.cur_action.item()

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        # batch array of states sampled randomly from replay memory to reduce correlation
        transitions = self.memory.sample(self.BATCH_SIZE)

        # convert to transitions of batch arrays
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # first compute Q(s_t, a_t)        
        state_action_values = self.policy_net.forward(state_batch).gather(1, action_batch)

        # compute V(s_t+1)
        next_state_values = self.target_net.forward(next_state_batch).max(1)[0].detach()

        # expected Q values for current state
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # accmulate loss
        self.episode_loss += loss.item()

        # update gradient and optimize the model, clamp the gradients between -1 and 1
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # update target net if needed
        if self.counter % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.counter += 1

    def episode_end(self, reward):
        # print('episode ends, reward: ', reward * self.WIN_LOSE_MULTIPLIER)
        print('Episode loss: ', self.episode_loss)
        self.episode_loss = 0

    def get_state_from_obs(self, obs):
        # extract three board information
        board = torch.FloatTensor(obs['board']).unsqueeze(0)
        bomb_life = torch.FloatTensor(obs['bomb_life']).unsqueeze(0)
        bomb_blast_strength = torch.FloatTensor(obs['bomb_blast_strength']).unsqueeze(0)

        # make the alive array of size 4, if some players died, its id will be excluded
        alive = [1. if i in obs['alive'] else 0. for i in range(10, 14)]

        # put extra information into the last dimension, *obs['message']
        extra_info = torch.zeros([1, 11, 11])
        extra_info[0, 0, :9] = torch.FloatTensor([
            obs['position'][0], obs['position'][1],
            obs['ammo'], obs['blast_strength'], float(obs['can_kick']), *alive
        ])

        # concat together
        return torch.cat((board, bomb_life, bomb_blast_strength, extra_info), dim=0).unsqueeze(0).to(self.device)

    def select_action(self, state):
        # generate an action based on e-greedy policy
        rand = random.random()
        if rand < self.EPS:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(6)]], device=self.device, dtype=torch.long)

    def get_reward(self):
        # based on the difference between current and previous obs, reshape reward
        if self.prev_obs is None:
            return 0

        total_rewards = 0
        # if we get collectible, we get positive rewards
        if self.prev_obs['ammo'] < self.cur_obs['ammo']:
            total_rewards += self.EXTRA_BOMB_REWARD
        if self.prev_obs['can_kick'] is False and self.cur_obs['can_kick'] is True:
            total_rewards += self.KICK_ENABLED_REWARD
        if self.prev_obs['blast_strength'] < self.cur_obs['blast_strength']:
            total_rewards += self.RANGE_UP_REWARD
        if self.prev_obs['position'] == self.cur_obs['position']:
            total_rewards += self.STAY_PUT_REWARD

        return total_rewards
