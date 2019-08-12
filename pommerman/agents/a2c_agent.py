from . import BaseAgent
from pommerman import constants

import os

import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()
label = 1


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
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

        # convolutional layers, 3x11x11
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(32)

        # fully connected layers
        # flattened out dimension + other info dimension
        self.fc1 = nn.Linear(32 * 3 * 3 + 9, 128)
        # middle layer dimension
        self.fc2 = nn.Linear(128, 64)

        # followed by two heads
        self.action_head = nn.Linear(64, 6)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        # board and other data, dim=(batch_size, 3, board_size, board_size)
        x1 = x[:, :3]
        # extra info off the board, dim=(batch_size, 1, board_size, 9)
        x2 = x[:, -1, 0, :9]

        # feed the board representation into conv layers
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))

        # put board and other information together
        x1 = x1.view(x1.shape[0], -1)
        x = torch.cat((x1, x2), dim=1)

        # feed concat data into fc layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        return F.softmax(action_scores, dim=-1), state_values


class A2CAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(A2CAgent, self).__init__(*args, **kwargs)

        global label
        self.label = label
        label += 1

        self.save_counter = 0
        self.training = True
        self.debug_mode = False

        # constants
        self.GAMMA = 0.99
        self.SAVE_FREQ = 50
        self.EPSILON = 0.90

        self.EXTRA_BOMB_REWARD = 10
        self.RANGE_UP_REWARD = 5
        self.KICK_ENABLED_REWARD = 5
        self.LAY_BOMB_REWARD = 10
        self.AVOID_BOMB_REWARD = 10

        self.WIN_LOSE_MULTIPLIER = 7

        # agent ids
        self.teammate_id = -1
        self.my_id = -1
        self.enemies_ids = [-1, -1]

        # saved action and rewards for each episode
        self.saved_actions = []
        self.saved_rewards = []
        self.saved_obs = []

        self.policy = Policy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)

        self.model_name = 'vs_simple_agent_%d_2000_episode' % self.label

        if not self.training:
            self.load_model()

    def act(self, obs, action_space):

        # choose an action based on the probability
        state = self.get_state_from_obs(obs)
        action_values, state_value = self.policy(state)

        m = Categorical(action_values)
        if self.training and np.random.uniform() < (1.0 - self.EPSILON):
            # if we're at training mode, then we do epsilon-greedy policy
            m = Categorical(torch.FloatTensor([[1.0 / 6] * 6]))

        action = m.sample()

        if self.training:
            # save the chosen action probability and the value of the state
            self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
            self.saved_obs.append(obs)

        # TEST:
        # a = action_space.sample()

        # print('board:\n', obs['board'])
        # print('bomb_life:\n', obs['bomb_life'])
        # print('action_taken: ', constants.Action(a[0]).name)
        # print('\n')

        # if self.save_counter == 0:
        #     a = constants.Action.Bomb.value
        # elif self.save_counter == 1:
        #     a = constants.Action.Right.value
        # elif self.save_counter == 2:
        #     a = constants.Action.Down.value
        # else:
        #     a = constants.Action.Stop.value
        #
        # self.save_counter += 1
        # return a
        # END_TEST;

        return action.item()

    def episode_end(self, reward):

        if not self.training:
            return

        self.reset_agent_id()

        # reshape reward
        self.saved_rewards = [0] * len(self.saved_obs)
        # final reward is given by the environment
        if reward < 0:
            self.saved_rewards[-1] = -150
        elif reward > 0:
            self.saved_rewards[-1] = 50
        else:
            self.saved_rewards[-1] = -50

        # fill in the reshaped rewards for each states except the final state
        self.fill_in_rewards()

        policy_losses = []
        value_losses = []
        returns = []

        # calculate discounted sum of rewards for each state
        R = 0
        for r in self.saved_rewards[::-1]:
            R = r + self.GAMMA * R
            returns.insert(0, R)

        # # TEST:
        # print('rewards: ', self.saved_rewards)
        # print('returns: ', returns)
        #
        # del self.saved_rewards[:]
        # del self.saved_actions[:]
        # del self.saved_obs[:]
        #
        # return
        # END_TEST;

        # normalize returns

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # calculate advantage for each state and loss
        for (log_prob, value), r in zip(self.saved_actions, returns):
            advantage = r - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([[r]])))

        # optimize policy network and value network together
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        loss = policy_loss + value_loss * 2.0
        loss.backward()
        self.optimizer.step()

        del self.saved_rewards[:]
        del self.saved_actions[:]
        del self.saved_obs[:]

        if self.save_counter % self.SAVE_FREQ == 0:
            self.save_model()
        self.save_counter += 1

        print(f'Episode loss: policy loss <{abs(policy_loss.item()):.02f}>, '
              f'value loss <{abs(value_loss.item()):.02f}>; ')

    def fill_in_rewards(self):
        max_ammo = -1
        # traverse from the first to the second of the last
        for i in range(len(self.saved_obs) - 1):
            total_reward = 0

            cur_obs = self.saved_obs[i]
            next_obs = self.saved_obs[i + 1]

            # 1. rewards for collectibles
            if max_ammo == -1:
                max_ammo = cur_obs['ammo']
            if max_ammo < next_obs['ammo']:
                max_ammo = next_obs['ammo']
                total_reward += self.EXTRA_BOMB_REWARD
                self.debug_print('REWARD: collected ammo!')
            if cur_obs['can_kick'] is False and next_obs['can_kick'] is True:
                total_reward += self.KICK_ENABLED_REWARD
                self.debug_print('REWARD: collected kick!')
            if cur_obs['blast_strength'] < next_obs['blast_strength']:
                total_reward += self.RANGE_UP_REWARD
                self.debug_print('REWARD: collected strength!')

            # 2. rewards for bomb-avoiding movements
            '''
                we first figure out if we move at this point
                if we moved, then we need to check if this move puts us
                into a safe zone with a specific vision range
                    OOOOOOO
                    OOOOOOO
                    OOOXOOO
                    OOOOOOO
                    OOOOOOO
                vision range is the square diameter we will be looking at, 
                if we can move to a place where the number of bombs we will be
                exposed to is fewer than before, we gain rewards.
            '''

            vision_range = 3
            safe_time = 3
            cur_pos = cur_obs['position']
            next_pos = next_obs['position']

            def clip_coord(x):
                return min(max(0, x), constants.BOARD_SIZE - 1)

            def dist(a, b, c, d):
                return abs(a - c) + abs(b - d)

            def cal_bomb_overlap(obs, x_pos, y_pos):
                board = obs['board']
                strength_map = obs['bomb_blast_strength']
                life_map = obs['bomb_life']
                num = 0

                for x in range(clip_coord(x_pos - vision_range), clip_coord(x_pos + vision_range) + 1):
                    for y in range(clip_coord(y_pos - vision_range), clip_coord(y_pos + vision_range) + 1):
                        if (x == x_pos or y == y_pos) and board[x][y] == constants.Item.Bomb.value:
                            # if the bomb and us are in the same line
                            strength = strength_map[x][y]
                            life = life_map[x][y]
                            if strength >= dist(x, y, x_pos, y_pos) and life <= safe_time:
                                # and we are in the range
                                num += 1
                return num

            if cur_pos[0] != next_pos[0] or cur_pos[1] != next_pos[1]:
                # in this case, we moved at current state
                cur_bomb_expose_num = cal_bomb_overlap(cur_obs, cur_pos[0], cur_pos[1])
                next_bomb_expose_num = cal_bomb_overlap(next_obs, next_pos[0], next_pos[1])
                if next_bomb_expose_num < cur_bomb_expose_num:
                    total_reward += self.AVOID_BOMB_REWARD * (cur_bomb_expose_num - next_bomb_expose_num)
                    self.debug_print('REWARD: avoid bomb!')

            # 3. rewards for laying bombs
            cur_x = cur_obs['position'][0]
            cur_y = cur_obs['position'][1]

            def is_fogged_out(board, x, y):
                return board[x][y] == constants.Item.Fog.value

            # test if current action is laying a bomb
            bomb_remain_time = int(next_obs['bomb_life'][cur_x][cur_y])
            if cur_obs['bomb_life'][cur_x][cur_y] == 0 and bomb_remain_time > 0:
                # we need to check if sooner or later this place becomes empty
                # and we're still alive
                for t in range(bomb_remain_time):
                    remain = bomb_remain_time - t - 1
                    state_idx = i + 2 + t
                    if state_idx >= len(self.saved_obs):
                        # if we died before this bomb explodes, then it is not useful
                        break
                    next_state = self.saved_obs[state_idx]
                    # either we can't observe it, or it must be 0 or equal to remain
                    target = int(next_state['bomb_life'][cur_x][cur_y])

                    # if its fogged out, then we don't care about it
                    if not is_fogged_out(next_state['board'], cur_x, cur_y):
                        if remain == target:
                            # if they are equal and equal to 0, then this bomb is good
                            if remain == 0:
                                total_reward += self.LAY_BOMB_REWARD
                                self.debug_print('REWARD: laid bomb!')
                                break
                        else:
                            # if not, then this bomb must have blown and replaced by some other bomb
                            total_reward += self.LAY_BOMB_REWARD
                            self.debug_print('REWARD: laid bomb!')
                            break

            self.saved_rewards[i] = total_reward

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

        # concat together, and convert to batch format
        return torch.cat((board, bomb_life, bomb_blast_strength, extra_info), dim=0).unsqueeze(0).to(self.device)

    def reset_agent_id(self):
        if self.my_id == -1 and len(self.saved_obs) > 0:
            self.teammate_id = self.saved_obs[0]['teammate'].value
            self.my_id = (self.teammate_id - 8) % 4 + 10
            self.enemies_ids = [(self.my_id - 9) % 4 + 10, (self.teammate_id - 9) % 4 + 10]

    def save_model(self):
        path = os.path.join(os.path.dirname(__file__), 'saved_model', self.model_name)
        torch.save(self.policy.state_dict(), path)

    def load_model(self):
        path = os.path.join(os.path.dirname(__file__), 'saved_model', self.model_name)
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

    def debug_print(self, string):
        if self.debug_mode:
            print(string)
