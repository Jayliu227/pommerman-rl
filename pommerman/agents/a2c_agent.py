from . import BaseAgent
from . import action_prune
from . import utils
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
        self.EPSILON = 0.10

        self.EXTRA_BOMB_REWARD = 10
        self.RANGE_UP_REWARD = 5
        self.KICK_ENABLED_REWARD = 5
        self.LAY_BOMB_REWARD = 10
        self.AVOID_BOMB_REWARD = 10

        # agent ids
        self.teammate_id = -1
        self.my_id = -1
        self.enemies_ids = [-1, -1]

        # episode entropy
        self.episode_dist_entropy = 0

        # saved action and rewards for each episode
        self.saved_actions = []
        self.saved_rewards = []
        self.saved_obs = []

        self.policy = utils.ActorCritic()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)

        self.model_name = 'af_%d_5000_epi' % self.label

        if not self.training:
            self.load_model()

    def act(self, obs, action_space):
        # choose an action based on the probability
        state = self.get_state_from_obs(obs)
        action_values, state_value = self.policy(state)

        # get action filter
        valid_actions = action_prune.get_filtered_actions(obs)
        modified_action_values = [0] * 6
        for i in range(6):
            if i in valid_actions:
                modified_action_values[i] = action_values[0][i].item()

        modified_action_values = torch.FloatTensor([modified_action_values])

        # get real and truncated distribution
        real_dist = Categorical(action_values)
        modified_dist = Categorical(modified_action_values)

        action = modified_dist.sample()

        if self.training:
            # epsilon-greedy
            if np.random.uniform() < self.EPSILON:
                action = torch.IntTensor([np.random.randint(0, 6)])
            # save the chosen action probability and the value of the state
            self.saved_actions.append(SavedAction(real_dist.log_prob(action), state_value))
            self.saved_obs.append(obs)
            self.episode_dist_entropy += real_dist.entropy()

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
            # losing reward
            self.saved_rewards[-1] = -150
        elif reward > 0:
            # winning reward
            self.saved_rewards[-1] = 100
        else:
            # drawing reward
            self.saved_rewards[-1] = -50

        # fill in the reshaped rewards for each states except the final state
        self.fill_in_rewards()

        actor_losses = []
        critic_losses = []
        returns = []

        # calculate discounted sum of rewards for each state
        R = 0
        for r in self.saved_rewards[::-1]:
            R = r + self.GAMMA * R
            returns.insert(0, R)

        # normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # calculate advantage for each state and loss
        for (log_prob, value), r in zip(self.saved_actions, returns):
            advantage = r - value.item()
            actor_losses.append(-log_prob * advantage)
            critic_losses.append(F.smooth_l1_loss(value, torch.tensor([[r]])))

        # optimize policy network and value network together
        self.optimizer.zero_grad()
        actor_loss = torch.stack(actor_losses).sum()
        critic_loss = torch.stack(critic_losses).sum()
        loss = actor_loss + critic_loss - 0.001 * self.episode_dist_entropy
        loss.backward()
        self.optimizer.step()

        print(f'Episode losses [ actor loss <{abs(actor_loss.item()):.02f}>, '
              f'critic loss <{abs(critic_loss.item()):.02f}>, '
              f'distribution entropy <{self.episode_dist_entropy.item():.02f}> ]')

        del self.saved_rewards[:]
        del self.saved_actions[:]
        del self.saved_obs[:]
        self.episode_dist_entropy = 0

        if self.save_counter % self.SAVE_FREQ == 0:
            self.save_model()
        self.save_counter += 1

        # decaying epsilon
        if self.save_counter > 1000 and self.save_counter % 100 == 0:
            self.EPSILON *= 0.99

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


class SmartRandomAgent(BaseAgent):
    """
        This agent uses the Action-Filter by Skynet;
        Reaching an average win rate of 0.6 against simple agent in 2v2 radio setting
    """

    def act(self, obs, action_space):
        valid_actions = action_prune.get_filtered_actions(obs)
        if len(valid_actions) == 0:
            valid_actions.append(constants.Action.Stop.value)
        action = np.random.choice(valid_actions)
        return action, 0, 0

    def episode_end(self, reward):
        pass
