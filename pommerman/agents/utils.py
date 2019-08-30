import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import os
from pommerman import constants

######## HYPERPARAMETERS ########
DEBUG_MODE = False

EXTRA_BOMB_REWARD = 0.01
KICK_ENABLED_REWARD = 0.02
RANGE_UP_REWARD = 0.01
AVOID_BOMB_REWARD = 0.002
LAY_BOMB_REWARD = 0.02
#################################


class Memory:
    """
        for storing information about each steps in an episode
    """

    def __init__(self):
        self.actions = []
        self.obs = []
        self.states = []
        self.log_probs = []
        self.rewards = []

    def clear(self):
        del self.actions[:]
        del self.obs[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.affine = nn.Sequential(
            # flattened out dimension + other info dimension
            nn.Linear(32 * 3 * 3 + 9, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.action_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, obs, memory):
        state = get_state_from_obs(obs)
        action_probs = self.action_head(self.common_layer_pass(state))
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.obs.append(obs)
        memory.actions.append(action)
        memory.log_probs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        x = self.common_layer_pass(state)

        action_probs = self.action_head(x)
        dist = Categorical(action_probs)

        log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_head(x)

        return log_prob, torch.squeeze(state_value), dist_entropy

    def common_layer_pass(self, x):
        # board and other data, dim=(batch_size, 3, board_size, board_size)
        x1 = x[:, :3]
        # extra info off the board, dim=(batch_size, 1, board_size, 9)
        x2 = x[:, -1, 0, :9]

        x1 = self.conv_layer(x1)

        # put board and other information together
        x1 = x1.view(x1.shape[0], -1)
        x = torch.cat((x1, x2), dim=1)

        return self.affine(x)


class PPO:
    def __init__(self, lr, gamma, k_epochs, eps, save_freq):
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps = eps
        self.save_freq = save_freq

        self.counter = 0

        self.policy = NN()
        self.behavior_policy = NN()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def update(self, memory):
        rewards = []
        R = 0
        for reward in reversed(memory.rewards):
            R = reward + self.gamma * R
            rewards.insert(0, R)

        # normalize rewards
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # get data and stack them into batch form
        old_states = torch.stack(memory.states).detach().squeeze(1)
        old_actions = torch.stack(memory.actions).detach()
        old_log_probs = torch.stack(memory.log_probs).detach()

        # update the new policy for k epochs
        for _ in range(self.k_epochs):
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # calculate ratio between pi(a|s) / pi_old(a|s)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # calculate A and surrogate losses
            advantages = rewards - state_values.detach()
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            loss = \
                - torch.min(surrogate1, surrogate2) \
                + 0.5 * F.smooth_l1_loss(rewards, state_values) \
                - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # update old policy by the new policy
        self.behavior_policy.load_state_dict(self.policy.state_dict())

    def save(self, model_name, directory='saved_model'):
        if self.counter % self.save_freq == 0:
            save_model(self.behavior_policy, model_name, directory)
        self.counter += 1


def fill_in_rewards(memory):
    max_ammo = -1
    # traverse from the first to the second of the last
    for i in range(len(memory.obs) - 1):
        total_reward = 0

        cur_obs = memory.obs[i]
        next_obs = memory.obs[i + 1]

        # 1. rewards for collectibles
        if max_ammo == -1:
            max_ammo = cur_obs['ammo']
        if max_ammo < next_obs['ammo']:
            max_ammo = next_obs['ammo']
            total_reward += EXTRA_BOMB_REWARD
            debug_print('REWARD: collected ammo!')
        if cur_obs['can_kick'] is False and next_obs['can_kick'] is True:
            total_reward += KICK_ENABLED_REWARD
            debug_print('REWARD: collected kick!')
        if cur_obs['blast_strength'] < next_obs['blast_strength']:
            total_reward += RANGE_UP_REWARD
            debug_print('REWARD: collected strength!')

        # 2. rewards for bomb-avoiding movements
        """
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
        """
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
                total_reward += AVOID_BOMB_REWARD * (cur_bomb_expose_num - next_bomb_expose_num)
                debug_print('REWARD: avoid bomb!')

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
                if state_idx >= len(memory.obs):
                    # if we died before this bomb explodes, then it is not useful
                    break
                next_state = memory.obs[state_idx]
                # either we can't observe it, or it must be 0 or equal to remain
                target = int(next_state['bomb_life'][cur_x][cur_y])

                # if its fogged out, then we don't care about it
                if not is_fogged_out(next_state['board'], cur_x, cur_y):
                    if remain == target:
                        # if they are equal and equal to 0, then this bomb is good
                        if remain == 0:
                            total_reward += LAY_BOMB_REWARD
                            debug_print('REWARD: laid bomb!')
                            break
                    else:
                        # if not, then this bomb must have blown and replaced by some other bomb
                        total_reward += LAY_BOMB_REWARD
                        debug_print('REWARD: laid bomb!')
                        break

        memory.rewards[i] = total_reward


def debug_print(string):
    if DEBUG_MODE:
        print(string)


def save_model(net, model_name, directory='saved_model'):
    path = os.path.join(os.path.dirname(__file__), directory, model_name)
    torch.save(net.state_dict(), path)


def load_model(net, model_name, directory='saved_model'):
    path = os.path.join(os.path.dirname(__file__), directory, model_name)
    net.load_state_dict(torch.load(path))
    net.eval()


def get_state_from_obs(obs):
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
    return torch.cat((board, bomb_life, bomb_blast_strength, extra_info), dim=0).unsqueeze(0)
