import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
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
