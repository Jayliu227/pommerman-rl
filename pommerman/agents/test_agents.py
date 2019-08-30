from . import BaseAgent
from . import action_prune
from pommerman import constants

import numpy as np

"""
    Different simple agents for training purposes
"""


class StaticAgent(BaseAgent):
    """
        This agent stays static, used for training rl agents
    """
    def act(self, obs, action_space):
        return constants.Action.Stop.value


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

