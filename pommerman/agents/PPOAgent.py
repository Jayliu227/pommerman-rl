from . import BaseAgent
from . import utils

from pommerman import constants


class PPOAgent(BaseAgent):
    def __init__(self):
        super(PPOAgent, self).__init__(*args, **kwargs)
        pass

    def act(self, obs, action_space):
        pass

    def episode_end(self, reward):
        pass
