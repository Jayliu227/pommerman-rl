from . import BaseAgent
from . import utils

from enum import Enum

label = 1


class Mode(Enum):
    Training = 1
    Testing = 2
    Transfer = 3


class PPOAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(PPOAgent, self).__init__(*args, **kwargs)

        # get a global label for the agent
        global label
        self.label = label
        label += 1

        # ids for each of the players in the game
        self.my_id = -1
        self.teammate_id = -1
        self.enemies_ids = [-1, -1]

        # model name and test mode switch
        self.mode = Mode.Testing
        self.load_model_name = 'ppo_static3000_{}'.format(self.label)
        self.save_model_name = 'ppo_simple3000_{}'.format(self.label)

        # PPO update and memory for storing transitions
        self.ppo = utils.PPO(lr=2e-3, gamma=0.99, k_epochs=4, eps=0.2, save_freq=50)
        self.memory = utils.Memory()

        if self.mode.value == Mode.Testing.value or self.mode.value == Mode.Transfer.value:
            utils.load_model(self.ppo.behavior_policy, self.load_model_name)

    def act(self, obs, action_space):
        # if we're in testing mode, we can skip remembering rewards
        if self.mode.value != Mode.Testing.value:
            self.memory.rewards.append(0)
            self.set_agent_id()

        return self.ppo.behavior_policy.act(obs, self.memory)

    def episode_end(self, reward):
        # if we're in testing mode, we can directly clear the memory and return
        if self.mode.value == Mode.Testing.value:
            self.memory.clear()
            return

        utils.calculate_final_reward(self.memory, reward, self.my_id)
        utils.fill_in_rewards(self.memory)
        self.ppo.update(self.memory)
        self.ppo.save(self.save_model_name)
        self.memory.clear()

    def set_agent_id(self):
        if self.my_id == -1 and len(self.memory.obs) > 0:
            self.teammate_id = self.memory.obs[0]['teammate'].value
            self.my_id = (self.teammate_id - 8) % 4 + 10
            self.enemies_ids = [(self.my_id - 9) % 4 + 10, (self.teammate_id - 9) % 4 + 10]
