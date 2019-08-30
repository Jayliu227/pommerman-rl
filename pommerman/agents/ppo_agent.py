from . import BaseAgent
from . import utils

label = 1


class PPOAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(PPOAgent, self).__init__(*args, **kwargs)

        # get a global label for the agent
        global label
        self.label = label
        label += 1

        self.TEST_MODE = False
        self.model_name = 'ppo_{}'.format(self.label)

        self.ppo = utils.PPO(lr=2e-3, gamma=0.99, k_epochs=5, eps=0.2, save_freq=50)
        self.memory = utils.Memory()

        if self.TEST_MODE:
            utils.load_model(self.ppo.behavior_policy, self.model_name)

    def act(self, obs, action_space):
        if not self.TEST_MODE:
            self.memory.rewards.append(0)

        return self.ppo.behavior_policy.act(obs, self.memory)

    def episode_end(self, reward):
        if self.TEST_MODE:
            return

        self.memory.rewards[-1] = reward
        utils.fill_in_rewards(self.memory)
        self.ppo.update(self.memory)
        self.ppo.save(self.model_name)
        self.memory.clear()

