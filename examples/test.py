import pommerman
import matplotlib.pyplot as plt
from pommerman import agents


def main():
    '''
        Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''

    # Create a set of agents (exactly four)
    agent_list = [
        agents.A2CAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
        agents.RandomAgent()
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeRadio-v2', agent_list)

    win_times = 0
    lose_times = 0

    win_rates = []

    # Run the episodes just like OpenAI Gym
    for i_episode in range(3001):
        state = env.reset()
        done = False
        while not done:
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

        print('Episode {} finished'.format(i_episode))

        if reward[0] > 0:
            win_times += 1
        elif reward[0] < 0:
            lose_times += 1

        if i_episode % 50 == 0 and i_episode != 0:
            if win_times + lose_times > 0:
                rate = win_times / (win_times + lose_times)
                print('Win Rate: {0:.2f}.'.format(rate))
                win_rates.append(rate)
            else:
                print('All draws.')
                win_rates.append(0)

            win_times = 0
            lose_times = 0

            plot_figure(win_rates)

    print('Average win rate is: ', sum(win_rates) / len(win_rates))
    env.close()


def plot_figure(win_rates, path='C:\\Users\\Jingyu Liu\\Documents\\pommerman-rl\\win_rate.png'):
    plt.plot(win_rates)
    plt.xlabel('per 50 episodes')
    plt.ylabel('winning rate')
    plt.savefig(path)


if __name__ == '__main__':
    main()
