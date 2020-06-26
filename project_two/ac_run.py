import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from unityagents import UnityEnvironment

from project_v2.ac_agent import A2CAgent


def train_ac(agent: A2CAgent, env: UnityEnvironment, num_of_agents: int, max_ep: int = 300, rollout: int = 5):
    """
    Advantage Actor-Critic agent to solve the continuous action space in the Reacher environment.

    :param agent: an agent with a RL learning algorithm
    :param env: Reacher UnityEnvironment
    :param num_of_agents: the number of agents present in the environment
    :param max_ep: maximum number of episodes to run for
    :param rollout: number of rollout steps to perform to get expected return
    """

    agent = agent
    env = env
    num_of_agents = num_of_agents

    print_every = 1

    average_score_for_ep = []
    scores_window = deque(maxlen=100)
    rolling_average = []

    brain_name = env.brain_names[0]

    for i_episode in range(1, max_ep + 1):
        scores = np.zeros(num_of_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        experience = []
        done = False

        n_steps_bootstrap = 0

        while not np.any(done):
            n_steps_bootstrap += 1
            action, log_prob, entropy, value = agent.get_action(state)

            env_info = env.step(action)
            new_state = env_info[brain_name].vector_observations
            reward = np.array(env_info[brain_name].rewards)
            done = env_info[brain_name].local_done
            _, _, _, value_new_state = agent.get_action(new_state)

            experience.append([log_prob, entropy, value, reward, done, value_new_state])

            if n_steps_bootstrap % rollout == 0:
                agent.update_policy(experience)
                experience = []

            state = new_state

            scores += reward
        average_score_for_ep.append(np.mean(scores))
        scores_window.append(average_score_for_ep)
        rolling_average.append(np.mean(scores_window))

        if i_episode % print_every == 0:
            print('Episode {}\tScore: {:.2f}\tRolling 100 Score: {:.2f}'.format(i_episode, np.mean(scores),
                                                                                np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            print('Environment solved in {:d} episodes!\tRolling 100 Score: {:.2f}'.format(i_episode - 100,
                                                                                           np.mean(scores_window)))
            torch.save(agent.policy.state_dict(), 'checkpoint.pth')
            break

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(average_score_for_ep)), rolling_average, c='b',
             label='Rolling average over last 100 episodes')
    plt.plot(np.arange(len(average_score_for_ep)), average_score_for_ep, c='g', label='Average score per agent')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    cur_dir = os.getcwd()
    v1 = 'Reacher_Windows_x86_64v1\\Reacher.exe'
    v2 = 'Reacher_Windows_x86_64v2\\Reacher.exe'
    reacher_env = UnityEnvironment(file_name=os.path.join(cur_dir, v2))

    brain = reacher_env.brain_names[0]
    environment_info = reacher_env.reset(train_mode=True)[brain]
    num_agents = len(environment_info.agents)

    actor_critic_agent = A2CAgent(num_of_agents=num_agents)

    train_ac(agent=actor_critic_agent, env=reacher_env, num_of_agents=num_agents)
