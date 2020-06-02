import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from unityagents import UnityEnvironment

from ac_model import PolicyNetwork
from ac_agent import A2CAgent


def train_ac(policy_net: PolicyNetwork, agent: A2CAgent, env: UnityEnvironment, num_of_agents: int):
    """
    Advantage Actor-Critic agent to solve the continuous action space in the Reacher environment.

    :param policy_net: neural network compatible with A2C
    :param agent: an agent with a RL learning algorithm
    :param env: Reacher UnityEnvironment
    :param num_of_agents: the number of agents present in the environment
    """

    policy_net = policy_net
    agent = agent
    env = env
    num_of_agents = num_of_agents

    max_episode_num = 1000
    max_steps = 2000

    print_every = 1

    scores = np.zeros(num_of_agents)
    average_scores = []
    scores_window = deque(maxlen=100)
    rolling_average = []

    brain_name = env.brain_names[0]

    for i_episode in range(1, max_episode_num + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        log_probs = []
        rewards = []
        values = []
        entropy = []
        done = False

        n_steps_boots = 0

        while not np.any(done):
            n_steps_boots += 1
            action, log_prob, vals, ent = agent.get_action(state)
            log_probs.append(log_prob)
            values.append(vals)
            entropy.append(ent)

            env_info = env.step(action)
            new_state = env_info[brain_name].vector_observations
            reward = env_info[brain_name].rewards
            done = env_info[brain_name].local_done
            rewards.append(np.array(reward))

            if n_steps_boots % 5 == 0:
                agent.update_policy(rewards=rewards, log_probs=log_probs, values=values, entropy=entropy)

                log_probs = []
                rewards = []
                values = []
                entropy = []

            state = new_state

            scores += reward
        average_scores.append(np.mean(scores))
        scores_window.append(np.mean(scores))
        rolling_average.append(np.mean(scores_window))

        if i_episode % print_every == 0:
            print('Episode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores),
                                                                            np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_window)))
            torch.save(agent.model.state_dict(), 'checkpoint.pth')
            break

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(average_scores)), rolling_average, c='b', label='Rolling average over last 100 episodes')
    plt.plot(np.arange(len(average_scores)), average_scores, c='g', label='Average score per agent')
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

    policy_network = PolicyNetwork()
    actor_critic_agent = A2CAgent(policy=policy_network, gamma=0.99, lr=1e-2, agents=num_agents)

    train_ac(policy_net=policy_network, agent=actor_critic_agent, env=reacher_env, num_of_agents=num_agents)
