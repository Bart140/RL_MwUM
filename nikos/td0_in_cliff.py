import gymnasium as gym
import numpy as np

def test_random_agent(env, episodes):
    total_rewards = []
    for e in range(episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = env.action_space.sample()
            next_state, reward, term, trunc, info = env.step(action)
            episode_reward += reward
            done = term or trunc
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)
