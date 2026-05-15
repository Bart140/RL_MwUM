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

def epsilon_greedy(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])

def train_td0(env, alpha, episodes = 500, gamma = 0.99):
    epsilon_start = 1.0
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    V_history = []

    for e in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            next_state, reward, term, trunc, info = env.step(action)
            done = term or trunc
            if term:
                V[state] = V[state] + alpha * (reward - V[state])
            else:
                V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
        V_history.append(np.mean(V))
    return V, V_history

