import numpy as np


def epsilon_greedy_action(q_table, state, epsilon, rng):
    """Sample action from epsilon-greedy policy."""
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))
    return int(np.argmax(q_table[state]))


def td0_update_sarsa(
    q_table,
    state,
    action,
    reward,
    next_state,
    next_action,
    done,
    alpha,
    gamma,
):
    """
    One-step TD(0) update for action-value function (SARSA(0)).
    """
    if done:
        td_target = reward
    else:
        td_target = reward + gamma * q_table[next_state, next_action]

    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error
