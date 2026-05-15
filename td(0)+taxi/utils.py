import numpy as np


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if values.size < window:
        return np.array([], dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def compute_reference_values(env, gamma=1.0, tol=1e-10, max_iter=50_000):
    """
    Iterative policy evaluation for uniform random policy in Taxi.
    Returns V_pi used as a convergence reference for TD(0) prediction.
    """
    base_env = env.unwrapped
    if not hasattr(base_env, "P"):
        raise AttributeError("Environment does not expose transition model P.")

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    action_prob = 1.0 / n_actions

    values = np.zeros(n_states, dtype=np.float64)
    new_values = np.zeros(n_states, dtype=np.float64)

    for _ in range(max_iter):
        max_delta = 0.0
        for state in range(n_states):
            estimate = 0.0
            for action in range(n_actions):
                for prob, next_state, reward, terminated in base_env.P[state][action]:
                    bootstrap = 0.0 if terminated else gamma * values[next_state]
                    estimate += action_prob * prob * (reward + bootstrap)

            new_values[state] = estimate
            max_delta = max(max_delta, abs(estimate - values[state]))

        values, new_values = new_values, values
        if max_delta < tol:
            break

    return values


def interact(
    env,
    agent,
    episodes_n=5000,
    reward_window=100,
    reference_values=None,
    tracked_states=None,
):
    """
    Run TD(0) interaction and collect convergence diagnostics.
    """
    if tracked_states is None:
        tracked_states = []

    reward_by_episode = []
    abs_td_by_episode = []
    abs_update_by_episode = []
    rmse_by_episode = []
    tracked_values = {state: [] for state in tracked_states}

    for i_episode in range(1, episodes_n + 1):
        state, _ = env.reset()

        episode_reward = 0.0
        abs_td_sum = 0.0
        abs_update_sum = 0.0
        step_count = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            td_error, update = agent.step(state, action, reward, next_state, done)

            episode_reward += reward
            abs_td_sum += abs(td_error)
            abs_update_sum += abs(update)
            step_count += 1
            state = next_state

            if done:
                break

        reward_by_episode.append(episode_reward)
        abs_td_by_episode.append(abs_td_sum / max(step_count, 1))
        abs_update_by_episode.append(abs_update_sum / max(step_count, 1))

        if reference_values is not None:
            current_values = agent.value_vector(env.observation_space.n)
            rmse = np.sqrt(np.mean((current_values - reference_values) ** 2))
            rmse_by_episode.append(rmse)

        for tracked_state in tracked_states:
            tracked_values[tracked_state].append(agent.V[tracked_state])

        if i_episode % 250 == 0 or i_episode == episodes_n:
            print(f"\rEpisode {i_episode}/{episodes_n}", end="")

    print()

    return {
        "avg_reward_window": moving_average(reward_by_episode, reward_window),
        "reward_by_episode": np.asarray(reward_by_episode, dtype=np.float64),
        "abs_td_by_episode": np.asarray(abs_td_by_episode, dtype=np.float64),
        "abs_update_by_episode": np.asarray(abs_update_by_episode, dtype=np.float64),
        "rmse_by_episode": np.asarray(rmse_by_episode, dtype=np.float64),
        "tracked_values": {
            state: np.asarray(values, dtype=np.float64)
            for state, values in tracked_values.items()
        },
    }
