import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from td0_taxi import epsilon_greedy_action, td0_update_sarsa

ENV_ID = "Taxi-v4"
EPISODES = 5000
GAMMA = 0.99
EPSILON = 0.1
ALPHAS = [0.01, 0.1, 0.9]
RUNS_PER_ALPHA = 8
SMOOTHING_WINDOW = 50
MAX_STEPS_PER_EPISODE = 200
SEED_BASE = 1234
EVAL_EPISODES = 200


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if values.size < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def train_td0(alpha, episodes=EPISODES, gamma=GAMMA, epsilon=EPSILON, seed=None):
    """
    Train TD(0) control with epsilon-greedy policy (SARSA(0)).
    Returns episode reward sums and learned Q-table.
    """
    rng = np.random.default_rng(seed)
    env = gym.make(ENV_ID)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions), dtype=np.float64)
    episode_rewards = np.zeros(episodes, dtype=np.float64)

    for episode in range(episodes):
        episode_seed = None if seed is None else int(rng.integers(0, 2**31 - 1))
        state, _ = env.reset(seed=episode_seed)
        action = epsilon_greedy_action(q_table, state, epsilon, rng)

        total_reward = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if done:
                td0_update_sarsa(
                    q_table,
                    state,
                    action,
                    reward,
                    next_state,
                    next_action=0,
                    done=True,
                    alpha=alpha,
                    gamma=gamma,
                )
                break

            next_action = epsilon_greedy_action(q_table, next_state, epsilon, rng)
            td0_update_sarsa(
                q_table,
                state,
                action,
                reward,
                next_state,
                next_action,
                done=False,
                alpha=alpha,
                gamma=gamma,
            )

            state = next_state
            action = next_action

        episode_rewards[episode] = total_reward

        if (episode + 1) % 500 == 0:
            print(f"alpha={alpha} | epizod {episode + 1}/{episodes}")

    env.close()
    return episode_rewards, q_table


def run_random_policy(episodes=EPISODES, seed=None):
    """Baseline: random policy without learning."""
    rng = np.random.default_rng(seed)
    env = gym.make(ENV_ID)
    n_actions = env.action_space.n
    episode_rewards = np.zeros(episodes, dtype=np.float64)

    for episode in range(episodes):
        episode_seed = None if seed is None else int(rng.integers(0, 2**31 - 1))
        state, _ = env.reset(seed=episode_seed)
        total_reward = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = int(rng.integers(n_actions))
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break

        episode_rewards[episode] = total_reward

    env.close()
    return episode_rewards


def evaluate_greedy_policy(q_table, episodes=EVAL_EPISODES, seed=None):
    """Evaluate learned policy with epsilon=0 (pure greedy)."""
    rng = np.random.default_rng(seed)
    env = gym.make(ENV_ID)
    rewards = np.zeros(episodes, dtype=np.float64)

    for episode in range(episodes):
        episode_seed = None if seed is None else int(rng.integers(0, 2**31 - 1))
        state, _ = env.reset(seed=episode_seed)
        total_reward = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = int(np.argmax(q_table[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break

        rewards[episode] = total_reward

    env.close()
    return rewards


def main():
    results_raw = {}
    results_smoothed = {}
    eval_scores = {}

    for alpha in ALPHAS:
        print(f"\nStart eksperymentu dla alpha={alpha}")
        runs = []
        q_tables = []

        for run_idx in range(RUNS_PER_ALPHA):
            run_seed = SEED_BASE + run_idx
            rewards, q_table = train_td0(alpha=alpha, seed=run_seed)
            runs.append(rewards)
            q_tables.append(q_table)

        runs_np = np.vstack(runs)
        mean_rewards = runs_np.mean(axis=0)
        results_raw[alpha] = mean_rewards
        results_smoothed[alpha] = moving_average(mean_rewards, SMOOTHING_WINDOW)

        eval_runs = []
        for run_idx, q_table in enumerate(q_tables):
            eval_seed = SEED_BASE + 10_000 + run_idx
            eval_rewards = evaluate_greedy_policy(q_table, seed=eval_seed)
            eval_runs.append(eval_rewards.mean())
        eval_scores[alpha] = np.asarray(eval_runs, dtype=np.float64)

    baseline_runs = []
    for run_idx in range(RUNS_PER_ALPHA):
        run_seed = SEED_BASE + 20_000 + run_idx
        baseline_rewards = run_random_policy(seed=run_seed)
        baseline_runs.append(baseline_rewards)

    baseline_np = np.vstack(baseline_runs)
    baseline_mean = baseline_np.mean(axis=0)
    baseline_smoothed = moving_average(baseline_mean, SMOOTHING_WINDOW)

    plt.figure(figsize=(12, 7))
    for alpha in ALPHAS:
        smoothed = results_smoothed[alpha]
        x_axis = np.arange(SMOOTHING_WINDOW, SMOOTHING_WINDOW + smoothed.size)
        plt.plot(x_axis, smoothed, linewidth=2.0, label=f"alpha={alpha}")
    baseline_x = np.arange(SMOOTHING_WINDOW, SMOOTHING_WINDOW + baseline_smoothed.size)
    plt.plot(
        baseline_x,
        baseline_smoothed,
        linewidth=2.2,
        linestyle="--",
        color="black",
        label="baseline (losowa polityka)",
    )

    plt.title("Porownanie zbieznosci TD(0) w Taxi-v4 dla alpha = 0.01, 0.1, 0.9")
    plt.xlabel("Epizod")
    plt.ylabel("Suma nagrod (srednia ruchoma, okno=50)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig("td0_taxi_alpha_comparison.png", bbox_inches="tight", dpi=180)
    print("\nZapisano wykres: td0_taxi_alpha_comparison.png")

    print("\nSzybkie podsumowanie koncowych srednich (ostatnie 200 epizodow):")
    for alpha in ALPHAS:
        tail = results_raw[alpha][-200:]
        print(f"alpha={alpha}: mean={tail.mean():.2f}, std={tail.std():.2f}")
    baseline_tail = baseline_mean[-200:]
    print(
        "baseline (losowa polityka): "
        f"mean={baseline_tail.mean():.2f}, std={baseline_tail.std():.2f}"
    )

    print("\nEwaluacja rozwiazania (greedy policy po uczeniu, 200 epizodow):")
    for alpha in ALPHAS:
        scores = eval_scores[alpha]
        print(f"alpha={alpha}: mean={scores.mean():.2f}, std={scores.std():.2f}")


if __name__ == "__main__":
    main()
