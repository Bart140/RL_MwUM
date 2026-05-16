import gymnasium as gym
import matplotlib
import argparse

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


def safe_segment_mean(values, start, end):
    start = max(0, min(start, values.size))
    end = max(start + 1, min(end, values.size))
    return float(np.mean(values[start:end]))


def compute_effect_metrics(results_raw):
    """
    Build quantitative checks for:
    - slow learning at very low alpha,
    - instability / overwriting at very high alpha.
    """
    smoothed = {alpha: moving_average(results_raw[alpha], SMOOTHING_WINDOW) for alpha in ALPHAS}

    low_curve = smoothed[0.01]
    mid_curve = smoothed[0.1]
    high_curve = smoothed[0.9]

    early_end = 300
    later_start = 1200
    later_end = 1800

    gain_low = safe_segment_mean(low_curve, later_start, later_end) - safe_segment_mean(
        low_curve, 0, early_end
    )
    gain_mid = safe_segment_mean(mid_curve, later_start, later_end) - safe_segment_mean(
        mid_curve, 0, early_end
    )

    # Overwriting proxy: stronger oscillation in late training.
    tail = 1200
    high_tail = results_raw[0.9][-tail:]
    mid_tail = results_raw[0.1][-tail:]
    rough_high = float(np.std(np.diff(high_tail)))
    rough_mid = float(np.std(np.diff(mid_tail)))

    metrics = {
        "gain_low": gain_low,
        "gain_mid": gain_mid,
        "rough_high": rough_high,
        "rough_mid": rough_mid,
        "slow_learning_ok": gain_low < 0.8 * gain_mid,
        "overwriting_ok": rough_high > 1.15 * rough_mid,
    }
    return metrics


def assert_effects(metrics):
    if not metrics["slow_learning_ok"]:
        raise AssertionError(
            "Brak efektu wolnego uczenia: gain(alpha=0.01) nie jest wystarczajaco mniejszy od gain(alpha=0.1)."
        )
    if not metrics["overwriting_ok"]:
        raise AssertionError(
            "Brak efektu nadpisywania: niestabilnosc alpha=0.9 nie jest wystarczajaco wieksza niz dla alpha=0.1."
        )


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


def main(assert_effects_enabled=False):
    results_raw = {}
    results_smoothed = {}
    eval_scores = {}
    value_profiles = {}

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

        value_runs = np.asarray([np.max(q_table, axis=1) for q_table in q_tables], dtype=np.float64)
        value_profiles[alpha] = {
            "mean": value_runs.mean(axis=0),
            "std": value_runs.std(axis=0),
        }

    baseline_runs = []
    for run_idx in range(RUNS_PER_ALPHA):
        run_seed = SEED_BASE + 20_000 + run_idx
        baseline_rewards = run_random_policy(seed=run_seed)
        baseline_runs.append(baseline_rewards)

    baseline_np = np.vstack(baseline_runs)
    baseline_mean = baseline_np.mean(axis=0)
    baseline_smoothed = moving_average(baseline_mean, SMOOTHING_WINDOW)

    metrics = compute_effect_metrics(results_raw)

    baseline_x = np.arange(SMOOTHING_WINDOW, SMOOTHING_WINDOW + baseline_smoothed.size)
    for alpha in ALPHAS:
        alpha_tag = f"{alpha:.2f}".replace(".", "_")

        plt.figure(figsize=(12, 7))
        smoothed = results_smoothed[alpha]
        x_axis = np.arange(SMOOTHING_WINDOW, SMOOTHING_WINDOW + smoothed.size)
        plt.plot(x_axis, smoothed, linewidth=2.2, label=f"alpha={alpha}")
        plt.plot(
            baseline_x,
            baseline_smoothed,
            linewidth=2.2,
            linestyle="--",
            color="black",
            label="baseline (losowa polityka)",
        )
        plt.title(f"Zbieznosc TD(0) w Taxi-v4 dla alpha={alpha}")
        plt.xlabel("Epizod")
        plt.ylabel("Suma nagrod (srednia ruchoma, okno=50)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        reward_filename = f"td0_taxi_reward_alpha_{alpha_tag}.png"
        plt.savefig(reward_filename, bbox_inches="tight", dpi=180)
        plt.close()
        print(f"\nZapisano wykres: {reward_filename}")

        plt.figure(figsize=(12, 7))
        mean_values = value_profiles[alpha]["mean"]
        std_values = value_profiles[alpha]["std"]
        states = np.arange(mean_values.size)
        plt.plot(states, mean_values, linewidth=2.2, label=f"alpha={alpha}")
        plt.fill_between(states, mean_values - std_values, mean_values + std_values, alpha=0.15)
        plt.title(f"Przyblizona funkcja stanu V(s)=max_a Q(s,a) po uczeniu, alpha={alpha}")
        plt.xlabel("Indeks stanu s")
        plt.ylabel("V(s)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        value_filename = f"td0_taxi_v_s_alpha_{alpha_tag}.png"
        plt.savefig(value_filename, bbox_inches="tight", dpi=180)
        plt.close()
        print(f"Zapisano wykres: {value_filename}")

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

    print("\nMetryki automatycznej walidacji efektow alpha:")
    print(
        f"- gain(alpha=0.01)={metrics['gain_low']:.3f}, "
        f"gain(alpha=0.1)={metrics['gain_mid']:.3f} | "
        f"slow_learning_ok={metrics['slow_learning_ok']}"
    )
    print(
        f"- rough(alpha=0.9)={metrics['rough_high']:.3f}, "
        f"rough(alpha=0.1)={metrics['rough_mid']:.3f} | "
        f"overwriting_ok={metrics['overwriting_ok']}"
    )

    if assert_effects_enabled:
        assert_effects(metrics)
        print("\nASSERTIONS OK: efekty wolnego uczenia i nadpisywania zostaly wykryte.")


def parse_args():
    parser = argparse.ArgumentParser(description="TD(0) alpha comparison in Taxi-v4")
    parser.add_argument(
        "--assert-effects",
        action="store_true",
        help="Wlacza twarde asercje efektow: wolne uczenie (alpha=0.01) i nadpisywanie (alpha=0.9).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(assert_effects_enabled=args.assert_effects)
