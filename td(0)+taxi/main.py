import gymnasium as gym
from utils import interact
from td0_taxi import Agent
from IPython.display import clear_output
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    env = gym.make('Taxi-v4', render_mode='ansi')
    
    alphas = [0.01, 0.1, 0.9]
    n_bootstraps = 10
    episodes_n = 5000
    
    results_mean = {}
    results_std = {}

    for a in alphas:
        print(f"\n Ewaluacja dla alpha={a} ---")
        all_runs_rewards = []
        for b in range(n_bootstraps):
            print(f"Bootstrap {b+1}/{n_bootstraps}...")
            agent = Agent(alpha=a)
            avg_rewards, _ = interact(env, agent, episodes_n=episodes_n)
            all_runs_rewards.append(avg_rewards)
            
        all_runs_np = np.array(all_runs_rewards)
        
        results_mean[a] = np.mean(all_runs_np, axis=0)
        results_std[a] = np.std(all_runs_np, axis=0)

    plt.figure(figsize=(10, 6))
    for a in alphas:
        mean_data = results_mean[a]
        std_data = results_std[a]
        x_axis = np.arange(len(mean_data))
        
        line, = plt.plot(x_axis, mean_data, label=f'alpha={a}')
        plt.fill_between(x_axis, mean_data - std_data, mean_data + std_data, alpha=0.2, color=line.get_color())
        
    plt.xlabel('Epizody')
    plt.ylabel('Średnia nagroda (z okna 100 epizodów)')
    plt.title(f'Krzywe uczenia TD(0) - uśrednione po {n_bootstraps} niezależnych próbach')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Zapis
    plt.savefig('wyniki_alfa_bootstrap.png', bbox_inches='tight')
    print("\nGotowe! Wykres zapisany jako wyniki_alfa_bootstrap.png")

if __name__ == '__main__':
    main()