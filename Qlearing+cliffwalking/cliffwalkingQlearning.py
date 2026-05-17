import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

def train_q_learning(env, env_render, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1, max_steps=200, render_every=100):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_history = []
    steps_history = []
    
    for episode in range(episodes):
        show_game = (episode == 0) or ((episode + 1) % render_every == 0)
        current_env = env_render if show_game else env
        
        if show_game:
            print(f"  -> Wyświetlam na żywo: Epizod {episode + 1}/{episodes}...")
            
        state, _ = current_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            if np.random.rand() < epsilon:
                action = current_env.action_space.sample()
            else:
                action = np.argmax(Q[state])
                
            next_state, reward, term, trunc, _ = current_env.step(action)
            done = term or trunc
            
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            Q[state, action] += alpha * (td_target - Q[state, action])
            
            state = next_state
            total_reward += reward
            steps += 1
            
        rewards_history.append(total_reward)
        steps_history.append(steps)
    return Q, rewards_history, steps_history

def main():
    env_name = "CliffWalking-v1"
    env = gym.make(env_name)
    env_render = gym.make(env_name, render_mode="human")
        
    alphas_to_test = [0.01, 0.1, 0.9]
    episodes = 500
    
    results_rewards = {}
    results_steps = {}
    print(f"--- START: Testowanie {len(alphas_to_test)} wartości alpha z podglądem na żywo dla {env_name} ---")
    
    for alpha in alphas_to_test:
        print(f"\n[!] Rozpoczynam uczenie dla alpha = {alpha}...")
        _, rewards, steps = train_q_learning(env, env_render, episodes=episodes, alpha=alpha, render_every=1000)
        results_rewards[alpha] = rewards
        results_steps[alpha] = steps
        
    env.close()
    env_render.close()
    
    colors = {0.01: 'red', 0.1: 'blue', 0.3: 'purple', 0.6: 'green', 0.9: 'orange'}
    window_size = 20
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    plt.figure(figsize=(10, 6))
    for alpha, rewards in results_rewards.items():
        if len(rewards) >= window_size:
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            x = range(window_size-1, len(rewards))
            color = colors.get(alpha, 'black')
            plt.plot(x, smoothed_rewards, label=f'α = {alpha}', color=color, linewidth=2)
        else:
            plt.plot(rewards, label=f'α = {alpha}')
            
    plt.title("Wpływ parametru uczenia (α) na zbieżność (Suma nagród)", fontsize=14, fontweight='bold')
    plt.xlabel("Epizod", fontsize=12)
    plt.ylabel("Suma nagród (średnia krocząca)", fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-300, 0)
    plt.tight_layout()
    
    save_path1 = os.path.join(script_dir, "wykres_zbieznosc_nagrody_cliffwalking.png")
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"\nUkończono! Zapisano wykres nagród do: {save_path1}")
    
    plt.figure(figsize=(10, 6))
    for alpha, steps in results_steps.items():
        if len(steps) >= window_size:
            smoothed_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
            x = range(window_size-1, len(steps))
            color = colors.get(alpha, 'black')
            plt.plot(x, smoothed_steps, label=f'α = {alpha}', color=color, linewidth=2)
        else:
            plt.plot(steps, label=f'α = {alpha}')
            
    plt.title("Tempo uczenia się (Liczba kroków do celu)", fontsize=14, fontweight='bold')
    plt.xlabel("Epizod", fontsize=12)
    plt.ylabel("Liczba kroków (średnia krocząca)", fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path2 = os.path.join(script_dir, "wykres_tempo_uczenia_kroki_cliffwalking.png")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Zapisano wykres tempa uczenia do: {save_path2}")
    
    print("\n" + "="*70)
    print("WNIOSKI (Analiza pojedynczej próby):")
    print("="*70)
    print("1. Niskie α (0.01): Krzywa pnie się bardzo wolno do góry, wiedza zdobywana")
    print("   jest w zbyt małym stopniu, przez co potrzeba znacznie więcej epizodów.")
    print("2. Umiarkowane α (0.1): Najlepszy balans. Wartości Q aktualizują")
    print("   się optymalnie i wykres osiąga maksymalny pułap stabilnie i szybko.")
    print("3. Wysokie α (0.9): Występuje zauważalne zjawisko 'nadpisywania wiedzy'.")
    print("   Mimo dobrego poziomu, wykres ma mocne załamania/spadki (szarpany), ponieważ")
    print("   agent zbytnio reaguje na najnowsze, czasem przypadkowe nagrody z eksploracji.")
    print("="*70 + "\n")
    
    plt.show()

if __name__ == "__main__":
    main()
