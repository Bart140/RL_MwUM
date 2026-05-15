import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from nikos.stuff.td0_in_cliff import train_td0

env = gym.make('CliffWalking-v1')
alphas = [0.01, 0.1, 0.9]
episodes = 500
plt.figure(figsize=(12,7))

for alpha in alphas:
    print(f"Uczenie dla alfa = {alpha}")
    V_table, V_history = train_td0(env, alpha = alpha, episodes=episodes)
    print(f"Średnia ocena planszy dla alfa = {alpha}: {np.mean(V_table):.2f}")
    plt.plot(V_history, label=f"alfa = {alpha}")

plt.title("Zbieżność algorytmu TD(0) - wpływ parametru alfa")
plt.xlabel("Numer epizodu")
plt.ylabel("Średnia wartość wszystkich stanów V(s)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()