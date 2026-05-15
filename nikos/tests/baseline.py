from nikos.stuff.td0_in_cliff import test_random_agent
import gymnasium as gym

env = gym.make('CliffWalking-v1')

avg_reward = test_random_agent(env, 500)
print(f"Skuteczność agenta losowego: {avg_reward}")

