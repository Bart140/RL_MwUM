from td0_in_cliff import test_random_agent
import gymnasium as gym

env = gym.make('CliffWalking-v1')

def main():
    avg_reward = test_random_agent(env, 500)
    print(f"Skuteczność agenta losowego: {avg_reward}")

if __name__ == "__main__":
    main()
