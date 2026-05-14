from collections import defaultdict, deque
import math
import numpy as np
import sys
import time

def interact(env, agent, episodes_n = 20000, window = 100):
    # initialize average rewards
    avg_rewards = deque(maxlen=episodes_n)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # for each episode
    for i_episode in range(1,episodes_n+1):
        # begin the episode
        state, _ = env.reset()
        # initialize the sampled reward
        samp_reward = 0
        while True:
            # agent selects an action
            action = agent.select_action(state)
            # agent performs the selected action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # agent performs internal updates based on sampled experience
            agent.step(state,action,reward,next_state,done)
            # update the sampled reward
            samp_reward +=reward
            # update the state (s <- s`) to next time step
            state = next_state
            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                break

            if i_episode >= window :
                # get average reward from last 100 episodes
                avg_reward = np.mean(samp_rewards)
                # append to deque
                avg_rewards.append(avg_reward)
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward

        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, episodes_n, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == episodes_n:
            print('\n')
        time.sleep(0.001)  # 1ms
    return avg_rewards, best_avg_reward