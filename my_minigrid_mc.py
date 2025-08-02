import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
import numpy as np
import random

#########ENVIRONMENT SETUP############
env = gym.make("MiniGrid-Empty-6x6-v0", render_mode='rgb_array')
obs_space = env.observation_space
act_space = [0, 1, 2]
######################################

############HYPERPARAMETERS###########
alpha=0.09 #exponential forgetting rate
gamma=0.99
episodes=2000
grid_size=6
######################################

###############Q TABLE################
Q = {((x, y), d): {z: 0 for z in range(3)} for d in range(0, 4) for y in range(1, grid_size) for x in range(1, grid_size)}
######################################

#######IMPLEMENTING MONTE CARLO#######

rewards=[] # for plotting graph

# GLIE epsilon initialization
epsilon = 1.0         # Start fully random
min_epsilon = 0.01    # Lower limit for exploration
#decay_rate = 0.9   # Decay rate for epsilon, but in this exponential forgetting rate produced better results

#Training loop

for ep in range(episodes):

    obs, _ = env.reset() #env.reset() returns obs dict {'image':array[...], 'direction': 0, 'agent_pos':[2,2]} and info dict{seed': 123, 'mission': 'get to the goal square'}
    state = ((1, 1), 0)
    net_reward = 0

    terminated=False #setting flag to false for entering the loop
    truncated =False #setting flag to false for entering the loop

    curr_ep=[] # stores history of current episode
    ctr_step = 0 # to store number os steps per episode

    #runs until end of episode
    while not terminated and not truncated:
        # Implement GLIE epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(act_space)  # Random action (explore)
        else:
            q_values_for_state = [Q[state][act] for act in act_space]
            action = act_space[np.argmax(q_values_for_state)]  # taking action in accordance to q values calculated so far

        observation, reward, terminated, truncated, info = env.step(action)
        curr_ep.append((state, action, reward)) # storing current episode history
        state = (env.unwrapped.agent_pos, env.unwrapped.agent_dir) # re-initialising to the next episode
        ctr_step += 1 # counting steps

        net_reward+=reward # keeping track of net reward in the current episode

    # checking if episode was terminated or truncated
    if net_reward > 0: # terminated
        print(f"{ep} Reached goal at:", state, "with", ctr_step, "steps")
        rewards.append(net_reward)
    else: # truncated
        print(f"{ep} Episode finished at:", state, "with", ctr_step, "steps")

    G = 0
    for curr in reversed(range(len(curr_ep))): # calculating q values of all the states visited in the terminated or truncated episode
        curr_state, curr_action, curr_reward = curr_ep[curr]
        G = curr_reward + gamma * G  # Compute return from end
        Q[curr_state][curr_action] += (alpha) * (G - Q[curr_state][curr_action])  # Update value estimate

    # epsilon decay
    epsilon = max(min_epsilon, epsilon*(min_epsilon / epsilon) ** (1 / episodes))

######################################

#printing output
print("Final Q Table:")
print(Q)

#plotting rewards per episode
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title('Rewards over episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.show()