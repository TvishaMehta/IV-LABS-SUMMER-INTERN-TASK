import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
import numpy as np
import random
import pygame

#########ENVIRONMENT SETUP############
env = gym.make("MiniGrid-Empty-6x6-v0", render_mode='rgb_array')
obs_space = env.observation_space
act_space = [0, 1, 2]  # 0: left, 1: right, 2: forward
######################################

##############PARAMETERS##############
alpha_initial = 0.1  # Initial learning rate
alpha_decay = 0.9995  # Decay factor for exponential decay
alpha_min = 0.01  # Minimum learning rate
gamma = 0.99  # Discount factor
lambda_ = 0.8  # Trace decay rate
episodes = 2000  # Increased episodes for better learning
grid_size = 6
# GLIE epsilon initialization
epsilon = 1.0  # Start fully random
min_epsilon = 0.01  # Lower limit for exploration
epsilon_decay_rate = 0.995  # A more standard exponential decay for epsilon
######################################

###############Q TABLE################
Q = {
    ((x, y), d): {act: 0.0 for act in act_space}
    for d in range(4)
    for y in range(1, grid_size)
    for x in range(1, grid_size)
}
######################################

########ALPHA DECAY FUNCTIONS#########
# Learning rate decay functions

def exponential_decay(alpha_initial, episode, decay_rate): # in this code we have also used this function for epsilon decay
    """Exponential decay: alpha = alpha_initial * decay_rate^episode"""
    return max(alpha_initial * (decay_rate ** episode), alpha_min)


def inverse_decay(alpha_initial, episode, decay_constant):
    """Inverse decay: alpha = alpha_initial / (1 + episode / decay_constant)"""
    return max(alpha_initial / (1 + episode / decay_constant), alpha_min)


def linear_decay(alpha_initial, episode, total_episodes):
    """Linear decay: alpha decreases linearly from initial to min over total episodes"""
    decay_rate = (alpha_initial - alpha_min) / total_episodes
    return max(alpha_initial - decay_rate * episode, alpha_min)
######################################

############CHOOSING ACTION###########
def choose_action(state, Q_table, action_space, epsilon):
    if state not in Q_table:
        return random.choice(action_space) # Explore in case state is not visited earlier
    if random.random() < epsilon:
        return random.choice(action_space)  # Explore
    else:
        q_values = Q_table[state] # storing all q values for a given state
        q_values_list = [q_values[act] for act in action_space] # converting the above to a list
        return action_space[np.argmax(q_values_list)] # Exploit
######################################


##########TD(ʎ) IMPLEMENTATION########
# Track metrics for plotting
learning_rates = []
success_rate_history = []
reward_per_episode=[]
total_successes = 0

current_alpha = alpha_initial

# Implementation loop
for ep in range(episodes):
    obs, _ = env.reset()
    state = (env.unwrapped.agent_pos, env.unwrapped.agent_dir)

    E = {s: {a: 0.0 for a in act_space} for s in Q.keys()}

    terminated = False
    truncated = False

    action = choose_action(state, Q, act_space, epsilon)

    net_reward = 0

    while not terminated and not truncated:
        # env.render()

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = (env.unwrapped.agent_pos, env.unwrapped.agent_dir)
        next_action = choose_action(next_state, Q, act_space, epsilon)

        # This handles the terminal state, which is not in our Q-table
        q_next = Q.get(next_state).get(next_action)

        # Calculate TD error
        delta = reward + gamma * q_next - Q[state][action]

        # updating eligibility traces
        E[state][action] += 1

        # updating q table with respect to eligibility traces and lambda, decaying eligibility traces
        for s in Q.keys():
            for a in act_space:
                Q[s][a] += current_alpha * delta * E[s][a]
                E[s][a] *= gamma * lambda_

        # updation for next iteration
        state = next_state
        action = next_action

        # calculating total reward gained per episode
        net_reward += reward

    # keeping track of rewards across episodes
    reward_per_episode.append(net_reward)

    # Printing episode stats
    if reward > 0:  # A positive reward at the end indicates reaching the goal in MiniGrid
        total_successes += 1
        print(f"Episode {ep + 1}/{episodes}: Success! Alpha: {current_alpha:.4f}, Epsilon: {epsilon:.4f}")
    else:
        print(f"Episode {ep + 1}/{episodes}: Failed. Alpha: {current_alpha:.4f}, Epsilon: {epsilon:.4f}")

    # Update learning rate and epsilon for the next episode

    # Calculate current learning rate using exponential decay
    # current_alpha = exponential_decay(alpha_initial, ep, alpha_decay)

    # Calculate current learning rate using inverse decay
    current_alpha = inverse_decay(alpha_initial, ep, alpha_decay)

    # Calculate current learning rate using linear decay
    # current_alpha = linear_decay(alpha_initial, ep, episodes)

    # epsilon decay in accordance with GLIE
    epsilon = exponential_decay(epsilon, epsilon_decay_rate, min_epsilon)

    # Log metrics
    learning_rates.append(current_alpha)
    success_rate_history.append(total_successes / (ep + 1))

env.close()
######################################

###############RESULTS################
print(f"\nFinal Q Table: {Q}")
print(f"\nFinal learning rate: {learning_rates[-1]:.4f}")
print(f"Final success rate: {success_rate_history[-1]:.2%}")

# Plot learning rate decay
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(learning_rates)
plt.title('Learning Rate Decay Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Learning Rate (α)')
plt.grid(True)

# Plot rewards per episode
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(reward_per_episode)
plt.title('Rewards per episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)

# Plotting success rate
plt.subplot(1, 2, 2)
plt.plot(success_rate_history)
plt.title('Success Rate over Episodes')
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.grid(True)

plt.tight_layout()
plt.show()
######################################