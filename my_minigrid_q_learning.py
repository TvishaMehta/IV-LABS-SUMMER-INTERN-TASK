import gymnasium as gym
import minigrid
import numpy as np
import matplotlib.pyplot as plt
import random

#######SETTING UP ENVIRONMENT#########
env = gym.make("MiniGrid-Empty-6x6-v0", render_mode = 'rgb_array')
obs_space = env.observation_space
act_space = [0, 1, 2] # Actions: 0: left, 1: right, 2: forward
######################################

###########HYPERPARAMETERS############
alpha_initial = 0.1  # Initial learning rate
alpha_decay = 0.9995  # Decay factor for exponential decay
alpha_min = 0.01  # Minimum learning rate
gamma = 0.99  # Discount factor
# Removed lambda_ as it's not used in basic Q-learning (it's for Sarsa(lambda) or Q(lambda))
episodes = 5000  # Increased episodes for better learning
grid_size = 6
# GLIE epsilon initialization
epsilon = 1.0  # Start fully random
min_epsilon = 0.01  # Lower limit for exploration
epsilon_decay_rate = 0.995  # exponential decay for epsilon
######################################

##############Q TABLE#################
Q = {
    ((x, y), d): {act: 0.0 for act in act_space}
    for d in range(4) # 0: East, 1: South, 2: West, 3: North
    for y in range(1, grid_size) # Grid coordinates start from 0, but agent is usually within 1 to grid_size-1
    for x in range(1, grid_size)
}
######################################

########ALPHA DECAY FUNCTIONS#########
# Learning rate decay functions

def exponential_decay(alpha_initial, episode, decay_rate):
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
    """
    Chooses an action using an epsilon-greedy policy.
    This is the behavior policy (b).
    """
    if random.random() < epsilon:
        return random.choice(action_space)  # Explore
    else:
        q_values = Q_table[state]
        # Get actions with the maximum Q-value to break ties randomly
        max_q = max(q_values.values())
        greedy_actions = [act for act, q_val in q_values.items() if q_val == max_q]
        return random.choice(greedy_actions) # Exploit

def get_max_q_for_state(state, Q_table, action_space):
    """
    Returns the maximum Q-value for a given state.
    This is used for the target policy (pi), which is greedy.
    """
    q_values = Q_table[state]
    return max(q_values.values()) # Return the maximum Q-value
######################################

###############Q LEARNING#############
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

    terminated = False
    truncated = False

    net_reward = 0

    # For Q-learning, the action is chosen by the behavior policy, but the update
    # uses the greedy action from the *next* state.
    while not terminated and not truncated:
        # Action chosen by the behavior policy (epsilon-greedy)
        action = choose_action(state, Q, act_space, epsilon) # This is A_t

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = (env.unwrapped.agent_pos, env.unwrapped.agent_dir) # This is S_{t+1}

        # Ensure next_state is initialized in Q-table if it's new
        if next_state not in Q:
            Q[next_state] = {act: 0.0 for act in act_space}

        q_curr = Q[state][action]

        max_q_next = get_max_q_for_state(next_state, Q, act_space)

        Q[state][action] += current_alpha * (reward + gamma * max_q_next - q_curr)

        # updation for next iteration
        state = next_state # S_t becomes S_{t+1} for the next loop iteration

        # calculating total reward gained per episode
        net_reward += reward

    # keeping track of rewards across episodes
    reward_per_episode.append(net_reward)

    # Printing episode stats
    if reward > 0:  # A positive reward at the end indicates reaching the goal in MiniGrid
        total_successes += 1
        print(f"Episode {ep + 1}/{episodes}: Success! Alpha: {current_alpha:.4f}, Epsilon: {epsilon:.4f}, Reward: {net_reward:.2f}")
    else:
        print(f"Episode {ep + 1}/{episodes}: Failed. Alpha: {current_alpha:.4f}, Epsilon: {epsilon:.4f}, Reward: {net_reward:.2f}")

    # Update learning rate and epsilon for the next episode
    current_alpha = linear_decay(alpha_initial, ep, episodes) # Using linear decay as in your original script

    # epsilon decay in accordance with GLIE
    epsilon = exponential_decay(epsilon, ep, epsilon_decay_rate) # Ensure epsilon decays over time

    # Log metrics
    learning_rates.append(current_alpha)
    success_rate_history.append(total_successes / (ep + 1))

env.close()
######################################

###############RESULTS################
print(f"\nFinal Q Table (sample for a few states):")
# Print a sample of the Q-table for verification
sample_states = list(Q.keys())[:5] # Print first 5 states
for s in sample_states:
    print(f"State {s}: {Q[s]}")

print(f"\nFinal learning rate: {learning_rates[-1]:.4f}")
print(f"Final success rate: {success_rate_history[-1]:.2%}")

# Plot learning rate decay
plt.figure(figsize=(15, 6)) # Adjusted figure size for better layout

plt.subplot(1, 3, 1) # Changed to 1 row, 3 columns
plt.plot(learning_rates)
plt.title('Learning Rate Decay Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Learning Rate (α)')
plt.grid(True)

# Plot rewards per episode
plt.subplot(1, 3, 2) # Changed to 1 row, 3 columns
plt.plot(reward_per_episode)
plt.title('Rewards per episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)

# Plotting success rate
plt.subplot(1, 3, 3) # Changed to 1 row, 3 columns
plt.plot(success_rate_history)
plt.title('Success Rate over Episodes')
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.grid(True)

plt.tight_layout()
plt.show()

