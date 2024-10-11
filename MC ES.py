import gym
import numpy as np
from collections import defaultdict

# Create the Blackjack environment
env = gym.make('Blackjack-v1')

# Initialize action-value function (Q) and policy
Q = defaultdict(lambda: np.zeros(env.action_space.n))
policy = defaultdict(lambda: 0)  # Initialize policy to "stick" for all states

# Parameters
num_episodes = 50000  # Number of episodes to run
gamma = 1.0  # Discount factor
epsilon=0.01
# Store returns for each state-action pair to compute the average return
returns_sum = defaultdict(float)
returns_count = defaultdict(float)

def generate_episode(policy):
    """Generate an episode using the current policy."""
    episode = []
    
    # Reset the environment and capture the initial state
    initial_state = env.reset()
    
    # Extract the actual state from the returned value
    state = initial_state[0]  # Access the first element (the state)
 

    while True:
        action = policy[state]  # Follow the current policy
        next_state, reward, done, _,crack= env.step(action)
        
        # Extract the next state
        next_state = next_state[0]  # Ensure you're getting the state part
        
        episode.append((state, action, reward))
        
        if done:
            break
            
        state = next_state  # Move to the next state


    return episode

# Monte Carlo On-policy control
for i_episode in range(1, num_episodes + 1):
    if i_episode % 10000 == 0:
        print(f"Episode {i_episode}/{num_episodes}")
    
    # Generate an episode using the current policy
    episode = generate_episode(policy)

    # Compute the total return for each state-action pair in the episode
    G = 0
    visited_state_action_pairs = set()
    for t in range(len(episode) - 1, -1, -1):
        state, action, reward = episode[t]
        G = gamma * G + reward

        # If this state-action pair hasn't been encountered in this episode
        if (state, action) not in visited_state_action_pairs:
            visited_state_action_pairs.add((state, action))
            returns_sum[(state, action)] += G
            returns_count[(state, action)] += 1
            Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

            # Update the policy to be greedy with respect to Q
            policy[state] = np.argmax(Q[state])
        

# Test the learned policy
wins, losses, draws = 0, 0, 0
win_rate=[]
for count in range(100):
    
    for _ in range(1000):
        state = env.reset()
        state = state[0]  # Extract the state from the reset
        while True:
            action = policy[state]
            next_state, reward, done, _ ,crack= env.step(action)
            next_state = next_state[0]  # Extract the state after taking an action
            if done:
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1
                break
            state = next_state
    win_rate.append((wins/1000)*100)
win_rate=np.array(win_rate)
np.save('ES_win_rate.npy',win_rate)
