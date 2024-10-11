#Q-learning
import gym
import numpy as np
from collections import defaultdict
import random
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create the Blackjack environment
env = gym.make('Blackjack-v1')

# Initialize action-value function (Q) and policy
Q = defaultdict(lambda: np.zeros(env.action_space.n))
value=defaultdict(float)
policy=defaultdict(lambda:0)
# Parameters
num_episodes = 500000  # Number of episodes to run
gamma = 1.0  # Discount factor
alpha=0.9
epsilon=0.001

# epsilon-soft policy for off-policy control
def follow_policy(state):
    if np.random.rand() < epsilon:
            return np.random.choice([0,1])  # Explore by choosing a random action
    else:
            return np.argmax(Q[state])  # Exploit by choosing the best action
        
# the final optimal deterministic policy based on Q
def optimum(state):
    return np.argmax(Q[state])    
 
#Q-learning implemented 
def generate_episode(num_episodes):
    """Generate an episode using the current policy."""  
    for _ in range (num_episodes):
    # Reset the environment and capture the initial state
        initial_state = env.reset()
        state = initial_state[0] 
        action=follow_policy(state)
        while (True):
            # Follow the current policy
            next_state, reward, done, _,crack= env.step(action)
            
            # Update Q value by SARSAMAX
            Q[state][action]+=  alpha*(reward+ gamma*max(Q[next_state]) - Q[state][action])
            
            state=next_state
            action=follow_policy(next_state)

            if done:
                break


        
generate_episode(num_episodes=50000)

# Use MC prediction to get value function for the newly estimated optimal policy w.r.t Q
#This part is just for visualization
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
np.save('q_learning_win_rate.npy',win_rate)
def generate_episode_prediction( env):
    
   
    states, actions, rewards = [], [], []
    
    # Initialize the gym environment
    observation = env.reset()
    observation=observation[0]
    
    while True:
        
        
        states.append(observation)
        
       
         
        action = optimum(observation)
        actions.append(action)
        
        # 
        observation, reward, done, info, crack = env.step(action)
        rewards.append(reward)
        
        # Break if the state is a terminal state
        if done:
             break
                
    return states, actions, rewards
def first_visit_mc_prediction(env, n_episodes):
    
    # First, we initialize the empty value table as a dictionary for storing the values of each state
    value_table = defaultdict(float)
    N = defaultdict(int)

    
    for _ in range(n_episodes):
        
        # Next, we generate the epsiode and store the states and rewards
        states, _, rewards = generate_episode_prediction(env)
        returns = 0
        
        for t in range(len(states) - 1, -1, -1):
            R = rewards[t]
            S = states[t]
            
            returns += R
            
            # first-visit MC
            
            if S not in states[:t]:
                N[S] += 1
                value_table[S] += (returns - value_table[S]) / N[S]
    
    return value_table
value=first_visit_mc_prediction(env, n_episodes=50000)

# Plot the 3D graph showing how good the policy can perform in all states available
def plot_blackjack(V, ax1, ax2):
    player_sum = np.arange(12, 21 + 1)
    dealer_show = np.arange(1, 10 + 1)
    usable_ace = np.array([False, True])
    state_values = np.zeros((len(player_sum), len(dealer_show), len(usable_ace)))

    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = V[player, dealer, ace]
    
    X, Y = np.meshgrid(player_sum, dealer_show)
 
    ax1.plot_wireframe(X, Y, state_values[:, :, 0])
    ax2.plot_wireframe(X, Y, state_values[:, :, 1])
 
    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('dealer showing')
        ax.set_xlabel('player sum')
        ax.set_zlabel('state-value')
fig, axes = pyplot.subplots(nrows=2, figsize=(6, 8),
subplot_kw={'projection': '3d'})

axes[0].set_title('value function without usable ace')

axes[1].set_title('value function with usable ace')
plot_blackjack(value, axes[0], axes[1])
plt.show()



    
