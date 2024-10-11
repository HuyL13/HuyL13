import numpy as np
import matplotlib.pyplot as plt

true_value=np.load("value_table.npy")

MSE=[]
Y=[]
class GridworldDP:
    def __init__(self, grid_size=(50, 50), targets=[(0, 0), (49, 49)], gamma=0.9):
        self.grid_size = grid_size
        self.targets = targets
        self.gamma = gamma  # Discount factor
        self.values = np.zeros(grid_size)  # Initialize value function for each state
        self.policy = np.full(grid_size, None)  # Policy: what action to take in each state
        self.actions = ['up', 'down', 'left', 'right']  # Available actions

    def get_possible_actions(self, state):
        """Returns the list of valid actions in the current state (to avoid moving out of the grid)."""
        row, col = state
        possible_actions = []
        if row > 0: possible_actions.append('up')
        if row < self.grid_size[0] - 1: possible_actions.append('down')
        if col > 0: possible_actions.append('left')
        if col < self.grid_size[1] - 1: possible_actions.append('right')
        return possible_actions

    def step(self, state, action):
        """Returns the next state and reward given the current state and action."""
        row, col = state
        if action == 'up':
            next_state = (row - 1, col)
        elif action == 'down':
            next_state = (row + 1, col)
        elif action == 'left':
            next_state = (row, col - 1)
        elif action == 'right':
            next_state = (row, col + 1)
        else:
            next_state = state

        # Check if we reached a target
        if next_state in self.targets:
            reward = -1  # Reaching target
        else:
            reward = -1  # Every step costs -1

        return next_state, reward
    
    def value_iteration(self, theta=0.0001):
        """Perform value iteration to find the optimal value function and policy."""
        dem=0
        while True:
            dem+=1
            delta = 0
            sum=0
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    state = (row, col)
                    
                    # Skip target states
                    if state in self.targets:
                        continue

                    v = self.values[state]
                    action_values = []

                    # Loop over all possible actions
                    for action in self.get_possible_actions(state):
                        next_state, reward = self.step(state, action)
                        action_value = reward + self.gamma * self.values[next_state]
                        action_values.append(action_value)

                    # Update the value of the state
                    self.values[state] = max(action_values)
                    
                    delta = max(delta, abs(v - self.values[state]))
            if delta < theta:
                break
            # Break if the change is smaller than the threshold (theta)
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                        sum+=(self.values[(row,col)]-true_value[row][col])**2
                
            sum=sum/2500
            MSE.append(sum)
            Y.append(dem)
       

        # Extract the policy from the computed value function
        self.extract_policy()

    def extract_policy(self):
        """After value iteration, extract the optimal policy based on the values."""
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                state = (row, col)

                # Skip target states
                if state in self.targets:
                    self.policy[state] = 'target'
                    continue

                action_values = []
                possible_actions = self.get_possible_actions(state)

                # Evaluate the value for each possible action
                for action in possible_actions:
                    next_state, reward = self.step(state, action)
                    action_value = reward + self.gamma * self.values[next_state]
                    action_values.append(action_value)

                # Assign the action that leads to the highest value
                best_action = possible_actions[np.argmax(action_values)]
                self.policy[state] = best_action

    def render_values(self):
        """Display the value function."""
        print("Value Function:")
        for row in range(self.grid_size[0]):
            print([round(self.values[row, col], 2) for col in range(self.grid_size[1])])
        print()

    def render_policy(self):
        """Display the policy."""
        print("Policy:")
        for row in range(self.grid_size[0]):
            print([self.policy[(row, col)] for col in range(self.grid_size[1])])
        print()


# Example usage
env = GridworldDP()
env.value_iteration()

MSE=np.array(MSE)
Y=np.array(Y)
MSE1=np.load('my_array.npy')
Y1=np.load('my_another_array.npy')
fig=plt.figure()
plt.plot(Y,MSE,label='value iteration',color='red',linestyle='--')
plt.plot(Y1,MSE1,label='policy iteration',color='blue',linestyle='-')
plt.xticks(np.arange(1,Y.size,step=5))
plt.xlim(0,50)
plt.title("Compare convergence speed of 2 DP method in Gridworld")
plt.xlabel("Number of iteration")
plt.ylabel("Mean Square Error")
plt.legend()
plt.show()
