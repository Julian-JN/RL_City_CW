import numpy as np
from matplotlib import pyplot as plt


class Maze_env:
    """
    Represents a maze navigation environment for reinforcement learning tasks.
    It manages the maze layout, positions of start, target, and coin, and rewards/transitions. 
    Functionality includes:
    - `__init__(start, target, coins, maze)`: Initializes the environment.
    - `plot_env()`: Visualizes the maze with important positions highlighted.
    - `plot_env_position(position, timestep)`: Visualizes  maze with agent's position at specific timestep.
    - `create_r_matrix()`: Generates a reward matrix based on the maze layout.
    - `reward(state, action)`: Calculates the reward for an action taken from a state.
    - `transition(state, action)`: Determines the new state after an action.
    - `done()`: Checks if the target has been reached, ending the episode.
    - `create_q_matrix()`: Initializes a Q-learning matrix for action selection.
    """

    def __init__(self, start, target, coins, maze):
        self.maze = maze
        self.target = target
        self.start = start
        self.coins = coins
        self.position = 0
        self.R = self.create_r_matrix(reward_type="limited_movement")
        print(f" Shape of the R matrix is {self.R.shape}")
        self.Q = self.create_q_matrix()
        print(f" Shape of the Q matrix is {self.Q.shape}")
        self.coin_collected = False
        self.terminate = False

    def plot_env(self):
        cmap = plt.cm.colors.ListedColormap(
            ["white", "orange", "red", "blue", "yellow"]
        )
        maze_plot = self.maze.copy()
        maze_plot[self.target] = 2
        maze_plot[self.start] = 3
        maze_plot[self.coins] = 4
        plt.imshow(maze_plot, cmap=cmap)
        plt.show()

    def plot_env_position(self, position, timestep):
        cmap = plt.cm.colors.ListedColormap(
            ["white", "orange", "red", "blue", "yellow"]
        )
        maze_plot = self.maze.copy()
        maze_plot[self.target] = 2
        maze_plot[position] = 3
        maze_plot[self.coins] = 4
        plt.imshow(maze_plot, cmap=cmap)
        plt.savefig(f"img/plot_{timestep:06d}.png", dpi=300)
        plt.show()
        plt.close()

    def create_r_matrix(self, reward_type="terminal_movement"):
        """
        This synthesizes the reward and transition functions.
        reward_type (str): The type of reward to use.
        Options are "terminal_movement" and "free_movement".
        """
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        num_states = self.maze.shape[0] * self.maze.shape[1] * 2  # times coin state
        coin_states = 2  # 0 for no coin collected, 1 for coin collected
        R = np.full(
            (self.maze.shape[0], self.maze.shape[1], coin_states, len(actions)), np.nan)
    
        if reward_type == "terminal_movement":
            print("Reward type: Terminal Movement")
            # actions beyond limits get -10 (and terminate)
            # actions to a 0 -10 (and terminate)
            # action to coin get 200
            # action to target get 100
            # allowed actions get -1 (for the time)

            for i in range(self.maze.shape[0]):
                for j in range(self.maze.shape[1]):
                    for coin_state in range(coin_states):
                        for action_index, action in enumerate(actions):
                            new_i, new_j = i + action[0], j + action[1]
                            if new_i >= 0 and new_i < self.maze.shape[0] and new_j >= 0 and new_j < self.maze.shape[1]:
                                # Actions to a wall (1 in the maze) get -1
                                if self.maze[new_i, new_j] == 1:
                                    R[i, j, coin_state, action_index] = -1 # for the fire
                                elif self.maze[new_i, new_j] == 0:
                                    R[i, j, coin_state, action_index] = -0.005  # for an allowed action
                                    if (new_i, new_j) == self.coins and not coin_state:
                                        print("Assigning coin")
                                        R[i, j, coin_state, action_index] = 10  # coin
                                    elif (new_i, new_j) == self.target:
                                        print("Assigning target")
                                        R[i, j, coin_state, action_index] = 1  # target
                            else:
                                R[i, j, coin_state, action_index] = -1  # actions beyond the limits are forbidden

            return R
            # then add the transition function so that if reward smaller than -1, then terminate.
    
        if reward_type == "limited_movement":
            print("Reward type: Limited Movement")
            # actions beyond limits get None (can't move)
            # actions to a 0 (get -10)
            # action to coin get 200
            # action to target get 100
            # allowed actions get -1 (for the time)

            for i in range(self.maze.shape[0]):
                for j in range(self.maze.shape[1]):
                    for coin_state in range(coin_states):
                        for action_index, action in enumerate(actions):
                            new_i, new_j = i + action[0], j + action[1]

                            if new_i >= 0 and new_i < self.maze.shape[0] and new_j >= 0 and new_j < self.maze.shape[1]: # inside of maze
                                # Actions to a wall (1 in the maze) get None
                                if self.maze[new_i, new_j] == 1:
                                    R[i, j, coin_state, action_index] = None # for the fire
                                elif self.maze[new_i, new_j] == 0:
                                    R[i, j, coin_state, action_index] = -0.005  # for an allowed action
                                    if (new_i, new_j) == self.coins and not coin_state:
                                        R[i, j, coin_state, action_index] = 10  # coin
                                    elif (new_i, new_j) == self.target:
                                        R[i, j, coin_state, action_index] = 1  # target
                            else:
                                R[i, j, coin_state, action_index] = None  # actions beyond the limits are forbidden
            return R

    def transition_R(self, state, action, reward_type="terminal_movement"):
        initial_state = state
        x, y = initial_state
        new_x = x
        new_y = y
        if action == 0:  # up
            new_x -= 1
        elif action == 1:  # down
            new_x += 1
        elif action == 2:  # left
            new_y -= 1
        elif action == 3:  # right
            new_y += 1

        if reward_type == "terminal_movement":
            if new_x >= 0 and new_x < self.maze.shape[0] and new_y >= 0 and new_y < self.maze.shape[1]:
                if self.R[x, y, int(self.coin_collected), action] == -1: # fire
                    # print("Fire")
                    self.terminate = True
                    return state
                elif (new_x,new_y) == self.coins and not self.coin_collected: # coin
                    # print("Coin")
                    self.coin_collected = True
                    # print(self.coin_collected)
                    return new_x,new_y
                elif (new_x,new_y) == self.target: # target
                    # print("Target")
                    self.terminate = True
                    return new_x,new_y
                elif self.R[x, y, int(self.coin_collected), action] == -0.005: # normal action
                    # print("Allowed")
                    return new_x,new_y
            else:
                self.terminate = True  # walls
                # print("Out of bounds")
                return state

        if reward_type == "limited_movement": # should not attempt to access fire or wall
            if new_x >= 0 and new_x < self.maze.shape[0] and new_y >= 0 and new_y < self.maze.shape[1]:
                if (new_x,new_y) == self.coins and not self.coin_collected: # coin
                    self.coin_collected = True
                    return new_x,new_y
                elif (new_x,new_y) == self.target: # target
                    self.terminate = True
                    return new_x,new_y
                elif self.R[x, y, int(self.coin_collected), action] == -0.005: # normal action
                    return new_x,new_y

    def done(self):
        return self.terminate

    def coin_reached(self):
        return self.coin_collected

    def create_q_matrix(self):
        Q = np.zeros_like(self.R)
        return Q

if __name__ == "__main__":
    maze = np.array(
        [
            [1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        ]
    )
    env = Maze_env((2, 0), (0, 8), (7, 5), maze)
    env.plot_env()
    print("Info")
    print(env.R[0, 7, 0])
    print(env.R[0, 7, 1])
    print(env.R[7, 5, 0])
    print(env.R[7, 5, 1])
    print(env.R[7, 4, 0])
    print(env.R[7, 4, 1])