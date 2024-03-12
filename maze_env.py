import numpy as np
from matplotlib import pyplot as plt


class Maze_env:
    """
    Represents a maze navigation environment for reinforcement learning tasks.
    It manages the maze layout, start, target, and coin positions.
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
        self.R = 0
        self.Q = 0
        self.states = []
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

    def create_r_matrix(self):
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        num_states = self.maze.shape[0] * self.maze.shape[1]
        R = np.full((num_states, 4), np.nan)

        state_index = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                # If the cell is not a wall
                if self.maze[i, j] == 0:
                    for index, action in enumerate(actions):
                        new_position = (i + action[0], j + action[1])
                        # If action leads to a valid state
                        if (
                            0 <= new_position[0] < self.maze.shape[0]
                            and 0 <= new_position[1] < self.maze.shape[1]
                            and self.maze[new_position] == 0
                        ):
                            # Calculate the state number for the new position
                            # Set reward to 0
                            R[state_index, index] = -5

                            # If action leads to goal state set reward to 100
                            if new_position == self.target:
                                R[state_index, index] = 1000
                            if new_position == self.coins:
                                R[state_index, index] = 200
                state_index += 1

        self.R = R
        print(self.R.shape)
        return self.R

    def reward(self, state, action):
        state = self.states[state]
        x, y = state
        if action == 0:  # up
            x -= 1
        elif action == 1:  # down
            x += 1
        elif action == 2:  # left
            y -= 1
        elif action == 3:  # right
            y += 1
        if (
            x < 0
            or x >= len(self.maze)
            or y < 0
            or y >= len(self.maze[0])
            or self.maze[x][y] == 1
        ):
            return -0.0  # hit a wall (including edges wall?)
        elif (x, y) == self.target:
            print("Reached Target!")
            print((x, y))

            return 0 + int(
                self.coin_collected == True
            )  # reached the target and bonus if collected coin
        elif (x, y) == self.coins and not self.coin_collected:
            print("DING DING DING")
            return 10
        else:
            return -0.001  # regular step

    def transition(self, state, action):
        state_new = self.states[state]
        x, y = state_new
        if action == 0:  # up
            x -= 1
        elif action == 1:  # down
            x += 1
        elif action == 2:  # left
            y -= 1
        elif action == 3:  # right
            y += 1

        if (
            x < 0
            or x >= len(self.maze)
            or y < 0
            or y >= len(self.maze[0])
            or self.maze[x][y] == 1
        ):
            #             self.terminate = True
            return self.states.index(state_new)  # hit a wall, stay in the same state
        else:
            if (x, y) == self.coins and not self.coin_collected:
                self.coin_collected = True
                return 100  # specific index for coin
            else:
                if (x, y) == self.target:
                    self.terminate = True
                return self.states.index((x, y))  # move to the new state

    def done(self):
        return self.terminate

    def create_q_matrix(self):
        coord_to_index = []
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                coord_to_index.append((i, j))
        coord_to_index.append(self.coins)
        print(coord_to_index)

        num_states = self.maze.shape[0] * self.maze.shape[1] + 1
        num_actions = 4
        self.Q = np.zeros((num_states, num_actions))
        print(self.Q.shape)
        self.states = coord_to_index
        return self.Q, coord_to_index
