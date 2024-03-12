#%%
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

    def create_r_matrix(self, reward_type="free_movement"):
        """
        reward_type (str): The type of reward to use. Options are:
        - "free_movement": ###-1 for each step, 100 for reaching the target, 200 for reaching the coin
        - "sparse": ###-1 for each step, 100 for reaching the target, 200 for reaching the coin
        """
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        num_states = self.maze.shape[0] * self.maze.shape[1] * 2  # times coin state
        coin_states = 2  # 0 for no coin collected, 1 for coin collected
        R = np.full(
            (self.maze.shape[0], self.maze.shape[1], coin_states, len(actions)), np.nan
        )

        if reward_type == "free_movement":
            # actions beyond limits get None
            # actions to a 0 (get -10
            # action to coin get 200
            # action to target get 100
            # allowed actions get -1

            for i in range(self.maze.shape[0]):
                for j in range(self.maze.shape[1]):
                    for coin_state in range(coin_states):
                        for action_index, action in enumerate(actions):
                            new_i, new_j = i + action[0], j + action[1]

                            if new_i >= 0 and new_i < self.maze.shape[0] and new_j >= 0 and new_j < self.maze.shape[1]:
                                # Actions to a wall (1 in the maze) get None
                                if self.maze[new_i, new_j] == 1:
                                    R[i, j, coin_state, action_index] = -10 # for the fire
                                elif self.maze[new_i, new_j] == 0:
                                    R[i, j, coin_state, action_index] = -1  # for an allowed action
                                    if (i, j) == self.coins and not coin_state:
                                        R[i, j, coin_state, action_index] = 100  # coin
                                    elif (i, j) == self.target:
                                        R[i, j, coin_state, action_index] = 200 # target
                            else:
                                R[i, j, coin_state, action_index] =  None  # actions beyond the limits are forbidden
        self.R = R
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

            return 0 + int(self.coin_collected == True)  # reached the target and bonus if collected coin
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

#%%
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

# %%
