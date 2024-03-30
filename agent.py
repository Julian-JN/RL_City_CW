import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from maze_env import Maze_env


class Q_learning:
    """
    Implements the Q-learning algorithm for reinforcement learning tasks within a predefined environment.
    This class is responsible for learning optimal action-selection policies to maximize rewards over episodes of interactions with the environment.

    - `__init__(alpha, gamma, epsilon, episodes, steps, env, states)`: Initializes the learning parameters, environment, and states.
    - `plot_rewards()`: Plots the rewards accumulated over each episode, visualizing the learning progress.
    - `show_Q_spec(coord)`: Displays Q-values for a specific coordinate/state.
    - `greedy_policy(state)`: Selects an action based on a greedy policy (highest Q-value) with an epsilon chance of random action for exploration.
    - `softmax_policy(state, temperature)`: Selects an action based on the softmax of Q-values, factoring in the temperature for exploration-exploitation balance.
    - `train()`: Conducts the learning process over a specified number of episodes and steps per episode, updating Q-values based on the received rewards.
    - `create_video()`: Generates a video from saved images of the agent's journey through the maze, illustrating the learned policy in action.
    - `test(limit)`: Evaluates the learned policy by navigating the environment for a given number of steps, visualizing the path taken and summarizing the rewards.

    The class utilizes epsilon-greedy and softmax policies for action selection, balancing the exploration of the state space with the exploitation of known rewards.
    """

    def __init__(self, alpha, gamma, epsilon, episodes, steps, env):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.temperature = 100.0
        self.R = env.R
        self.R_mod = self.R
        self.Q = env.Q
        self.episodes = episodes
        self.steps = steps
        self.start = env.start
        self.target = env.target
        self.coins = env.coins
        self.env = env
        self.episodes_rewards = []
        self.max_list_size = 10
        self.list_rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.threshold = 2
        self.window_size = 4
        self.current_average = 0

        print("Initial Q matrix shape is '{}'".format(self.Q.shape))
        print("Initial Q matrix values are '{}'".format(self.Q))

    def plot_rewards(self):
        plt.plot(self.episodes_rewards)
        plt.show()

    def show_Q_spec(self, coord):
        i, j = coord
        print(self.Q[i, j, int(self.env.coin_reached()), :])

    def greedy_policy(self, state):
        i, j = state
        available_actions = np.where(~np.isnan(self.R_mod[i, j, int(self.env.coin_reached())]))[0]
        print(available_actions)
        q_values = [self.Q[i, j, int(self.env.coin_reached()), a] for a in available_actions]
        best_actions = available_actions[np.where(q_values == np.max(q_values))[0]]
        # print(best_actions)

        # available_actions = np.array([0, 1, 2, 3])
        # q_values = [self.Q[state, a] for a in available_actions]
        # best_actions = available_actions[np.where(q_values == np.max(q_values))[0]]

        if np.random.uniform() < self.epsilon:
            # a = np.random.choice(4)
            a = np.random.choice(available_actions)
        else:
            #                     a = np.argmax(self.Q[s,:])
            a = np.random.choice(best_actions)
        # print(a)
        return a

    def softmax_policy(self, state, temperature=1.0):
        i, j = state
        available_actions = np.where(~np.isnan(self.R_mod[i, j, int(self.env.coin_reached())]))[0]
        q_values = [self.Q[i, j, int(self.env.coin_reached()), a] for a in available_actions]
        max_q_value = np.max(q_values)
        exp_values = np.exp((q_values - max_q_value) / temperature)
        action_probs = exp_values / np.sum(exp_values)

        # Sample an action based on the probabilities
        selected_action = np.random.choice(len(action_probs), p=action_probs)
        return selected_action

    def train(self):
        print("Target is '{}'".format(self.target))
        print("Starting state is '{}'".format(self.start))

        for episode in range(self.episodes):
            # print("New episode")
            s = self.start
            episode_reward = 0
            self.env.coin_collected = False
            self.env.terminate = False
            # print("New episode")
            for timestep in range(self.steps):
                # print(self.env.coin_reached())
                i, j = s
                # Epsilon-greedy action choice
                a = self.greedy_policy(s)
                #                 a = self.softmax_policy(s, self.temperature)
                # Environment updating
                # r = env.reward(s, a)
                # print(self.R_mod[i,j,int(self.env.coin_collected)])
                r = self.R_mod[i, j, int(self.env.coin_reached()), a]
                # print(r)
                # if self.env.coin_reached():
                #     print()
                #     print("Coin collected")
                # print(self.env.coin_reached())
                # print("Reward")
                episode_reward += r
                new_state = self.env.transition_R((i, j), a, reward_type="limited_movement")
                new_i, new_j = new_state

                if r == 10: # picked up coin for first time
                    # print("COOOOOOOOOOOOOOOOIIIIIIIINNNNNNNN")
                    # Current q in state o and next in state 1 for coin_collected
                    self.Q[i, j, 0, a] = self.Q[i, j, 0, a] + self.alpha * (r + self.gamma * np.max(
                        self.Q[new_i, new_j, 1, :]) - self.Q[i, j, 0, a])
                else:
                    self.Q[i, j, int(self.env.coin_reached()), a] = self.Q[i, j, int(
                        self.env.coin_reached()), a] + self.alpha * (r + self.gamma * np.max(
                        self.Q[new_i, new_j, int(self.env.coin_reached()), :]) - self.Q[
                                                                         i, j, int(self.env.coin_reached()), a])

                if self.env.done():
                    # print("Death")
                    # print(self.env.terminate)
                    break
                s = new_state

            self.episodes_rewards.append(episode_reward)

            self.list_rewards.append(episode_reward)
            if len(self.list_rewards) > self.max_list_size:
                self.list_rewards.pop(0)
            window = self.list_rewards[-self.window_size:]
            window_average = sum(window) / self.window_size
            self.current_average = window_average

            if episode % 5 == 0:
                print('Episode {} finished. Episode Reward {}. Timesteps {}. Average {}. Epsilon {}'.format(episode,
                                                                                                         episode_reward,
                                                                                                         timestep,
                                                                                                         window_average,
                                                                                                         self.epsilon))
            self.epsilon = np.interp(episode, [0, self.episodes], [1, 0.05])

    def create_video(self):
        image_folder = "img"  # Directory containing your saved plot images
        video_name = "video_agent.mp4"

        images = [
            img
            for img in os.listdir(image_folder)
            if img.endswith((".jpg", ".jpeg", ".png"))
        ]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(
            video_name, cv2.VideoWriter_fourcc(*"mp4v"), 1, (width, height)
        )

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

    def test(self, limit=40):
        s = self.start
        print("Starting state is '{}'".format(s))
        episode_reward = 0
        env.coin_collected = False
        env.terminate = False
        for timestep in range(limit):
            i, j = s
            print("Step {}".format(timestep))
            self.env.plot_env_position(s, timestep)
            print(self.Q[i, j, int(self.env.coin_reached())])
            a = np.argmax(self.Q[i, j, int(self.env.coin_reached())])
            print(a)

            # Environment updating
            r = self.R_mod[i, j, int(self.env.coin_reached()), a]
            print(r)
            episode_reward += r
            new_state = self.env.transition_R((i, j), a, reward_type="terminal_movement")
            new_i, new_j = new_state

            if env.done():
                self.env.plot_env_position(new_state, timestep+1)
                break
            s = new_state
        print('Episode Reward {}.Q matrix values:\n{}'.format(episode_reward, self.Q.round(1)))
        self.create_video()


if "main":
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

    q_learning = Q_learning(alpha=1, gamma=0.999, epsilon=1, episodes=10000, steps=200, env=env)
    print("Info")
    print(q_learning.R_mod[0, 7, 0])
    print(q_learning.R_mod[0, 7, 1])
    print(q_learning.R_mod[7, 5, 0])
    print(q_learning.R_mod[7, 5, 1])
    print(q_learning.R_mod[7, 4, 0])
    print(q_learning.R_mod[7, 4, 1])

    q_learning.train()
    q_learning.plot_rewards()
    q_learning.test()
    print(q_learning.Q[3, 3, 0])
    print(q_learning.Q[7, 4, 0])
