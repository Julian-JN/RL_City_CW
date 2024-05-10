import gymnasium as gym
from gymnasium.utils.save_video import save_video

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_units=512):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.layer3 = nn.Linear(hidden_units, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNCNN(nn.Module): # DQN/DDQN
    def __init__(self, input_shape, n_actions, hidden_units=512):
        super(DQNCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self.get_conv_out_size(input_shape)

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, n_actions)
        )

    def get_conv_out_size(self, shape):
        conv_size = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_size.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.value(conv_out)


class Agent:
    def __init__(self, env, per=False, double=False, logger=None):
        self.logger = logger
        self.max_timesteps = 5000
        self.number_timesteps = 0
        self.number_episodes = 0
        self.epsilon = 1
        # Get number of actions from gym action space
        self.env = env
        self.n_actions = 4
        self.number_lives = 5
        # self.n_actions = env.action_space.shape[0]
        # num_bins = 61  # Number of bins for each action dimension
        # self.n_actions = num_bins ** self.n_actions
        print(self.n_actions)
        print(f"Number actions: {self.n_actions}")
        seed = None
        self.random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)

        # Get the number of state observations
        self.state, self.info = env.reset()
        print(f"State shape: {self.state.shape}")
        # self.n_observations = len(self.state)
        self.n_observations = self.state.shape
        checkpoint = torch.load(f"rl_chk/dqn-per-checkpoint_4mil.pth")
        self.policy_net = DQNCNN(self.n_observations, self.n_actions, hidden_units=512).to(device)
        self.policy_net.load_state_dict(checkpoint['q-network-state'])

        print(self.n_observations)
        print(env.observation_space.shape)
        # self.policy_net = DQN(env.observation_space.shape, self.n_actions).to(device)
        # self.target_net = DQN(env.observation_space.shape, self.n_actions).to(device)

        self.video = []
    # def discrete2cont_action(self, action):
    #     # Map the discrete action index to continuous torques
    #     num_bins = 61
    #     action_indices = np.unravel_index(action, (num_bins, num_bins, num_bins))
    #     torque_min = -1.0
    #     torque_max = 1.0
    #     torques = [torque_min + (torque_max - torque_min) * idx / (num_bins - 1) for idx in action_indices]
    #     return np.array(torques)

    def choose_action(self, state):
        # need to reshape state array and convert to tensor
        state_tensor = (torch.from_numpy(np.array(state)).unsqueeze(dim=0).to(device)).float()
        action = self.epsilon_greedy_policy(state_tensor, self.epsilon)
        return action

    def epsilon_greedy_policy(self, state, epsilon):
        """With probability epsilon explore randomly; otherwise exploit knowledge optimally."""
        action = self.greedy_policy(state)
        return action

    def uniform_random_policy(self, state):
        """Choose an action uniformly at random."""
        # random_vector = np.random.(low=-1, high=1, size=self.n_actions)
        # return random_vector
        return self.random_state.randint(self.n_actions)

    def greedy_policy(self, state):
        # print(state.shape)
        # print(state.dtype)
        """Choose an action that maximizes the action_values given the current state."""
        action = (self.policy_net(state)
                  .argmax()
                  .cpu()  # action_values might reside on the GPU!
                  .item())
        return action

    def select_greedy_actions(self, states, q_network):
        _, actions = q_network(states).max(dim=1, keepdim=True)
        # print(actions)
        return actions

    def step(self, state, action, reward, next_state, done):
        if not done:
            self.number_timesteps += 1

    def train_for_at_most(self):
        """Train agent for a maximum number of timesteps."""
        state, self.info = self.env.reset()
        score = 0
        done = False
        episode_timestep = 0
        state, _, _, _, _ = self.env.step(1)
        self.policy_net.eval()
        with torch.no_grad():
            for t in range(self.max_timesteps):
        #     while not done:

                action = self.choose_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                if info.get("lives") < self.number_lives:
                    self.number_lives = info.get("lives")
                    next_state, _, _, _, _ = self.env.step(1)

                self.video.append(self.env.render())
                self.step(state, action, reward, next_state, done)
                episode_timestep +=1
                state = next_state
                score += reward
                if done or truncated:
                    print("GAME OVER!")
                    save_video(self.video, "videos", fps=25, name_prefix="video-inference")
                    self.number_episodes += 1
                    self.video = []
                    break
            print(f"Episode {self.number_episodes} finished in {episode_timestep} timesteps score: {score}")
            if not done:
                print("TOO LONG!")
                save_video(self.video, "videos", fps=25, name_prefix="video-inference")
                self.number_episodes += 1
                self.video = []
        return score

    def train(self):
        scores = []
        target_score = float("inf")
        most_recent_scores = deque(maxlen=100)
        score = self.train_for_at_most()
        scores.append(score)
        most_recent_scores.append(score)
        return scores


def Preprocessing_env(env):

    env = gym.wrappers.AtariPreprocessing(env, noop_max=30,
                                      screen_size=84, terminal_on_life_loss=False,
                                      grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)

    env = gym.wrappers.FrameStack(env, 4)
    return env

if "main":
    # env = gym.make('CartPole-v1', render_mode="rgb_array")
    # env = gym.make('Hopper-v4', render_mode="rgb_array")
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = Preprocessing_env(env)
    dqn = Agent(env, per=False, double=True)
    scores = dqn.train()
    # plt.plot(scores)
    # plt.savefig("rewards.png")
    # plt.show()
