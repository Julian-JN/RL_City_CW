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

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        """Add a new experience to memory."""
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_units=64):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.layer3 = nn.Linear(hidden_units, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Agent:
    def __init__(self, env):
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.TAU = 0.005
        self.LR = 1e-4
        self.ALPHA = 1
        self.GAMMA = 1
        self.update_frequency = 4
        self.update_target_frequency = 1000
        self.batch_size = 64

        # Get number of actions from gym action space
        self.env = env
        self.n_actions = env.action_space.n
        seed = None
        self.random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)

        # Get the number of state observations
        self.state, self.info = env.reset()
        print(self.state.shape)
        self.n_observations = len(self.state)

        self.double_dqn = False
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(50000)
        self.max_episodes = 2000
        self.number_episodes = 0
        self.max_timesteps = 2000
        self.number_timesteps = 0
        self.epsilon = 1

        self.video = []

    def has_sufficient_experience(self):
        """True if agent has enough experience to train on a batch of samples; False otherwise."""
        return len(self.memory) >= self.batch_size

    def save(self, filepath):
        checkpoint = {
            "q-network-state": self.policy_net.state_dict(),
            "optimizer-state": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)

    def choose_action(self, state):
        # need to reshape state array and convert to tensor
        state_tensor = (torch.from_numpy(state).unsqueeze(dim=0).to(device))
        # choose uniform at random if agent has insufficient experience
        if not self.has_sufficient_experience():
            action = self.uniform_random_policy(state_tensor)
        else:
            action = self.epsilon_greedy_policy(state_tensor, self.epsilon)
        return action

    def epsilon_greedy_policy(self, state, epsilon):
        """With probability epsilon explore randomly; otherwise exploit knowledge optimally."""
        if self.random_state.random() < epsilon:
            action = self.uniform_random_policy(state)
        else:
            action = self.greedy_policy(state)
        return action

    def uniform_random_policy(self, state):
        """Choose an action uniformly at random."""
        return self.random_state.randint(self.n_actions)

    def greedy_policy(self, state: torch.Tensor) -> int:
        """Choose an action that maximizes the action_values given the current state."""
        action = (self.policy_net(state)
                  .argmax()
                  .cpu()  # action_values might reside on the GPU!
                  .item())
        return action

    def select_greedy_actions(self, states, q_network):
        _, actions = q_network(states).max(dim=1, keepdim=True)
        return actions

    def evaluate_selected_actions(self, states,actions,rewards,dones,gamma,q_network):
        """Compute the Q-values by evaluating the actions given the current states and Q-network."""
        next_q_values = q_network(states).gather(dim=1, index=actions)
        q_values = rewards + (gamma * next_q_values * (1 - dones))
        return q_values

    def q_learning_update(self, states,rewards,dones,gamma,q_network):
        """Q-Learning update with explicitly decoupled action selection and evaluation steps."""
        actions = self.select_greedy_actions(states, q_network)
        q_values = self.evaluate_selected_actions(states, actions, rewards, dones, gamma, q_network)
        return q_values

    def double_q_learning_update(self, states,rewards,dones,gamma,q_network1, q_network2):
        """Q-Learning update with explicitly decoupled action selection and evaluation steps."""
        actions = self.select_greedy_actions(states, q_network1)
        q_values = self.evaluate_selected_actions(states, actions, rewards, dones, gamma, q_network2)
        return q_values

    def learn(self, experiences):
        """Update the agent's state based on a collection of recent experiences."""
        states, actions, rewards, next_states, dones = (torch.Tensor(vs).to(device) for vs in zip(*experiences))

        # need to add second dimension to some tensors
        actions = (actions.long().unsqueeze(dim=1))
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)

        if self.double_dqn:
            target_q_values = self.double_q_learning_update(next_states,rewards, dones,self.GAMMA,self.policy_net,
                                                            self.target_net)
        else:
            target_q_values = self.q_learning_update(next_states,rewards,dones,self.GAMMA,self.target_net)

        online_q_values = (self.policy_net(states).gather(dim=1, index=actions))
        # compute the mean squared loss
        loss = F.mse_loss(online_q_values, target_q_values)
        # updates the parameters of the online network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.number_timesteps % self.update_target_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def step(self, state, action, reward, next_state, done):
        experience = Transition(state, action, reward, next_state, done)
        self.memory.push(experience)

        if done:
            self.number_episodes += 1
        else:
            self.number_timesteps += 1

            # every so often the agent should learn from experiences
            if self.number_timesteps % self.update_frequency == 0 and self.has_sufficient_experience():
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def train_for_at_most(self):
        """Train agent for a maximum number of timesteps."""
        state, self.info = self.env.reset()
        score = 0
        done = False
        episode_timestep = 0
        # for t in range(self.max_timesteps):
        while not done:
            # if episode_timestep > 1000:
            #     print("Long training")
            action = self.choose_action(state)
            # print(self.env.step(action))
            next_state, reward, done, _, _ = self.env.step(action)
            # self.env.render()
            self.video.append(self.env.render())
            # print(self.video)
            save_video(self.video, "videos", fps=25,
                       episode_trigger=lambda x: x % 200 == 0,
                       episode_index=self.number_episodes)
            self.step(state, action, reward, next_state, done)
            episode_timestep +=1
            state = next_state
            score += reward
            if done:
                self.video = []
                break
        print(f"Episode {self.number_episodes} finished in {episode_timestep} timesteps")
        return score

    def train(self):
        scores = []
        checkpoint_filepath = "double-dqn-checkpoint.pth"
        target_score = float("inf")
        most_recent_scores = deque(maxlen=100)

        for i in range(self.max_episodes):
            # self.number_timesteps = 0
            score = self.train_for_at_most()
            self.epsilon = np.interp(i, [0, self.max_episodes], [1, 0.01])

            scores.append(score)
            most_recent_scores.append(score)

            average_score = sum(most_recent_scores) / len(most_recent_scores)
            if average_score >= target_score:
                print(f"\nEnvironment solved in {i:d} episodes!\tAverage Score: {average_score:.2f}")
                self.save(checkpoint_filepath)
                break
            if (i + 1) % 50 == 0:
                print(f"\rEpisode {i + 1}\tAverage Score: {average_score:.2f} Epsilon: {self.epsilon}")

        return scores


if "main":
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    dqn = Agent(env)
    scores = dqn.train()
    plt.plot(scores)
    plt.show()
