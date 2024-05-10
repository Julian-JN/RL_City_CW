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
import os
from buffer import ReplayMemory
from logger import Logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"


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
    def __init__(self, env, per=False, double = False, logger = None):

        self.logger = logger
        self.GAMMA = 0.99 # 0.99 for breakout
        print("GAMMA VALUE IS: ")
        print(self.GAMMA)
        self.LR = 1e-4 # 1e-4 for breakout
        self.ALPHA = 1
        self.update_frequency = 4
        self.update_target_frequency = 20000 # 20k for tuned ddqn
        self.batch_size = 64
        self.per = per
        self.double_dqn = double

        self.replay = ReplayMemory(100000, use_per=self.per)
        if self.per:
            self.alpha = self.replay.alpha
            self.sum_tree = self.replay.sum_tree
            self.max_priority = self.replay.max_priority
        self.memory = self.replay.memory

        self.max_episodes = 5000
        self.number_episodes = 0
        self.max_timesteps = 2000
        self.number_timesteps = 0
        self.epsilon = 1

        # Get number of actions from gym action space
        self.env = env
        self.n_actions = 18 # 4 for breakout
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
        self.policy_net = DQNCNN(self.n_observations, self.n_actions, hidden_units=512).to(device)
        self.target_net = DQNCNN(self.n_observations, self.n_actions, hidden_units=512).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
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

    def has_sufficient_experience(self):
        """True if agent has enough experience to train on a batch of samples; False otherwise."""
        # return len(self.memory) >= self.batch_size
        if len(self.memory) == 5000:
            print("Sufficient experience recently obtained!!!")
        return len(self.memory) >= 5000

    def has_full_experience(self):
        """True if agent has enough experience to train on a batch of samples; False otherwise."""
        # return len(self.memory) >= self.batch_size
        if len(self.memory) == 100000:
            return len(self.memory) >= 100000

    def save(self, filepath):
        checkpoint = {
            "q-network-state": self.policy_net.state_dict(),
            "optimizer-state": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)

    def choose_action(self, state):
        # print(state.shape)
        # need to reshape state array and convert to tensor
        state_tensor = (torch.from_numpy(np.array(state)).unsqueeze(dim=0).to(device)).float()
        # choose uniform at random if agent has insufficient experience
        if not self.has_sufficient_experience():
            action = self.uniform_random_policy(state_tensor)
        else:
            # print("Sufficient experience")
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

    def learn(self, experiences, is_weights, idxs):
        """Update the agent's state based on a collection of recent experiences."""
        states, actions, rewards, next_states, dones = (torch.Tensor(np.array(vs)).to(device) for vs in zip(*experiences))

        actions = (actions.long()).unsqueeze(dim=1)
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)

        if self.double_dqn:
            target_q_values = self.double_q_learning_update(next_states,rewards, dones,self.GAMMA,self.policy_net,
                                                            self.target_net)
        else:
            target_q_values = self.q_learning_update(next_states,rewards,dones,self.GAMMA,self.target_net)
        online_q_values = (self.policy_net(states).gather(dim=1, index=actions))
        # losses = F.mse_loss(online_q_values, target_q_values, reduction='none')
        criterion = nn.HuberLoss(reduction='none') # Indiv test
        losses = criterion(online_q_values, target_q_values)
        td_errors = torch.sqrt(losses)  # used for PER
        is_weights_tensor = torch.tensor(np.array(is_weights), dtype=torch.float32, device=device)
        weighted_losses = losses * is_weights_tensor  # Apply IS weights
        loss = weighted_losses.mean()
        # updates the parameters of the online network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.replay.use_per:
            self.replay.update_priority(idxs, td_errors.cpu().detach().numpy()) #necessary?


    def step(self, state, action, reward, next_state, done):
        experience = Transition(state, action, reward, next_state, done)
        self.replay.push(experience)
        if not done:
            self.number_timesteps += 1
            # every so often the agent should learn from experiences
            if self.number_timesteps % self.update_frequency == 0 and self.has_sufficient_experience():

                batch, idxs, is_weights = self.replay.sample(self.batch_size)
                self.learn(experiences=batch, is_weights=is_weights, idxs=idxs)

            if self.number_timesteps % self.update_target_frequency == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_for_at_most(self):
        """Train agent for a maximum number of timesteps."""
        state, info = self.env.reset()
        state, _, _, _, _ = self.env.step(1)

        self.number_lives = 5
        score = 0
        done = False
        episode_timestep = 0
        # for t in range(self.max_timesteps):
        while not done:
            action = self.choose_action(state)
            # print(f"Action Dis: {action} Timestep: {episode_timestep}")
            # action_cont = self.discrete2cont_action(action)
            next_state, reward, done, truncated, info = self.env.step(action)
            reward = min(1, reward)
            if info.get("lives") < self.number_lives:
                self.number_lives = info.get("lives")
                self.step(state, action, reward, next_state, True)
                next_state, _, _, _, _ = self.env.step(1)

            else:
                self.step(state, action, reward, next_state, done)
            self.epsilon = np.interp(self.number_timesteps, [0, 500000], [1, 0.01]) # 500000 for breakout
            episode_timestep +=1
            state = next_state
            score += reward
            if done or truncated:
                print(f"Episode: {self.number_episodes} Timesteps {episode_timestep}")
                self.number_episodes += 1
                self.video = []
                break
        if self.number_episodes % 200 == 0:
            print(f"Episode: {self.number_episodes} finished in {episode_timestep} timesteps score: {score}")
            with open('prints.txt', 'a') as f:
                f.write(f"\nEpisode: {self.number_episodes} finished in {episode_timestep} timesteps score: {score}")
        return score

    def train(self):
        scores = []
        target_score = float("inf")
        most_recent_scores = deque(maxlen=100)
        best_score = float("-inf")
        self.policy_net.train()
        self.target_net.train()
        with open('prints.txt', 'w') as f:
            f.write("Starting prints")
        for i in range(self.max_episodes):
            score = self.train_for_at_most()
            if self.logger:
                logger.log({'Score': score})
            scores.append(score)
            most_recent_scores.append(score)
            average_score = np.mean(most_recent_scores)
            if self.logger:
                logger.log({'Mean Score 100 Episodes': average_score})

            if average_score >= target_score or self.number_timesteps >= 4000000: # 4 million episode limit for breakout
                print(f"\nEnvironment solved in {i:d} episodes!\tAverage Score: {average_score:.2f}")
                checkpoint_filepath = f"rl_chk/indiv-HUBER-ddqn-per-checkpoint{self.number_episodes}.pth"
                os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
                self.save(checkpoint_filepath)
                break
            elif average_score > best_score:
                best_score = average_score
                plt.plot(average_score)
                plt.savefig("rewards.png")
                with open('prints.txt', 'a') as f:
                    f.write("\nSaving checkpoint")
                print("Saving checkpoint")
                checkpoint_filepath = f"rl_chk/indiv-HUBER-ddqn-per-checkpoint_4mil.pth"
                self.save(checkpoint_filepath)
            if (i + 1) % 100 == 0:
                plt.plot(scores)
                plt.savefig("rewards.png")
                with open('prints.txt', 'a') as f:
                    f.write(f"\n\rEpisode: {i + 1}\tAverage Score: {average_score:.2f} Epsilon: {self.epsilon} N_Frames: {self.number_timesteps}")
                print(f"\rEpisode: {i + 1}\tAverage Score: {average_score:.2f} Epsilon: {self.epsilon} N_Frames: {self.number_timesteps}")

        return scores


def Preprocessing_env(env):

    env = gym.wrappers.AtariPreprocessing(env, noop_max=30,
                                      screen_size=84, terminal_on_life_loss=False,
                                      grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)

    env = gym.wrappers.FrameStack(env, 4)
    return env

if "main":
    # env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = gym.make("BattleZoneNoFrameskip-v4", render_mode="rgb_array")
    # env = gym.make('Hopper-v4')
    env = Preprocessing_env(env)

    # wandb_logger = Logger(
    #     f"PER-DDQN-Individual_HUBER",
    #     project='INM707-Task2')
    # logger = wandb_logger.get_logger()

    # If you want to use wandb
    dqn = Agent(env, per=True, double=True, logger = False)
    scores = dqn.train()
    plt.plot(scores)
    plt.savefig("rewards.png")
    # plt.show()
