"""DQN and DDQN implementations in Pytorch for the OpenAI pendulum task

Done as part of my Deep Learning module assignment in Singapore Polytechnic
"""

import pickle
import time
import os

import numpy as np
import torch

from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from torchrl.data import ReplayBuffer, ListStorage

import gymnasium as gym  # open ai environment
import pygame

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

from lib.plot_q_value import plot_state_q

MODEL_LOCATION = "set a folder"


class QNetwork(nn.Module):
    """Q Network MLP"""

    def __init__(self, state_dim, action_dim, q_lr):
        super(QNetwork, self).__init__()

        # Define the layers of the Q Network
        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, action_dim)

        # Learning rate for optimizer
        self.lr = q_lr

        # Adam optimizer for Q Network
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # Move the Q Network to CUDA (GPU)
        self.to("cuda")

    def forward(self, x):
        # Define the forward pass of the Q Network
        q = F.leaky_relu(self.fc_1(x))
        q = F.leaky_relu(self.fc_2(q))
        q = self.fc_out(q)
        return q


class DQNAgent:
    """DQN Agent class, no target network"""

    def __init__(self, n=9, distribution="linear", render_mode="human"):
        # Environment string and rendering mode
        self.env_string = "Pendulum-v1"
        self.render_mode = render_mode

        # Define parameters for the DQN agent
        self.state_dim = 3
        self.action_dim = n
        self.lr = 0.01
        self.gamma = 0.98
        self.tau = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.01
        self.buffer_size = 100000
        self.batch_size = 256

        # Name for saving the model
        self.save_name = "dqn"
        self.save_image = False

        self.episodes = 0

        self.curr_mem_size = 0
        self.min_memory = 1000

        self.max_score = -np.inf

        # Distribution type for action space discretization
        self.distribution = distribution

    def build(self):
        # Initialize environment, replay buffer, and action space
        self.init_env()
        self.init_replay_buffer()
        self.set_action_space(self.action_dim)

        # Build the Q Network
        self.Q = QNetwork(self.state_dim, self.action_dim, self.lr)

    def init_env(self):
        # Initialize the environment
        if self.render_mode is not None:
            self.env = gym.make(self.env_string, g=9.81, render_mode=self.render_mode)
        else:
            self.env = gym.make(self.env_string)

    def init_replay_buffer(self):
        # Create the replay buffer
        self.memory = ReplayBuffer(
            pin_memory=False,
            prefetch=5,
            storage=ListStorage(max_size=self.buffer_size),
            batch_size=self.batch_size,
        )

    def bell_curve_linspace(self, start, end, num_points):
        # Generate points on a bell curve
        nums = np.linspace(0.01, 0.99, num_points)
        result = [norm.ppf(num) for num in nums]

        # Scale the generated points to the specified range
        result = [
            (x * 2 / (np.max(result) - np.min(result)) + (end + start) / 2)
            * (end - (end + start) / 2)
            for x in result
        ]
        result = [list(x) for x in result]

        return result

    def set_action_space(self, n):
        # Set up the discrete action space
        self.discretize_action_space(n, self.distribution)
        self.action_space = self.discrete_action_space[:]

    def discretize_action_space(self, n, distribution):
        # Discretize the continuous action space
        print("Action Space Discretization: ")
        print("Lower Bound: ", self.env.action_space.low)
        print("Upper Bound: ", self.env.action_space.high)
        self.discrete_action_space = np.array(list(range(n)))
        if distribution == "linear":
            # Linearly spaced discrete action values
            self.discrete_action_space_actual = np.linspace(
                self.env.action_space.low, self.env.action_space.high, num=n
            )
        elif distribution == "normal":
            # Bell curve spaced discrete action values
            self.discrete_action_space_actual = self.bell_curve_linspace(
                self.env.action_space.low, self.env.action_space.high, n
            )
        print(self.discrete_action_space_actual)

    def continuize_action(self, action_no):
        # Convert discrete action to continuous action
        return self.discrete_action_space_actual[action_no]

    def choose_action(self, state):
        # Exploration-exploitation trade-off
        state = state.to("cuda")
        random_number = np.random.rand()
        if self.epsilon < random_number:
            with torch.no_grad():
                action = int(torch.argmax(self.Q(state)).cpu().numpy())
        else:
            action = np.random.choice(self.discrete_action_space)
        real_action = self.discrete_action_space_actual[action]

        return action, real_action

    def calc_target(self, mini_batch):
        # Calculate the target Q-values for training
        s, a, r, s_prime, done = mini_batch
        done_ = []
        for _done in done:
            done_.append(0.0 if _done else 1.0)
        done = torch.tensor(done_, dtype=torch.float).to("cuda").unsqueeze(1)

        with torch.no_grad():
            q_target = self.Q(s_prime.to("cuda")).max(1)[0].unsqueeze(1)
            target = r.to("cuda") + self.gamma * done * q_target
        return target

    def optimize(self):
        # Optimize the Q Network using the sampled mini-batch
        mini_batch = self.memory.sample()
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch

        s_batch = s_batch.to("cuda")
        a_batch = a_batch.to("cuda")

        a_batch = a_batch.type(torch.int64)

        td_target = self.calc_target(mini_batch)

        Q_a = self.Q(s_batch).gather(1, a_batch)
        q_loss = F.smooth_l1_loss(Q_a, td_target)  # L1 loss
        self.Q.optimizer.zero_grad()
        q_loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
        self.Q.optimizer.step()

        self.episode_loss.append(q_loss.detach().cpu().numpy())

    def train(self, episode):
        # Train the DQN agent over a specified number of episodes
        self.score_list = []

        for ep in range(episode):
            self.episode_loss = []
            state, _ = self.env.reset()
            score, terminate, truncate = 0.0, False, False
            start = time.time()

            while not terminate and not truncate:
                action, real_action = self.choose_action(torch.FloatTensor(state))

                state_prime, reward, terminate, truncate, _ = self.env.step(real_action)

                self.curr_mem_size = self.memory.add(
                    (
                        state,
                        np.array([action]),
                        np.array([reward]),
                        state_prime,
                        terminate or truncate,
                    )
                )

                score += reward

                state = state_prime

                if self.curr_mem_size > self.min_memory:
                    self.optimize()

            end = time.time()

            if ep % 10 == 0:
                self.save(f"{os.getcwd()}{MODEL_LOCATION}/{self.save_name}-ep")

            if len(self.episode_loss) < 1:
                self.episode_loss = 0

            print(
                f"[Episode: {ep}] - Total Rewards:{score:.1f} - Epsilon: {self.epsilon:.5f} - Loss: {np.mean(self.episode_loss):.5f} - {end-start:.2f}s"
            )

            self.episodes += 1
            self.score_list.append(score)

            if self.save_image:
                plot_state_q(
                    self.Q,
                    self.discrete_action_space_actual,
                    episode=self.episodes,
                    save_image=True,
                    save_name=f"img/{self.save_name}",
                )

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def save(self, path):
        # Save the current Q Network's state dict to a file
        torch.save(self.Q.state_dict(), f"{path}-{self.episodes}.pt")

    def load(self, path):
        # Load a saved Q Network state dict from a file
        self.Q.load_state_dict(torch.load(f"{path}.pt"))

    def plot(self, title):
        # Plot the episode rewards and rolling mean
        plt.figure(figsize=(10, 10))
        sns.lineplot(
            x=list(range(len(self.score_list))),
            y=self.score_list,
            label="Episode Rewards",
        )

        # Calculate rolling mean using pandas
        rolling_mean = (
            pd.Series(self.score_list).rolling(window=100, min_periods=1).mean()
        )
        sns.lineplot(
            x=list(range(len(self.score_list))),
            y=rolling_mean,
            label="Rolling Last 100 Episode Mean",
        )

        plt.ylabel("Total Rewards each Episode")
        plt.xlabel("Episodes")
        plt.title(title)
        plt.legend()  # Show legend with labels
        plt.show()

    def log(self, path):
        # Save the episode rewards to a text file
        np.savetxt(f"{path}-{self.episodes}.txt", self.score_list)

    def eval(self, episodes, render_mode="human"):
        # Evaluate the DQN agent over a specified number of episodes
        self.render_mode = render_mode
        self.init_env()

        eval_score_list = []
        self.epsilon = 0
        for ep in range(episodes):
            state, _ = self.env.reset()
            score, terminate, truncate = 0.0, False, False

            while not terminate and not truncate:
                _, real_action = self.choose_action(torch.FloatTensor(state))

                state_prime, reward, terminate, truncate, _ = self.env.step(real_action)

                score += reward

                state = state_prime

            print(f"[Episode: {ep}] - Total Rewards:{score:.1f}")

            eval_score_list.append(score)

        print(
            f"{self.episodes} Reward - Mean: {np.mean(eval_score_list)} - Max: {np.max(eval_score_list)} - Min: {np.min(eval_score_list)}"
        )
        return eval_score_list


class DQNAgent_Target(DQNAgent):
    def build(self):
        # Build DQNAgent_Target, inheriting from DQNAgent
        super().build()

        # Initialize a target Q Network
        self.Q_target = QNetwork(self.state_dim, self.action_dim, self.lr)
        self.Q_target.load_state_dict(self.Q.state_dict())

    def calc_target(self, mini_batch):
        # Calculate the target Q-values for training with a target Q Network
        s, a, r, s_prime, done = mini_batch
        done_ = []
        for _done in done:
            done_.append(0.0 if _done else 1.0)
        done = torch.tensor(done_, dtype=torch.float).to("cuda")

        with torch.no_grad():
            q_target = self.Q_target(s_prime.to("cuda")).max(1)[0].unsqueeze(1)
            target = r.to("cuda") + self.gamma * done * q_target
        return target

    def optimize(self):
        # Optimize both the Q Network and the target Q Network
        super().optimize()

        # Soft update (taken from DDPG) as opposed to hard update that was proposed in the original DQN paper
        for param_target, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )


class DDQNAgent(DQNAgent_Target):
    def calc_target(self, mini_batch):
        # Calculate the target Q-values for training using Double DQN
        s, a, r, s_prime, done = mini_batch
        done_ = []
        for _done in done:
            done_.append(0.0 if _done else 1.0)
        done = torch.tensor(done_, dtype=torch.float).to("cuda")

        s_prime = s_prime.to("cuda")

        with torch.no_grad():
            # DDQN target
            q_a = self.Q(s_prime.to("cuda")).argmax(1).unsqueeze(1)
            q_target = self.Q_target(s_prime.to("cuda")).gather(1, q_a)
            target = r.to("cuda") + self.gamma * done * q_target
        return target
