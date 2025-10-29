import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from src.models.advanced_networks import DQN
from src.utils.schedules import LinearSchedule

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_shape, action_size, config, device):
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = device

        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 1e-4)
        self.batch_size = config.get("batch_size", 64)
        self.buffer_size = config.get("buffer_size", 100000)
        self.target_update_freq = config.get("target_update_freq", 1000)

        self.epsilon = config.get("epsilon_start", 1.0)

        self.epsilon_schedule = LinearSchedule(
            start=self.epsilon,
            end=config.get("epsilon_end", 0.01),
            duration=config.get("epsilon_decay", 50000)
        )

        self.memory = ReplayBuffer(self.buffer_size)

        self.policy_net = DQN(state_shape, action_size).to(device)
        self.target_net = DQN(state_shape, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.update_target_network()

        self.steps_done = 0

    def act(self, state, training=True):
        self.epsilon = eps_threshold = self.epsilon_schedule.get_value(self.steps_done)
        if training and random.random() < eps_threshold:
            action = random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.policy_net(state_tensor).argmax().item()
        self.steps_done += 1

        return action

    def remember(self, state, action, reward, next_state, done):
        exp = Experience(state, action, reward, next_state, done)
        self.memory.add(exp)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(np.stack([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor([float(e.done) for e in batch]).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.update_target_network()


def load_trained_agent(model_path, state_shape, action_size, device):
    agent = DQNAgent(state_shape, action_size, {}, device)
    agent.load(model_path)
    return agent

