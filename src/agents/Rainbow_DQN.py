import math
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

# ------------------------
# 1) Noisy Linear (factorized)
# ------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.reset_parameters(sigma0)
        self.reset_noise()

    def reset_parameters(self, sigma0):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, +mu_range)
        self.bias_mu.data.uniform_(-mu_range, +mu_range)

        # Factorized Gaussian
        self.weight_sigma.data.fill_(sigma0 / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(sigma0 / math.sqrt(self.in_features))

    def _f(self, x):
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = torch.randn(self.in_features, device=self.weight_eps.device)
        eps_out = torch.randn(self.out_features, device=self.weight_eps.device)
        f_in = self._f(eps_in)
        f_out = self._f(eps_out)
        self.weight_eps.copy_(f_out.ger(f_in))
        self.bias_eps.copy_(f_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ------------------------
# 2) Dueling + Distributional (C51) head with Noisy layers
# ------------------------
class DuelingNoisyC51(nn.Module):
    def __init__(self, state_shape, action_size, atoms=51, vmin=-10.0, vmax=10.0, hidden=256):
        super().__init__()
        self.atoms = atoms
        self.action_size = action_size
        self.vmin = vmin
        self.vmax = vmax
        self.register_buffer("support", torch.linspace(vmin, vmax, atoms))

        in_dim = int(np.prod(state_shape))
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )

        # Dueling streams, both with noisy layers
        self.advantage1 = NoisyLinear(hidden, hidden)
        self.advantage2 = NoisyLinear(hidden, action_size * atoms)

        self.value1 = NoisyLinear(hidden, hidden)
        self.value2 = NoisyLinear(hidden, atoms)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        # Flatten state
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.feature(x)

        a = F.relu(self.advantage1(h))
        a = self.advantage2(a).view(-1, self.action_size, self.atoms)

        v = F.relu(self.value1(h))
        v = self.value2(v).view(-1, 1, self.atoms)

        # Dueling combine (distributional logits)
        q_atoms = v + (a - a.mean(dim=1, keepdim=True))
        # Return log-probabilities over atoms
        log_probs = F.log_softmax(q_atoms, dim=2)
        return log_probs

    def q_values(self, x):
        """Compute expected Q(s,a) from distributional head."""
        with torch.no_grad():
            log_p = self.forward(x)                    # [B, A, atoms]
            p = log_p.exp()
            q = torch.sum(p * self.support, dim=2)     # [B, A]
        return q


# ------------------------
# 3) Prioritized Replay (simple proportional, numpy-based)
# ------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.eps = 1e-6

    def add(self, experience, priority=None):
        max_prio = self.priorities.max() if self.buffer else 1.0
        p = max_prio if priority is None else priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = float(p) + self.eps

    def __len__(self):
        return len(self.buffer)


# ------------------------
# 4) N-step helper
# ------------------------
class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.deque = deque()

    def push(self, exp: Experience):
        self.deque.append(exp)

    def pop_ready(self):
        """Return a single n-step aggregated Experience if ready, else None."""
        if len(self.deque) < self.n:
            return None
        R, next_state, done = 0.0, None, False
        for i, e in enumerate(self.deque):
            R += (self.gamma ** i) * e.reward
            next_state = e.next_state
            done = e.done
            if done:
                break
        first = self.deque[0]
        # Pop only one from the left; rolling window
        self.deque.popleft()
        return Experience(first.state, first.action, R, next_state, done)

    def flush_all(self):
        outs = []
        while len(self.deque) > 0:
            outs.append(self.pop_ready() or self.deque.popleft())
        self.deque.clear()
        return outs


# ------------------------
# 5) Rainbow Agent
# ------------------------
class DQNAgent:
    def __init__(self, state_shape, action_size, config, device):
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = device

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.n_step = config.get("n_step", 3)
        self.gamma_n = self.gamma ** self.n_step

        self.lr = config.get("learning_rate", 1e-4)
        self.batch_size = config.get("batch_size", 64)
        self.buffer_size = config.get("buffer_size", 100000)
        self.target_update_freq = config.get("target_update_freq", 2000)

        # PER params
        self.per_alpha = config.get("per_alpha", 0.6)
        self.per_beta_start = config.get("per_beta_start", 0.4)
        self.per_beta_frames = config.get("per_beta_frames", 200_000)

        # C51 params
        self.atoms = config.get("atoms", 51)
        self.vmin = config.get("vmin", -10.0)
        self.vmax = config.get("vmax", 10.0)

        # Exploration defaults to NoisyNet (disable epsilon)
        self.use_noisy = config.get("use_noisy", True)

        # Schedules (reuse your LinearSchedule if available)
        try:
            from src.utils.schedules import LinearSchedule
            self.beta_schedule = LinearSchedule(self.per_beta_start, 1.0, self.per_beta_frames)
        except Exception:
            # Simple fallback schedule
            class _Lin:
                def __init__(self, start, end, dur): self.s, self.e, self.d = start, end, max(1, dur)
                def get_value(self, t): 
                    f = min(1.0, t / self.d)
                    return self.s + f * (self.e - self.s)
            self.beta_schedule = _Lin(self.per_beta_start, 1.0, self.per_beta_frames)

        # Buffers
        self.memory = PrioritizedReplayBuffer(self.buffer_size, alpha=self.per_alpha)
        self.nstep_buf = NStepBuffer(self.n_step, self.gamma)

        # Networks
        self.policy_net = DuelingNoisyC51(state_shape, action_size, atoms=self.atoms, vmin=self.vmin, vmax=self.vmax).to(device)
        self.target_net = DuelingNoisyC51(state_shape, action_size, atoms=self.atoms, vmin=self.vmin, vmax=self.vmax).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Counters
        self.steps_done = 0
        self.learn_steps = 0

        # Precompute support & delta_z
        support = torch.linspace(self.vmin, self.vmax, self.atoms)
        self.register_buffers(support)

    def register_buffers(self, support):
        self.support = support.to(self.device)  # [atoms]
        self.delta_z = (self.vmax - self.vmin) / (self.atoms - 1)

    def act(self, state, training=True):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.steps_done += 1
        self.policy_net.train(training)
        if self.use_noisy:
            # No ε-greedy; NoisyNet handles exploration. Just pick argmax Q.
            with torch.no_grad():
                q = self.policy_net.q_values(s)
                return int(q.argmax(dim=1).item())
        else:
            # Optional: epsilon path (not typical in Rainbow)
            eps = max(0.01, 1.0 - self.steps_done / 50_000)
            if training and random.random() < eps:
                return random.randrange(self.action_size)
            with torch.no_grad():
                q = self.policy_net.q_values(s)
                return int(q.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.nstep_buf.push(Experience(state, action, reward, next_state, done))
        exp_n = self.nstep_buf.pop_ready()
        if exp_n is not None:
            # Start with high priority so it gets sampled soon
            self.memory.add(exp_n, priority=1.0)

    def _dist_projection(self, rewards, dones, next_log_probs):
        """
        Categorical (C51) projection of target distribution onto the fixed support.
        rewards, dones: [B,1]
        next_log_probs: [B, A, atoms] for next_state evaluated by target_net
        """
        with torch.no_grad():
            # Double DQN: select next actions via policy_net expected Q
            # (we already computed selection before calling this)
            p_next = next_log_probs.exp()  # [B, A, atoms]
            # Expected Q for each action under next-state distribution
            q_next = torch.sum(p_next * self.support.view(1, 1, -1), dim=2)  # [B, A]
            next_actions = q_next.argmax(dim=1, keepdim=True)  # [B,1]

            # Get the distribution for chosen actions
            p_next_a = p_next.gather(1, next_actions.unsqueeze(-1).expand(-1, -1, self.atoms)).squeeze(1)  # [B, atoms]

            Tz = rewards + (1.0 - dones) * self.gamma_n * self.support.view(1, -1)  # [B, atoms]
            Tz = Tz.clamp(min=self.vmin, max=self.vmax)
            b = (Tz - self.vmin) / self.delta_z  # [B, atoms]
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)

            B = rewards.size(0)
            m = torch.zeros(B, self.atoms, device=self.device)

            # Distribute probability mass
            offset = torch.linspace(0, (B - 1) * self.atoms, B, device=self.device).long().unsqueeze(1)
            m.view(-1).index_add_(0, (l + offset).view(-1), (p_next_a * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (p_next_a * (b - l.float())).view(-1))
        return m  # [B, atoms]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        beta = self.beta_schedule.get_value(self.learn_steps)
        batch, indices, weights = self.memory.sample(self.batch_size, beta=beta)

        states = torch.FloatTensor(np.stack([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor([float(e.done) for e in batch]).unsqueeze(1).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)  # [B,1]

        # Reset NoisyNet noise each update (recommended)
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # Current log-probs over atoms for taken actions
        log_p = self.policy_net(states)                 # [B, A, atoms]
        log_p_a = log_p.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.atoms)).squeeze(1)  # [B, atoms]

        with torch.no_grad():
            next_log_p = self.target_net(next_states)   # [B, A, atoms]

        # C51 projection with Double DQN action selection
        target_dist = self._dist_projection(rewards, dones, next_log_p)  # [B, atoms]

        # Cross-entropy loss (minimize KL: target (fixed) vs current)
        loss_per = -(target_dist * log_p_a).sum(dim=1, keepdim=True)     # [B,1]
        loss = (loss_per * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update PER priorities from per-sample loss (use detached TD-ish proxy)
        prios = loss_per.detach().squeeze(1).cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, prios)

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_network()

    # Optional: call at episode end to flush remaining n-step transitions
    def end_episode(self):
        for e in self.nstep_buf.flush_all():
            self.memory.add(e, priority=1.0)

def load_trained_agent(model_path, state_shape, action_size, device):
    agent = DQNAgent(state_shape, action_size, {}, device)
    agent.load(model_path)
    return agent
