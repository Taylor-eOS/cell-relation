import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import settings
from model import Encoder

class GridWorld:
    def __init__(self):
        self.size = settings.grid_size
        self.agent_pos = [self.size // 2] * 2
        self.goal_pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]

    def reset(self):
        self.agent_pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
        self.goal_pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
        while self.goal_pos == self.agent_pos:
            self.goal_pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
        return self._get_obs()

    def step(self, action):
        if action == 0:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        obs = self._get_obs()
        done = (self.agent_pos == self.goal_pos)
        return obs, done

    def _get_obs(self):
        agent_map = np.zeros((self.size, self.size), dtype=np.float32)
        goal_map = np.zeros((self.size, self.size), dtype=np.float32)
        agent_map[self.agent_pos[0], self.agent_pos[1]] = 1.0
        goal_map[self.goal_pos[0], self.goal_pos[1]] = 1.0
        return np.stack([agent_map, goal_map], axis=0)

    def sample(self):
        corners = [[0, 0], [0, self.size - 1], [self.size - 1, 0], [self.size - 1, self.size - 1]]
        self.goal_pos = random.choice(corners)
        self.agent_pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
        while self.agent_pos == self.goal_pos:
            self.agent_pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
        return self._get_obs()

class WorldModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.action_embed = nn.Linear(4, 8)
        num_cells = settings.grid_size * settings.grid_size
        self.head = nn.Linear(34 + 8, num_cells)

    def forward(self, obs, action):
        hvec = self.encoder(obs)
        a_onehot = F.one_hot(torch.tensor(action), num_classes=4).float()
        a_emb = F.relu(self.action_embed(a_onehot))
        logits = self.head(torch.cat([hvec, a_emb], dim=-1).unsqueeze(0))
        return logits

class PolicyModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(34, 4)

    def forward(self, obs):
        hvec = self.encoder(obs)
        logits = self.head(hvec.unsqueeze(0))
        return logits

def train_world():
    env = GridWorld()
    encoder = Encoder()
    model = WorldModel(encoder)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_correct = 0
    interval = settings.step_interval
    for step in range(1, settings.pretraining_steps + 1):
        obs = env._get_obs()
        action = random.randrange(4)
        next_obs, _ = env.step(action)
        true_pos = int(next_obs[0].flatten().argmax())
        logits = model(obs, action)
        loss = F.cross_entropy(logits, torch.tensor([true_pos]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_pos = torch.argmax(logits, dim=1).item()
        total_correct += int(pred_pos == true_pos)
        if step % interval == 0:
            acc = total_correct / interval
            print(f"Step {step}, predictive accuracy: {acc:.2f}")
            total_correct = 0
    torch.save(model.state_dict(), "world_model.pt")

def train_policy():
    env = GridWorld()
    encoder = Encoder()
    pretrained = torch.load("world_model.pt")
    encoder_state = {k.replace("encoder.", ""): v for k, v in pretrained.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    policy = PolicyModel(encoder)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    success_count = 0
    step_sum = 0
    interval = 100
    for ep in range(1, settings.training_steps + 1):
        trajectory, reached_goal = run_episode(env, policy)
        step_sum += len(trajectory)
        if reached_goal:
            success_count += 1
            rewards = assign_rewards(trajectory)
        else:
            rewards = [0.0] * len(trajectory)
        loss = 0.0
        for (_, _, _, logp), r in zip(trajectory, rewards):
            loss = loss - logp * r
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % interval == 0:
            success_rate = success_count / interval
            avg_steps = step_sum / interval
            print(f"Episode {ep}, success rate: {success_rate:.2f}, avg steps: {avg_steps:.2f}")
            success_count = 0
            step_sum = 0
    torch.save(policy.state_dict(), "policy.pt")

def assign_rewards(trajectory):
    rewards = [0.0] * len(trajectory)
    if not trajectory:
        return rewards
    goal_idx = trajectory[-1][2]
    current = goal_idx
    for i in range(len(trajectory) - 1, -1, -1):
        prev_s, _, next_s, _ = trajectory[i]
        if next_s == current:
            r_prev = divmod(prev_s, settings.grid_size)
            r_cur = divmod(current, settings.grid_size)
            if abs(r_prev[0] - r_cur[0]) + abs(r_prev[1] - r_cur[1]) == 1:
                rewards[i] = 1.0
                current = prev_s
    return rewards

def run_episode(env, policy):
    obs = env.sample()
    trajectory = []
    reached_goal = False
    for _ in range(8):
        logits = policy(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        logp = dist.log_prob(torch.tensor(action))
        next_obs, done = env.step(action)
        prev_state = obs[0].flatten().argmax().item()
        next_state = next_obs[0].flatten().argmax().item()
        trajectory.append((prev_state, action, next_state, logp))
        obs = next_obs
        if done:
            reached_goal = True
            break
    return trajectory, reached_goal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action="store_true")
    args = parser.parse_args()
    if args.pretrain:
        train_world()
    else:
        train_policy()

