import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class GridWorld3x3:
    def __init__(self):
        pass

    def reset(self):
        self.agent_pos = [1, 1]
        cardinal_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.goal_pos = random.choice([(self.agent_pos[0] + dx, self.agent_pos[1] + dy) for dx, dy in cardinal_offsets])
        return self._get_obs()

    def step(self, action):
        # Perform the action
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # down
            self.agent_pos[0] = min(2, self.agent_pos[0] + 1)
        elif action == 2:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # right
            self.agent_pos[1] = min(2, self.agent_pos[1] + 1)
        # Compute reward
        reward = 1.0 if tuple(self.agent_pos) == self.goal_pos else 0.0
        # Return new observation and reward
        return self._get_obs(), reward

    def _get_obs(self):
        agent_map = np.zeros((3, 3), dtype=np.float32)
        goal_map  = np.zeros((3, 3), dtype=np.float32)
        agent_map[self.agent_pos[0], self.agent_pos[1]] = 1.0
        goal_map[self.goal_pos[0], self.goal_pos[1]]   = 1.0
        return np.stack([agent_map, goal_map], axis=0)

class SpatialRelationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        enc_layer = nn.TransformerEncoderLayer(d_model=34, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.policy_head = nn.Linear(34, 4)

    def forward(self, obs):
        x = torch.from_numpy(obs).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        cell_emb = x.view(1, 16, 9).permute(0, 2, 1).squeeze(0)
        tokens = []
        for i in range(9):
            for j in range(9):
                dx = ((i // 3) - (j // 3)) / 2.0
                dy = ((i % 3) - (j % 3)) / 2.0
                rel = torch.tensor([dx, dy], dtype=torch.float32)
                tokens.append(torch.cat([cell_emb[i], cell_emb[j], rel]))
        seq = torch.stack(tokens).unsqueeze(0)
        t_out = self.transformer(seq).squeeze(0)
        agent_idx = 4
        agent_tokens = t_out[agent_idx * 9: agent_idx * 9 + 9]
        h = agent_tokens.mean(dim=0)
        return self.policy_head(h)

def train():
    env = GridWorld3x3()
    model = SpatialRelationModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_reward = 0.0
    print_interval = 100
    for ep in range(1, 1001):
        obs = env.reset()
        logits = model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        next_obs, reward = env.step(action)
        total_reward += reward
        loss = -dist.log_prob(torch.tensor(action)) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % print_interval == 0:
            avg_reward = total_reward / print_interval
            print(f"Episode {ep}, average reward: {avg_reward:.2f}")
            total_reward = 0.0

if __name__ == "__main__":
    train()

