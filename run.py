import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class GridWorld3x3:
    def __init__(self):
        self.reset()

    def reset(self):
        self.agent_pos = [1, 1]
        # Only cardinal neighbors of center
        cardinal_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.goal_pos = random.choice([(self.agent_pos[0] + dx, self.agent_pos[1] + dy) for dx, dy in cardinal_offsets])
        return self._get_obs()

    def get_correct_action(self):
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        if gx < ax: return 0  # up
        if gx > ax: return 1  # down
        if gy < ay: return 2  # left
        if gy > ay: return 3  # right

    def _get_obs(self):
        agent_map = np.zeros((3, 3), dtype=np.float32)
        goal_map = np.zeros((3, 3), dtype=np.float32)
        agent_map[self.agent_pos[0], self.agent_pos[1]] = 1.0
        goal_map[self.goal_pos[0], self.goal_pos[1]] = 1.0
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
    correct_count = 0
    for ep in range(1000):
        obs = env.reset()
        correct_action = env.get_correct_action()
        if correct_action is None:
            continue  # skip accidental goal-on-agent
        logits = model(obs)
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([correct_action]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_action = torch.argmax(logits).item()
        correct_count += int(pred_action == correct_action)
        if ep % 10 == 0 and ep > 0:
            acc = correct_count / 10
            print(f"Episode {ep}, last 10 accuracy: {acc:.2f}")
            correct_count = 0

if __name__ == "__main__":
    train()

