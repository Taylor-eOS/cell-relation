import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class GridWorld:
    def __init__(self):
        self.agent_pos = [1, 1]

    def reset(self):
        self.agent_pos = [random.randint(0, 3), random.randint(0, 3)]
        return self._get_obs()

    def step(self, action):
        if action == 0:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:
            self.agent_pos[0] = min(3, self.agent_pos[0] + 1)
        elif action == 2:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:
            self.agent_pos[1] = min(3, self.agent_pos[1] + 1)
        return self._get_obs()

    def _get_obs(self):
        agent_map = np.zeros((4, 4), dtype=np.float32)
        agent_map[self.agent_pos[0], self.agent_pos[1]] = 1.0
        zero_map = np.zeros((4, 4), dtype=np.float32)
        return np.stack([agent_map, zero_map], axis=0)

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        enc_layer = nn.TransformerEncoderLayer(d_model=34, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.action_embed = nn.Linear(4, 8)
        self.head = nn.Linear(34 + 8, 16)

    def forward(self, obs, action):
        x = torch.from_numpy(obs).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        b, c, h, w = x.shape
        cell_emb = x.view(b, c, h*w).permute(0, 2, 1).squeeze(0)
        tokens = []
        for i in range(h*w):
            ri, ci = divmod(i, w)
            for j in range(h*w):
                rj, cj = divmod(j, w)
                dx = (ri - rj) / ((h - 1) / 2.0)
                dy = (ci - cj) / ((w - 1) / 2.0)
                rel = torch.tensor([dx, dy], dtype=torch.float32)
                tokens.append(torch.cat([cell_emb[i], cell_emb[j], rel]))
        seq = torch.stack(tokens).unsqueeze(0)
        t_out = self.transformer(seq).squeeze(0)
        flat = obs[0].flatten()
        agent_idx = int(flat.argmax())
        start = agent_idx * (h*w)
        agent_tokens = t_out[start : start + (h*w)]
        hvec = agent_tokens.mean(dim=0)
        a_onehot = F.one_hot(torch.tensor(action), num_classes=4).float()
        a_emb = F.relu(self.action_embed(a_onehot))
        logits = self.head(torch.cat([hvec, a_emb], dim=-1).unsqueeze(0))
        return logits

def train_world_model():
    env = GridWorld()
    model = WorldModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_correct = 0
    interval = 100

    for step in range(1, 30001):
        obs = env._get_obs()
        action = random.randrange(4)
        next_obs = env.step(action)
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

train_world_model()

