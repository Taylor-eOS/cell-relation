import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Encoder
from gridworld import GridWorld
import settings

class WorldModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.action_embed = nn.Linear(4, 8)
        num_cells = settings.grid_size * settings.grid_size
        self.head = nn.Linear(64 + 8, num_cells)

    def forward(self, obs, action):
        hvec = self.encoder(obs)
        a_onehot = F.one_hot(torch.tensor(action), num_classes=4).float()
        a_emb = F.relu(self.action_embed(a_onehot))
        logits = self.head(torch.cat([hvec, a_emb], dim=-1).unsqueeze(0))
        return logits

def train_world():
    env = GridWorld()
    encoder = Encoder()
    model = WorldModel(encoder)
    optimizer = optim.Adam(model.parameters(), lr=settings.learning_rate)
    total_correct = 0
    interval = settings.step_interval
    for step in range(1, settings.pretraining_steps + 1):
        obs = env._get_obs()
        action = random.randrange(4)
        next_obs, _, _, _ = env.step(action)
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
            print(f"Step {step:5}, predictive accuracy: {acc:.2f}")
            total_correct = 0
    torch.save(model.state_dict(), settings.world_model)

if __name__ == "__main__":
    train_world()

