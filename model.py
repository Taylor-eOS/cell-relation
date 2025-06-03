import torch
import torch.nn as nn
import torch.nn.functional as F
import settings

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        C = 16
        d_model = 2 * C + 2
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        size = settings.grid_size
        num_cells = size * size
        coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij'), dim=-1).view(num_cells, 2).float()
        coords = coords / (size - 1) * 2 - 1
        ri = coords[:, None, 0]
        rj = coords[None, :, 0]
        ci = coords[:, None, 1]
        cj = coords[None, :, 1]
        rel_full = torch.stack([ri - rj, ci - cj], dim=-1)
        pairs = []
        for i in range(num_cells):
            for j in range(i, num_cells):
                pairs.append((i, j))
        pair_i = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        pair_j = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        self.register_buffer('pair_i', pair_i)
        self.register_buffer('pair_j', pair_j)
        rel = rel_full[pair_i, pair_j]
        self.register_buffer('rel', rel)
        self.num_cells = num_cells

    def forward(self, obs):
        device = self.rel.device
        x = torch.from_numpy(obs).unsqueeze(0).to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        B, C, h, w = x.shape
        cell_emb = x.view(B, C, h * w).permute(0, 2, 1)
        ce_i = cell_emb[:, self.pair_i, :]
        ce_j = cell_emb[:, self.pair_j, :]
        rel = self.rel.unsqueeze(0).expand(B, -1, -1)
        flat_obs = torch.from_numpy(obs).view(3, -1).to(device)
        obs_i = flat_obs[:, self.pair_i]
        obs_j = flat_obs[:, self.pair_j]
        is_empty_i = (obs_i == 0).all(dim=0)
        is_empty_j = (obs_j == 0).all(dim=0)
        is_wall_i = (obs_i[2] == 1.0)
        is_wall_j = (obs_j[2] == 1.0)
        pad_mask = (is_empty_i & is_empty_j) | (is_wall_i & is_empty_j) | (is_empty_i & is_wall_j) | (is_wall_i & is_wall_j)
        pad_mask_batch = pad_mask.unsqueeze(0).expand(B, -1)
        tokens = torch.cat([ce_i, ce_j, rel], dim=-1)
        t_out = self.transformer(tokens, src_key_padding_mask=pad_mask_batch)
        flat_agent_layer = torch.from_numpy(obs[0]).flatten().to(device)
        agent_idx = int(flat_agent_layer.argmax())
        mask = (self.pair_i == agent_idx) | (self.pair_j == agent_idx)
        agent_tokens = t_out[0, mask]
        hvec = agent_tokens.mean(dim=0)
        return hvec

class PolicyModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(34, 4)

    def forward(self, obs):
        hvec = self.encoder(obs) 
        logits = self.head(hvec.unsqueeze(0))
        return logits

