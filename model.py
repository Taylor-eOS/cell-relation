import torch
import torch.nn as nn
import torch.nn.functional as F
import settings

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
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
        self.rel = torch.stack([ri - rj, ci - cj], dim=-1)
        self.num_cells = num_cells

    def forward(self, obs):
        x = torch.from_numpy(obs).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        B, C, h, w = x.shape
        cell_emb = x.view(B, C, h * w).permute(0, 2, 1)
        num_cells = self.num_cells
        ce_i = cell_emb[:, :, None, :].expand(-1, -1, num_cells, -1)
        ce_j = cell_emb[:, None, :, :].expand(-1, num_cells, -1, -1)
        rel = self.rel.to(x.device)[None, :, :, :]
        tokens = torch.cat([ce_i, ce_j, rel], dim=-1)
        seq = tokens.view(B, num_cells * num_cells, 2 * C + 2)
        t_out = self.transformer(seq)
        flat = obs[0].flatten()
        agent_idx = int(flat.argmax())
        start = agent_idx * num_cells
        agent_tokens = t_out[0, start : start + num_cells]
        hvec = agent_tokens.mean(dim=0)
        return hvec

