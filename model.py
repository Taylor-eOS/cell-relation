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
        d_model = 2 * C + 2 + 3
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, batch_first=True)
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
        self.register_buffer('rel_full', rel_full)
        self.num_cells = num_cells

    def forward(self, obs):
        device = self.rel_full.device
        x = torch.from_numpy(obs).unsqueeze(0).to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        B, C, h, w = x.shape
        num_cells = h * w
        cell_emb = x.view(B, C, num_cells).permute(0, 2, 1)
        flat_obs = torch.from_numpy(obs).to(device).view(3, num_cells)
        tokens_list = []
        for b in range(B):
            flat_agent_layer = torch.from_numpy(obs[0]).flatten().to(device)
            agent_idx = int(flat_agent_layer.argmax())
            ce_agent = cell_emb[b, agent_idx, :].unsqueeze(0)
            ce_agent_exp = ce_agent.expand(self.num_cells - 1, C)
            idxs = [j for j in range(self.num_cells) if j != agent_idx]
            other_idxs = torch.tensor(idxs, dtype=torch.long, device=device)
            ce_others = cell_emb[b, other_idxs, :]
            rel_offsets = self.rel_full[agent_idx, other_idxs]
            cell_types = flat_obs[:, other_idxs].permute(1, 0)
            tokens_b = torch.cat([ce_agent_exp, ce_others, rel_offsets, cell_types], dim=-1)
            tokens_list.append(tokens_b)
        tokens = torch.stack(tokens_list, dim=0)
        t_out = self.transformer(tokens)
        hvecs = t_out.mean(dim=1)
        return hvecs.squeeze(0)

class PolicyModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(37, 4)

    def forward(self, obs):
        hvec = self.encoder(obs)
        logits = self.head(hvec.unsqueeze(0))
        return logits

