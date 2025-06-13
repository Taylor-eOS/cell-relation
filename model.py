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
        size = settings.grid_size
        num_cells = size * size
        self.abs_pos_embed = nn.Embedding(num_cells, 4)
        self.rel_pos_embed = nn.Linear(2, 4)
        self.type_rel_embed = nn.Embedding(16, 4)
        self.angle_embed = nn.Linear(2, 4)
        d_model = 2 * C + 4 + 4 + 4 + 4
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=settings.attention_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=settings.transformer_layers)
        coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij'), dim=-1).view(num_cells, 2).float()
        coords = coords / (size - 1) * 2 - 1
        ri = coords[:, None, 0]
        rj = coords[None, :, 0]
        ci = coords[:, None, 1]
        cj = coords[None, :, 1]
        rel_full = torch.stack([ri - rj, ci - cj], dim=-1)
        self.register_buffer('rel_full', rel_full)
        dx = rel_full[..., 0]
        dy = rel_full[..., 1]
        angles = torch.atan2(dy, dx)
        angles_sin_cos = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        self.register_buffer('angles_sin_cos', angles_sin_cos)
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
        flat_agent = flat_obs[0]
        type_ids_all = torch.zeros(num_cells, dtype=torch.long, device=device)
        type_ids_all[flat_obs[2] == 1] = 1
        type_ids_all[flat_obs[1] == 1] = 2
        type_ids_all[flat_obs[0] == 1] = 3
        tokens_list = []
        for b in range(B):
            agent_idx = int(flat_agent.argmax())
            ce_agent = cell_emb[b, agent_idx, :].unsqueeze(0)
            indices = torch.arange(num_cells, dtype=torch.long, device=device)
            ce_agent_exp = ce_agent.expand(num_cells, C)
            ce_all = cell_emb[b, indices, :]
            rel_offsets = self.rel_full[agent_idx, indices]
            rel_embs = self.rel_pos_embed(rel_offsets)
            angle_feats = self.angles_sin_cos[agent_idx, indices]
            angle_embs = self.angle_embed(angle_feats)
            abs_embs = self.abs_pos_embed(indices)
            agent_type = type_ids_all[agent_idx].unsqueeze(0)
            other_types = type_ids_all[indices]
            type_pair_ids = agent_type * 4 + other_types
            type_embs = self.type_rel_embed(type_pair_ids)
            tokens_b = torch.cat([ce_agent_exp, ce_all, rel_embs, angle_embs, abs_embs, type_embs], dim=-1)
            tokens_list.append(tokens_b)
        tokens = torch.stack(tokens_list, dim=0)
        t_out = self.transformer(tokens)
        hvecs = t_out.mean(dim=1)
        return hvecs.squeeze(0)

class PolicyModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(48, 4)

    def forward(self, obs):
        hvec = self.encoder(obs)
        logits = self.head(hvec.unsqueeze(0))
        return logits

