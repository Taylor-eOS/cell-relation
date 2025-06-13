import torch
import torch.nn as nn
import torch.nn.functional as F
import settings

import torch
import torch.nn as nn
import settings

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        C = 32
        size = settings.grid_size
        num_cells = size * size

        self.cell_feature_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, C)
        )

        self.abs_pos_embed  = nn.Embedding(num_cells, 8)
        self.rel_pos_embed  = nn.Linear(2, 8)
        self.angle_embed    = nn.Linear(2, 8)
        self.type_rel_embed = nn.Embedding(16, 8)
        self.neighbor_proj  = nn.Linear(C, 8)

        d_model = C + 8
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=settings.attention_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=settings.transformer_layers
        )

        coords = torch.stack(
            torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij'),
            dim=-1
        ).view(num_cells, 2).float()
        coords = coords / (size - 1) * 2 - 1
        ri, ci = coords[:, 0], coords[:, 1]
        rj, cj = ri[None, :], ci[None, :]
        rel_full = torch.stack([ri[:, None] - rj, ci[:, None] - cj], dim=-1)
        angles = torch.atan2(rel_full[..., 1], rel_full[..., 0])
        angles_sin_cos = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        self.register_buffer('rel_full', rel_full)
        self.register_buffer('angles_sin_cos', angles_sin_cos)

        neighbors = []
        for idx in range(num_cells):
            i, j = divmod(idx, size)
            nbrs = []
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                nbrs.append(ni*size+nj if 0<=ni<size and 0<=nj<size else idx)
            neighbors.append(nbrs)
        self.register_buffer('neighbor_idx', torch.tensor(neighbors, dtype=torch.long))
        self.num_cells = num_cells

    def forward(self, obs):
        device = self.rel_full.device
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        flat_obs = obs_tensor.view(1, 3, self.num_cells)
        cell_emb = self.cell_feature_mlp(flat_obs.permute(0, 2, 1))

        indices     = torch.arange(self.num_cells, device=device)
        abs_embs    = self.abs_pos_embed(indices)
        rel_offsets = self.rel_full[flat_obs[0,0].argmax(), indices]
        rel_embs    = self.rel_pos_embed(rel_offsets)
        angle_feats = self.angles_sin_cos[flat_obs[0,0].argmax(), indices]
        angle_embs  = self.angle_embed(angle_feats)

        nbr_idxs    = self.neighbor_idx[indices]
        nbr_embs    = cell_emb[0, nbr_idxs]
        nbr_avg     = nbr_embs.mean(dim=1)
        nbr_feats   = self.neighbor_proj(nbr_avg)

        ce_all = cell_emb[0]
        t_abs    = torch.cat([ce_all, abs_embs],   dim=-1)
        t_rel    = torch.cat([ce_all, rel_embs],   dim=-1)
        t_angle  = torch.cat([ce_all, angle_embs], dim=-1)
        t_nbr    = torch.cat([ce_all, nbr_feats],  dim=-1)

        tokens = torch.stack([torch.cat([t_abs, t_rel, t_angle, t_nbr], dim=0)], dim=0)
        t_out  = self.transformer(tokens)
        return t_out.mean(dim=1).squeeze(0)

class PolicyModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(40, 4)

    def forward(self, obs):
        hvec = self.encoder(obs)
        logits = self.head(hvec.unsqueeze(0))
        return logits

