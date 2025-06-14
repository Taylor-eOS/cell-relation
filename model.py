import torch
import torch.nn as nn
import settings

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        C = 16
        size = settings.grid_size
        num_cells = size * size
        self.cell_feature_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, C))
        self.abs_pos_embed  = nn.Embedding(num_cells, 4)
        self.rel_pos_embed  = nn.Linear(2, 4)
        self.angle_embed    = nn.Linear(2, 4)
        self.neighbor_proj  = nn.Linear(C, 4)
        d_model = C + 4 + 4 + 4 + 4
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=settings.attention_heads,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=settings.transformer_layers)
        coords = torch.stack(torch.meshgrid(
            torch.arange(size), torch.arange(size),
            indexing='ij'), dim=-1).view(num_cells, 2).float()
        coords = coords / (size - 1) * 2 - 1
        ri, ci = coords[:, 0], coords[:, 1]
        rj, cj = ri[None, :], ci[None, :]
        rel_full = torch.stack([ri[:, None] - rj, ci[:, None] - cj], dim=-1)
        angles = torch.atan2(rel_full[..., 1], rel_full[..., 0])
        angles_sin_cos = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        neighbors = []
        for idx in range(num_cells):
            i, j = divmod(idx, size)
            nbrs = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                nbrs.append(ni * size + nj if 0 <= ni < size and 0 <= nj < size else idx)
            neighbors.append(nbrs)
        self.register_buffer('rel_full', rel_full)
        self.register_buffer('angles_sin_cos', angles_sin_cos)
        self.register_buffer('neighbor_idx', torch.tensor(neighbors, dtype=torch.long))
        self.num_cells = num_cells

    def simulate_hypotheticals(self, obs):
        H, W = obs.shape[1], obs.shape[2]
        agent = obs[0]
        walls = obs[2]
        goal  = obs[1]
        hypos = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_agent = torch.zeros_like(agent)
            ai, aj = (agent == 1).nonzero(as_tuple=True)
            ai, aj = ai.item(), aj.item()
            ni, nj = ai + dy, aj + dx
            if 0 <= ni < H and 0 <= nj < W and walls[ni, nj] == 0:
                new_agent[ni, nj] = 1
            else:
                new_agent[ai, aj] = 1
            hypos.append(torch.stack([new_agent, goal, walls], dim=0))
        return torch.stack(hypos, dim=0)

    def forward(self, obs):
        device = self.rel_full.device
        hypos = self.simulate_hypotheticals(torch.from_numpy(obs).to(device))
        B = hypos.size(0)
        flat = hypos.view(B, 3, self.num_cells)
        cell_emb = self.cell_feature_mlp(flat.permute(0, 2, 1))
        indices = torch.arange(self.num_cells, device=device)
        abs_embs = self.abs_pos_embed(indices)
        tokens = []
        for b in range(B):
            agent_idx = int(flat[b, 0].argmax())
            rel_offsets = self.rel_full[agent_idx]
            rel_embs    = self.rel_pos_embed(rel_offsets)
            angle_feats = self.angles_sin_cos[agent_idx]
            angle_embs  = self.angle_embed(angle_feats)
            nbrs        = self.neighbor_idx[indices]
            nbr_embs    = cell_emb[b, nbrs]
            nbr_avg     = nbr_embs.mean(dim=1)
            nbr_feats   = self.neighbor_proj(nbr_avg)
            ce_all = cell_emb[b]
            token  = torch.cat([ce_all, abs_embs, rel_embs,
                                angle_embs, nbr_feats], dim=-1)
            tokens.append(token)
        tokens = torch.stack(tokens, dim=0)
        t_out  = self.transformer(tokens)
        return t_out.mean(dim=1)

class PolicyModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(32, 4)

    def forward(self, obs):
        hvec = self.encoder(obs)
        logits = self.head(hvec.unsqueeze(0))
        return logits

