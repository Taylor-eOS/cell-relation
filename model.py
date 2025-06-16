import torch
import torch.nn as nn
import numpy as np
import settings

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        size = settings.grid_size
        C = 64
        self.size = size
        feat_dim = 55
        self.feat_mlp = nn.Sequential(nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, C))
        enc_layer = nn.TransformerEncoderLayer(d_model=C, nhead=settings.attention_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=settings.transformer_layers)

    def simulate_hypotheticals(self, obs):
        H, W = self.size, self.size
        agent, goal, walls = obs[0], obs[1], obs[2]
        hypos = []
        ai, aj = (agent == 1).nonzero(as_tuple=True)
        ai, aj = ai.item(), aj.item()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ai + dy, aj + dx
            new_agent = torch.zeros_like(agent)
            if 0 <= ni < H and 0 <= nj < W:
                new_agent[ni, nj] = 1
            else:
                new_agent[ai, aj] = 1
            hypos.append(torch.stack([new_agent, goal, walls], dim=0))
        return torch.stack(hypos, dim=0)

    def extract_features(self, state):
        agent_map, goal_map, wall_map = state
        H = W = self.size
        device = state.device
        ai, aj = (agent_map == 1).nonzero(as_tuple=True)
        gi, gj = (goal_map == 1).nonzero(as_tuple=True)
        ai, aj = ai.item(), aj.item()
        gi, gj = gi.item(), gj.item()
        ain = torch.tensor(2 * ai / (H - 1) - 1, device=device)
        ajn = torch.tensor(2 * aj / (W - 1) - 1, device=device)
        gin = torch.tensor(2 * gi / (H - 1) - 1, device=device)
        gjn = torch.tensor(2 * gj / (W - 1) - 1, device=device)
        dxg, dyg = gjn - ajn, gin - ain
        eps = torch.tensor(1e-8, device=device)
        dg = torch.sqrt(dxg * dxg + dyg * dyg + eps)
        sing, cosg = dyg / dg, dxg / dg
        goal_feats = torch.stack([ain, ajn, gin, gjn, dxg, dyg, dg, sing, cosg])
        wall_locs = (wall_map == 1).nonzero(as_tuple=False).float()
        agent_wall_feats, goal_wall_feats = [], []
        for i in range(min(3, wall_locs.size(0))):
            wi, wj = wall_locs[i]
            win = 2 * wi / (H - 1) - 1
            wjn = 2 * wj / (W - 1) - 1
            dxaw, dyaw = wjn - ajn, win - ain
            daw = torch.sqrt(dxaw * dxaw + dyaw * dyaw + eps)
            sing_aw, cosg_aw = dyaw / daw, dxaw / daw
            agent_wall_feats += [win, wjn, dxaw, dyaw, daw, sing_aw, cosg_aw]
            dxgw, dygw = wjn - gjn, win - gin
            dgw = torch.sqrt(dxgw * dxgw + dygw * dygw + eps)
            sing_gw, cosg_gw = dygw / dgw, dxgw / dgw
            goal_wall_feats += [win, wjn, dxgw, dygw, dgw, sing_gw, cosg_gw]
        pad = lambda lst, n: lst + [0.] * (n - len(lst))
        agent_wall_feats = torch.tensor(pad(agent_wall_feats, 21), device=device)
        goal_wall_feats = torch.tensor(pad(goal_wall_feats, 21), device=device)
        walls = wall_map
        neighbors = torch.tensor([
            walls[ai - 1, aj] if ai > 0 else 1,
            walls[ai + 1, aj] if ai < H - 1 else 1,
            walls[ai, aj - 1] if aj > 0 else 1,
            walls[ai, aj + 1] if aj < W - 1 else 1
        ], device=device).float()
        return torch.cat([goal_feats, agent_wall_feats, goal_wall_feats, neighbors])

    def forward(self, obs_np):
        device = next(self.parameters()).device
        obs = torch.from_numpy(obs_np).to(device)
        current_tok = self.feat_mlp(self.extract_features(obs))
        hypos = self.simulate_hypotheticals(obs)
        hypo_toks = [self.feat_mlp(self.extract_features(h)) for h in hypos]
        tokens = torch.stack([current_tok] + hypo_toks, dim=0).unsqueeze(0)
        return self.transformer(tokens)

class PolicyModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(64, 1)
        
    def forward(self, obs):
        token_vecs = self.encoder(obs)
        action_vecs = token_vecs[:, 1:, :]
        logits = self.head(action_vecs)
        return logits.squeeze(-1)

