import torch
import torch.nn as nn
import torch.nn.functional as F
import settings

omit_empty_cells = False

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2,16,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(16,16,kernel_size=3,padding=1)
        C = 16
        d_model = 2 * C + 2
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=2,batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer,num_layers=2)
        size = settings.grid_size
        num_cells = size * size
        coords = torch.stack(torch.meshgrid(torch.arange(size),torch.arange(size),indexing='ij'),dim=-1).view(num_cells,2).float()
        coords = coords / (size - 1) * 2 - 1
        ri = coords[:,None,0]
        rj = coords[None,:,0]
        ci = coords[:,None,1]
        cj = coords[None,:,1]
        rel_full = torch.stack([ri - rj,ci - cj],dim=-1)
        pairs = []
        for i in range(num_cells):
            for j in range(i,num_cells):
                pairs.append((i,j))
        pair_i = torch.tensor([p[0] for p in pairs],dtype=torch.long)
        pair_j = torch.tensor([p[1] for p in pairs],dtype=torch.long)
        self.register_buffer('pair_i',pair_i)
        self.register_buffer('pair_j',pair_j)
        rel = rel_full[pair_i,pair_j]
        self.register_buffer('rel',rel)
        self.num_cells = num_cells

    def forward(self, obs, omit_empty_cells=True):
        x = torch.from_numpy(obs).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        B, C, h, w = x.shape
        cell_emb = x.view(B, C, h * w).permute(0, 2, 1)
        M = self.rel.shape[0]
        ce_i = cell_emb[:, self.pair_i, :]
        ce_j = cell_emb[:, self.pair_j, :]
        rel = self.rel.unsqueeze(0).expand(B, -1, -1)
        flat_obs = torch.from_numpy(obs).view(2, -1)
        obs_i = flat_obs[:, self.pair_i]
        obs_j = flat_obs[:, self.pair_j]
        if omit_empty_cells:
            empty_i = (obs_i == 0).all(dim=0)
            empty_j = (obs_j == 0).all(dim=0)
            keep_mask = ~(empty_i & empty_j)
            ce_i = ce_i[:, keep_mask, :]
            ce_j = ce_j[:, keep_mask, :]
            rel = rel[:, keep_mask, :]
            mask_i = self.pair_i[keep_mask]
            mask_j = self.pair_j[keep_mask]
        else:
            mask_i = self.pair_i
            mask_j = self.pair_j
        tokens = torch.cat([ce_i, ce_j, rel], dim=-1)
        t_out = self.transformer(tokens)
        flat = obs[0].flatten()
        agent_idx = int(flat.argmax())
        mask = (mask_i == agent_idx) | (mask_j == agent_idx)
        agent_tokens = t_out[0, mask]
        hvec = agent_tokens.mean(dim=0)
        return hvec

class PolicyModel(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(34,4)

    def forward(self,obs):
        hvec = self.encoder(obs)
        logits = self.head(hvec.unsqueeze(0))
        return logits

def sample_action(logits):
    probs = F.softmax(logits,dim=-1)
    dist = torch.distributions.Categorical(probs)
    action_tensor = dist.sample()
    return action_tensor.item(),dist.log_prob(action_tensor)

def assign_rewards(trajectory):
    rewards = [0.0] * len(trajectory)
    if not trajectory:
        return rewards
    goal_idx = trajectory[-1][2]
    current = goal_idx
    for i in range(len(trajectory)-1,-1,-1):
        prev_s,_,next_s,_ = trajectory[i]
        if next_s == current:
            r_prev = divmod(prev_s,settings.grid_size)
            r_cur = divmod(current,settings.grid_size)
            if abs(r_prev[0]-r_cur[0]) + abs(r_prev[1]-r_cur[1]) == 1:
                rewards[i] = 1.0
                current = prev_s
    return rewards

def run_episode(env,policy,max_steps):
    obs = env._get_obs()
    trajectory = []
    reached_goal = False
    for _ in range(max_steps):
        logits = policy(obs)
        action,logp = sample_action(logits)
        next_obs,done = env.step(action)
        prev_state = obs[0].flatten().argmax().item()
        next_state = next_obs[0].flatten().argmax().item()
        trajectory.append((prev_state,action,next_state,logp))
        obs = next_obs
        if done:
            reached_goal = True
            break
    return trajectory,reached_goal

