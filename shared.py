import os, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gridworld import GridWorld
from model import Encoder, PolicyModel
from analysis import render_grid
import numpy as np
import settings
import utils

def initialize_policy():
    env = GridWorld()
    encoder = Encoder()
    if os.path.exists(settings.world_model):
        pretrained = torch.load(settings.world_model)
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in pretrained.items()
            if k.startswith("encoder.")}
        encoder.load_state_dict(encoder_state)
    policy = PolicyModel(encoder)
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 1.0)
    return env, policy, optimizer, scheduler

def load_wall_stage_data(filename):
    data = np.load(filename)
    stages = utils.stage_offsets(preliminary=True).keys()
    stage_data = {}
    for stage in stages:
        prefix = f"stage_{stage}_"
        key_obs = prefix + "obs"
        if key_obs not in data:
            continue
        obs = data[key_obs]
        agent = data[prefix + "agent_pos"]
        goal = data[prefix + "goal_pos"]
        wall = data[prefix + "wall_pos"]
        nw = data[prefix + "num_walls"]
        if obs.shape[0] > 0:
            stage_data[stage] = {"obs": obs, "agent": agent, "goal": goal, "wall": wall, "nw": nw}
    return stage_data, stages

def sample_wall_example(env, stage, stage_data, prev_data):
    curr = stage_data[stage]
    curr_N = curr["obs"].shape[0]
    prev_N = prev_data["obs"].shape[0] if prev_data else 0
    use_prev = prev_N > 0 and random.random() < 0.5
    if use_prev:
        idx = random.randrange(prev_N)
        agent_arr, goal_arr, wall_arr, num_walls = prev_data["agent"], prev_data["goal"], prev_data["wall"], prev_data["nw"]
    else:
        idx = random.randrange(curr_N)
        agent_arr, goal_arr, wall_arr, num_walls = curr["agent"], curr["goal"], curr["wall"], curr["nw"]
    wcount = int(num_walls[idx])
    env.agent_pos = [int(agent_arr[idx, 0]), int(agent_arr[idx, 1])]
    env.goal_pos = [int(goal_arr[idx, 0]), int(goal_arr[idx, 1])]
    env.wall_positions = [tuple(wall_arr[idx, i]) for i in range(wcount)]

def run_episode(env, policy, max_steps, render_prefix=None, render_dir=None):
    obs = env._get_obs()
    trajectory = []
    reached_goal = False
    if render_prefix:
        render_grid(env, f'{render_prefix}_step0', render_dir)
    for i in range(1, max_steps + 1):
        prev_state = obs[0].flatten().argmax().item()
        logits = policy(obs)
        action, logp = sample_action(logits)
        next_obs, done, on_wall, off_grid = env.step(action)
        next_state = next_obs[0].flatten().argmax().item()
        collision = on_wall or off_grid
        trajectory.append((prev_state, action, next_state, logp, collision))
        obs = next_obs
        if render_prefix:
            render_grid(env, f'{render_prefix}_step{i}', render_dir)
        if collision or done:
            if done:
                reached_goal = True
            break
    return trajectory, reached_goal

def sample_action(logits):
    logits = logits.view(-1)
    probs = F.softmax(logits, dim=0)
    dist = Categorical(probs)
    action = dist.sample()
    logp   = dist.log_prob(action)
    return int(action), logp

