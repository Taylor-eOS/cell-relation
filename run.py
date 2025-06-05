import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shutil
from gridworld import GridWorld
from model import Encoder, PolicyModel
from analysis import render_grid, analyze_episode
import settings

def train_policy():
    env, policy, optimizer = initialize_policy()
    stages = sorted(settings.stage_offsets.keys())
    for stage in stages:
        train_stage(env, policy, optimizer, stage)
    train_free_roam(env, policy, optimizer)
    torch.save(policy.state_dict(), "policy_model.pt")

def train_stage(env, policy, optimizer, stage):
    cumulative_offsets = []
    for s in range(1, stage + 1):
        if s in settings.stage_offsets:
            cumulative_offsets.extend(settings.stage_offsets[s])
    #max_steps = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3}.get(stage, 4)
    max_steps = {1: 1, 2: 1}.get(stage, 8)
    success_count = 0
    step_sum = 0
    episode_logged = False
    for ep in range(1, settings.training_steps + 1):
        env.sample_stage(stage)
        if not episode_logged:
            trajectory, reached_goal = run_episode(env, policy, max_steps, render_prefix=f'stage{stage}_ep{ep}', render_dir=f'images/stage{stage}')
            analyze_episode(trajectory, reached_goal, ep, stage=stage)
            episode_logged = True
        else:
            trajectory, reached_goal = run_episode(env, policy, max_steps)
        step_sum += len(trajectory)
        if reached_goal:
            success_count += 1
            rewards = assign_rewards(trajectory, True)
        else:
            rewards = assign_rewards(trajectory, False)
        loss = compute_loss(trajectory, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % settings.step_interval == 0:
            success_rate = success_count / settings.step_interval
            avg_steps = step_sum / settings.step_interval
            print(f"Stage {stage}, episode {ep:5}, success rate: {success_rate:.2f}, avg steps: {avg_steps:.2f}")
            if success_rate >= settings.threshold:
                break
            success_count = 0
            step_sum = 0
    print(f"Stage {stage + 1}")

def train_free_roam(env, policy, optimizer):
    success_count = 0
    step_sum = 0
    for ep in range(1, settings.training_steps + 1):
        env.reset()
        if ep % settings.free_roam_log == 0:
            render_subdir = os.path.join('images/free_roam', f'ep{ep}')
            trajectory, reached_goal = run_episode(env, policy, 8, render_prefix=f'freeroam_ep{ep}', render_dir=render_subdir)
            analyze_episode(trajectory, reached_goal, ep, free_roam=True)
        else:
            trajectory, reached_goal = run_episode(env, policy, 8)
        step_sum += len(trajectory)
        if reached_goal:
            success_count += 1
        rewards = assign_rewards(trajectory, reached_goal)
        loss = compute_loss(trajectory, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % settings.step_interval == 0:
            success_rate = success_count / settings.step_interval
            avg_steps = step_sum / settings.step_interval
            print(f"Free roam episode {ep:5}, success rate: {success_rate:.2f}, avg steps: {avg_steps:.2f}")
            success_count = 0
            step_sum = 0

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
        next_obs, done, collision = env.step(action)
        next_state = next_obs[0].flatten().argmax().item()
        trajectory.append((prev_state, action, next_state, logp, collision))
        obs = next_obs
        if render_prefix:
            render_grid(env, f'{render_prefix}_step{i}', render_dir)
        if collision or done:
            if done:
                reached_goal = True
            break
    return trajectory, reached_goal

def assign_rewards(trajectory, reached_goal):
    rewards = []
    if reached_goal:
        seq_len = len(trajectory)
        for (_, _, _, _, collision) in trajectory:
            rewards.append(1.0)
    else:
        for (_, _, _, _, collision) in trajectory:
            if collision:
                rewards.append(-1.0)
            else:
                rewards.append(0.0)
    return rewards

def initialize_policy():
    env = GridWorld()
    encoder = Encoder()
    pretrained = torch.load("world_model.pt")
    encoder_state = {
        k.replace("encoder.", ""): v 
        for k, v in pretrained.items() 
        if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    policy = PolicyModel(encoder)
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings.policy_lr)
    return env, policy, optimizer

def sample_action(logits):
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action_tensor = dist.sample()
    return action_tensor.item(), dist.log_prob(action_tensor)

def compute_loss(trajectory, rewards):
    loss = torch.tensor(0.0)
    for (_, _, _, logp, _), r in zip(trajectory, rewards):
        loss = loss - logp * r
    return loss

if __name__ == "__main__":
    if os.path.isdir('images'):
        shutil.rmtree('images')
    train_policy()

