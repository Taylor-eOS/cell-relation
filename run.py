import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import settings
from pretrain import train_world
from gridworld import GridWorld
from model import Encoder, PolicyModel

def run_episode(env, policy, max_steps):
    obs = env._get_obs()
    trajectory = []
    reached_goal = False
    for _ in range(max_steps):
        logits = policy(obs)
        action, logp = sample_action(logits)
        next_obs, done = env.step(action)
        prev_state = obs[0].flatten().argmax().item()
        next_state = next_obs[0].flatten().argmax().item()
        trajectory.append((prev_state, action, next_state, logp))
        obs = next_obs
        if done:
            reached_goal = True
            break
    return trajectory, reached_goal

def train_stage(env, policy, optimizer, stage):
    cumulative_offsets = []
    for s in range(1, stage+1):
        if s in GridWorld.STAGE_OFFSETS:
            cumulative_offsets.extend(GridWorld.STAGE_OFFSETS[s])
    max_steps = max(abs(dx)+abs(dy) for dx, dy in cumulative_offsets)
    success_count = 0
    step_sum = 0
    for ep in range(1, settings.training_steps+1):
        env.sample_stage(stage)
        trajectory, reached_goal = run_episode(env, policy, max_steps)
        step_sum += len(trajectory)
        if reached_goal:
            success_count += 1
            rewards = assign_rewards(trajectory)
        else:
            rewards = [0.0]*len(trajectory)
        loss = torch.tensor(0.0)
        for (_, _, _, logp), r in zip(trajectory, rewards):
            loss = loss-logp*r
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep%settings.step_interval == 0:
            success_rate = success_count/settings.step_interval
            avg_steps = step_sum/settings.step_interval
            print(f"Stage {stage}, Episode {ep}, success rate: {success_rate:.2f}, avg steps: {avg_steps:.2f}")
            if success_rate >= settings.threshold:
                break
            success_count = 0
            step_sum = 0
    print(f"Completed stage {stage}, moving to stage {stage+1}")

def train_policy():
    env = GridWorld()
    encoder = Encoder()
    pretrained = torch.load("world_model.pt")
    encoder_state = {k.replace("encoder.", ""):v for k, v in pretrained.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    policy = PolicyModel(encoder)
    optimizer = torch.optim.Adam(policy.parameters(), lr = 1e-4)
    stages = sorted(GridWorld.STAGE_OFFSETS.keys())
    for stage in stages:
        cumulative_offsets = []
        for s in range(1, stage+1):
            if s in GridWorld.STAGE_OFFSETS:
                cumulative_offsets.extend(GridWorld.STAGE_OFFSETS[s])
        max_steps_stage = max(abs(dx)+abs(dy) for dx, dy in cumulative_offsets)
        success_count_stage = 0
        step_sum_stage = 0
        for ep in range(1, settings.training_steps+1):
            env.sample_stage(stage)
            trajectory, reached_goal = run_episode(env, policy, max_steps_stage)
            step_sum_stage += len(trajectory)
            if reached_goal:
                success_count_stage += 1
                rewards = assign_rewards(trajectory)
            else:
                rewards = [0.0]*len(trajectory)
            loss = torch.tensor(0.0)
            for (_, _, _, logp), r in zip(trajectory, rewards):
                loss = loss-logp*r
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ep%settings.step_interval == 0:
                success_rate = success_count_stage/settings.step_interval
                avg_steps = step_sum_stage/settings.step_interval
                print(f"Stage {stage}, Episode {ep}, success rate: {success_rate:.2f}, avg steps: {avg_steps:.2f}")
                if success_rate >= settings.threshold:
                    break
                success_count_stage = 0
                step_sum_stage = 0
        print(f"Completed stage {stage}, moving to stage {stage+1}")
    max_steps_free = 8
    success_count = 0
    step_sum = 0
    for ep in range(1, settings.training_steps+1):
        obs = env.reset()
        trajectory, reached_goal = run_episode(env, policy, max_steps_free)
        step_sum += len(trajectory)
        if reached_goal:
            success_count += 1
            rewards = assign_rewards(trajectory)
        else:
            rewards = [0.0]*len(trajectory)
        loss = torch.tensor(0.0)
        for (_, _, _, logp), r in zip(trajectory, rewards):
            loss = loss-logp*r
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep%settings.step_interval == 0:
            success_rate = success_count/settings.step_interval
            avg_steps = step_sum/settings.step_interval
            print(f"Free roam episode {ep}, success rate: {success_rate:.2f}, avg steps: {avg_steps:.2f}")
            success_count = 0
            step_sum = 0
    torch.save(policy.state_dict(), "policy_model.pt")

def sample_action(logits):
    probs = F.softmax(logits, dim = -1)
    dist = torch.distributions.Categorical(probs)
    action_tensor = dist.sample()
    return action_tensor.item(), dist.log_prob(action_tensor)

def assign_rewards(trajectory):
    return [1.0] * len(trajectory) if trajectory else []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action = "store_true")
    args = parser.parse_args()
    if args.pretrain:
        train_world()
    else:
        train_policy()

