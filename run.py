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
import utils

def train_policy():
    env, policy, optimizer = initialize_policy()
    stages = sorted(utils.stage_offsets().keys())
    if not settings.skip_to_wall:
        for stage in stages:
            train_stage(env, policy, optimizer, stage)
        torch.save(policy.state_dict(), "policy_model.pt")
    train_curriculum(env, policy, optimizer)
    torch.save(policy.state_dict(), "policy_model.pt")
    train_free_roam(env, policy, optimizer)
    torch.save(policy.state_dict(), "policy_model.pt")

def train_stage(env, policy, optimizer, stage):
    max_steps = 8
    success_count = 0
    step_sum = 0
    for ep in range(1, settings.training_steps + 1):
        env.sample_stage(stage)
        if ep == 1:
            trajectory, reached_goal = run_episode(env, policy, max_steps, render_prefix=f'stage{stage}_ep{ep}', render_dir=f'images/stage{stage}')
            analyze_episode(trajectory, reached_goal, ep, stage=stage)
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
            print(f"Stage {stage}, episode {ep}, success rate: {success_rate:.2f}, avg steps: {avg_steps:.2f}")
            if success_rate >= settings.threshold:
                break
            success_count = 0
            step_sum = 0
    print(f"Stage {stage + 1}")

def train_curriculum(env, policy, optimizer):
    stage_data, stages = load_wall_stage_data("curriculum.npz")
    max_steps = 8
    for stage in stages:
        if stage not in stage_data:
            print(f"Skipping curriculum for stage {stage}: no data.")
            continue
        prev_list = [stage_data[s] for s in stages if s < stage and s in stage_data]
        if prev_list:
            prev_data = {
                "obs": np.concatenate([d["obs"] for d in prev_list], axis=0),
                "agent": np.concatenate([d["agent"] for d in prev_list], axis=0),
                "goal": np.concatenate([d["goal"] for d in prev_list], axis=0),
                "wall": np.concatenate([d["wall"] for d in prev_list], axis=0),
                "nw": np.concatenate([d["nw"] for d in prev_list], axis=0),}
        else:
            prev_data = None
        success_count = 0
        step_sum = 0
        for ep in range(1, settings.curriculum_steps + 1):
            sample_wall_example(env, stage, stage_data, prev_data)
            if ep == 1:
                trajectory, reached_goal = run_episode(env, policy, max_steps, render_prefix=f'stage{stage}_ep{ep}', render_dir=f'images/wall_curriculum_stage{stage}')
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
                print(f"Curriculum stage {stage}, episode {ep}, success rate: {success_rate:.2f}, avg steps: {avg_steps:.2f}")
                if success_rate >= settings.threshold:
                    break
                success_count = 0
                step_sum = 0
        print(f"Stage {stage + 1}")

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

def load_wall_stage_data(filename):
    data = np.load(filename)
    stages = sorted(utils.stage_offsets().keys())
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

