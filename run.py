import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shutil
from gridworld import GridWorld
from analysis import analyze_episode
from shared import initialize_policy, load_wall_stage_data, sample_wall_example, run_episode
import wall_curriculum
import evaluate
import settings
import utils

def train_policy():
    env, policy, optimizer = initialize_policy()
    stages = utils.stage_offsets(preliminary=True).keys()
    if settings.skip_to_wall:
        policy.load_state_dict(torch.load("policy_model.pt"))
    else:
        for i in range(settings.epochs):
            print(f"Epoch {i + 1}")
            for stage in stages:
                train_stage(env, policy, optimizer, stage)
            torch.save(policy.state_dict(), "policy_model.pt")
    if settings.skip_curriculum:
        policy.load_state_dict(torch.load("policy_model.pt"))
    else:
        if settings.run_wall_curriculum_evaluation:
            wall_curriculum.main()
            evaluate.evaluate_and_cache_performance(model_path="policy_model.pt")
        for i in range(settings.epochs):
            print(f"Epoch {i + 1}")
            train_curriculum(env, policy, optimizer)
            torch.save(policy.state_dict(), "policy_model.pt")
    train_free_roam(env, policy, optimizer)
    torch.save(policy.state_dict(), "policy_model.pt")

def train_stage(env, policy, optimizer, stage):
    max_steps = settings.max_steps
    success_count = 0
    step_sum = 0
    for ep in range(1, settings.training_steps + 1):
        env.sample_stage(stage, preliminary=True)
        if ep == 1 or settings.full_state_render:
            trajectory, reached_goal = run_episode(env, policy, max_steps, render_prefix=f'stage{stage}_ep{ep}', render_dir=f'images')
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
    max_steps = settings.max_steps
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
            if ep < settings.wall_renders + 1 or settings.full_state_render:
                trajectory, reached_goal = run_episode(env, policy, max_steps, render_prefix=f'stage{stage}_ep{ep}', render_dir=f'images/wall_curriculum')
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
                print(f"Wall stage {stage}, episode {ep}, success rate: {success_rate:.2f}, avg steps: {avg_steps:.2f}")
                if success_rate >= settings.threshold:
                    break
                success_count = 0
                step_sum = 0
        print(f"Stage {stage + 1}")

def train_free_roam(env, policy, optimizer):
    success_count = 0
    step_sum = 0
    for ep in range(1, settings.roam_steps + 1):
        env.reset()
        if ep % settings.free_roam_log == 0 or settings.full_state_render:
            render_subdir = os.path.join('images/free_roam', f'ep{ep}')
            trajectory, reached_goal = run_episode(env, policy, settings.max_steps, render_prefix=f'freeroam_ep{ep}', render_dir=render_subdir)
            analyze_episode(trajectory, reached_goal, ep, free_roam=True)
        else:
            trajectory, reached_goal = run_episode(env, policy, settings.max_steps)
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

def assign_rewards(trajectory, reached_goal):
    rewards = []
    if reached_goal:
        seq_len = len(trajectory)
        for (_, _, _, _, collision) in trajectory:
            rewards.append(1.0 - 0.05 * seq_len)
    else:
        for (_, _, _, _, collision) in trajectory:
            if collision:
                rewards.append(-1.0)
            else:
                rewards.append(-0.1)
    return rewards

def compute_loss(trajectory, rewards):
    loss = torch.tensor(0.0)
    for (_, _, _, logp, _), r in zip(trajectory, rewards):
        loss = loss - logp * r
    return loss

if __name__ == "__main__":
    if os.path.isdir('images'):
        shutil.rmtree('images')
    train_policy()

