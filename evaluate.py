import torch
import random
import json
import numpy as np
from shared import initialize_policy, load_wall_stage_data, sample_wall_example, run_episode
import settings
import utils

def evaluate_wall_curriculum(num_episodes, max_steps=settings.max_steps, model_path=settings.policy_model, render_samples=settings.evaluation_render_samples):
    env, policy, _, _ = initialize_policy()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    stage_data, stages = load_wall_stage_data("curriculum.npz")
    success_rates = []
    for stage in stages:
        if stage not in stage_data:
            print(f"Skipping stage {stage}: no data in curriculum.")
            success_rates.append(0.0)
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
        rendered_count = 0
        for ep in range(num_episodes):
            sample_wall_stage_only(env, stage, stage_data)
            if rendered_count < render_samples:
                if settings.create_grid_log:
                    print_grid(env.agent_pos, env.wall_positions, env.goal_pos)
                trajectory, reached_goal = run_episode(env, policy, max_steps, render_prefix=f'wall_stage{stage}_ep{ep+1}', render_dir='images')
                rendered_count += 1
            else:
                _, reached_goal = run_episode(env, policy, max_steps)
            if reached_goal:
                success_count += 1
        success_rate = success_count / num_episodes
        success_rates.append(success_rate)
        print(f"Stage {stage}: {success_count}/{num_episodes}")
    return success_rates

def evaluate_and_cache_performance(num_episodes=settings.evaluation_episodes, max_steps=settings.max_steps, model_path=settings.policy_model):
    print("Evaluating curriculum")
    success_rates = evaluate_wall_curriculum(num_episodes, max_steps, model_path)
    size = settings.grid_size
    all_raw = []
    for dx in range(-size+1, size):
        for dy in range(-size+1, size):
            if (dx, dy) == (0, 0):
                continue
            all_raw.append((dx, dy))
    all_uniq = {(abs(dx), abs(dy)) for dx, dy in all_raw}
    all_distance_sorted = sorted(all_uniq, key=lambda t: (t[0]*t[0] + t[1]*t[1], t[0], t[1]))
    back = [off for off in all_distance_sorted if (off[0] == 0 or off[1] == 0) and (off[0] > 1 or off[1] > 1)]
    front = [off for off in all_distance_sorted if off not in back]
    pretraining_ordered = front + back
    ordered_offsets = [off for off in pretraining_ordered if off not in ((1, 0), (0, 1))]
    performance_list = [{"offset": [dx, dy], "rate": rate} for (dx, dy), rate in zip(ordered_offsets, success_rates)]
    with open("offset_performance.json", "w") as f:
        json.dump(performance_list, f, indent=2)
    return success_rates

def sample_wall_stage_only(env, stage, stage_data):
    curr = stage_data[stage]
    N = curr["obs"].shape[0]
    idx = random.randrange(N)
    agent_arr, goal_arr, wall_arr, num_walls = curr["agent"], curr["goal"], curr["wall"], curr["nw"]
    wcount = int(num_walls[idx])
    env.agent_pos = [int(agent_arr[idx, 0]), int(agent_arr[idx, 1])]
    env.goal_pos = [int(goal_arr[idx, 0]), int(goal_arr[idx, 1])]
    env.wall_positions = [tuple(wall_arr[idx, i]) for i in range(wcount)]

def print_grid(agent_pos, wall_positions, goal_pos, size=settings.grid_size):
    grid = [['.'] * size for _ in range(size)]
    for x, y in wall_positions:
        grid[y][x] = '□'
    ax, ay = agent_pos
    grid[ay][ax] = '⚉'
    gx, gy = goal_pos
    grid[gy][gx] = '○'
    print('\n'.join(''.join(row) for row in grid))
    print()

if __name__ == "__main__":
    evaluate_and_cache_performance()

