import os
import shutil
import random
import numpy as np
import settings
import utils
import analysis

def make_obs(agent_pos, goal_pos, wall_positions, size):
    agent_map = np.zeros((size, size), dtype=np.float32)
    goal_map = np.zeros((size, size), dtype=np.float32)
    wall_map = np.zeros((size, size), dtype=np.float32)
    agent_map[agent_pos[0], agent_pos[1]] = 1.0
    goal_map[goal_pos[0], goal_pos[1]] = 1.0
    for wx, wy in wall_positions:
        wall_map[wx, wy] = 1.0
    return np.stack([agent_map, goal_map, wall_map], axis=0)

def filter_states(states, offsets, stage):
    filtered = []
    target_dx, target_dy = offsets[stage]
    for wall, agent_pos, goal_pos in states:
        dx = agent_pos[0] - goal_pos[0]
        dy = agent_pos[1] - goal_pos[1]
        if abs(dx) == abs(target_dx) and abs(dy) == abs(target_dy):
            filtered.append((wall, agent_pos, goal_pos))
    return filtered

def generate_all_states(size, walls):
    all_states = []
    for wall in walls:
        wall_set = set(wall)
        for ax in range(size):
            for ay in range(size):
                agent_pos = (ax, ay)
                if agent_pos in wall_set:
                    continue
                for gx in range(size):
                    for gy in range(size):
                        goal_pos = (gx, gy)
                        if goal_pos in wall_set or agent_pos == goal_pos:
                            continue
                        all_states.append((wall, agent_pos, goal_pos))
    return all_states

def save_states(all_states, size):
    offsets = utils.stage_offsets()
    os.makedirs("images", exist_ok=True)
    combined_data = {}

    def filter_by_offset(states, target_dx, target_dy):
        abs_dx = abs(target_dx)
        abs_dy = abs(target_dy)
        return [(wall, a, g) for wall, a, g in states if abs(a[0] - g[0]) == abs_dx and abs(a[1] - g[1]) == abs_dy]

    def filter_valid_positions(states):
        line_hits = [(w, a, g) for w, a, g in states if wall_in_line(a, g, set(w))]
        if line_hits:
            return line_hits
        adjacent_hits = []
        for wall, a, g in states:
            for wx, wy in wall:
                agent_adjacent = any(abs(a[0] - wx) + abs(a[1] - wy) == 1 for wx, wy in wall)
                goal_adjacent = any(abs(g[0] - wx) + abs(g[1] - wy) == 1 for wx, wy in wall)
                if agent_adjacent and goal_adjacent:
                    adjacent_hits.append((wall, a, g))
        return adjacent_hits

    def build_data_arrays(states):
        obs_list, agent_list, goal_list, wall_list = [], [], [], []
        for i, (wall, a, g) in enumerate(states):
            if settings.create_log:
                dx = g[0] - a[0]
                dy = g[1] - a[1]
                with open(f"images/wall_states_dx{dx}_dy{dy}.txt", "a") as f_txt:
                    f_txt.write(f"{wall}, {a}, {g}\n")
            obs = make_obs(a, g, wall, size)
            def render_to(path, obs, name): analysis.render_obs(obs, name, path, render_images=settings.render_state_images)
            if i == 0 or settings.full_state_render:
                render_to("images", obs, f"stage_{stage}_{i}")
            obs_list.append(obs)
            agent_list.append(a)
            goal_list.append(g)
            wall_list.append(wall)
        np_obs = np.stack(obs_list, axis=0)
        agent_pos = np.array(agent_list, dtype=np.int32)
        goal_pos = np.array(goal_list, dtype=np.int32)
        wall_arrays = [np.array(w, dtype=np.int32).reshape(-1, 2) for w in wall_list]
        max_walls = max(w.shape[0] for w in wall_arrays)
        wall_pos = np.full((len(wall_arrays), max_walls, 2), -1, dtype=np.int32)
        num_walls = np.zeros((len(wall_arrays),), dtype=np.int32)
        for i, w_arr in enumerate(wall_arrays):
            wall_pos[i, :w_arr.shape[0], :] = w_arr
            num_walls[i] = w_arr.shape[0]
        return np_obs, agent_pos, goal_pos, wall_pos, num_walls
    for stage, (dx, dy) in offsets.items():
        stage_states = filter_by_offset(all_states, dx, dy)
        if not stage_states:
            print(f"Stage {stage}: 0")
            continue
        filtered = filter_valid_positions(stage_states)
        if not filtered:
            print(f"Stage {stage}: 0 (no valid positions)")
            continue
        print(f"Stage {stage}: {len(filtered)}")
        np_obs, agent_pos, goal_pos, wall_pos, num_walls = build_data_arrays(filtered)
        combined_data[f"stage_{stage}_obs"] = np_obs
        combined_data[f"stage_{stage}_agent_pos"] = agent_pos
        combined_data[f"stage_{stage}_goal_pos"] = goal_pos
        combined_data[f"stage_{stage}_wall_pos"] = wall_pos
        combined_data[f"stage_{stage}_num_walls"] = num_walls
    np.savez_compressed("curriculum.npz", **combined_data)

def generate_all_walls(size):
    walls = []
    for x in range(size):
        for y in range(size - 2):
            walls.append([(x, y), (x, y + 1), (x, y + 2)])
    for y in range(size):
        for x in range(size - 2):
            walls.append([(x, y), (x + 1, y), (x + 2, y)])
    return walls

def wall_in_line(agent_pos, goal_pos, wall_set):
    line_points = utils.bresenham_line(agent_pos[0], agent_pos[1], goal_pos[0], goal_pos[1])
    intermediate_points = line_points[1:-1]
    return any(p in wall_set for p in intermediate_points)

def main():
    os.makedirs("images", exist_ok=True)
    size = settings.grid_size
    walls = generate_all_walls(size)
    print(f"Total walls generated: {len(walls)}")
    all_states = generate_all_states(size, walls)
    print(f"Total states: {len(all_states)}")
    save_states(all_states, size)

if __name__ == "__main__":
    if os.path.isdir('images'):
        shutil.rmtree('images')
    main()

