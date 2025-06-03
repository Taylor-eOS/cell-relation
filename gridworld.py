import random
import numpy as np
import settings

class GridWorld:
    STAGE_OFFSETS = {
        1: [(-1, 0), (1, 0), (0, -1), (0, 1)],
        2: [(1, 1), (-1, 1)],
        3: [(1, -1), (-1, -1)],
        4: [(2, 0), (-2, 0)],
        5: [(0, 2), (0, -2)],
        6: [(2, 1), (-2, 1)],
        7: [(2, -1), (-2, -1)],
        8: [(1, 2), (-1, 2)],
        9: [(1, -2), (-1, -2)],
        10: [(2, 2), (-2, 2)],
        11: [(2, -2), (-2, -2)]}

    def __init__(self):
        self.size = settings.grid_size
        self.center = [self.size // 2, self.size // 2]
        self.agent_pos = list(self.center)
        self.goal_pos = list(self.center)
        self.wall_positions = []

    def _generate_wall(self, avoid_positions):
        walls = []
        for x in range(self.size):
            for y in range(self.size - 2):
                wall = [(x, y), (x, y+1), (x, y+2)]
                walls.append(wall)
        for y in range(self.size):
            for x in range(self.size - 2):
                wall = [(x, y), (x+1, y), (x+2, y)]
                walls.append(wall)
        avoid_set = set(avoid_positions)
        valid_walls = [wall for wall in walls if not set(wall) & avoid_set]
        if valid_walls:
            return random.choice(valid_walls)
        else:
            min_overlap = float('inf')
            best_wall = walls[0]
            for wall in walls:
                overlap = len(set(wall) & avoid_set)
                if overlap < min_overlap:
                    min_overlap = overlap
                    best_wall = wall
            return best_wall

    def reset(self):
        self.wall_positions = self._generate_wall(avoid_positions=[])
        free_cells = set((i, j) for i in range(self.size) for j in range(self.size))
        free_cells -= set(map(tuple, self.wall_positions))
        free_cells = list(free_cells)
        self.agent_pos = list(random.choice(free_cells))
        free_cells.remove(tuple(self.agent_pos))
        self.goal_pos = list(random.choice(free_cells))
        return self._get_obs()

    def sample_stage(self, stage):
        self.wall_positions = self._generate_wall(avoid_positions=[tuple(self.center)])
        self.agent_pos = list(self.center)
        x, y = self.center
        all_offsets = []
        for s in range(1, stage + 1):
            if s in GridWorld.STAGE_OFFSETS:
                all_offsets.extend(GridWorld.STAGE_OFFSETS[s])
        valid_positions = []
        for dx, dy in all_offsets:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.size and 0 <= ny < self.size and 
                (nx, ny) not in self.wall_positions):
                valid_positions.append((nx, ny))
        if valid_positions:
            self.goal_pos = list(random.choice(valid_positions))
        else:
            free_cells = set((i, j) for i in range(self.size) for j in range(self.size))
            free_cells -= set(map(tuple, self.wall_positions))
            free_cells -= {tuple(self.center)}
            free_cells = list(free_cells)
            if free_cells:
                self.goal_pos = list(random.choice(free_cells))
            else:
                self.goal_pos = list(self.center)
        return self._get_obs()

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:
            new_pos = [max(0, x - 1), y]
        elif action == 1:
            new_pos = [min(self.size - 1, x + 1), y]
        elif action == 2:
            new_pos = [x, max(0, y - 1)]
        elif action == 3:
            new_pos = [x, min(self.size - 1, y + 1)]
        else:
            new_pos = [x, y]
        if tuple(new_pos) not in self.wall_positions:
            self.agent_pos = new_pos
        obs = self._get_obs()
        done = (self.agent_pos == self.goal_pos)
        return obs, done

    def _get_obs(self):
        agent_map = np.zeros((self.size, self.size), dtype=np.float32)
        goal_map = np.zeros((self.size, self.size), dtype=np.float32)
        wall_map = np.zeros((self.size, self.size), dtype=np.float32)
        agent_map[self.agent_pos[0], self.agent_pos[1]] = 1.0
        goal_map[self.goal_pos[0], self.goal_pos[1]] = 1.0
        for wx, wy in self.wall_positions:
            wall_map[wx, wy] = 1.0
        return np.stack([agent_map, goal_map, wall_map], axis=0)

