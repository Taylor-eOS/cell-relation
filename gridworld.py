import random
import numpy as np
import settings

class GridWorld:
    def __init__(self):
        self.size = settings.grid_size
        self.center = [self.size//2, self.size//2]
        self.agent_pos = list(self.center)
        self.goal_pos = list(self.center)
        self.wall_positions = []

    def _get_obs(self):
        agent_map = np.zeros((self.size, self.size), dtype = np.float32)
        goal_map = np.zeros((self.size, self.size), dtype = np.float32)
        wall_map = np.zeros((self.size, self.size), dtype = np.float32)
        agent_map[self.agent_pos[0], self.agent_pos[1]] = 1.0
        goal_map[self.goal_pos[0], self.goal_pos[1]] = 1.0
        for wx, wy in self.wall_positions:
            wall_map[wx, wy] = 1.0
        return np.stack([agent_map, goal_map, wall_map], axis = 0)

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
        stepped_on_wall = tuple(new_pos) in self.wall_positions
        if not stepped_on_wall:
            self.agent_pos = new_pos
        obs = self._get_obs()
        done = (self.agent_pos == self.goal_pos)
        return obs, done, stepped_on_wall

    def sample_stage(self, stage):
        free_cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        self.agent_pos = list(random.choice(free_cells))
        free_cells.remove(tuple(self.agent_pos))
        all_offsets = []
        max_stage = max(settings.stage_offsets.keys())
        actual_stage = min(stage, max_stage)
        for s in range(1, actual_stage + 1):
            all_offsets.extend(settings.stage_offsets[s])
        candidates = []
        ax, ay = self.agent_pos
        for dx, dy in all_offsets:
            gx = ax + dx
            gy = ay + dy
            if 0 <= gx < self.size and 0 <= gy < self.size:
                candidates.append((gx, gy))
        candidates = [c for c in set(candidates) if c != tuple(self.agent_pos)]
        if candidates:
            self.goal_pos = list(random.choice(candidates))
        else:
            self.goal_pos = list(random.choice(free_cells))
        self.wall_positions = self.generate_wall(avoid_positions=[], agent_pos=self.agent_pos, goal_pos=self.goal_pos, require_blocking=False)
        return self._get_obs()

    def reset(self):
        free_cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        self.agent_pos = list(random.choice(free_cells))
        free_cells.remove(tuple(self.agent_pos))
        self.goal_pos = list(random.choice(free_cells))
        free_cells.remove(tuple(self.goal_pos))
        self.wall_positions = self.generate_wall(
            avoid_positions=[],
            agent_pos=self.agent_pos,
            goal_pos=self.goal_pos,
            require_blocking=True)
        return self._get_obs()

    def generate_wall(self, avoid_positions, agent_pos=None, goal_pos=None, require_blocking=True):
        size = self.size
        all_walls = []
        for x in range(size):
            for y in range(size - 2):
                all_walls.append([(x, y), (x, y + 1), (x, y + 2)])
        for y in range(size):
            for x in range(size - 2):
                all_walls.append([(x, y), (x + 1, y), (x + 2, y)])
        avoid_set = set(avoid_positions)
        if agent_pos is not None:
            avoid_set.add(tuple(agent_pos))
        if goal_pos is not None:
            avoid_set.add(tuple(goal_pos))
        valid_walls = [w for w in all_walls if not (set(w) & avoid_set)]
        if agent_pos is not None and goal_pos is not None:
            blocking_walls = [w for w in valid_walls if self.is_blocking(agent_pos, goal_pos, w)]
            if require_blocking:
                if blocking_walls:
                    return random.choice(blocking_walls)
                if valid_walls:
                    return random.choice(valid_walls)
            else:
                if valid_walls:
                    return random.choice(valid_walls)
        min_overlap = float('inf')
        best_wall = all_walls[0]
        for w in all_walls:
            overlap = len(set(w) & avoid_set)
            if overlap < min_overlap:
                min_overlap = overlap
                best_wall = w
        return best_wall

    def is_blocking(self, agent_pos, goal_pos, wall):
        ax, ay = agent_pos
        gx, gy = goal_pos
        if ay == gy:
            min_x, max_x = min(ax, gx), max(ax, gx)
            for wx, wy in wall:
                if wy == ay and min_x < wx < max_x:
                    return True
        if ax == gx:
            min_y, max_y = min(ay, gy), max(ay, gy)
            for wx, wy in wall:
                if wx == ax and min_y < wy < max_y:
                    return True
        for wx, wy in wall:
            if ay == wy and min(ax, wx) < wx < max(ax, wx):
                return True
            if ax == wx and min(ay, wy) < wy < max(ay, wy):
                return True
        return False

