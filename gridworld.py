import random
import numpy as np
import settings

class GridWorld:
    STAGE_OFFSETS = {
        1: [(-1,0),(1,0),(0,-1),(0,1)],
        2: [(-2,0),(2,0)],
        3: [(0,-2),(0,2)],
        4: [(-1,-1),(-1,1),(1,-1),(1,1)],
        5: [(-2,-2),(-2,2),(2,-2),(2,2)],}

    def __init__(self):
        self.size = settings.grid_size
        self.center = [self.size//2,self.size//2]
        self.agent_pos = list(self.center)
        self.goal_pos = list(self.center)

    def reset(self):
        self.agent_pos = [random.randint(0,self.size-1),random.randint(0,self.size-1)]
        self.goal_pos = [random.randint(0,self.size-1),random.randint(0,self.size-1)]
        while self.goal_pos == self.agent_pos:
            self.goal_pos = [random.randint(0,self.size-1),random.randint(0,self.size-1)]
        return self._get_obs()

    def step(self,action):
        if action == 0:
            self.agent_pos[0] = max(0,self.agent_pos[0]-1)
        elif action == 1:
            self.agent_pos[0] = min(self.size-1,self.agent_pos[0]+1)
        elif action == 2:
            self.agent_pos[1] = max(0,self.agent_pos[1]-1)
        elif action == 3:
            self.agent_pos[1] = min(self.size-1,self.agent_pos[1]+1)
        obs = self._get_obs()
        done = (self.agent_pos == self.goal_pos)
        return obs,done

    def _get_obs(self):
        agent_map = np.zeros((self.size,self.size),dtype=np.float32)
        goal_map = np.zeros((self.size,self.size),dtype=np.float32)
        agent_map[self.agent_pos[0],self.agent_pos[1]] = 1.0
        goal_map[self.goal_pos[0],self.goal_pos[1]] = 1.0
        return np.stack([agent_map,goal_map],axis=0)

    def sample_stage(self,stage):
        self.agent_pos = list(self.center)
        x,y = self.center
        all_offsets = []
        for s in range(1,stage+1):
            if s in GridWorld.STAGE_OFFSETS:
                all_offsets.extend(GridWorld.STAGE_OFFSETS[s])
        valid_positions = []
        for dx,dy in all_offsets:
            nx,ny = x+dx,y+dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                valid_positions.append((nx,ny))
        self.goal_pos = list(random.choice(valid_positions))
        return self._get_obs()

