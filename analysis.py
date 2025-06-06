import os
import numpy as np
import matplotlib.pyplot as plt
import settings
import utils

def render_grid(env, state_name, output_dir):
        obs = env._get_obs()
        utils.render_obs(obs, state_name, output_dir, render_images=settings.render_images)

def analyze_episode(trajectory, reached_goal, ep, stage=None, free_roam=False):
    if settings.create_log:
        if free_roam:
            filename = 'images/free_roam/free_roam_log.txt'
            context = f'Free roam episode {ep}'
        else:
            filename = 'images/stage_log.txt'
            context = f'Stage {stage}, Episode {ep}'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(f'{context}: reached_goal={reached_goal}, '
                    f'steps={len(trajectory)}, '
                    f'collisions={sum(1 for (_, _, _, _, collision) in trajectory if collision)}\n')
            for i, (prev_state, action, next_state, logp, collision) in enumerate(trajectory, start=1):
                logp_val = logp.item() if hasattr(logp, "item") else float(logp)
                f.write(f'  Step {i:>2}: prev_state={prev_state}, '
                        f'action={action}, '
                        f'next_state={next_state}, '
                        f'logp={logp_val:.4f}, '
                        f'collision={collision}\n')
            f.write('\n')

