import os
import numpy as np
import matplotlib.pyplot as plt
from settings import create_log

def render_grid(env, state_name, output_dir):
    if create_log:
        os.makedirs(output_dir, exist_ok=True)
        obs = env._get_obs()
        size = obs.shape[1]
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        wall_map = obs[2]
        agent_map = obs[0]
        goal_map = obs[1]
        img[wall_map == 1.0] = [100, 100, 100]
        img[goal_map == 1.0] = [255, 0, 0]
        img[agent_map == 1.0] = [0, 0, 255]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(output_dir, f'{state_name}.png'))
        plt.close(fig)

def analyze_episode(trajectory, reached_goal, ep, stage=None, free_roam=False):
    if create_log:
        if free_roam:
            filename = 'images/free_roam/free_roam_log.txt'
            context = f'Free roam episode {ep}'
        else:
            filename = 'images/stage_log.txt'
            context = f'Stage {stage}, Episode {ep}'
        with open(filename, 'a') as f:
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

