import os
import numpy as np
import matplotlib.pyplot as plt
import settings

def stage_offsets():
    offsets = {}
    stage = 1
    size = settings.grid_size
    for dist in range(1, size):
        cardinals = []
        for dx, dy in [(-dist, 0), (dist, 0), (0, -dist), (0, dist)]:
            cardinals.append((dx, dy))
        for offset in cardinals:
            offsets[stage] = offset
            stage += 1
        diagonals = []
        for m in range(1, dist + 1):
            if m > dist:
                break
            pairs = []
            if m == dist:
                pairs = [(dist, dist), (dist, -dist), (-dist, dist), (-dist, -dist)]
            else:
                for dx in (-dist, dist):
                    for dy in (-m, m):
                        pairs.append((dx, dy))
                for dx in (-m, m):
                    for dy in (-dist, dist):
                        pairs.append((dx, dy))
            for dx, dy in sorted(set(pairs), key=lambda t: (abs(t[0]), abs(t[1]), t[0], t[1])):
                diagonals.append((dx, dy))
        for offset in diagonals:
            offsets[stage] = offset
            stage += 1
    if settings.offsets_test:
        offsets = {1: (4, 4)}
    return offsets

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx // 2
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points

def render_obs(obs, state_name, output_dir, render_images=False):
    if not render_images:
        return
    os.makedirs(output_dir, exist_ok=True)
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

if __name__ == "__main__":
    print(stage_offsets())

