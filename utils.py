import os
import json
import numpy as np
import matplotlib.pyplot as plt
import settings

def stage_offsets(preliminary=True):
    size = settings.grid_size
    all_raw = []
    for dx in range(-size+1, size):
        for dy in range(-size+1, size):
            if (dx, dy) == (0, 0):
                continue
            all_raw.append((dx, dy))
    all_uniq = {(abs(dx), abs(dy)) for dx, dy in all_raw}
    all_distance_sorted = sorted(all_uniq, key=lambda t: (t[0]*t[0] + t[1]*t[1], t[0], t[1]))
    if preliminary:
        back = [off for off in all_distance_sorted
                if (off[0] == 0 or off[1] == 0) and (off[0] > 1 or off[1] > 1)]
        front = [off for off in all_distance_sorted if off not in back]
        ordered = front + back
        return {i+1: off for i, off in enumerate(ordered)}
    else:
        perf_file = "offset_performance.json"
        if not os.path.exists(perf_file):
            raise FileNotFoundError(f"{perf_file} not found")
        with open(perf_file) as f:
            perf_list = json.load(f)
        filtered = [
            (tuple(entry["offset"]), entry["rate"])
            for entry in perf_list
            if entry["offset"] not in ([1, 0], [0, 1])]
        filtered.sort(key=lambda p: p[1], reverse=True)
        sorted_offsets = [off for off, _ in filtered]
        sorted_list = {i+1: off for i, off in enumerate(sorted_offsets)}
        return sorted_list

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

def block_options_for(offset, size=settings.grid_size):
    ax, ay = 0, 0
    gx, gy = offset
    line_pts = set(bresenham_line(ax, ay, gx, gy)[1:-1])
    legal = 0
    for x in range(size):
        for y in range(size - 2):
            wall = {(x, y), (x, y+1), (x, y+2)}
            if wall & {(ax, ay), (gx, gy)}: continue
            if wall & line_pts: legal += 1
    for y in range(size):
        for x in range(size - 2):
            wall = {(x, y), (x+1, y), (x+2, y)}
            if wall & {(ax, ay), (gx, gy)}: continue
            if wall & line_pts: legal += 1
    return legal

def scored_offsets():
    size    = settings.grid_size
    offsets = utils.stage_offsets()
    scores  = []
    for stage, (dx, dy) in utils.items():
        b = block_options_for((dx, dy), size)
        a = dx * dy
        score = a / (b + 1)
        scores.append((stage, (dx, dy), b, a, score))
    return sorted(scores, key=lambda x: x[-1])

if __name__ == "__main__":
    s1 = stage_offsets()
    s2 = stage_offsets(preliminary=False)
    with open(f'offsets.txt', "w") as f:
        f.write(f"Preliminary mode:\n")
        f.write(f"{s1}\n")
        f.write(f"Curriculum mode:\n")
        f.write(f"{s2}\n")
    print("Preliminary mode:")
    print(f"{s1}")
    print("Curriculum mode:")
    print(f"{s2}")
    if False:
        for stage, off, blockers, area, score in scored_offsets():
            print(f"Stage {stage:2d}: off={off} blockers={blockers:2d} area={area:2d} score={score:.3f}")

