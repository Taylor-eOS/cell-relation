grid_size = 5
step_interval = 100
pretraining_steps = 50000
training_steps = 20000
threshold = 0.95
stage_offsets = {
    1: [(-1, 0), (1, 0)],
    2: [(0, -1), (0, 1)],
    3: [(1, 1), (-1, 1)],
    4: [(1, -1), (-1, -1)],
    5: [(2, 0), (-2, 0)],
    6: [(0, 2), (0, -2)],
    7: [(2, 1), (2, -1)],
    8: [(-2, 1), (-2, -1)],
    9: [(1, 2), (-1, 2)],
    10: [(1, -2), (-1, -2)],
    11: [(2, 2), (-2, 2)],
    12: [(2, -2), (-2, -2)]}
free_roam_log = 1000
inference_cell_size = 50
inference_sleep = 0.3
transformer_layers = 2
attention_heads = 2
world_lr = 1e-4
policy_lr = 1e-4
create_log = True
