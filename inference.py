import pygame
import torch
import torch.nn.functional as F
import time
from run import sample_action
from model import Encoder, PolicyModel
from gridworld import GridWorld
import settings

def draw_grid(env, screen, grid_size, cell_size):
    screen.fill((255, 255, 255))
    for i in range(grid_size):
        for j in range(grid_size):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)
    for wx, wy in env.wall_positions:
        pygame.draw.rect(screen, (100, 100, 100), (wy * cell_size, wx * cell_size, cell_size, cell_size))
    gx, gy = env.goal_pos
    ax, ay = env.agent_pos
    pygame.draw.rect(screen, (255, 0, 0), (gy * cell_size, gx * cell_size, cell_size, cell_size))
    pygame.draw.rect(screen, (0, 0, 255), (ay * cell_size, ax * cell_size, cell_size, cell_size))
    pygame.display.flip()

def inference_loop():
    pygame.init()
    env = GridWorld()
    grid_size = env.size
    cell_size = settings.inference_cell_size
    screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
    pygame.display.set_caption("GridWorld Inference")
    encoder = Encoder()
    state_dict = torch.load("policy_model.pt")
    encoder_state = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    policy = PolicyModel(encoder)
    policy.load_state_dict(state_dict)
    policy.eval()
    running = True
    while running:
        obs = env.reset()
        draw_grid(env, screen, grid_size, cell_size)
        done = False
        while not done and running:
            with torch.no_grad():
                logits = policy(obs)
            action, _ = sample_action(logits)
            obs, done, stepped_on_wall = env.step(action)
            draw_grid(env, screen, grid_size, cell_size)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            time.sleep(settings.inference_sleep)
    pygame.quit()

if __name__ == "__main__":
    inference_loop()
    pygame.quit()

