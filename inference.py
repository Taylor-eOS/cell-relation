import pygame, time, torch
import numpy as np
from model import Encoder, PolicyModel, sample_action
from gridworld import GridWorld
import settings

pygame.init()
cell_size = 40
grid_size = settings.grid_size
screen = pygame.display.set_mode((cell_size * grid_size, cell_size * grid_size))
clock = pygame.time.Clock()

def draw_grid(env):
    screen.fill((255, 255, 255))
    for i in range(grid_size):
        for j in range(grid_size):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)
    gx, gy = env.goal_pos
    ax, ay = env.agent_pos
    pygame.draw.rect(screen, (255, 0, 0), (gy * cell_size, gx * cell_size, cell_size, cell_size))
    pygame.draw.rect(screen, (0, 0, 255), (ay * cell_size, ax * cell_size, cell_size, cell_size))
    pygame.display.flip()

def inference_loop():
    env = GridWorld()
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
        draw_grid(env)
        time.sleep(0.5)
        done = False
        while not done:
            logits = policy(obs)
            action, _ = sample_action(logits)
            obs, done = env.step(action)
            draw_grid(env)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            time.sleep(0.4)
        time.sleep(1)

def draw_grid(env):
    screen.fill((255, 255, 255))
    for i in range(grid_size):
        for j in range(grid_size):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)
    gx, gy = env.goal_pos
    ax, ay = env.agent_pos
    pygame.draw.rect(screen, (160, 160, 160), (gy * cell_size, gx * cell_size, cell_size, cell_size))
    pygame.draw.rect(screen, (0, 0, 0), (ay * cell_size, ax * cell_size, cell_size, cell_size))
    pygame.display.flip()

if __name__ == "__main__":
    inference_loop()
    pygame.quit()

