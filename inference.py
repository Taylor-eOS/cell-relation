import pygame
import torch
import torch.nn.functional as F
import time
from model import Encoder, PolicyModel
from gridworld import GridWorld
from shared import sample_action
import settings

def draw_grid(env, screen, grid_size, cell_size, button_height, hover, pressing):
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
    button_rect = pygame.Rect(0, grid_size * cell_size, grid_size * cell_size, button_height)
    base_color = (220, 220, 220)
    hover_color = (200, 200, 200)
    press_color = (180, 180, 180)
    if pressing:
        color = press_color
    elif hover:
        color = hover_color
    else:
        color = base_color
    pygame.draw.rect(screen, color, button_rect)
    font = pygame.font.Font(None, 36)
    text = font.render("Restart", True, (0, 0, 0))
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)
    pygame.display.flip()

def inference_loop():
    pygame.init()
    env = GridWorld()
    grid_size = env.size
    cell_size = settings.inference_cell_size
    button_height = 40
    screen_width = grid_size * cell_size
    screen_height = grid_size * cell_size + button_height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("GridWorld Inference")
    encoder = Encoder()
    state_dict = torch.load(settings.policy_model)
    encoder_state = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    policy = PolicyModel(encoder)
    policy.load_state_dict(state_dict)
    policy.eval()
    running = True
    while running:
        obs = env.reset()
        hover_button = False
        pressing_button = False
        draw_grid(env, screen, grid_size, cell_size, button_height, hover_button, pressing_button)
        time.sleep(settings.inference_sleep)
        done = False
        while not done and running:
            with torch.no_grad():
                logits = policy(obs)
            action = logits.argmax(dim=-1).item()
            obs, done, _, _ = env.step(action)
            mx, my = pygame.mouse.get_pos()
            hover_button = (my >= grid_size * cell_size and mx < grid_size * cell_size)
            draw_grid(env, screen, grid_size, cell_size, button_height, hover_button, pressing_button)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    hover_button = (my >= grid_size * cell_size and mx < grid_size * cell_size)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if my >= grid_size * cell_size and mx < grid_size * cell_size:
                        pressing_button = True
                        draw_grid(env, screen, grid_size, cell_size, button_height, hover_button, pressing_button)
                        done = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    pressing_button = False
            time.sleep(settings.inference_sleep)
    pygame.quit()

if __name__ == "__main__":
    inference_loop()
    pygame.quit()

