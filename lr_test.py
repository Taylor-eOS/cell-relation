import sys
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import linregress
import numpy as np
import settings
from run import compute_loss, assign_rewards
from shared import initialize_policy, run_episode

def evaluate_accuracy(lr):
    env, policy, _ = initialize_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    steps_limit = settings.lr_test_steps
    success_count = 0
    total_steps = 0
    total_reward = 0
    loss_history = []
    grad_norms = []
    for ep in range(1, steps_limit + 1):
        trajectory, reached_goal = run_episode(env, policy, settings.max_steps)
        total_steps += len(trajectory)
        total_reward += sum(assign_rewards(trajectory, reached_goal))
        if reached_goal:
            success_count += 1
        loss = compute_loss(trajectory, assign_rewards(trajectory, reached_goal))
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'))
        grad_norms.append(grad_norm)
        loss_history.append(loss.item())
        optimizer.step()
        percent = ep / steps_limit * 100
        sys.stdout.write(f"\rProgress: {percent:.0f}%")
        sys.stdout.flush()
    print()
    success_rate = success_count / steps_limit
    avg_steps = total_steps / steps_limit
    avg_reward = total_reward / steps_limit
    losses = np.array(loss_history)
    grads = np.array(grad_norms)
    steps = np.arange(steps_limit)
    loss_slope = linregress(steps, losses).slope
    grad_slope = linregress(steps, grads).slope
    early_loss = losses[:steps_limit // 2]
    late_loss = losses[steps_limit // 2:]
    early_grad = grads[:steps_limit // 2]
    late_grad = grads[steps_limit // 2:]
    loss_drop = np.mean(early_loss) - np.mean(late_loss)
    grad_drop = np.mean(early_grad) - np.mean(late_grad)
    loss_cv = np.std(losses) / (np.mean(losses) + 1e-8)
    grad_cv = np.std(grads) / (np.mean(grads) + 1e-8)
    exploding = np.any(np.isnan(grads)) or np.any(grads > 1000)
    if settings.lr_print_each: print(f"{lr}: success_rate={success_rate:.2f}, avg_steps={avg_steps:.2f}, avg_reward={avg_reward:.2f}")
    if settings.lr_print_each: print(f"      loss_drop={loss_drop:.4f}, loss_slope={loss_slope:.6f}, loss_cv={loss_cv:.3f}")
    if settings.lr_print_each: print(f"      grad_drop={grad_drop:.4f}, grad_slope={grad_slope:.6f}, grad_cv={grad_cv:.3f}, exploding={exploding}")
    return {
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'loss_slope': loss_slope,
        'loss_cv': loss_cv,
        'grad_slope': grad_slope,
        'grad_cv': grad_cv,
        'exploding': exploding}

def average_metrics(metrics_list):
    keys = metrics_list[0].keys()
    avg_metrics = {}
    for k in keys:
        values = [m[k] for m in metrics_list]
        avg_metrics[k] = np.mean(values)
    return avg_metrics

def evaluate_multiple_runs(lr, runs=settings.lr_test_runs):
    run_metrics = []
    for i in range(runs):
        print(f"Test {i + 1}")
        metrics = evaluate_accuracy(lr)
        run_metrics.append(metrics)
    avg = average_metrics(run_metrics)
    print(f"\nLR {lr}: averaged results over {runs} runs:")
    print(f"  success_rate={avg['success_rate']:.3f}, avg_steps={avg['avg_steps']:.3f}, avg_reward={avg['avg_reward']:.3f}")
    print(f"  loss_slope={avg['loss_slope']:.6f}, loss_cv={avg['loss_cv']:.3f}")
    print(f"  grad_slope={avg['grad_slope']:.6f}, grad_cv={avg['grad_cv']:.3f}, exploding={avg['exploding']}")
    return avg

if __name__ == "__main__":
    evaluate_multiple_runs(settings.learning_rate)

