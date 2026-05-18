import torch
import wandb
import torch.nn as nn
from torch.distributions import Normal
from tqdm.auto import tqdm
from util.logger import get_logger

logger = get_logger()


class EtaPolicyNetwork(nn.Module):
    def __init__(self, hidden_dim=128, eta_max=5.0, std=0.3, state_norm_config=None):
        super().__init__()
        self.eta_max = eta_max
        self.std = std

        # State normalization parameters
        self.log_consistency_offset = 15.0
        self.log_consistency_scale = 0.1
        if state_norm_config:
            self.log_consistency_offset = float(state_norm_config.get('log_consistency_offset', 15.0))
            self.log_consistency_scale = float(state_norm_config.get('log_consistency_scale', 0.1))

        # Input is 2 (normalized timestep + normalized log_consistency)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # state: [B, 2] -> [t_norm, log_consistency]
        t_norm = state[:, 0:1]
        log_consist = state[:, 1:2]

        # Normalize log_consistency to be roughly in [0, 1] range
        # log_consistency is usually negative (e.g. -5 to -15). 
        # Adding offset 15 and scaling by 0.1 makes -15 -> 0.0, -5 -> 1.0
        norm_log_consist = (log_consist + self.log_consistency_offset) * self.log_consistency_scale

        normalized_state = torch.cat([t_norm, norm_log_consist], dim=-1)

        h = self.net(normalized_state)
        return self.mu_head(h)

    def sample_eta(self, state, deterministic=False):
        mu = self.forward(state)
        if deterministic:
            action = mu
            log_prob = torch.zeros_like(mu)
            entropy = torch.zeros_like(mu)
        else:
            std = torch.ones_like(mu) * self.std
            dist = Normal(mu, std)

            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        # Scale eta using the configured maximum
        eta = torch.sigmoid(action) * self.eta_max

        return eta, log_prob, entropy

def train_rl_policy(diffusion_model, model, policy_net, optimizer, loader, operator, noiser, measurement_cond_fn, num_episodes, reward_fn, max_steps=64, device='cuda', baseline_decay=0.99, grad_clip=1.0, entropy_beta=0.01, save_path=None, save_interval=500):
    """
    REINFORCE algorithm with Timestep-Dependent Baselines and Periodic Saving.
    """
    policy_net.train()

    # Timestep-dependent baseline tensor and visit counters
    num_ts = diffusion_model.num_timesteps + 1
    baselines = torch.zeros(num_ts, device=device)
    baseline_initialized = torch.zeros(num_ts, device=device, dtype=torch.bool)
    visit_counts = torch.zeros(num_ts, device=device)

    pbar = tqdm(total=num_episodes, desc="RL Training")
    
    episode = 0
    while episode < num_episodes:
        for ref_img in loader:
            if episode >= num_episodes:
                break

            optimizer.zero_grad()

            ref_img = ref_img.to(device)
            y = operator.forward(ref_img)
            noisy_measurement = noiser(y)
            x_start = torch.randn_like(ref_img, device=device).requires_grad_()

            # 2. Sample transitions (batched) using RL mode
            _, trajectory_log_probs, trajectory_rewards_info, trajectory_ts, trajectory_etas, trajectory_entropies = diffusion_model.p_sample_loop(
                model=model,
                x_start=x_start,
                measurement=noisy_measurement,
                measurement_cond_fn=measurement_cond_fn,
                record=False,
                save_root=None,
                rl_mode=True,
                policy_net=policy_net,
                reward_fn=reward_fn,
                max_steps=max_steps,
                ref_img=ref_img,
                operator=operator
            )

            # Update visit counts
            for t in trajectory_ts.view(-1):
                visit_counts[t.item()] += 1

            # trajectory_rewards_info is now a list of (reward_tensor, metrics_dict)
            rewards = torch.cat([item[0].view(-1) for item in trajectory_rewards_info])

            # Aggregate metrics across the trajectory for logging
            metrics_list = [item[1] for item in trajectory_rewards_info]
            avg_metrics = {}
            for k in metrics_list[0].keys():
                avg_metrics[f"metrics/{k}"] = torch.stack([m[k].mean() for m in metrics_list]).mean().item()

            # 3. Calculate Per-Step Advantage
            log_probs = trajectory_log_probs.view(-1)
            ts = trajectory_ts.view(-1)
            etas = trajectory_etas.view(-1)
            entropies = trajectory_entropies.view(-1)

            advantages = torch.zeros_like(rewards)
            for i in range(len(rewards)):
                t_idx = ts[i].item()
                r_i = rewards[i].detach()

                if not baseline_initialized[t_idx]:
                    baselines[t_idx] = r_i
                    baseline_initialized[t_idx] = True
                else:
                    baselines[t_idx] = baseline_decay * baselines[t_idx] + (1 - baseline_decay) * r_i

                advantages[i] = rewards[i] - baselines[t_idx]

            # 4. Loss: -log_prob * advantage - entropy_beta * entropy
            pg_loss = -(log_probs * advantages.detach()).mean()
            entropy_loss = -entropy_beta * entropies.mean()
            loss = pg_loss + entropy_loss

            # 5. Backprop
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)

            optimizer.step()

            # Metrics for WandB
            avg_reward = rewards.mean().item()
            avg_eta = etas.mean().item()

            # Coverage calculation: (Total visits / number of possible non-zero timesteps)
            num_valid_ts = diffusion_model.num_timesteps - 1
            coverage = (visit_counts.sum().item() / num_valid_ts) * 100.0
            unique_visited = baseline_initialized.sum().item()

            wandb.log({
                "episode": episode,
                "loss/total": loss.item(),
                "loss/policy_gradient": pg_loss.item(),
                "loss/entropy": entropy_loss.item(),
                "reward/mean": avg_reward,
                "reward/min": rewards.min().item(),
                "reward/max": rewards.max().item(),
                "eta/mean": avg_eta,
                "eta/std": etas.std().item(),
                "eta/max": etas.max().item(),
                "eta/min": etas.min().item(),
                "baseline/mean": baselines[baseline_initialized].mean().item() if baseline_initialized.any() else 0.0,
                "stats/coverage_percent": coverage,
                "stats/unique_timesteps": unique_visited,
                "stats/advantage_mean": advantages.mean().item(),
                "stats/entropy_mean": entropies.mean().item(),
                **avg_metrics
            })

            # Update tqdm postfix with episode-level summary
            current_metrics = {
                "Ep": episode,
                "Reward": f"{avg_reward:.1f}",
                "PSNR": f"{avg_metrics['metrics/psnr']:.2f}",
                "MSE": f"{avg_metrics['metrics/consistency_mse']:.3f}",
                "Coverage": f"{coverage:.1f}%",
                "Unique Visits": f"{unique_visited}",
            }
            pbar.set_postfix(current_metrics)
            pbar.update(1)
            
            # Print a permanent record of the episode completion to the terminal history
            summary = f"Ep {episode:4d} | R: {avg_reward:7.1f} | PSNR: {avg_metrics['metrics/psnr']:5.2f} | MSE: {avg_metrics['metrics/consistency_mse']:6.4f} | Coverage: {coverage:6.1f}% | Unique Visits: {unique_visited}"
            pbar.write(summary)

            # Periodic saving and uploading to wandb
            if episode > 0 and episode % save_interval == 0 and save_path:
                torch.save(policy_net.state_dict(), save_path)
                wandb.save(save_path)
                logger.info(f"\nPolicy checkpoint saved at episode {episode}")

            episode += 1

    pbar.close()
    # Final save
    if save_path:
        torch.save(policy_net.state_dict(), save_path)
        wandb.save(save_path)
        logger.info(f"Final policy saved and uploaded to WandB: {save_path}")