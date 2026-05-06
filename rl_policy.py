import torch
import wandb
import torch.nn as nn
from torch.distributions import Normal


class EtaPolicyNetwork(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Input is 2 (normalized timestep + current measurement error
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # State already contains [t_norm, error]
        h = self.net(state)
        mu = self.mu_head(h)
        std = torch.exp(self.log_std_head(h).clamp(min=-20, max=2))
        return mu, std

    def sample_eta(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        eta = torch.sigmoid(action)

        return eta, log_prob


def train_rl_policy(diffusion_model, policy_net, optimizer, num_episodes, reward_fn, data_kwargs, max_steps=50, step_penalty=0.01):
    """
    REINFORCE algorithm with an advantage function (moving average baseline).
    """
    policy_net.train()

    device = next(policy_net.parameters()).device

    num_bins = 20  # 1000 timesteps / 20 bins = 50 steps per zone
    time_baselines = torch.zeros(num_bins, device=device)
    time_visits = torch.zeros(num_bins, device=device)  # Track if we've visited a bin
    global_baseline = 0.0

    # Create an array of 1000 baselines (one for each possible timestep)
    # We initialize them to 0.0.
    time_baselines = torch.zeros(1000, device=next(policy_net.parameters()).device)

    for episode in range(num_episodes):
        optimizer.zero_grad()

        # 1. Sample trajectory using RL mode
        final_img, trajectory_log_probs, trajectory_reward, start_t = diffusion_model.p_sample_loop(
            **data_kwargs,
            rl_mode=True,
            policy_net=policy_net,
            reward_fn=reward_fn,
            max_steps=max_steps,
            step_penalty=step_penalty
        )

        # 2. Update Global Baseline
        if episode == 0:
            global_baseline = trajectory_reward.item()
        else:
            global_baseline = 0.9 * global_baseline + 0.1 * trajectory_reward.item()

        # 3. Determine which "Time Zone" (bin) we started in
        # Assuming num_timesteps is 1000. E.g., start_t=843 -> bin_idx=16
        bin_idx = start_t // (1000 // num_bins)

        # 4. Calculate Advantage
        if time_visits[bin_idx] == 0:
            # First visit to this zone! Fallback to global baseline so we don't get 0 loss
            advantage = trajectory_reward.item() - global_baseline
            time_baselines[bin_idx] = trajectory_reward.item()
        else:
            # We've been here before. Use the zone's specific baseline
            current_baseline = time_baselines[bin_idx].item()
            advantage = trajectory_reward.item() - current_baseline
            # Update the zone's baseline
            time_baselines[bin_idx] = 0.9 * current_baseline + 0.1 * trajectory_reward.item()

        time_visits[bin_idx] += 1

        # 3. REINFORCE Loss: -log_prob * advantage
        loss = -trajectory_log_probs * advantage

        # 5. Backprop
        loss.backward()
        optimizer.step()

        wandb.log({
            "Episode": episode,
            "Reward": trajectory_reward.item(),
            "Advantage": advantage,
            "Loss": loss.item(),
            "Zone Baseline": time_baselines[bin_idx].item(),
            "Start Timestep": start_t
        })

        print(f"Episode {episode} | Reward: {trajectory_reward.item():.4f} | Advantage: {advantage:.4f} | Loss: {loss.item():.4f}")