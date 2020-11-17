import torch
import torch.nn as nn
import torch.nn.functional as F
from procgenac.utils import orthogonal_init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=1024, out_features=feature_dim),
            nn.ReLU(),
        )
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)


class Policy(nn.Module):
    def __init__(self, encoder, feature_dim, num_actions, c1, c2, eps, device):
        super().__init__()
        self.encoder = encoder
        self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=0.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.0)
        self.c1 = c1
        self.c2 = c2
        self.eps = eps
        self.device = device

    def act(self, x):
        with torch.no_grad():
            x = x.to(device=self.device).contiguous()
            dist, value = self.forward(x)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu(), log_prob.cpu(), value.cpu()

    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)

        return dist, value

    def pi_loss(self, log_pi, old_log_pi, advantage):
        # Clipped policy objective
        ratio = torch.exp(log_pi - old_log_pi)
        clipped_ratio = ratio.clamp(min=1.0 - self.eps, max=1.0 + self.eps)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)
        return policy_reward

    def value_loss(self, value, old_reward):
        # Clipped value objective
        # TODO: Decide if this should be reward/return/advantage?
        vf_loss = F.mse_loss(value, old_reward)
        return self.c1 * vf_loss

    def entropy_loss(self, dist):
        # Entropy loss - we think of entropy as a regularization term,
        # maximizing entropy makes sure policy remains stochastic.
        return self.c2 * dist.entropy()
