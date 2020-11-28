import torch
import torch.nn as nn
import torch.nn.functional as F
from procgenac.utils import orthogonal_init


class A2C(nn.Module):
    def __init__(self, encoder, feature_dim, num_actions, c1, c2, device):
        super().__init__()
        self.encoder = encoder
        self.actor = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=0.01)
        self.critic = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.0)
        self.c1 = c1
        self.c2 = c2
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
        logits = self.actor(x)
        value = self.critic(x).squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value

    def actor_objective(self, log_pi, advantage):
        return log_pi * advantage

    def value_loss(self, value, future_reward):
        # Clipped value objective
        return self.c1 * F.mse_loss(value, future_reward)

    def entropy_objective(self, dist):
        # Entropy loss - we think of entropy as a regularization term,
        # maximizing entropy makes sure policy remains stochastic.
        return self.c2 * dist.entropy()

    def criterion(self, actor_objective, value_loss, entropy_objective):
        return torch.mean(-1 * (actor_objective - value_loss + entropy_objective))
