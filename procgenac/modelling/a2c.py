import torch
import torch.nn as nn
import torch.nn.functional as F
from procgenac.utils import orthogonal_init


class A2C(nn.Module):
    def __init__(self, encoder, feature_dim, num_actions, c1, c2, grad_eps, device):
        super().__init__()
        self.encoder = encoder
        self.actor = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=0.01)
        self.critic = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.0)
        self.c1 = c1
        self.c2 = c2
        self.grad_eps = grad_eps
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

    def _actor_objective(self, log_pi, advantage):
        return log_pi * advantage

    def _value_loss(self, value, future_reward):
        # Clipped value objective
        return self.c1 * F.mse_loss(value, future_reward)

    def _entropy_objective(self, dist):
        # Entropy loss - we think of entropy as a regularization term,
        # maximizing entropy makes sure policy remains stochastic.
        return self.c2 * dist.entropy()

    def criterion(self, batch, policy, value):
        b_obs, b_action, b_log_pi, b_value, b_returns, b_delta, b_advantage = batch

        # Log prob of action given policy
        log_pi = policy.log_prob(b_action)

        # Clipped policy objective
        pi_loss = self._actor_objective(log_pi, b_advantage)

        # Clipped value function objective
        vf_loss = self._value_loss(value, b_returns)

        # Entropy loss
        entropy = self._entropy_objective(policy)

        # Return sum of loss (with appropriate sign)
        return torch.mean(-1 * (pi_loss - vf_loss + entropy))
