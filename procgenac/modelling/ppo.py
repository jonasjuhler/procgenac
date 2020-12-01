import torch
import torch.nn as nn
import torch.nn.functional as F
from procgenac.utils import orthogonal_init


class PPO(nn.Module):
    def __init__(self, encoder, feature_dim, num_actions, c1, c2, eps, grad_eps, device):
        super().__init__()
        self.name = "PPO"
        self.encoder = encoder
        self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=0.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.0)
        self.c1 = c1
        self.c2 = c2
        self.eps = eps
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
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)

        return dist, value

    def _pi_loss(self, log_pi, old_log_pi, advantage):
        # Clipped policy objective
        ratio = torch.exp(log_pi - old_log_pi)
        clipped_ratio = ratio.clamp(min=1.0 - self.eps, max=1.0 + self.eps)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)
        return policy_reward

    def _value_loss(self, value, future_reward):
        # Clipped value objective
        vf_loss = F.mse_loss(value, future_reward)
        return vf_loss

    def _entropy_loss(self, dist):
        # Entropy loss - we think of entropy as a regularization term,
        # maximizing entropy makes sure policy remains stochastic.
        return dist.entropy()

    def criterion(self, batch, policy, value):
        b_obs, b_action, b_log_pi, b_value, b_returns, b_delta, b_advantage = batch

        # Calculate log prob of action given policy
        log_pi = policy.log_prob(b_action)

        # Clipped policy objective
        pi_loss = self._pi_loss(log_pi, b_log_pi, b_advantage)

        # Clipped value function objective
        vf_loss = self._value_loss(value, b_returns)

        # Entropy loss
        entropy = self._entropy_loss(policy)

        # Return sum of weighted losses
        return torch.mean(-1 * (pi_loss - self.c1 * vf_loss + self.c2 * entropy))
