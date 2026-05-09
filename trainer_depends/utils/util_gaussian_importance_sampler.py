import math

import torch
import torch.nn as nn


class NormalizedGaussianSampler(nn.Module):
    """Sample bounded normalized 4D coordinates around normalized centers."""

    def __init__(self, norm_std_devs, device="cuda"):
        super().__init__()
        self.device = device
        self.std_devs = torch.tensor(norm_std_devs, device=device).view(1, 1, -1)
        self.lower = torch.tensor([-1.0] * 4, device=device).view(1, 1, -1)
        self.upper = torch.tensor([1.0] * 4, device=device).view(1, 1, -1)
        self.periods = torch.tensor([2.0] * 4, device=device).view(1, 1, -1)
        self.circular_mask = torch.tensor([False, False, True, False], device=device).view(1, 1, -1)
        self.sqrt_2 = math.sqrt(2)
        self.epsilon = 1e-6

    def _phi(self, x):
        return 0.5 * (1 + torch.erf(x / self.sqrt_2))

    def _phi_inv(self, p):
        return self.sqrt_2 * torch.erfinv(2 * p - 1)

    def sample_importance(self, centers_linear, num_samples=16, include_center=False):
        batch_size = centers_linear.shape[0]
        centers_expanded = centers_linear.unsqueeze(1)

        z_min = (self.lower - centers_expanded) / (self.std_devs + 1e-8)
        z_max = (self.upper - centers_expanded) / (self.std_devs + 1e-8)
        p_min = self._phi(z_min)
        p_max = self._phi(z_max)

        zeros = torch.zeros_like(p_min)
        ones = torch.ones_like(p_max)
        eff_p_min = torch.where(self.circular_mask, zeros, p_min)
        eff_p_max = torch.where(self.circular_mask, ones, p_max)
        eff_p_max = torch.maximum(eff_p_max, eff_p_min + self.epsilon)

        rand_u = torch.rand(batch_size, num_samples, 4, device=self.device)
        p_sample = eff_p_min + rand_u * (eff_p_max - eff_p_min)
        p_sample = torch.clamp(p_sample, self.epsilon, 1.0 - self.epsilon)

        z_sample = self._phi_inv(p_sample)
        sampled_coords = centers_expanded + z_sample * self.std_devs
        final_coords = torch.where(
            self.circular_mask,
            torch.remainder(sampled_coords - self.lower, self.periods) + self.lower,
            sampled_coords,
        )

        if include_center:
            return torch.cat([centers_expanded, final_coords], dim=1)
        return final_coords


class GaussianSampler(nn.Module):
    """Bounded Gaussian sampler for physical coordinate spaces."""

    def __init__(self, std_devs, limits, circular_dims=None, device="cuda"):
        super().__init__()
        self.device = device
        self.std_devs = torch.tensor(std_devs, device=device, dtype=torch.float32).view(1, 1, -1)
        self.lower_bounds = torch.tensor([lim[0] for lim in limits], device=device, dtype=torch.float32).view(1, 1, -1)
        self.upper_bounds = torch.tensor([lim[1] for lim in limits], device=device, dtype=torch.float32).view(1, 1, -1)
        self.periods = self.upper_bounds - self.lower_bounds
        self.circular_mask = torch.zeros(len(std_devs), device=device, dtype=torch.bool).view(1, 1, -1)
        if circular_dims:
            self.circular_mask[:, :, circular_dims] = True
        self.sqrt_2 = math.sqrt(2)
        self.epsilon = 1e-6

    def _phi(self, x):
        return 0.5 * (1 + torch.erf(x / self.sqrt_2))

    def _phi_inv(self, p):
        return self.sqrt_2 * torch.erfinv(2 * p - 1)

    def sample_importance(self, centers, num_samples=16):
        batch_size, dim = centers.shape
        centers_expanded = centers.unsqueeze(1)
        z_min = (self.lower_bounds - centers_expanded) / (self.std_devs + 1e-8)
        z_max = (self.upper_bounds - centers_expanded) / (self.std_devs + 1e-8)
        p_min = self._phi(z_min)
        p_max = self._phi(z_max)

        zeros = torch.zeros_like(p_min)
        ones = torch.ones_like(p_max)
        effective_p_min = torch.where(self.circular_mask, zeros, p_min)
        effective_p_max = torch.where(self.circular_mask, ones, p_max)
        effective_p_max = torch.maximum(effective_p_max, effective_p_min + self.epsilon)

        rand_u = torch.rand(batch_size, num_samples, dim, device=self.device)
        p_sample = effective_p_min + rand_u * (effective_p_max - effective_p_min)
        p_sample = torch.clamp(p_sample, self.epsilon, 1.0 - self.epsilon)
        sampled_coords = centers_expanded + self._phi_inv(p_sample) * self.std_devs
        return torch.where(
            self.circular_mask,
            torch.remainder(sampled_coords - self.lower_bounds, self.periods) + self.lower_bounds,
            sampled_coords,
        )

    def compute_weights(self, centers, query_points):
        centers_expanded = centers.unsqueeze(1)
        delta = query_points - centers_expanded
        delta_abs = torch.abs(delta)
        delta_cyclic = torch.min(delta_abs, self.periods - delta_abs)
        effective_delta = torch.where(self.circular_mask, delta_cyclic, delta)
        normalized_delta = effective_delta / (self.std_devs + 1e-8)
        dist_sq = torch.sum(normalized_delta ** 2, dim=-1)
        return torch.exp(-0.5 * dist_sq)
