from __future__ import annotations

import logging
import math
from abc import ABC
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DifferentialPrivacy(ABC):
    def __init__(self, epsilon: float, delta: float = 1e-5, clip_norm: float | None = None):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm if clip_norm is not None else 0.0
    
    def add_noise(self, weights: List[np.ndarray], sensitivity: float = 1.0) -> List[np.ndarray]:
        return weights
    
    def clip_gradients(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return weights
    
    def compute_noise_scale(self, sensitivity: float) -> float:
        return 0.0


class GaussianDP(DifferentialPrivacy):
    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
    ):
        super().__init__(epsilon, delta)
        self.clip_norm = clip_norm
    
    def compute_noise_scale(self, sensitivity: float) -> float:
        if self.epsilon == 0:
            raise ValueError("Epsilon must be greater than 0")
        
        noise_scale = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        return noise_scale
    
    def clip_gradients(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        total_norm = 0.0
        for w in weights:
            total_norm += np.sum(np.square(w))
        total_norm = np.sqrt(total_norm)
        
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-6))
        
        clipped_weights = [w * clip_coef for w in weights]
        return clipped_weights
    
    def add_noise(self, weights: List[np.ndarray], sensitivity: float) -> List[np.ndarray]:
        clipped_weights = self.clip_gradients(weights)
        
        noise_scale = self.compute_noise_scale(sensitivity)
        
        noisy_weights = []
        for w in clipped_weights:
            noise = np.random.normal(0, noise_scale, w.shape)
            noisy_weights.append(w + noise)
        
        return noisy_weights


class LaplacianDP(DifferentialPrivacy):
    def __init__(
        self,
        epsilon: float,
        delta: float = 0.0,
        clip_norm: float = 1.0,
    ):
        super().__init__(epsilon, delta)
        self.clip_norm = clip_norm
    
    def compute_noise_scale(self, sensitivity: float) -> float:
        if self.epsilon == 0:
            raise ValueError("Epsilon must be greater than 0")
        
        return sensitivity / self.epsilon
    
    def clip_gradients(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        total_norm = 0.0
        for w in weights:
            total_norm += np.sum(np.square(w))
        total_norm = np.sqrt(total_norm)
        
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-6))
        
        clipped_weights = [w * clip_coef for w in weights]
        return clipped_weights
    
    def add_noise(self, weights: List[np.ndarray], sensitivity: float) -> List[np.ndarray]:
        clipped_weights = self.clip_gradients(weights)
        
        noise_scale = self.compute_noise_scale(sensitivity)
        
        noisy_weights = []
        for w in clipped_weights:
            noise = np.random.laplace(0, noise_scale, w.shape)
            noisy_weights.append(w + noise)
        
        return noisy_weights


class PrivacyAccountant:
    def __init__(self, epsilon_total: float, delta_total: float):
        self.epsilon_total = epsilon_total
        self.delta_total = delta_total
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self.privacy_history = []
    
    def compute_rdp_epsilon(
        self,
        sigma: float,
        q: float,
        steps: int,
        orders: Optional[List[float]] = None,
    ) -> float:
        if orders is None:
            orders = [1 + x / 10.0 for x in range(1, 100)]
        
        rdp = []
        for order in orders:
            if order == 1:
                continue
            rdp_at_order = q * q * order / (2 * sigma * sigma)
            rdp.append((order, rdp_at_order * steps))
        
        eps = float('inf')
        for order, rdp_val in rdp:
            eps_at_order = rdp_val + math.log(1 / self.delta_total) / (order - 1)
            eps = min(eps, eps_at_order)
        
        return eps
    
    def compute_privacy_spent(
        self,
        noise_multiplier: float,
        sample_rate: float,
        steps: int,
    ) -> Tuple[float, float]:
        epsilon = self.compute_rdp_epsilon(noise_multiplier, sample_rate, steps)
        return epsilon, self.delta_total
    
    def spend_privacy_budget(
        self,
        epsilon: float,
        delta: float,
        operation: str = "unknown",
    ) -> bool:
        if self.epsilon_spent + epsilon > self.epsilon_total:
            logger.warning(
                f"Privacy budget exceeded: {self.epsilon_spent + epsilon} > {self.epsilon_total}"
            )
            return False
        
        if self.delta_spent + delta > self.delta_total:
            logger.warning(
                f"Delta budget exceeded: {self.delta_spent + delta} > {self.delta_total}"
            )
            return False
        
        self.epsilon_spent += epsilon
        self.delta_spent += delta
        
        self.privacy_history.append({
            'operation': operation,
            'epsilon': epsilon,
            'delta': delta,
            'epsilon_spent': self.epsilon_spent,
            'delta_spent': self.delta_spent,
        })
        
        logger.info(
            f"Privacy spent - Operation: {operation}, "
            f"ε: {epsilon:.4f}, δ: {delta:.4e}, "
            f"Total ε: {self.epsilon_spent:.4f}/{self.epsilon_total:.4f}"
        )
        
        return True
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        return (
            self.epsilon_total - self.epsilon_spent,
            self.delta_total - self.delta_spent,
        )
    
    def reset(self):
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self.privacy_history = []


class LocalDP:
    def __init__(
        self,
        epsilon: float,
        mechanism: str = 'randomized_response',
    ):
        self.epsilon = epsilon
        self.mechanism = mechanism
    
    def randomized_response(self, value: float, domain_size: int = 2) -> float:
        p = math.exp(self.epsilon) / (math.exp(self.epsilon) + domain_size - 1)
        
        if np.random.rand() < p:
            return value
        else:
            return np.random.choice([v for v in range(domain_size) if v != value])
    
    def laplace_mechanism(self, value: float, sensitivity: float) -> float:
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def apply_local_dp(
        self,
        weights: List[np.ndarray],
        sensitivity: float = 1.0,
    ) -> List[np.ndarray]:
        if self.mechanism == 'laplace':
            noisy_weights = []
            for w in weights:
                scale = sensitivity / self.epsilon
                noise = np.random.laplace(0, scale, w.shape)
                noisy_weights.append(w + noise)
            return noisy_weights
        else:
            return weights


class ShuffleDP:
    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
    ):
        self.epsilon = epsilon
        self.delta = delta
    
    def shuffle_and_aggregate(
        self,
        client_weights: List[List[np.ndarray]],
        num_samples_list: List[int],
    ) -> List[np.ndarray]:
        num_clients = len(client_weights)
        shuffled_indices = np.random.permutation(num_clients)
        
        shuffled_weights = [client_weights[i] for i in shuffled_indices]
        shuffled_samples = [num_samples_list[i] for i in shuffled_indices]
        
        total_samples = sum(shuffled_samples)
        num_layers = len(shuffled_weights[0])
        
        aggregated = []
        for layer_idx in range(num_layers):
            weighted_sum = np.zeros_like(shuffled_weights[0][layer_idx])
            for weights, samples in zip(shuffled_weights, shuffled_samples):
                weighted_sum += (samples / total_samples) * weights[layer_idx]
            
            noise_scale = self.compute_shuffle_noise_scale(num_clients)
            noise = np.random.normal(0, noise_scale, weighted_sum.shape)
            aggregated.append(weighted_sum + noise)
        
        return aggregated
    
    def compute_shuffle_noise_scale(self, num_clients: int) -> float:
        amplification = math.sqrt(num_clients)
        effective_epsilon = self.epsilon * amplification
        
        if effective_epsilon == 0:
            return float('inf')
        
        return 1.0 / effective_epsilon


class AdaptivePrivacy:
    def __init__(
        self,
        epsilon_total: float,
        delta_total: float,
        num_rounds: int,
    ):
        self.epsilon_total = epsilon_total
        self.delta_total = delta_total
        self.num_rounds = num_rounds
        self.current_round = 0
        self.accountant = PrivacyAccountant(epsilon_total, delta_total)
    
    def get_round_privacy_budget(self) -> Tuple[float, float]:
        remaining_rounds = self.num_rounds - self.current_round
        
        if remaining_rounds <= 0:
            return 0.0, 0.0
        
        epsilon_remaining, delta_remaining = self.accountant.get_remaining_budget()
        
        epsilon_per_round = epsilon_remaining / remaining_rounds
        delta_per_round = delta_remaining / remaining_rounds
        
        return epsilon_per_round, delta_per_round
    
    def update_round(self, epsilon_spent: float, delta_spent: float):
        self.accountant.spend_privacy_budget(
            epsilon_spent,
            delta_spent,
            operation=f"round_{self.current_round}",
        )
        self.current_round += 1
    
    def get_noise_multiplier(
        self,
        sample_rate: float,
        target_epsilon: float,
    ) -> float:
        left, right = 0.1, 100.0
        tolerance = 0.01
        
        while right - left > tolerance:
            mid = (left + right) / 2
            epsilon, _ = self.accountant.compute_privacy_spent(mid, sample_rate, 1)
            
            if epsilon < target_epsilon:
                right = mid
            else:
                left = mid
        
        return (left + right) / 2
