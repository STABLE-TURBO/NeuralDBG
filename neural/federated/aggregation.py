from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(
        self,
        weights_list: List[List[np.ndarray]],
        num_samples_list: List[int],
        **kwargs
    ) -> List[np.ndarray]:
        pass


class FedAvg(AggregationStrategy):
    def aggregate(
        self,
        weights_list: List[List[np.ndarray]],
        num_samples_list: List[int],
        **kwargs
    ) -> List[np.ndarray]:
        total_samples = sum(num_samples_list)
        
        if total_samples == 0:
            logger.warning("Total samples is 0, returning first client weights")
            return weights_list[0]
        
        num_layers = len(weights_list[0])
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            weighted_sum = np.zeros_like(weights_list[0][layer_idx])
            
            for client_weights, num_samples in zip(weights_list, num_samples_list):
                weight = num_samples / total_samples
                weighted_sum += weight * client_weights[layer_idx]
            
            aggregated_weights.append(weighted_sum)
        
        return aggregated_weights


class FedProx(AggregationStrategy):
    def __init__(self, mu: float = 0.01):
        self.mu = mu
    
    def aggregate(
        self,
        weights_list: List[List[np.ndarray]],
        num_samples_list: List[int],
        **kwargs
    ) -> List[np.ndarray]:
        return FedAvg().aggregate(weights_list, num_samples_list, **kwargs)


class FedAdam(AggregationStrategy):
    def __init__(
        self,
        learning_rate: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def aggregate(
        self,
        weights_list: List[List[np.ndarray]],
        num_samples_list: List[int],
        **kwargs
    ) -> List[np.ndarray]:
        avg_weights = FedAvg().aggregate(weights_list, num_samples_list, **kwargs)
        
        if self.m is None:
            self.m = [np.zeros_like(w) for w in avg_weights]
            self.v = [np.zeros_like(w) for w in avg_weights]
        
        self.t += 1
        
        updated_weights = []
        for i, w in enumerate(avg_weights):
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * w
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * np.square(w)
            
            m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta_2 ** self.t)
            
            updated_w = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_weights.append(updated_w)
        
        return updated_weights


class FedYogi(AggregationStrategy):
    def __init__(
        self,
        learning_rate: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def aggregate(
        self,
        weights_list: List[List[np.ndarray]],
        num_samples_list: List[int],
        **kwargs
    ) -> List[np.ndarray]:
        avg_weights = FedAvg().aggregate(weights_list, num_samples_list, **kwargs)
        
        if self.m is None:
            self.m = [np.zeros_like(w) for w in avg_weights]
            self.v = [np.zeros_like(w) for w in avg_weights]
        
        self.t += 1
        
        updated_weights = []
        for i, w in enumerate(avg_weights):
            gradient = w - self.m[i]
            
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * gradient
            
            self.v[i] = self.v[i] - (1 - self.beta_2) * np.square(gradient) * np.sign(
                self.v[i] - np.square(gradient)
            )
            
            m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta_2 ** self.t)
            
            updated_w = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_weights.append(updated_w)
        
        return updated_weights


class SecureAggregator:
    def __init__(self, threshold: int = 2):
        self.threshold = threshold
    
    def encrypt(self, data: Any) -> Any:
        return data
    
    def generate_masks(
        self,
        num_clients: int,
        weight_shapes: List[tuple],
        seed: int = 42,
    ) -> List[List[np.ndarray]]:
        np.random.seed(seed)
        masks = []
        
        for client_idx in range(num_clients):
            client_masks = []
            for shape in weight_shapes:
                mask = np.random.randn(*shape) * 0.01
                client_masks.append(mask)
            masks.append(client_masks)
        
        return masks
    
    def add_noise(
        self,
        weights: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        noisy_weights = []
        for w, mask in zip(weights, masks):
            noisy_weights.append(w + mask)
        return noisy_weights
    
    def remove_noise(
        self,
        aggregated_weights: List[np.ndarray],
        all_masks: List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        num_layers = len(aggregated_weights)
        denoised_weights = []
        
        for layer_idx in range(num_layers):
            total_mask = np.zeros_like(aggregated_weights[layer_idx])
            for masks in all_masks:
                total_mask += masks[layer_idx]
            
            denoised_w = aggregated_weights[layer_idx] - total_mask
            denoised_weights.append(denoised_w)
        
        return denoised_weights
    
    def secure_aggregate(
        self,
        weights_list: List[List[np.ndarray]],
        num_samples_list: List[int],
        aggregation_strategy: Optional[AggregationStrategy] = None,
    ) -> List[np.ndarray]:
        if len(weights_list) < self.threshold:
            logger.warning(
                f"Number of clients ({len(weights_list)}) below threshold ({self.threshold}), "
                f"skipping secure aggregation"
            )
            strategy = aggregation_strategy or FedAvg()
            return strategy.aggregate(weights_list, num_samples_list)
        
        weight_shapes = [w.shape for w in weights_list[0]]
        masks = self.generate_masks(len(weights_list), weight_shapes)
        
        noisy_weights_list = []
        for client_weights, client_masks in zip(weights_list, masks):
            noisy_weights = self.add_noise(client_weights, client_masks)
            noisy_weights_list.append(noisy_weights)
        
        strategy = aggregation_strategy or FedAvg()
        noisy_aggregated = strategy.aggregate(noisy_weights_list, num_samples_list)
        
        denoised_weights = self.remove_noise(noisy_aggregated, masks)
        
        return denoised_weights
    
    def compute_verification_hash(self, weights: List[np.ndarray]) -> str:
        concatenated = b''.join([w.tobytes() for w in weights])
        return hashlib.sha256(concatenated).hexdigest()
    
    def verify_aggregation(
        self,
        client_hashes: List[str],
        aggregated_hash: str,
    ) -> bool:
        expected_hash = hashlib.sha256(''.join(sorted(client_hashes)).encode()).hexdigest()
        return expected_hash == aggregated_hash


class FedMA(AggregationStrategy):
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
    
    def _compute_layer_similarity(
        self,
        layer1: np.ndarray,
        layer2: np.ndarray,
    ) -> float:
        if layer1.shape != layer2.shape:
            return 0.0
        
        norm1 = np.linalg.norm(layer1)
        norm2 = np.linalg.norm(layer2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.sum(layer1 * layer2) / (norm1 * norm2)
        return float(cosine_sim)
    
    def aggregate(
        self,
        weights_list: List[List[np.ndarray]],
        num_samples_list: List[int],
        **kwargs
    ) -> List[np.ndarray]:
        num_layers = len(weights_list[0])
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            layer_weights = [w[layer_idx] for w in weights_list]
            
            similarity_matrix = np.zeros((len(weights_list), len(weights_list)))
            for i in range(len(weights_list)):
                for j in range(i + 1, len(weights_list)):
                    sim = self._compute_layer_similarity(layer_weights[i], layer_weights[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
            
            weights = np.exp(similarity_matrix.sum(axis=1) / self.sigma)
            weights = weights / weights.sum()
            
            weighted_sum = np.zeros_like(layer_weights[0])
            for w, layer_w in zip(weights, layer_weights):
                weighted_sum += w * layer_w
            
            aggregated_weights.append(weighted_sum)
        
        return aggregated_weights


class AdaptiveAggregation(AggregationStrategy):
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.previous_weights = None
    
    def aggregate(
        self,
        weights_list: List[List[np.ndarray]],
        num_samples_list: List[int],
        **kwargs
    ) -> List[np.ndarray]:
        current_weights = FedAvg().aggregate(weights_list, num_samples_list, **kwargs)
        
        if self.previous_weights is None:
            self.previous_weights = current_weights
            return current_weights
        
        aggregated_weights = []
        for curr_w, prev_w in zip(current_weights, self.previous_weights):
            adapted_w = self.alpha * curr_w + (1 - self.alpha) * prev_w
            aggregated_weights.append(adapted_w)
        
        self.previous_weights = aggregated_weights
        return aggregated_weights

# Aliases expected by tests
FederatedAveraging = FedAvg
SecureAggregation = SecureAggregator
