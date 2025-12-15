from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neural.federated.aggregation import FedAvg

logger = logging.getLogger(__name__)


class FederatedServer:
    def __init__(
        self,
        model: Any = None,
        aggregation_strategy: Optional[Any] = None,
        backend: str = 'tensorflow',
        num_clients: Optional[int] = None,
        min_available_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: Optional[int] = None,
        min_evaluate_clients: Optional[int] = None,
    ):
        self.model = model
        self.backend = backend
        self.aggregation_strategy = aggregation_strategy or FedAvg()
        self.num_clients = num_clients if num_clients is not None else 0
        self.min_available_clients = min_available_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients or min_available_clients
        self.min_evaluate_clients = min_evaluate_clients or min_available_clients
        
        self.current_round = 0
        self.global_weights = self.get_weights()
        self.history = {
            'rounds': [],
            'loss': [],
            'accuracy': [],
            'num_clients': [],
            'communication_cost': [],
            'round_time': [],
        }
    
    def aggregate_weights(self, weights_list: List[List[float]]) -> List[float]:
        if not weights_list:
            return []
        num_items = len(weights_list[0])
        aggregated = [0.0] * num_items
        for i in range(num_items):
            aggregated[i] = float(np.mean([client_weights[i] for client_weights in weights_list]))
        return aggregated
    
    def get_weights(self) -> List[np.ndarray]:
        if self.backend == 'tensorflow':
            try:
                import tensorflow as tf
                return [w.numpy() for w in self.model.get_weights()]
            except ImportError:
                logger.warning("TensorFlow not available")
                return []
        elif self.backend == 'pytorch':
            try:
                import torch
                return [param.detach().cpu().numpy() for param in self.model.parameters()]
            except ImportError:
                logger.warning("PyTorch not available")
                return []
        return []
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        if self.backend == 'tensorflow':
            try:
                import tensorflow as tf
                self.model.set_weights(weights)
            except ImportError:
                logger.warning("TensorFlow not available")
        elif self.backend == 'pytorch':
            try:
                import torch
                state_dict = self.model.state_dict()
                for i, (key, param) in enumerate(state_dict.items()):
                    if i < len(weights):
                        state_dict[key] = torch.tensor(weights[i])
                self.model.load_state_dict(state_dict)
            except ImportError:
                logger.warning("PyTorch not available")
    
    def select_clients(
        self,
        all_clients: List[Any],
        num_clients: int,
        mode: str = 'random',
    ) -> List[Any]:
        if mode == 'random':
            if num_clients > len(all_clients):
                return all_clients
            indices = np.random.choice(len(all_clients), num_clients, replace=False)
            return [all_clients[i] for i in indices]
        elif mode == 'resource_aware':
            scores = [c.compute_capability * c.bandwidth for c in all_clients]
            sorted_indices = np.argsort(scores)[::-1]
            return [all_clients[i] for i in sorted_indices[:num_clients]]
        else:
            return all_clients[:num_clients]
    
    def aggregate_fit(
        self,
        client_results: List[Dict[str, Any]],
        use_secure_aggregation: bool = False,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not client_results:
            return self.global_weights, {}
        
        weights_list = [r['weights'] for r in client_results]
        num_samples_list = [r['num_samples'] for r in client_results]
        
        aggregated_weights = self.aggregation_strategy.aggregate(
            weights_list,
            num_samples_list,
            secure=use_secure_aggregation,
        )
        
        avg_loss = np.mean([r['metrics'].get('loss', 0) for r in client_results])
        avg_accuracy = np.mean([r['metrics'].get('accuracy', 0) for r in client_results])
        total_computation_time = sum([r.get('computation_time', 0) for r in client_results])
        
        metrics = {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'num_clients': len(client_results),
            'total_samples': sum(num_samples_list),
            'computation_time': total_computation_time,
        }
        
        return aggregated_weights, metrics
    
    def aggregate_evaluate(
        self,
        client_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not client_results:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        total_samples = sum([r['num_samples'] for r in client_results])
        
        weighted_loss = sum([
            r['metrics'].get('loss', 0) * r['num_samples']
            for r in client_results
        ]) / total_samples
        
        weighted_accuracy = sum([
            r['metrics'].get('accuracy', 0) * r['num_samples']
            for r in client_results
        ]) / total_samples
        
        return {
            'loss': weighted_loss,
            'accuracy': weighted_accuracy,
            'num_clients': len(client_results),
            'total_samples': total_samples,
        }
    
    def fit_round(
        self,
        clients: List[Any],
        use_secure_aggregation: bool = False,
        proximal_mu: float = 0.0,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        num_fit_clients = max(
            self.min_fit_clients,
            int(len(clients) * self.fraction_fit)
        )
        
        selected_clients = self.select_clients(clients, num_fit_clients)
        
        logger.info(f"Round {self.current_round}: Training on {len(selected_clients)} clients")
        
        client_results = []
        for client in selected_clients:
            try:
                result = client.local_train(
                    global_weights=self.global_weights,
                    proximal_mu=proximal_mu,
                )
                client_results.append(result)
            except Exception as e:
                logger.error(f"Client {client.client_id} training failed: {e}")
        
        if not client_results:
            logger.warning("No successful client results, keeping current weights")
            return self.global_weights, {}
        
        aggregated_weights, metrics = self.aggregate_fit(
            client_results,
            use_secure_aggregation=use_secure_aggregation,
        )
        
        return aggregated_weights, metrics
    
    def evaluate_round(
        self,
        clients: List[Any],
        test_data: Optional[Tuple] = None,
    ) -> Dict[str, Any]:
        num_eval_clients = max(
            self.min_evaluate_clients,
            int(len(clients) * self.fraction_evaluate)
        )
        
        selected_clients = self.select_clients(clients, num_eval_clients)
        
        logger.info(f"Round {self.current_round}: Evaluating on {len(selected_clients)} clients")
        
        for client in selected_clients:
            client.set_weights(self.global_weights)
        
        client_results = []
        for client in selected_clients:
            try:
                metrics = client.evaluate(test_data=test_data)
                client_results.append({
                    'client_id': client.client_id,
                    'num_samples': client.num_samples,
                    'metrics': metrics,
                })
            except Exception as e:
                logger.error(f"Client {client.client_id} evaluation failed: {e}")
        
        if not client_results:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        return self.aggregate_evaluate(client_results)
    
    def train(
        self,
        clients: List[Any],
        num_rounds: int,
        use_secure_aggregation: bool = False,
        proximal_mu: float = 0.0,
        evaluate_every: int = 1,
        test_data: Optional[Tuple] = None,
    ) -> Dict[str, List]:
        logger.info(f"Starting federated training for {num_rounds} rounds")
        
        for round_num in range(num_rounds):
            self.current_round = round_num + 1
            round_start_time = time.time()
            
            aggregated_weights, train_metrics = self.fit_round(
                clients,
                use_secure_aggregation=use_secure_aggregation,
                proximal_mu=proximal_mu,
            )
            
            self.global_weights = aggregated_weights
            self.set_weights(self.global_weights)
            
            round_time = time.time() - round_start_time
            
            self.history['rounds'].append(self.current_round)
            self.history['loss'].append(train_metrics.get('loss', 0.0))
            self.history['accuracy'].append(train_metrics.get('accuracy', 0.0))
            self.history['num_clients'].append(train_metrics.get('num_clients', 0))
            self.history['round_time'].append(round_time)
            
            logger.info(
                f"Round {self.current_round}/{num_rounds} - "
                f"Loss: {train_metrics.get('loss', 0.0):.4f}, "
                f"Accuracy: {train_metrics.get('accuracy', 0.0):.4f}, "
                f"Time: {round_time:.2f}s"
            )
            
            if evaluate_every > 0 and self.current_round % evaluate_every == 0:
                eval_metrics = self.evaluate_round(clients, test_data=test_data)
                logger.info(
                    f"Evaluation - Loss: {eval_metrics.get('loss', 0.0):.4f}, "
                    f"Accuracy: {eval_metrics.get('accuracy', 0.0):.4f}"
                )
        
        logger.info("Federated training completed")
        return self.history
    
    def save_model(self, filepath: str) -> None:
        if self.backend == 'tensorflow':
            try:
                import tensorflow as tf
                self.model.save(filepath)
                logger.info(f"Model saved to {filepath}")
            except ImportError:
                logger.warning("TensorFlow not available")
        elif self.backend == 'pytorch':
            try:
                import torch
                torch.save(self.model.state_dict(), filepath)
                logger.info(f"Model saved to {filepath}")
            except ImportError:
                logger.warning("PyTorch not available")
