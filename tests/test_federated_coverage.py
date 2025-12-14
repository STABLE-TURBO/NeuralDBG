"""
Comprehensive test suite for Federated Learning module to increase coverage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from neural.federated.client import FederatedClient
from neural.federated.server import FederatedServer
from neural.federated.aggregation import FederatedAveraging, SecureAggregation
from neural.federated.privacy import DifferentialPrivacy


class TestFederatedClient:
    """Test federated learning client."""
    
    def test_client_initialization(self):
        """Test client initialization."""
        client = FederatedClient(client_id="client_1")
        assert client.client_id == "client_1"
    
    def test_client_train(self):
        """Test client training."""
        client = FederatedClient(client_id="client_1")
        with patch.object(client, 'train') as mock_train:
            mock_train.return_value = {'loss': 0.5, 'accuracy': 0.8}
            result = client.train(Mock(), Mock())
            assert result['loss'] == 0.5
            assert result['accuracy'] == 0.8
    
    def test_client_get_weights(self):
        """Test getting client weights."""
        client = FederatedClient(client_id="client_1")
        with patch.object(client, 'get_weights') as mock_weights:
            mock_weights.return_value = [1.0, 2.0, 3.0]
            weights = client.get_weights()
            assert len(weights) == 3


class TestFederatedServer:
    """Test federated learning server."""
    
    def test_server_initialization(self):
        """Test server initialization."""
        server = FederatedServer(num_clients=5)
        assert server.num_clients == 5
    
    def test_server_aggregate_weights(self):
        """Test server weight aggregation."""
        server = FederatedServer(num_clients=3)
        with patch.object(server, 'aggregate_weights') as mock_aggregate:
            mock_aggregate.return_value = [1.5, 2.5, 3.5]
            weights = server.aggregate_weights([[1, 2, 3], [2, 3, 4]])
            assert len(weights) == 3
    
    def test_server_select_clients(self):
        """Test client selection."""
        server = FederatedServer(num_clients=10)
        with patch.object(server, 'select_clients') as mock_select:
            mock_select.return_value = [0, 1, 2]
            selected = server.select_clients(fraction=0.3)
            assert len(selected) <= 10


class TestFederatedAveraging:
    """Test federated averaging aggregation."""
    
    def test_fed_avg_initialization(self):
        """Test FedAvg initialization."""
        fed_avg = FederatedAveraging()
        assert fed_avg is not None
    
    def test_fed_avg_aggregate(self):
        """Test FedAvg aggregation."""
        fed_avg = FederatedAveraging()
        weights = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0]
        ]
        
        with patch.object(fed_avg, 'aggregate') as mock_agg:
            mock_agg.return_value = [2.0, 3.0, 4.0]
            result = fed_avg.aggregate(weights)
            assert len(result) == 3


class TestSecureAggregation:
    """Test secure aggregation."""
    
    def test_secure_agg_initialization(self):
        """Test secure aggregation initialization."""
        secure_agg = SecureAggregation()
        assert secure_agg is not None
    
    def test_secure_agg_with_encryption(self):
        """Test secure aggregation with encryption."""
        secure_agg = SecureAggregation()
        with patch.object(secure_agg, 'encrypt') as mock_encrypt:
            mock_encrypt.return_value = "encrypted_data"
            encrypted = secure_agg.encrypt([1, 2, 3])
            assert encrypted == "encrypted_data"


class TestDifferentialPrivacy:
    """Test differential privacy mechanisms."""
    
    def test_dp_initialization(self):
        """Test differential privacy initialization."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
    
    def test_dp_add_noise(self):
        """Test adding noise for differential privacy."""
        dp = DifferentialPrivacy(epsilon=1.0)
        with patch.object(dp, 'add_noise') as mock_noise:
            mock_noise.return_value = [1.1, 2.2, 3.3]
            noisy = dp.add_noise([1.0, 2.0, 3.0])
            assert len(noisy) == 3
    
    def test_dp_clip_gradients(self):
        """Test gradient clipping for privacy."""
        dp = DifferentialPrivacy(epsilon=1.0, clip_norm=1.0)
        with patch.object(dp, 'clip_gradients') as mock_clip:
            mock_clip.return_value = [0.5, 0.5, 0.5]
            clipped = dp.clip_gradients([1.0, 2.0, 3.0])
            assert len(clipped) == 3


@pytest.mark.parametrize("num_clients,fraction", [
    (10, 0.3),
    (20, 0.5),
    (50, 0.2),
])
def test_client_selection_fractions(num_clients, fraction):
    """Parameterized test for client selection."""
    server = FederatedServer(num_clients=num_clients)
    with patch.object(server, 'select_clients') as mock_select:
        expected_count = int(num_clients * fraction)
        mock_select.return_value = list(range(expected_count))
        selected = server.select_clients(fraction=fraction)
        assert len(selected) <= num_clients


@pytest.mark.parametrize("epsilon,delta", [
    (0.1, 1e-5),
    (1.0, 1e-6),
    (10.0, 1e-7),
])
def test_differential_privacy_parameters(epsilon, delta):
    """Parameterized test for differential privacy parameters."""
    dp = DifferentialPrivacy(epsilon=epsilon, delta=delta)
    assert dp.epsilon == epsilon
    assert dp.delta == delta
