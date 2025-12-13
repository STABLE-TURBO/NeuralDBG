"""
Tests for the Shape Propagation API
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pytest
import json
from neural.aquarium.api.shape_api import app, initialize_propagator


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_get_shape_propagation_no_propagator(client):
    response = client.get('/api/shape-propagation')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'shape_history' in data
    assert 'errors' in data
    assert len(data['shape_history']) == 0


def test_propagate_simple_model(client):
    model_config = {
        "input_shape": [None, 28, 28, 1],
        "framework": "tensorflow",
        "layers": [
            {
                "type": "Conv2D",
                "params": {
                    "filters": 32,
                    "kernel_size": [3, 3],
                    "padding": "same",
                    "stride": 1
                }
            },
            {
                "type": "Flatten",
                "params": {}
            },
            {
                "type": "Dense",
                "params": {
                    "units": 10
                }
            }
        ]
    }
    
    response = client.post(
        '/api/shape-propagation/propagate',
        data=json.dumps(model_config),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'shape_history' in data
    assert len(data['shape_history']) == 3


def test_reset_propagator(client):
    response = client.post('/api/shape-propagation/reset')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
