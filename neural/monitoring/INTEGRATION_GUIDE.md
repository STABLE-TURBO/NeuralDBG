# Integration Guide

This guide shows how to integrate Neural monitoring into your ML pipelines and applications.

## Table of Contents

- [Basic Integration](#basic-integration)
- [Flask/FastAPI Integration](#flaskfastapi-integration)
- [Batch Processing Integration](#batch-processing-integration)
- [Alert Configuration](#alert-configuration)

## Basic Integration

### Step 1: Install Dependencies

```bash
pip install neural-dsl[monitoring]
# For dashboard support
pip install neural-dsl[visualization]
```

### Step 2: Initialize Monitor

```python
from neural.monitoring import ModelMonitor

monitor = ModelMonitor(
    model_name="my-model",
    model_version="1.0",
    storage_path="/path/to/monitoring/data",
    enable_alerting=True
)
```

### Step 3: Set Reference Data

```python
import numpy as np

# Load your training/validation data
reference_data = np.load("reference_data.npy")
reference_predictions = np.load("reference_predictions.npy")
reference_performance = {
    'accuracy': 0.95,
    'f1': 0.94
}

monitor.set_reference_data(
    data=reference_data,
    predictions=reference_predictions,
    performance=reference_performance
)
```

### Step 4: Monitor Predictions

```python
# In your prediction endpoint
def predict(input_data):
    # Make prediction
    prediction = model.predict(input_data)
    
    # Track prediction
    monitor.record_prediction()
    
    return prediction
```

## Flask/FastAPI Integration

### Flask Example

```python
from flask import Flask, request, jsonify
from neural.monitoring import ModelMonitor
import time

app = Flask(__name__)

# Initialize monitor
monitor = ModelMonitor(
    model_name="flask-model",
    model_version="1.0",
    enable_alerting=True
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input
        data = request.get_json()
        
        # Make prediction
        prediction = model.predict(data['features'])
        
        # Track prediction
        monitor.record_prediction()
        
        return jsonify({
            'prediction': prediction
        })
        
    except Exception as e:
        monitor.record_error(type(e).__name__)
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    report = monitor.generate_health_report()
    return jsonify(report)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neural.monitoring import ModelMonitor

app = FastAPI()

# Initialize monitor
monitor = ModelMonitor(
    model_name="fastapi-model",
    model_version="1.0",
    enable_alerting=True
)

class PredictionRequest(BaseModel):
    features: dict

@app.post('/predict')
async def predict(request: PredictionRequest):
    try:
        # Make prediction
        prediction = model.predict(request.features)
        
        # Track prediction
        monitor.record_prediction()
        
        return {'prediction': prediction}
        
    except Exception as e:
        monitor.record_error(type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health():
    report = monitor.generate_health_report()
    return report
```

## Batch Processing Integration

```python
from neural.monitoring import ModelMonitor
import numpy as np

class BatchPredictor:
    def __init__(self, model, monitor: ModelMonitor):
        self.model = model
        self.monitor = monitor
        
    def predict_batch(self, batch_data):
        """Process a batch of predictions with monitoring."""
        results = []
        
        for data in batch_data:
            try:
                # Make prediction
                prediction = self.model.predict(data)
                self.monitor.record_prediction()
                results.append(prediction)
                
            except Exception as e:
                self.monitor.record_error(type(e).__name__)
                results.append(None)
        
        # Check drift periodically
        if len(batch_data) >= 100:
            drift_report = self.monitor.check_drift(
                data=np.array(batch_data)
            )
            if drift_report['is_drifting']:
                print(f"Warning: Drift detected with severity {drift_report['drift_severity']}")
        
        return results

# Usage
monitor = ModelMonitor(model_name="batch-model", model_version="1.0")
predictor = BatchPredictor(model, monitor)

batch_data = load_batch_data()
predictions = predictor.predict_batch(batch_data)
```

## Alert Configuration

### Slack Integration

1. Create incoming webhook in Slack
2. Configure in monitor:

```python
alert_config = {
    'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
}

monitor = ModelMonitor(
    model_name="my-model",
    model_version="1.0",
    alert_config=alert_config,
    enable_alerting=True
)
```

### Email Integration

```python
alert_config = {
    'email_config': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your-email@gmail.com',
        'password': 'your-app-password',
        'from_addr': 'your-email@gmail.com',
        'to_addrs': ['team@example.com', 'oncall@example.com']
    }
}
```

### Custom Alert Rules

```python
from neural.monitoring import AlertRule, AlertSeverity, AlertChannel

# Create custom rule
custom_rule = AlertRule(
    name="Custom Metric Alert",
    condition=lambda data: data.get('custom_metric', 0) > threshold,
    severity=AlertSeverity.WARNING,
    channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
    cooldown_seconds=300,
    message_template="Custom metric exceeded: {custom_metric}"
)

monitor.alert_manager.add_rule(custom_rule)
```

## Environment Variables

Set these environment variables for configuration:

```bash
# Storage
export NEURAL_MONITORING_PATH=/data/monitoring

# Alerts
export NEURAL_SLACK_WEBHOOK=https://hooks.slack.com/services/...
export NEURAL_ALERT_EMAIL=alerts@example.com
```

## Best Practices

1. **Initialize Early**: Set up monitoring before deploying to production
2. **Set Reference Data**: Always establish baseline metrics
3. **Monitor Gradually**: Start with basic logging, add features incrementally
4. **Tune Thresholds**: Adjust based on your model's behavior
5. **Regular Reviews**: Check monitoring dashboards regularly
6. **Test Alerting**: Verify alert delivery before relying on it
7. **Monitor the Monitor**: Ensure monitoring system is healthy

## Troubleshooting

### Alerts Not Sending

1. Test webhook URLs manually
2. Verify email credentials
3. Check network connectivity
4. Review cooldown periods

### Dashboard Not Loading

1. Check Dash dependencies: `pip install dash plotly`
2. Verify storage path exists
3. Check port availability

## Support

For issues and questions:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- Documentation: See README.md
