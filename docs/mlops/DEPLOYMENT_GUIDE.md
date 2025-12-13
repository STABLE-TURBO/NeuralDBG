# Deployment Guide

Complete guide for deploying models using Neural DSL MLOps.

## Table of Contents

1. [Deployment Strategies](#deployment-strategies)
2. [Shadow Deployment](#shadow-deployment)
3. [Canary Deployment](#canary-deployment)
4. [Blue-Green Deployment](#blue-green-deployment)
5. [Automated Rollback](#automated-rollback)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Best Practices](#best-practices)

## Deployment Strategies

Neural DSL supports multiple deployment strategies for different use cases.

### Strategy Overview

| Strategy | Risk | Complexity | Use Case |
|----------|------|------------|----------|
| Direct | High | Low | Development/testing |
| Shadow | Low | Medium | Pre-production validation |
| Canary | Medium | Medium | Gradual production rollout |
| Blue-Green | Low | High | Zero-downtime deployment |
| Rolling | Medium | Medium | Large-scale updates |

## Shadow Deployment

Shadow deployment runs the new model alongside production without affecting user responses.

### When to Use

- Pre-production validation
- Performance comparison
- Risk-free testing with production traffic
- Validating new model versions

### Implementation

```python
from neural.mlops.deployment import DeploymentManager

manager = DeploymentManager("./deployments")

# Create shadow deployment
shadow = manager.shadow_deploy(
    primary_model="fraud_detector:v1.0.0",
    shadow_model="fraud_detector:v2.0.0",
    traffic_percentage=100.0  # Copy 100% of production traffic
)

print(f"Shadow deployment created: {shadow.shadow_id}")
```

### Recording Comparisons

```python
# In your prediction service
import time

# Get predictions from both models
start_primary = time.time()
primary_result = primary_model.predict(features)
primary_latency = time.time() - start_primary

start_shadow = time.time()
shadow_result = shadow_model.predict(features)
shadow_latency = time.time() - start_shadow

# Check if predictions agree
agreement = (primary_result == shadow_result)

# Record comparison
manager.record_shadow_comparison(
    shadow_id=shadow.shadow_id,
    primary_result=primary_result,
    shadow_result=shadow_result,
    primary_latency=primary_latency,
    shadow_latency=shadow_latency,
    agreement=agreement
)
```

### Analyzing Results

```python
# After collecting sufficient data
comparison = manager.compare_shadow_deployment(shadow.shadow_id)

print(f"Total requests: {comparison['total_requests']}")
print(f"Agreement rate: {comparison['agreement_rate']:.2%}")
print(f"\nPrimary model latency:")
print(f"  P50: {comparison['primary_latency']['p50']:.3f}s")
print(f"  P95: {comparison['primary_latency']['p95']:.3f}s")
print(f"  P99: {comparison['primary_latency']['p99']:.3f}s")
print(f"\nShadow model latency:")
print(f"  P50: {comparison['shadow_latency']['p50']:.3f}s")
print(f"  P95: {comparison['shadow_latency']['p95']:.3f}s")
print(f"  P99: {comparison['shadow_latency']['p99']:.3f}s")

# Decision criteria
if comparison['agreement_rate'] > 0.95 and \
   comparison['shadow_latency']['p95'] < comparison['primary_latency']['p95'] * 1.2:
    print("\n✓ Shadow model is ready for production!")
else:
    print("\n✗ Shadow model needs improvement")
```

## Canary Deployment

Canary deployment gradually shifts traffic to the new model.

### When to Use

- Production rollouts
- Risk mitigation
- Performance validation under load
- Gradual traffic shift

### Implementation

```python
from neural.mlops.deployment import (
    DeploymentManager,
    DeploymentStrategy,
    PerformanceMetrics,
    RollbackConfig
)

manager = DeploymentManager("./deployments")

# Get baseline metrics from current production model
baseline_metrics = PerformanceMetrics(
    latency_p50=0.035,
    latency_p95=0.050,
    latency_p99=0.075,
    error_rate=0.001,
    requests_per_second=1000.0
)

# Create canary deployment
deployment = manager.create_deployment(
    model_name="fraud_detector",
    model_version="v2.0.0",
    strategy=DeploymentStrategy.CANARY,
    created_by="devops@company.com",
    environment="production",
    baseline_metrics=baseline_metrics,
    rollback_config=RollbackConfig(
        enabled=True,
        error_rate_threshold=0.01,  # 1% error rate
        latency_threshold_multiplier=1.5,  # 50% latency increase
        min_requests_before_check=1000,
        check_interval_seconds=60
    ),
    metadata={
        "initial_traffic": 0.05,  # Start with 5%
        "target_traffic": 1.0,
        "increment": 0.05,
        "increment_interval_hours": 24
    }
)

# Start deployment
manager.start_deployment(deployment.deployment_id)
```

### Traffic Shift Schedule

```python
# Recommended canary rollout schedule
canary_schedule = [
    {"day": 1, "traffic": 0.05, "action": "Monitor closely"},
    {"day": 2, "traffic": 0.10, "action": "Check metrics"},
    {"day": 3, "traffic": 0.25, "action": "Validate performance"},
    {"day": 5, "traffic": 0.50, "action": "Extended monitoring"},
    {"day": 7, "traffic": 1.00, "action": "Complete rollout"}
]

for stage in canary_schedule:
    print(f"Day {stage['day']}: {stage['traffic']:.0%} traffic - {stage['action']}")
```

### Monitoring Canary

```python
# Collect metrics during canary
current_metrics = PerformanceMetrics(
    latency_p50=0.038,
    latency_p95=0.052,
    latency_p99=0.078,
    error_rate=0.002,
    requests_per_second=50.0  # 5% of traffic
)

# Update deployment metrics
manager.update_metrics(deployment.deployment_id, current_metrics)

# Check if rollback needed
needs_rollback, reasons = manager.check_deployment_health(
    deployment.deployment_id,
    request_count=5000
)

if needs_rollback:
    print("⚠️ ROLLBACK TRIGGERED!")
    for reason in reasons:
        print(f"  - {reason}")
else:
    print("✓ Canary is healthy, proceeding with rollout")
```

## Blue-Green Deployment

Blue-green deployment maintains two identical production environments.

### When to Use

- Zero-downtime deployments
- Instant rollback capability
- Database migration testing
- High-availability requirements

### Implementation

```python
from neural.mlops.deployment import DeploymentManager, DeploymentStrategy

manager = DeploymentManager("./deployments")

# Deploy to green environment (blue is currently active)
green_deployment = manager.create_deployment(
    model_name="fraud_detector",
    model_version="v2.0.0",
    strategy=DeploymentStrategy.BLUE_GREEN,
    created_by="devops@company.com",
    environment="production-green",
    metadata={
        "active_environment": "blue",
        "target_environment": "green"
    }
)

# Start green deployment
manager.start_deployment(green_deployment.deployment_id)

# Test green environment
# ... run smoke tests ...

# Switch traffic from blue to green
# (Implementation specific to your infrastructure)

# Mark as complete
manager.complete_deployment(green_deployment.deployment_id)
```

## Automated Rollback

Automated rollback protects production from degraded performance.

### Rollback Configuration

```python
from neural.mlops.deployment import RollbackConfig

# Conservative rollback (quick to rollback)
conservative_config = RollbackConfig(
    enabled=True,
    error_rate_threshold=0.005,  # 0.5%
    latency_threshold_multiplier=1.2,  # 20% increase
    min_requests_before_check=500,
    check_interval_seconds=30
)

# Balanced rollback
balanced_config = RollbackConfig(
    enabled=True,
    error_rate_threshold=0.01,  # 1%
    latency_threshold_multiplier=1.5,  # 50% increase
    min_requests_before_check=1000,
    check_interval_seconds=60
)

# Aggressive rollback (slow to rollback)
aggressive_config = RollbackConfig(
    enabled=True,
    error_rate_threshold=0.02,  # 2%
    latency_threshold_multiplier=2.0,  # 100% increase
    min_requests_before_check=5000,
    check_interval_seconds=300
)
```

### Rollback Triggers

The system automatically rolls back when:

1. **Error Rate**: Exceeds configured threshold
2. **Latency Degradation**: P95 or P99 latency increases beyond multiplier
3. **Error Rate Increase**: Significant increase from baseline
4. **Custom Metrics**: Optional custom metric thresholds

### Manual Rollback

```python
# Manual rollback if needed
manager.rollback_deployment(
    deployment_id=deployment.deployment_id,
    reason="Manual rollback due to business metric degradation",
    triggered_by="sre-engineer@company.com"
)
```

## Monitoring and Observability

### Health Checks

```python
from neural.mlops.deployment import HealthStatus, PerformanceMetrics

# Add health check
manager.add_health_check(
    deployment_id=deployment.deployment_id,
    status=HealthStatus.HEALTHY,
    metrics=PerformanceMetrics(
        latency_p95=0.052,
        error_rate=0.001
    ),
    message="All systems operational"
)

# Get deployment with health history
deployment = manager.get_deployment(deployment.deployment_id)
print(f"Health checks: {len(deployment.health_checks)}")
for check in deployment.health_checks[-5:]:
    print(f"  {check.timestamp}: {check.status.value} - {check.message}")
```

### Metrics Collection

```python
import time
from collections import deque

class MetricsCollector:
    def __init__(self, window_size=1000):
        self.latencies = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.successes = deque(maxlen=window_size)
    
    def record_request(self, latency: float, success: bool):
        self.latencies.append(latency)
        if success:
            self.successes.append(1)
            self.errors.append(0)
        else:
            self.successes.append(0)
            self.errors.append(1)
    
    def get_metrics(self) -> PerformanceMetrics:
        import numpy as np
        return PerformanceMetrics(
            latency_p50=np.percentile(list(self.latencies), 50),
            latency_p95=np.percentile(list(self.latencies), 95),
            latency_p99=np.percentile(list(self.latencies), 99),
            error_rate=sum(self.errors) / len(self.errors),
            requests_per_second=len(self.latencies) / 60.0  # Last minute
        )

# Usage
collector = MetricsCollector()

# In prediction loop
start = time.time()
try:
    result = model.predict(features)
    collector.record_request(time.time() - start, success=True)
except Exception:
    collector.record_request(time.time() - start, success=False)

# Periodically update deployment
if request_count % 100 == 0:
    metrics = collector.get_metrics()
    manager.update_metrics(deployment.deployment_id, metrics)
    manager.check_deployment_health(deployment.deployment_id, request_count)
```

## Best Practices

### Pre-Deployment Checklist

- [ ] Model tested in development environment
- [ ] Shadow deployment completed successfully
- [ ] Performance benchmarks meet requirements
- [ ] Rollback plan documented
- [ ] Monitoring and alerting configured
- [ ] Stakeholders notified
- [ ] Approval workflow completed

### During Deployment

1. **Monitor Continuously**: Check metrics every minute during initial rollout
2. **Start Small**: Begin with 5% traffic for canary deployments
3. **Gradual Increase**: Increase traffic slowly (daily increments)
4. **Document**: Log all decisions and observations
5. **Communication**: Keep stakeholders informed of progress

### Post-Deployment

1. **Validate Metrics**: Compare with baseline for 24-48 hours
2. **Business Metrics**: Monitor business KPIs (conversion, revenue, etc.)
3. **User Feedback**: Monitor support tickets and user reports
4. **Performance**: Track latency, error rates, resource usage
5. **Cost**: Monitor infrastructure costs

### Rollback Decision Tree

```
Is error rate > threshold?
├─ Yes → ROLLBACK
└─ No → Continue

Is latency > threshold?
├─ Yes → ROLLBACK
└─ No → Continue

Is business metric degraded?
├─ Yes → ROLLBACK
└─ No → Continue

User complaints increasing?
├─ Yes → Investigate, consider rollback
└─ No → Continue monitoring
```

### Emergency Procedures

If immediate rollback needed:

```python
# Emergency rollback
manager.rollback_deployment(
    deployment_id=deployment.deployment_id,
    reason="EMERGENCY: Critical production issue",
    triggered_by="on-call-engineer@company.com"
)

# Notify team
from neural.mlops.audit import AuditLogger, EventType, EventSeverity

logger = AuditLogger()
logger.log_event(
    event_type=EventType.MODEL_ROLLED_BACK,
    user="on-call-engineer@company.com",
    resource_type="deployment",
    resource_id=deployment.deployment_id,
    action="emergency_rollback",
    severity=EventSeverity.CRITICAL,
    details={"reason": "Critical production issue"}
)
```

## Example Deployment Workflow

Complete end-to-end example:

```python
from neural.mlops.registry import ModelRegistry, ModelStage
from neural.mlops.deployment import DeploymentManager, DeploymentStrategy
from neural.mlops.audit import AuditLogger

# 1. Register model
registry = ModelRegistry("./models")
metadata = registry.register_model(
    name="fraud_detector",
    version="v2.0.0",
    model_path="./model.pt",
    framework="pytorch",
    created_by="data_scientist@company.com",
    metrics={"accuracy": 0.96}
)

# 2. Request and approve promotion
registry.request_promotion(
    name="fraud_detector",
    version="v2.0.0",
    target_stage=ModelStage.PRODUCTION,
    requested_by="ml_engineer@company.com",
    justification="Model exceeds accuracy requirements"
)

registry.approve_promotion(
    name="fraud_detector",
    version="v2.0.0",
    approver="ml_manager@company.com"
)

# 3. Shadow deployment
deployment_mgr = DeploymentManager("./deployments")
shadow = deployment_mgr.shadow_deploy(
    primary_model="fraud_detector:v1.0.0",
    shadow_model="fraud_detector:v2.0.0"
)

# ... collect shadow data ...

# 4. Analyze and decide
comparison = deployment_mgr.compare_shadow_deployment(shadow.shadow_id)
if comparison['agreement_rate'] > 0.95:
    # 5. Canary deployment
    deployment = deployment_mgr.create_deployment(
        model_name="fraud_detector",
        model_version="v2.0.0",
        strategy=DeploymentStrategy.CANARY,
        created_by="devops@company.com"
    )
    
    deployment_mgr.start_deployment(deployment.deployment_id)
    
    # 6. Monitor and complete
    # ... monitor metrics ...
    deployment_mgr.complete_deployment(deployment.deployment_id)
    
    # 7. Audit log
    logger = AuditLogger()
    logger.log_model_deployment(
        model_name="fraud_detector",
        version="v2.0.0",
        user="devops@company.com",
        environment="production",
        strategy="canary"
    )
```

## Troubleshooting

### Common Issues

**Issue**: Rollback triggered too frequently
- **Solution**: Adjust thresholds to be less sensitive
- **Check**: Baseline metrics are accurate

**Issue**: Shadow deployment showing disagreements
- **Solution**: Investigate prediction differences
- **Check**: Feature preprocessing consistency

**Issue**: Canary deployment never completes
- **Solution**: Review health check criteria
- **Check**: Monitoring data collection

**Issue**: Deployment fails to start
- **Solution**: Verify model files and configuration
- **Check**: Deployment permissions and resources

## Additional Resources

- [Model Registry Guide](./registry.md)
- [A/B Testing Guide](./ab_testing.md)
- [Audit Logging Guide](./audit.md)
- [CI/CD Integration](./ci_cd.md)
