"""
Basic monitoring example for Neural DSL models.
"""

import time
import numpy as np
from neural.monitoring import ModelMonitor


def main():
    """Basic monitoring example."""
    
    print("=== Basic Monitoring Example ===\n")
    
    # Initialize monitor
    print("1. Initializing monitor...")
    monitor = ModelMonitor(
        model_name="example-model",
        model_version="1.0",
        storage_path="monitoring_data_example",
        enable_alerting=True
    )
    print("   ✓ Monitor initialized\n")
    
    # Set reference data
    print("2. Setting reference data...")
    np.random.seed(42)
    reference_data = np.random.randn(1000, 10)
    reference_predictions = np.random.randint(0, 2, 1000)
    reference_performance = {
        'accuracy': 0.95,
        'precision': 0.94,
        'recall': 0.96,
        'f1': 0.95
    }
    
    monitor.set_reference_data(
        data=reference_data,
        predictions=reference_predictions,
        performance=reference_performance
    )
    print("   ✓ Reference data set\n")
    
    # Record some predictions
    print("3. Recording predictions...")
    for i in range(100):
        monitor.record_prediction()
        
        if (i + 1) % 20 == 0:
            print(f"   Recorded {i + 1} predictions...")
    
    print("   ✓ Predictions recorded\n")
    
    # Check drift
    print("4. Checking for drift...")
    new_data = np.random.randn(100, 10) + 0.1  # Slight shift
    new_predictions = np.random.randint(0, 2, 100)
    new_performance = {
        'accuracy': 0.93,
        'precision': 0.92,
        'recall': 0.94,
        'f1': 0.93
    }
    
    drift_report = monitor.check_drift(
        data=new_data,
        predictions=new_predictions,
        performance=new_performance
    )
    
    print(f"   Drift detected: {drift_report['is_drifting']}")
    print(f"   Drift severity: {drift_report['drift_severity']}")
    print(f"   Prediction drift: {drift_report['prediction_drift']:.4f}")
    print(f"   Performance drift: {drift_report['performance_drift']:.4f}")
    print()
    
    # Get monitoring summary
    print("5. Getting monitoring summary...")
    summary = monitor.get_monitoring_summary()
    
    print(f"   Total predictions: {summary['total_predictions']}")
    print(f"   Total errors: {summary['total_errors']}")
    print(f"   Error rate: {summary['error_rate']:.4f}")
    print(f"   Uptime: {summary['uptime_seconds']:.1f} seconds")
    print()
    
    # Generate health report
    print("6. Generating health report...")
    health = monitor.generate_health_report()
    
    print(f"   Status: {health['status']}")
    print(f"   Issues: {len(health['issues'])}")
    for issue in health['issues']:
        print(f"     - {issue}")
    print()
    
    print("=== Example Complete ===")
    print(f"\nMonitoring data stored in: monitoring_data_example/")
    print("View status with: neural monitor status --storage-path monitoring_data_example")


if __name__ == "__main__":
    main()
