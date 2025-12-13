"""
Complete MLOps Workflow Example.

Demonstrates end-to-end model lifecycle management including:
- Model registration with metadata
- Approval workflow for production deployment
- Shadow deployment for validation
- A/B testing for comparison
- Canary deployment with automated rollback
- Comprehensive audit logging
"""

from datetime import datetime, timedelta
import time

from neural.mlops.ab_testing import ABTestManager, TrafficSplitStrategy
from neural.mlops.audit import AuditLogger, EventSeverity, EventType
from neural.mlops.deployment import (
    DeploymentManager,
    DeploymentStrategy,
    HealthStatus,
    PerformanceMetrics,
    RollbackConfig,
)
from neural.mlops.registry import ModelRegistry, ModelStage


def simulate_model_training():
    """Simulate model training and return metrics."""
    print("📊 Training model...")
    time.sleep(0.5)
    return {
        "accuracy": 0.96,
        "precision": 0.94,
        "recall": 0.95,
        "f1_score": 0.945,
        "training_time_hours": 2.5
    }


def main():
    print("=" * 80)
    print("Neural DSL MLOps - Complete Workflow Example")
    print("=" * 80)
    print()
    
    # Initialize MLOps components
    print("🔧 Initializing MLOps components...")
    registry = ModelRegistry("./example_models")
    deployment_mgr = DeploymentManager("./example_deployments")
    ab_manager = ABTestManager("./example_ab_tests")
    audit_logger = AuditLogger("./example_audit_logs")
    print("✓ Components initialized\n")
    
    # Step 1: Train and register a new model
    print("=" * 80)
    print("Step 1: Model Registration")
    print("=" * 80)
    
    metrics = simulate_model_training()
    
    metadata = registry.register_model(
        name="fraud_detector",
        version="v2.0.0",
        model_path="./dummy_model.pt",
        framework="pytorch",
        created_by="data_scientist@company.com",
        metrics=metrics,
        tags=["fraud-detection", "production-ready", "xgboost"],
        description="Improved fraud detection model with 96% accuracy"
    )
    
    print(f"✓ Model registered: {metadata.name} v{metadata.version}")
    print(f"  Metrics: {metrics}")
    print(f"  Stage: {metadata.stage.value}")
    
    audit_logger.log_model_registration(
        model_name="fraud_detector",
        version="v2.0.0",
        user="data_scientist@company.com",
        framework="pytorch",
        metrics=metrics
    )
    print("✓ Audit log created\n")
    
    # Step 2: Approval workflow
    print("=" * 80)
    print("Step 2: Approval Workflow")
    print("=" * 80)
    
    approval = registry.request_promotion(
        name="fraud_detector",
        version="v2.0.0",
        target_stage=ModelStage.STAGING,
        requested_by="ml_engineer@company.com",
        justification="Model exceeds accuracy requirements (96% vs 95% target)",
        reviewers=["ml_manager@company.com"]
    )
    
    print(f"✓ Promotion requested to {approval.target_stage.value}")
    print(f"  Requested by: {approval.requested_by}")
    print(f"  Justification: {approval.justification}")
    
    audit_logger.log_approval_request(
        model_name="fraud_detector",
        version="v2.0.0",
        user="ml_engineer@company.com",
        target_stage=approval.target_stage.value,
        justification=approval.justification
    )
    
    registry.approve_promotion(
        name="fraud_detector",
        version="v2.0.0",
        approver="ml_manager@company.com",
        comment="Metrics validated, approved for staging"
    )
    
    print("✓ Promotion approved")
    
    audit_logger.log_approval_granted(
        model_name="fraud_detector",
        version="v2.0.0",
        approver="ml_manager@company.com",
        comment="Metrics validated"
    )
    print("✓ Audit logs updated\n")
    
    # Step 3: Shadow deployment
    print("=" * 80)
    print("Step 3: Shadow Deployment")
    print("=" * 80)
    
    shadow = deployment_mgr.shadow_deploy(
        primary_model="fraud_detector:v1.0.0",
        shadow_model="fraud_detector:v2.0.0",
        traffic_percentage=100.0
    )
    
    print(f"✓ Shadow deployment created: {shadow.shadow_id}")
    print(f"  Primary: {shadow.primary_model}")
    print(f"  Shadow: {shadow.shadow_model}")
    print(f"  Traffic: {shadow.traffic_percentage}%")
    
    print("\n📊 Simulating shadow traffic...")
    for i in range(100):
        agreement = i % 10 != 0  # 90% agreement rate
        deployment_mgr.record_shadow_comparison(
            shadow_id=shadow.shadow_id,
            primary_result={"prediction": 0 if i % 2 == 0 else 1},
            shadow_result={"prediction": 0 if (i % 2 == 0 or agreement) else 1},
            primary_latency=0.035 + (i % 5) * 0.001,
            shadow_latency=0.038 + (i % 5) * 0.001,
            agreement=agreement
        )
    
    comparison = deployment_mgr.compare_shadow_deployment(shadow.shadow_id)
    print("\n✓ Shadow deployment analysis:")
    print(f"  Total requests: {comparison['total_requests']}")
    print(f"  Agreement rate: {comparison['agreement_rate']:.2%}")
    print(f"  Primary P95 latency: {comparison['primary_latency']['p95']:.3f}s")
    print(f"  Shadow P95 latency: {comparison['shadow_latency']['p95']:.3f}s")
    
    if comparison['agreement_rate'] > 0.85:
        print("  ✓ Shadow model validated for production")
    print()
    
    # Step 4: A/B Testing
    print("=" * 80)
    print("Step 4: A/B Testing")
    print("=" * 80)
    
    ab_test = ab_manager.create_test(
        name="Model V2 Production Test",
        description="A/B test comparing v1.0.0 vs v2.0.0 in production",
        control_variant="fraud_detector:v1.0.0",
        treatment_variant="fraud_detector:v2.0.0",
        traffic_split=0.1,  # 10% to treatment
        strategy=TrafficSplitStrategy.HASH_BASED,
        created_by="ml_engineer@company.com",
        metadata={"business_metric": "false_positive_rate"}
    )
    
    print(f"✓ A/B test created: {ab_test.test_id}")
    print(f"  Control: {ab_test.control_variant}")
    print(f"  Treatment: {ab_test.treatment_variant}")
    print(f"  Traffic split: {ab_test.traffic_split * 100}%")
    
    ab_manager.start_test(ab_test.test_id)
    print("✓ A/B test started")
    
    audit_logger.log_event(
        event_type=EventType.AB_TEST_STARTED,
        user="ml_engineer@company.com",
        resource_type="ab_test",
        resource_id=ab_test.test_id,
        action="start",
        severity=EventSeverity.INFO
    )
    
    print("\n📊 Simulating A/B test traffic...")
    for i in range(1000):
        # Control: 90% success, 0.035s avg latency
        ab_manager.record_request(
            test_id=ab_test.test_id,
            variant="control",
            success=(i % 10) != 0,
            latency=0.035 + (i % 10) * 0.002,
            custom_metrics={"false_positive_rate": 0.05}
        )
        
        # Treatment: 95% success, 0.038s avg latency
        if i % 10 == 0:  # 10% to treatment
            ab_manager.record_request(
                test_id=ab_test.test_id,
                variant="treatment",
                success=(i % 20) != 0,
                latency=0.038 + (i % 10) * 0.002,
                custom_metrics={"false_positive_rate": 0.03}
            )
    
    analysis = ab_manager.analyze_test(ab_test.test_id, confidence_level=0.95)
    
    print("\n✓ A/B test analysis:")
    print(f"  Control success rate: {analysis['control']['success_rate']:.2%}")
    print(f"  Treatment success rate: {analysis['treatment']['success_rate']:.2%}")
    print(f"  Statistically significant: {analysis['statistically_significant']}")
    print(f"  Relative improvement: {analysis['improvement']['relative_improvement']:.2f}%")
    
    if analysis['statistically_significant']:
        print("  ✓ Treatment variant shows significant improvement!")
    
    ab_manager.complete_test(ab_test.test_id)
    print("✓ A/B test completed\n")
    
    # Step 5: Production deployment with rollback
    print("=" * 80)
    print("Step 5: Production Deployment")
    print("=" * 80)
    
    baseline_metrics = PerformanceMetrics(
        latency_p50=0.035,
        latency_p95=0.050,
        latency_p99=0.075,
        error_rate=0.001,
        requests_per_second=1000.0
    )
    
    deployment = deployment_mgr.create_deployment(
        model_name="fraud_detector",
        model_version="v2.0.0",
        strategy=DeploymentStrategy.CANARY,
        created_by="devops@company.com",
        environment="production",
        baseline_metrics=baseline_metrics,
        rollback_config=RollbackConfig(
            enabled=True,
            error_rate_threshold=0.01,
            latency_threshold_multiplier=1.5,
            min_requests_before_check=1000
        ),
        metadata={"initial_traffic": 0.05}
    )
    
    print(f"✓ Deployment created: {deployment.deployment_id}")
    print(f"  Strategy: {deployment.strategy.value}")
    print(f"  Environment: {deployment.environment}")
    print("  Rollback enabled: Yes")
    
    deployment_mgr.start_deployment(deployment.deployment_id)
    print("✓ Deployment started")
    
    audit_logger.log_model_deployment(
        model_name="fraud_detector",
        version="v2.0.0",
        user="devops@company.com",
        environment="production",
        strategy="canary"
    )
    print("✓ Deployment audit logged")
    
    # Simulate healthy deployment
    print("\n📊 Monitoring deployment health...")
    healthy_metrics = PerformanceMetrics(
        latency_p50=0.037,
        latency_p95=0.052,
        latency_p99=0.078,
        error_rate=0.002,
        requests_per_second=50.0
    )
    
    deployment_mgr.update_metrics(deployment.deployment_id, healthy_metrics)
    deployment_mgr.add_health_check(
        deployment_id=deployment.deployment_id,
        status=HealthStatus.HEALTHY,
        metrics=healthy_metrics,
        message="All systems operational"
    )
    
    needs_rollback, reasons = deployment_mgr.check_deployment_health(
        deployment.deployment_id,
        request_count=2000
    )
    
    if not needs_rollback:
        print("✓ Deployment health check passed")
        deployment_mgr.complete_deployment(deployment.deployment_id)
        print("✓ Deployment completed successfully\n")
    else:
        print(f"⚠️ Rollback triggered: {reasons}\n")
    
    # Step 6: Compliance reporting
    print("=" * 80)
    print("Step 6: Compliance Reporting")
    print("=" * 80)
    
    report = audit_logger.generate_compliance_report(
        start_date=datetime.now() - timedelta(hours=1),
        end_date=datetime.now()
    )
    
    print(f"✓ Compliance report generated: {report.report_id}")
    print(f"  Period: {report.period_start} to {report.period_end}")
    print(f"  Total events: {report.total_events}")
    print("  Events by type:")
    for event_type, count in report.events_by_type.items():
        print(f"    - {event_type}: {count}")
    print(f"  Security violations: {report.summary['total_security_violations']}")
    print(f"  Critical events: {report.summary['total_critical_events']}")
    
    # Export audit trail
    audit_logger.export_audit_trail(
        output_path="./example_audit_trail.json",
        format="json",
        start_date=datetime.now() - timedelta(hours=1),
        end_date=datetime.now()
    )
    print("✓ Audit trail exported to example_audit_trail.json\n")
    
    # Step 7: Model comparison
    print("=" * 80)
    print("Step 7: Model Comparison")
    print("=" * 80)
    
    # Register v1.0.0 for comparison
    registry.register_model(
        name="fraud_detector",
        version="v1.0.0",
        model_path="./dummy_model_v1.pt",
        framework="pytorch",
        created_by="data_scientist@company.com",
        metrics={"accuracy": 0.94, "precision": 0.92, "recall": 0.93}
    )
    
    comparison = registry.compare_models(
        name="fraud_detector",
        version1="v1.0.0",
        version2="v2.0.0"
    )
    
    print("✓ Model comparison:")
    print(f"  Version 1: {comparison['version1']}")
    print(f"  Version 2: {comparison['version2']}")
    print("\n  Metrics comparison:")
    for metric, values in comparison['metrics_comparison'].items():
        if values['difference'] is not None:
            print(f"    {metric}:")
            print(f"      v1.0.0: {values['version1']:.3f}")
            print(f"      v2.0.0: {values['version2']:.3f}")
            print(f"      Δ: {values['difference']:+.3f}")
    
    print("\n" + "=" * 80)
    print("✓ Complete MLOps workflow executed successfully!")
    print("=" * 80)
    print("\nGenerated artifacts:")
    print("  - Model registry: ./example_models/")
    print("  - Deployments: ./example_deployments/")
    print("  - A/B tests: ./example_ab_tests/")
    print("  - Audit logs: ./example_audit_logs/")
    print("  - Audit trail: ./example_audit_trail.json")


if __name__ == "__main__":
    main()
