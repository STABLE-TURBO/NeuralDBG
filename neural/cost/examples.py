"""
Example usage of the cost optimization module.
"""

from neural.cost import (
    CostEstimator,
    CloudProvider,
    SpotInstanceOrchestrator,
    SpotStrategy,
    ResourceOptimizer,
    ResourceMetrics,
    TrainingPredictor,
    CarbonTracker,
    CostAnalyzer,
    PerformanceMetrics,
    BudgetManager,
    CostDashboard,
)


def example_cost_estimation():
    """Example: Estimate training costs."""
    print("=" * 60)
    print("Cost Estimation Example")
    print("=" * 60)
    
    estimator = CostEstimator()
    
    estimate = estimator.estimate_cost(
        provider=CloudProvider.AWS,
        instance_name="p3.2xlarge",
        training_hours=10.0,
        storage_gb=100.0,
        use_spot=True
    )
    
    print(f"Provider: {estimate.provider.value}")
    print(f"Instance: {estimate.instance_type.name}")
    print(f"Training Hours: {estimate.estimated_hours}")
    print(f"On-Demand Cost: ${estimate.on_demand_cost:.2f}")
    print(f"Spot Cost: ${estimate.spot_cost:.2f}")
    print(f"Potential Savings: ${estimate.potential_savings:.2f}")
    print(f"Total Cost (Spot): ${estimate.total_spot_cost:.2f}")
    
    print("\nComparing Providers:")
    estimates = estimator.compare_providers(gpu_count=1, training_hours=10.0)
    
    for est in estimates[:5]:
        print(f"  {est.provider.value:8s} - {est.instance_type.name:20s}: ${est.total_spot_cost:.2f}")
    
    cheapest = estimator.get_cheapest_option(gpu_count=1, training_hours=10.0, max_cost=100.0)
    if cheapest:
        print(f"\nCheapest Option: {cheapest.provider.value} {cheapest.instance_type.name}")


def example_spot_orchestration():
    """Example: Spot instance orchestration."""
    print("\n" + "=" * 60)
    print("Spot Instance Orchestration Example")
    print("=" * 60)
    
    estimator = CostEstimator()
    instance_type = estimator.instance_types[CloudProvider.AWS][0]
    
    orchestrator = SpotInstanceOrchestrator(
        provider=CloudProvider.AWS,
        strategy=SpotStrategy.BALANCED
    )
    
    instance = orchestrator.launch_spot_instance(instance_type)
    
    if instance:
        print(f"Launched spot instance: {instance.instance_id}")
        print(f"Instance type: {instance.instance_type.name}")
        
        statuses = orchestrator.monitor_instances()
        for status in statuses:
            print(f"  Status: {status['status']}, Uptime: {status['uptime_hours']:.2f}h")
        
        savings = orchestrator.get_cost_savings()
        print(f"\nTotal Savings: ${savings['total_cost_saved']:.2f}")
        print(f"Total Interruptions: {savings['total_interruptions']}")


def example_resource_optimization():
    """Example: Resource right-sizing."""
    print("\n" + "=" * 60)
    print("Resource Optimization Example")
    print("=" * 60)
    
    estimator = CostEstimator()
    instance_type = estimator.instance_types[CloudProvider.AWS][1]
    
    optimizer = ResourceOptimizer(estimator)
    
    metrics = ResourceMetrics(
        cpu_utilization=0.45,
        memory_utilization=0.50,
        gpu_utilization=0.35,
        gpu_memory_utilization=0.40,
        samples=100
    )
    
    recommendation = optimizer.analyze_utilization(instance_type, metrics)
    
    print(f"Current Instance: {recommendation.current_instance.name}")
    print(f"Recommended Instance: {recommendation.recommended_instance.name}")
    print(f"Current Cost: ${recommendation.current_cost_per_hour:.2f}/hr")
    print(f"Recommended Cost: ${recommendation.recommended_cost_per_hour:.2f}/hr")
    print(f"Estimated Savings: {recommendation.estimated_savings_percent:.1f}%")
    print(f"Reason: {recommendation.reason}")
    print(f"Confidence: {recommendation.confidence_score:.2f}")


def example_training_prediction():
    """Example: Training time prediction."""
    print("\n" + "=" * 60)
    print("Training Time Prediction Example")
    print("=" * 60)
    
    estimator = CostEstimator()
    instance_type = estimator.instance_types[CloudProvider.AWS][0]
    
    predictor = TrainingPredictor(estimator)
    
    model_params = {
        'total_params': 50_000_000,
        'layers': 50,
    }
    
    estimate = predictor.predict_training_time(
        model_params=model_params,
        dataset_size=100_000,
        batch_size=32,
        epochs=100,
        instance_type=instance_type
    )
    
    print(f"Model Size: {model_params['total_params']:,} parameters")
    print(f"Dataset Size: 100,000 samples")
    print(f"Batch Size: 32")
    print(f"Epochs: 100")
    print(f"\nEstimated Training Time: {estimate.estimated_hours:.2f} hours")
    print(f"Estimated Iterations: {estimate.estimated_iterations:,}")
    print(f"Compute Cost: ${estimate.compute_cost:.2f}")
    print(f"Total Cost: ${estimate.total_cost:.2f}")
    ci_lower = estimate.confidence_interval[0]
    ci_upper = estimate.confidence_interval[1]
    print(f"Confidence Interval: [{ci_lower:.1f}, {ci_upper:.1f}] hours")
    
    print("\nOptimizing Batch Size:")
    optimal = predictor.optimize_batch_size(
        model_params=model_params,
        dataset_size=100_000,
        instance_type=instance_type,
        target_epochs=100
    )
    
    print(f"Optimal Batch Size: {optimal['optimal_batch_size']}")
    print(f"Estimated Time: {optimal['estimated_hours']:.2f} hours")
    print(f"Estimated Cost: ${optimal['estimated_cost']:.2f}")


def example_carbon_tracking():
    """Example: Carbon footprint tracking."""
    print("\n" + "=" * 60)
    print("Carbon Footprint Tracking Example")
    print("=" * 60)
    
    estimator = CostEstimator()
    instance_type = estimator.instance_types[CloudProvider.AWS][0]
    
    tracker = CarbonTracker()
    
    report = tracker.track_training(
        job_id="train-example-001",
        provider=CloudProvider.AWS,
        region="us-west-2",
        instance_type=instance_type,
        duration_hours=10.0
    )
    
    print(f"Job ID: {report.job_id}")
    print(f"Provider: {report.provider.value}")
    print(f"Region: {report.region}")
    print(f"Duration: {report.duration_hours} hours")
    print(f"\nEnergy Consumed: {report.energy_kwh:.2f} kWh")
    print(f"Carbon Emissions: {report.carbon_kg_co2:.2f} kg CO₂")
    print(f"Carbon Intensity: {report.carbon_intensity:.2f} g CO₂/kWh")
    print(f"\nEquivalents:")
    print(f"  🚗 {report.equivalent_miles_driven:.0f} miles driven")
    print(f"  🌳 {report.equivalent_trees_needed:.1f} trees needed to offset")
    
    greenest = tracker.get_greenest_region(CloudProvider.AWS, instance_type)
    print(f"\nGreenest Region: {greenest}")
    
    print("\nRegional Comparison (top 5):")
    comparisons = tracker.compare_regions(instance_type, 10.0)
    for comp in comparisons[:5]:
        print(f"  {comp['region']:15s}: {comp['carbon_kg_co2']:6.2f} kg CO₂")


def example_cost_analysis():
    """Example: Cost-performance analysis."""
    print("\n" + "=" * 60)
    print("Cost-Performance Analysis Example")
    print("=" * 60)
    
    analyzer = CostAnalyzer()
    
    configs = [
        {'instance': 'p3.2xlarge', 'batch_size': 32},
        {'instance': 'p3.8xlarge', 'batch_size': 128},
        {'instance': 'g4dn.xlarge', 'batch_size': 16},
    ]
    
    metrics = [
        PerformanceMetrics(accuracy=0.85, training_time_hours=12.0, inference_latency_ms=50),
        PerformanceMetrics(accuracy=0.87, training_time_hours=4.0, inference_latency_ms=45),
        PerformanceMetrics(accuracy=0.82, training_time_hours=20.0, inference_latency_ms=60),
    ]
    
    costs = [92.0, 367.0, 31.6]
    
    analysis = analyzer.analyze_tradeoffs(configs, metrics, costs)
    
    print("Analysis Results:")
    print(f"Total Configurations: {len(analysis.points)}")
    print(f"Pareto Optimal: {len(analysis.pareto_frontier)}")
    
    print("\nBest Value:")
    print(f"  Config: {analysis.best_value.configuration}")
    print(f"  Cost: ${analysis.best_value.cost:.2f}")
    print(f"  Accuracy: {analysis.best_value.performance.accuracy:.3f}")
    print(f"  Efficiency: {analysis.best_value.efficiency_score:.6f}")
    
    print("\nBest Performance:")
    print(f"  Config: {analysis.best_performance.configuration}")
    print(f"  Accuracy: {analysis.best_performance.performance.accuracy:.3f}")
    
    print("\nLowest Cost:")
    print(f"  Config: {analysis.best_cost.configuration}")
    print(f"  Cost: ${analysis.best_cost.cost:.2f}")
    
    roi = analyzer.calculate_roi(
        training_cost=367.0,
        improvement_accuracy=0.02,
        business_value_per_point=1000.0
    )
    
    print(f"\nROI Analysis:")
    print(f"  Training Cost: ${roi['training_cost']:.2f}")
    print(f"  Value Gained: ${roi['value_gained']:.2f}")
    print(f"  ROI: {roi['roi_percent']:.1f}%")
    print(f"  Payback Period: {roi['payback_period_days']:.0f} days")


def example_budget_management():
    """Example: Budget management."""
    print("\n" + "=" * 60)
    print("Budget Management Example")
    print("=" * 60)
    
    manager = BudgetManager()
    
    budget = manager.create_budget(
        name="Q1-Training-Budget",
        total_amount=10000.0,
        period_days=90,
        providers=[CloudProvider.AWS, CloudProvider.GCP],
        alert_thresholds=[0.5, 0.8, 0.95]
    )
    
    print(f"Created Budget: {budget.name}")
    print(f"Total Amount: ${budget.total_amount:.2f}")
    print(f"Period: {budget.period_days} days")
    
    manager.record_expense(
        budget_name="Q1-Training-Budget",
        amount=250.0,
        provider=CloudProvider.AWS,
        description="Training run #1"
    )
    
    manager.record_expense(
        budget_name="Q1-Training-Budget",
        amount=180.0,
        provider=CloudProvider.GCP,
        description="Training run #2"
    )
    
    status = manager.get_budget_status("Q1-Training-Budget")
    
    print(f"\nBudget Status:")
    print(f"  Spent: ${status['spent_amount']:.2f}")
    print(f"  Remaining: ${status['remaining_amount']:.2f}")
    print(f"  Utilization: {status['utilization_percent']:.1f}%")
    print(f"  Days Remaining: {status['days_remaining']:.0f}")
    print(f"  Burn Rate: ${status['burn_rate_per_day']:.2f}/day")
    print(f"  Projected Spend: ${status['projected_spend']:.2f}")
    print(f"  Status: {status['status']}")
    
    spending_report = manager.get_spending_report(days=30)
    print(f"\nSpending Report (30 days):")
    print(f"  Total Expenses: {spending_report['total_expenses']}")
    print(f"  Total Amount: ${spending_report['total_amount']:.2f}")
    print(f"  By Provider: {spending_report['by_provider']}")


def example_integrated_workflow():
    """Example: Integrated cost optimization workflow."""
    print("\n" + "=" * 60)
    print("Integrated Cost Optimization Workflow")
    print("=" * 60)
    
    estimator = CostEstimator()
    manager = BudgetManager()
    tracker = CarbonTracker()
    predictor = TrainingPredictor(estimator)
    
    manager.create_budget(
        name="Project-Alpha",
        total_amount=5000.0,
        period_days=30
    )
    
    print("Step 1: Estimate costs")
    model_params = {'total_params': 100_000_000}
    instance_type = estimator.instance_types[CloudProvider.AWS][2]
    
    time_estimate = predictor.predict_training_time(
        model_params=model_params,
        dataset_size=500_000,
        batch_size=64,
        epochs=50,
        instance_type=instance_type
    )
    
    print(f"  Estimated time: {time_estimate.estimated_hours:.1f} hours")
    print(f"  Estimated cost: ${time_estimate.total_cost:.2f}")
    
    budget_status = manager.get_budget_status("Project-Alpha")
    if time_estimate.total_cost > budget_status['remaining_amount']:
        print("  ⚠️  WARNING: Insufficient budget!")
        
        print("\nStep 2: Find cheaper alternative")
        cheapest = estimator.get_cheapest_option(
            gpu_count=4,
            training_hours=time_estimate.estimated_hours,
            max_cost=budget_status['remaining_amount']
        )
        
        if cheapest:
            print(f"  Alternative: {cheapest.provider.value} {cheapest.instance_type.name}")
            print(f"  Cost: ${cheapest.total_spot_cost:.2f}")
    
    print("\nStep 3: Track carbon footprint")
    carbon_report = tracker.track_training(
        job_id="project-alpha-001",
        provider=CloudProvider.AWS,
        region="us-west-2",
        instance_type=instance_type,
        duration_hours=time_estimate.estimated_hours
    )
    
    print(f"  Carbon: {carbon_report.carbon_kg_co2:.2f} kg CO₂")
    print(f"  Energy: {carbon_report.energy_kwh:.2f} kWh")
    
    print("\nStep 4: Record expense")
    manager.record_expense(
        budget_name="Project-Alpha",
        amount=time_estimate.total_cost,
        provider=CloudProvider.AWS,
        description="Model training - Project Alpha"
    )
    
    final_status = manager.get_budget_status("Project-Alpha")
    print(f"  Budget remaining: ${final_status['remaining_amount']:.2f}")
    print(f"  Utilization: {final_status['utilization_percent']:.1f}%")


def run_all_examples():
    """Run all examples."""
    example_cost_estimation()
    example_spot_orchestration()
    example_resource_optimization()
    example_training_prediction()
    example_carbon_tracking()
    example_cost_analysis()
    example_budget_management()
    example_integrated_workflow()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
