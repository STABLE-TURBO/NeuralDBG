"""
Command-line interface for cost optimization.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

try:
    import click
except ImportError:
    print("Click is required for CLI. Install with: pip install click")
    sys.exit(1)

from neural.cost import (
    CostEstimator,
    CloudProvider,
    BudgetManager,
    CarbonTracker,
    TrainingPredictor,
    ResourceOptimizer,
    ResourceMetrics,
)


@click.group()
def cost_cli():
    """Neural DSL cost optimization commands."""
    pass


@cost_cli.command()
@click.option('--provider', type=click.Choice(['aws', 'gcp', 'azure']), required=True)
@click.option('--instance', required=True, help='Instance type name')
@click.option('--hours', type=float, required=True, help='Training hours')
@click.option('--storage-gb', type=float, default=100.0)
@click.option('--spot/--on-demand', default=True)
def estimate(provider: str, instance: str, hours: float, storage_gb: float, spot: bool):
    """Estimate training costs."""
    estimator = CostEstimator()
    
    cloud_provider = CloudProvider(provider)
    
    try:
        estimate_result = estimator.estimate_cost(
            provider=cloud_provider,
            instance_name=instance,
            training_hours=hours,
            storage_gb=storage_gb,
            use_spot=spot
        )
        
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Cost Estimate - {cloud_provider.value.upper()}")
        click.echo(f"{'=' * 60}")
        click.echo(f"Instance:        {estimate_result.instance_type.name}")
        click.echo(f"Duration:        {estimate_result.estimated_hours:.2f} hours")
        gpu_info = f"{estimate_result.instance_type.gpu_type or 'None'}"
        gpu_count = estimate_result.instance_type.gpu_count
        click.echo(f"GPU:             {gpu_info} x{gpu_count}")
        click.echo(f"\nOn-Demand Cost:  ${estimate_result.on_demand_cost:.2f}")
        click.echo(f"Spot Cost:       ${estimate_result.spot_cost:.2f}")
        click.echo(f"Storage Cost:    ${estimate_result.storage_cost:.2f}")
        click.echo(f"Transfer Cost:   ${estimate_result.data_transfer_cost:.2f}")
        click.echo(f"\nTotal (On-Demand): ${estimate_result.total_on_demand_cost:.2f}")
        click.echo(f"Total (Spot):      ${estimate_result.total_spot_cost:.2f}")
        if estimate_result.total_on_demand_cost > 0:
            savings_pct = (
                estimate_result.potential_savings / 
                estimate_result.total_on_demand_cost * 100
            )
        else:
            savings_pct = 0
        click.echo(
            f"Potential Savings: ${estimate_result.potential_savings:.2f} "
            f"({savings_pct:.1f}%)"
        )
        
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cost_cli.command()
@click.option('--gpu-count', type=int, default=1)
@click.option('--hours', type=float, required=True)
@click.option('--max-cost', type=float, help='Maximum acceptable cost')
@click.option('--output', type=click.Path(), help='Output JSON file')
def compare(gpu_count: int, hours: float, max_cost: Optional[float], output: Optional[str]):
    """Compare costs across providers."""
    estimator = CostEstimator()
    
    estimates = estimator.compare_providers(
        gpu_count=gpu_count,
        training_hours=hours
    )
    
    if max_cost:
        estimates = [e for e in estimates if e.total_spot_cost <= max_cost]
    
    click.echo(f"\n{'=' * 80}")
    click.echo(f"Cost Comparison - {gpu_count} GPU(s), {hours:.1f} hours")
    click.echo(f"{'=' * 80}")
    click.echo(f"{'Provider':<10} {'Instance':<25} {'Spot Cost':<12} {'On-Demand':<12} {'Savings':<10}")
    click.echo(f"{'-' * 80}")
    
    for est in estimates[:10]:
        if est.total_on_demand_cost > 0:
            savings_pct = (est.potential_savings / est.total_on_demand_cost * 100)
        else:
            savings_pct = 0
        click.echo(
            f"{est.provider.value:<10} "
            f"{est.instance_type.name:<25} "
            f"${est.total_spot_cost:<11.2f} "
            f"${est.total_on_demand_cost:<11.2f} "
            f"{savings_pct:<9.1f}%"
        )
    
    if output:
        data = [
            {
                'provider': e.provider.value,
                'instance': e.instance_type.name,
                'spot_cost': e.total_spot_cost,
                'on_demand_cost': e.total_on_demand_cost,
                'savings_percent': (e.potential_savings / e.total_on_demand_cost * 100) if e.total_on_demand_cost > 0 else 0
            }
            for e in estimates
        ]
        
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
        
        click.echo(f"\nResults saved to {output}")


@cost_cli.command()
@click.option('--name', required=True, help='Budget name')
@click.option('--amount', type=float, required=True, help='Total budget amount')
@click.option('--days', type=int, required=True, help='Budget period in days')
@click.option('--providers', multiple=True, help='Cloud providers (aws, gcp, azure)')
def create_budget(name: str, amount: float, days: int, providers: tuple):
    """Create a new budget."""
    manager = BudgetManager()
    
    provider_list = [CloudProvider(p) for p in providers] if providers else []
    
    budget = manager.create_budget(
        name=name,
        total_amount=amount,
        period_days=days,
        providers=provider_list
    )
    
    click.echo(f"✓ Created budget '{budget.name}'")
    click.echo(f"  Amount: ${budget.total_amount:.2f}")
    click.echo(f"  Period: {budget.period_days} days")
    if provider_list:
        click.echo(f"  Providers: {', '.join(p.value for p in provider_list)}")


@cost_cli.command()
@click.option('--budget', required=True, help='Budget name')
@click.option('--amount', type=float, required=True, help='Expense amount')
@click.option('--provider', type=click.Choice(['aws', 'gcp', 'azure']), required=True)
@click.option('--description', default='', help='Expense description')
def record_expense(budget: str, amount: float, provider: str, description: str):
    """Record an expense."""
    manager = BudgetManager()
    
    try:
        manager.record_expense(
            budget_name=budget,
            amount=amount,
            provider=CloudProvider(provider),
            description=description
        )
        
        click.echo(f"✓ Recorded ${amount:.2f} expense to '{budget}'")
        
        status = manager.get_budget_status(budget)
        click.echo(f"  Remaining: ${status['remaining_amount']:.2f}")
        click.echo(f"  Utilization: {status['utilization_percent']:.1f}%")
        
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cost_cli.command()
@click.option('--budget', help='Budget name (optional, shows all if not specified)')
def budget_status(budget: Optional[str]):
    """Show budget status."""
    manager = BudgetManager()
    
    if budget:
        try:
            status = manager.get_budget_status(budget)
            
            click.echo(f"\n{'=' * 60}")
            click.echo(f"Budget: {status['name']}")
            click.echo(f"{'=' * 60}")
            click.echo(f"Total:       ${status['total_amount']:.2f}")
            click.echo(f"Spent:       ${status['spent_amount']:.2f}")
            click.echo(f"Remaining:   ${status['remaining_amount']:.2f}")
            click.echo(f"Utilization: {status['utilization_percent']:.1f}%")
            click.echo(f"Days Left:   {status['days_remaining']:.0f}")
            click.echo(f"Burn Rate:   ${status['burn_rate_per_day']:.2f}/day")
            click.echo(f"Projected:   ${status['projected_spend']:.2f}")
            click.echo(f"Status:      {status['status'].upper()}")
            
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    else:
        statuses = manager.get_all_budgets_status()
        
        if not statuses:
            click.echo("No budgets configured.")
            return
        
        click.echo(f"\n{'=' * 100}")
        header = f"{'Budget':<20} {'Total':<12} {'Spent':<12} {'Remaining':<12}"
        click.echo(f"{header} {'Util%':<8} {'Status':<10}")
        click.echo(f"{'-' * 100}")
        
        for status in statuses:
            click.echo(
                f"{status['name']:<20} "
                f"${status['total_amount']:<11.2f} "
                f"${status['spent_amount']:<11.2f} "
                f"${status['remaining_amount']:<11.2f} "
                f"{status['utilization_percent']:<7.1f}% "
                f"{status['status']:<10}"
            )


@cost_cli.command()
@click.option('--job-id', required=True, help='Job identifier')
@click.option('--provider', type=click.Choice(['aws', 'gcp', 'azure']), required=True)
@click.option('--region', required=True, help='Cloud region')
@click.option('--instance', required=True, help='Instance type')
@click.option('--hours', type=float, required=True, help='Training duration')
def track_carbon(job_id: str, provider: str, region: str, instance: str, hours: float):
    """Track carbon footprint."""
    estimator = CostEstimator()
    tracker = CarbonTracker()
    
    cloud_provider = CloudProvider(provider)
    
    instance_type = None
    for inst in estimator.instance_types[cloud_provider]:
        if inst.name == instance:
            instance_type = inst
            break
    
    if not instance_type:
        click.echo(f"Error: Instance type '{instance}' not found", err=True)
        sys.exit(1)
    
    report = tracker.track_training(
        job_id=job_id,
        provider=cloud_provider,
        region=region,
        instance_type=instance_type,
        duration_hours=hours
    )
    
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Carbon Footprint Report")
    click.echo(f"{'=' * 60}")
    click.echo(f"Job ID:      {report.job_id}")
    click.echo(f"Provider:    {report.provider.value}")
    click.echo(f"Region:      {report.region}")
    click.echo(f"Duration:    {report.duration_hours:.2f} hours")
    click.echo(f"\nEnergy:      {report.energy_kwh:.2f} kWh")
    click.echo(f"Carbon:      {report.carbon_kg_co2:.2f} kg CO₂")
    click.echo(f"Intensity:   {report.carbon_intensity:.2f} g CO₂/kWh")
    click.echo(f"\nEquivalents:")
    click.echo(f"  🚗 {report.equivalent_miles_driven:.0f} miles driven")
    click.echo(f"  🌳 {report.equivalent_trees_needed:.1f} trees to offset")


@cost_cli.command()
@click.option('--days', type=int, help='Number of days to report')
def carbon_summary(days: Optional[int]):
    """Show carbon emission summary."""
    tracker = CarbonTracker()
    tracker.load_reports()
    
    summary = tracker.get_total_emissions(time_period_days=days)
    
    period_str = f"({days} days)" if days else "(all time)"
    
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Carbon Emission Summary {period_str}")
    click.echo(f"{'=' * 60}")
    click.echo(f"Total Jobs:  {summary['total_jobs']}")
    click.echo(f"Total CO₂:   {summary['total_carbon_kg_co2']:.2f} kg")
    click.echo(f"Total Energy: {summary['total_energy_kwh']:.2f} kWh")
    click.echo("\nEquivalents:")
    click.echo(f"  🚗 {summary['equivalent_miles_driven']:.0f} miles")
    click.echo(f"  🌳 {summary['equivalent_trees_needed']:.1f} trees")
    
    if summary['by_provider']:
        click.echo("\nBy Provider:")
        for provider, carbon in summary['by_provider'].items():
            click.echo(f"  {provider}: {carbon:.2f} kg CO₂")


@cost_cli.command()
@click.option('--model-params', type=int, required=True, help='Model parameters')
@click.option('--dataset-size', type=int, required=True, help='Dataset size')
@click.option('--batch-size', type=int, required=True, help='Batch size')
@click.option('--epochs', type=int, required=True, help='Number of epochs')
@click.option('--provider', type=click.Choice(['aws', 'gcp', 'azure']), required=True)
@click.option('--instance', required=True, help='Instance type')
def predict(model_params: int, dataset_size: int, batch_size: int, epochs: int, 
            provider: str, instance: str):
    """Predict training time and cost."""
    estimator = CostEstimator()
    predictor = TrainingPredictor(estimator)
    
    cloud_provider = CloudProvider(provider)
    
    instance_type = None
    for inst in estimator.instance_types[cloud_provider]:
        if inst.name == instance:
            instance_type = inst
            break
    
    if not instance_type:
        click.echo(f"Error: Instance type '{instance}' not found", err=True)
        sys.exit(1)
    
    model_spec = {'total_params': model_params}
    
    estimate = predictor.predict_training_time(
        model_params=model_spec,
        dataset_size=dataset_size,
        batch_size=batch_size,
        epochs=epochs,
        instance_type=instance_type
    )
    
    click.echo(f"\n{'=' * 60}")
    click.echo("Training Prediction")
    click.echo(f"{'=' * 60}")
    click.echo(f"Model Parameters: {model_params:,}")
    click.echo(f"Dataset Size:     {dataset_size:,}")
    click.echo(f"Batch Size:       {batch_size}")
    click.echo(f"Epochs:           {epochs}")
    click.echo(f"Instance:         {instance_type.name}")
    click.echo(f"\nEstimated Time:   {estimate.estimated_hours:.2f} hours")
    click.echo(f"Iterations:       {estimate.estimated_iterations:,}")
    click.echo(f"Compute Cost:     ${estimate.compute_cost:.2f}")
    click.echo(f"Storage Cost:     ${estimate.storage_cost:.2f}")
    click.echo(f"Total Cost:       ${estimate.total_cost:.2f}")
    ci_lower = estimate.confidence_interval[0]
    ci_upper = estimate.confidence_interval[1]
    click.echo(f"\nConfidence Interval: [{ci_lower:.1f}, {ci_upper:.1f}] hours")


@cost_cli.command()
@click.option('--port', type=int, default=8052, help='Dashboard port')
def dashboard(port: int):
    """Launch cost optimization dashboard."""
    try:
        from neural.cost.dashboard import create_dashboard
        
        dash = create_dashboard(port=port)
        click.echo(f"Starting cost dashboard on http://localhost:{port}")
        dash.run(debug=False)
        
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Install dashboard dependencies: pip install neural-dsl[dashboard]")
        sys.exit(1)


if __name__ == '__main__':
    cost_cli()
