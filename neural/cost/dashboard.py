"""
Interactive cost monitoring dashboard.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc
    import plotly.graph_objs as go
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from neural.cost.estimator import CostEstimator, CloudProvider
from neural.cost.budget_manager import BudgetManager
from neural.cost.carbon_tracker import CarbonTracker
from neural.cost.analyzer import CostAnalyzer


class CostDashboard:
    """Interactive dashboard for cost monitoring."""
    
    def __init__(
        self,
        cost_estimator: Optional[CostEstimator] = None,
        budget_manager: Optional[BudgetManager] = None,
        carbon_tracker: Optional[CarbonTracker] = None,
        cost_analyzer: Optional[CostAnalyzer] = None,
        port: int = 8052
    ):
        """
        Initialize cost dashboard.
        
        Parameters
        ----------
        cost_estimator : CostEstimator, optional
            Cost estimator instance
        budget_manager : BudgetManager, optional
            Budget manager instance
        carbon_tracker : CarbonTracker, optional
            Carbon tracker instance
        cost_analyzer : CostAnalyzer, optional
            Cost analyzer instance
        port : int
            Dashboard port
        """
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dash is required for cost dashboard. "
                "Install with: pip install neural-dsl[dashboard]"
            )
        
        self.cost_estimator = cost_estimator or CostEstimator()
        self.budget_manager = budget_manager or BudgetManager()
        self.carbon_tracker = carbon_tracker or CarbonTracker()
        self.cost_analyzer = cost_analyzer or CostAnalyzer(self.cost_estimator)
        self.port = port
        
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        self._setup_layout()
        self._setup_callbacks()
    
    def run(self, debug: bool = False):
        """Run the dashboard."""
        self.app.run_server(debug=debug, port=self.port)
    
    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Neural DSL - Cost Optimization Dashboard",
                           className="text-center mb-4")
                ])
            ]),
            
            dbc.Tabs([
                dbc.Tab(label="Overview", tab_id="overview"),
                dbc.Tab(label="Budgets", tab_id="budgets"),
                dbc.Tab(label="Cost Estimation", tab_id="estimation"),
                dbc.Tab(label="Carbon Footprint", tab_id="carbon"),
                dbc.Tab(label="Analysis", tab_id="analysis"),
            ], id="tabs", active_tab="overview"),
            
            html.Div(id="tab-content", className="mt-4"),
            
            dcc.Interval(
                id='interval-component',
                interval=5000,
                n_intervals=0
            )
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output("tab-content", "children"),
            Input("tabs", "active_tab")
        )
        def render_tab_content(active_tab):
            if active_tab == "overview":
                return self._create_overview_tab()
            elif active_tab == "budgets":
                return self._create_budgets_tab()
            elif active_tab == "estimation":
                return self._create_estimation_tab()
            elif active_tab == "carbon":
                return self._create_carbon_tab()
            elif active_tab == "analysis":
                return self._create_analysis_tab()
            return html.Div("Select a tab")
    
    def _create_overview_tab(self) -> dbc.Container:
        """Create overview tab."""
        budgets = self.budget_manager.get_all_budgets_status()
        carbon_summary = self.carbon_tracker.get_total_emissions()
        spending_report = self.budget_manager.get_spending_report(days=30)
        
        total_spent = spending_report.get('total_amount', 0.0)
        total_carbon = carbon_summary.get('total_carbon_kg_co2', 0.0)
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Spend (30 days)", className="card-title"),
                            html.H2(f"${total_spent:.2f}", className="text-primary"),
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Budgets", className="card-title"),
                            html.H2(f"{len(budgets)}", className="text-info"),
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Carbon Footprint", className="card-title"),
                            html.H2(f"{total_carbon:.1f} kg", className="text-success"),
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Jobs", className="card-title"),
                            html.H2(f"{carbon_summary.get('total_jobs', 0)}", 
                                   className="text-warning"),
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Spending by Provider", className="card-title"),
                            dcc.Graph(
                                id='spending-by-provider',
                                figure=self._create_spending_chart(spending_report)
                            )
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Budget Status", className="card-title"),
                            dcc.Graph(
                                id='budget-status',
                                figure=self._create_budget_chart(budgets)
                            )
                        ])
                    ])
                ], width=6),
            ]),
        ])
    
    def _create_budgets_tab(self) -> dbc.Container:
        """Create budgets tab."""
        budgets = self.budget_manager.get_all_budgets_status()
        
        budget_cards = []
        for budget in budgets:
            status_color = {
                'healthy': 'success',
                'warning': 'warning',
                'critical': 'danger',
                'exceeded': 'danger',
                'expired': 'secondary'
            }.get(budget['status'], 'info')
            
            budget_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(budget['name']),
                        dbc.CardBody([
                            html.P(f"Total: ${budget['total_amount']:.2f}"),
                            html.P(f"Spent: ${budget['spent_amount']:.2f}"),
                            html.P(f"Remaining: ${budget['remaining_amount']:.2f}"),
                            dbc.Progress(
                                value=budget['utilization_percent'],
                                color=status_color,
                                className="mb-2"
                            ),
                            html.P(f"Utilization: {budget['utilization_percent']:.1f}%"),
                            html.P(f"Days Remaining: {budget['days_remaining']:.0f}"),
                            html.P(f"Burn Rate: ${budget['burn_rate_per_day']:.2f}/day"),
                            dbc.Badge(budget['status'].upper(), color=status_color)
                        ])
                    ], className="mb-3")
                ], width=4)
            )
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Budget Management"),
                    html.P("Monitor and manage your training budgets")
                ])
            ], className="mb-4"),
            
            dbc.Row(budget_cards if budget_cards else [
                dbc.Col([
                    html.P("No budgets configured. Create a budget to start tracking costs.")
                ])
            ])
        ])
    
    def _create_estimation_tab(self) -> dbc.Container:
        """Create cost estimation tab."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Cost Estimation"),
                    html.P("Estimate training costs for different configurations")
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Instance Comparison"),
                            html.P("Compare costs across cloud providers"),
                            
                            dbc.Label("GPUs Required:"),
                            dcc.Slider(
                                id='gpu-count-slider',
                                min=1,
                                max=8,
                                value=1,
                                marks={i: str(i) for i in [1, 2, 4, 8]}
                            ),
                            
                            dbc.Label("Training Hours:"),
                            dcc.Input(
                                id='training-hours-input',
                                type='number',
                                value=10,
                                className="form-control mb-3"
                            ),
                            
                            dbc.Button(
                                "Compare Providers",
                                id='compare-button',
                                color="primary"
                            ),
                            
                            html.Div(id='comparison-results', className="mt-3")
                        ])
                    ])
                ])
            ])
        ])
    
    def _create_carbon_tab(self) -> dbc.Container:
        """Create carbon footprint tab."""
        carbon_summary = self.carbon_tracker.get_total_emissions()
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Carbon Footprint Tracking"),
                    html.P("Monitor environmental impact of training jobs")
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Emissions", className="card-title"),
                            html.H2(f"{carbon_summary.get('total_carbon_kg_co2', 0):.2f} kg CO₂",
                                   className="text-success"),
                            html.P(f"Energy: {carbon_summary.get('total_energy_kwh', 0):.2f} kWh"),
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Equivalents", className="card-title"),
                            html.P(
                                f"🚗 {carbon_summary.get('equivalent_miles_driven', 0):.0f} "
                                f"miles driven"
                            ),
                            html.P(
                                f"🌳 {carbon_summary.get('equivalent_trees_needed', 0):.1f} "
                                f"trees needed"
                            ),
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Jobs", className="card-title"),
                            html.H2(f"{carbon_summary.get('total_jobs', 0)}",
                                   className="text-info"),
                        ])
                    ])
                ], width=4),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Emissions by Provider"),
                            dcc.Graph(
                                id='carbon-by-provider',
                                figure=self._create_carbon_chart(carbon_summary)
                            )
                        ])
                    ])
                ])
            ])
        ])
    
    def _create_analysis_tab(self) -> dbc.Container:
        """Create cost analysis tab."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Cost-Performance Analysis"),
                    html.P("Analyze cost-performance tradeoffs")
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Provider Comparison"),
                            html.P("Cost comparison across cloud providers"),
                            html.Div(id='provider-comparison')
                        ])
                    ])
                ])
            ])
        ])
    
    def _create_spending_chart(self, spending_report: Dict[str, Any]) -> go.Figure:
        """Create spending pie chart."""
        by_provider = spending_report.get('by_provider', {})
        
        if not by_provider:
            return go.Figure()
        
        fig = go.Figure(data=[go.Pie(
            labels=list(by_provider.keys()),
            values=list(by_provider.values()),
            hole=.3
        )])
        
        fig.update_layout(
            title="Spending Distribution",
            height=300
        )
        
        return fig
    
    def _create_budget_chart(self, budgets: List[Dict[str, Any]]) -> go.Figure:
        """Create budget status chart."""
        if not budgets:
            return go.Figure()
        
        names = [b['name'] for b in budgets]
        spent = [b['spent_amount'] for b in budgets]
        remaining = [b['remaining_amount'] for b in budgets]
        
        fig = go.Figure(data=[
            go.Bar(name='Spent', x=names, y=spent),
            go.Bar(name='Remaining', x=names, y=remaining)
        ])
        
        fig.update_layout(
            barmode='stack',
            title="Budget Status",
            height=300,
            xaxis_title="Budget",
            yaxis_title="Amount ($)"
        )
        
        return fig
    
    def _create_carbon_chart(self, carbon_summary: Dict[str, Any]) -> go.Figure:
        """Create carbon emissions chart."""
        by_provider = carbon_summary.get('by_provider', {})
        
        if not by_provider:
            return go.Figure()
        
        fig = go.Figure(data=[go.Bar(
            x=list(by_provider.keys()),
            y=list(by_provider.values()),
            marker_color='green'
        )])
        
        fig.update_layout(
            title="Carbon Emissions by Provider",
            height=300,
            xaxis_title="Provider",
            yaxis_title="CO₂ (kg)"
        )
        
        return fig


def create_dashboard(
    cost_estimator: Optional[CostEstimator] = None,
    budget_manager: Optional[BudgetManager] = None,
    carbon_tracker: Optional[CarbonTracker] = None,
    port: int = 8052
) -> CostDashboard:
    """
    Create and return cost dashboard instance.
    
    Parameters
    ----------
    cost_estimator : CostEstimator, optional
        Cost estimator
    budget_manager : BudgetManager, optional
        Budget manager
    carbon_tracker : CarbonTracker, optional
        Carbon tracker
    port : int
        Dashboard port
        
    Returns
    -------
    CostDashboard
        Dashboard instance
    """
    return CostDashboard(
        cost_estimator=cost_estimator,
        budget_manager=budget_manager,
        carbon_tracker=carbon_tracker,
        port=port
    )


if __name__ == "__main__":
    dashboard = create_dashboard()
    print(f"Starting cost dashboard on port {dashboard.port}...")
    dashboard.run(debug=True)
