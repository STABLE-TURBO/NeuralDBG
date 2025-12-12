"""
Visualization tools for hyperparameter optimization results.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union, Tuple
from .parameter_importance import ParameterImportanceAnalyzer

logger = logging.getLogger(__name__)

def plot_optimization_history(trials: List[Dict[str, Any]],
                             metric: str = 'score',
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the optimization history.

    Args:
        trials: List of trial dictionaries
        metric: The metric to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if not trials:
        logger.warning("No trials provided for optimization history plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No trials available", ha='center', va='center')
        return fig

    # Extract trial numbers and scores
    trial_numbers = []
    scores = []
    best_so_far = []
    current_best = float('-inf')

    for i, trial in enumerate(trials):
        trial_numbers.append(i + 1)
        score = trial.get(metric, trial.get('score', None))

        if score is None:
            continue

        scores.append(score)
        current_best = max(current_best, score)
        best_so_far.append(current_best)

    if not scores:
        logger.warning(f"No {metric} values found in trials")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No {metric} values found", ha='center', va='center')
        return fig

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot individual trial scores
    ax.scatter(trial_numbers, scores, alpha=0.6, label=f"Trial {metric}", s=50)

    # Plot best score so far
    ax.plot(trial_numbers, best_so_far, 'r-', label=f"Best {metric} so far", linewidth=2)

    # Add labels and legend
    ax.set_xlabel("Trial number")
    ax.set_ylabel(metric)
    ax.set_title(f"Optimization History ({metric})")
    ax.legend()

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def plot_param_importance(trials: List[Dict[str, Any]],
                         metric: str = 'score',
                         method: str = 'random_forest',
                         figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot parameter importance.

    Args:
        trials: List of trial dictionaries
        metric: The metric to use for importance analysis
        method: Method for importance analysis
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    analyzer = ParameterImportanceAnalyzer(method=method)
    importance_dict = analyzer.analyze(trials, target_metric=metric)
    return analyzer.plot_importance(importance_dict,
                                   title=f"Hyperparameter Importance ({metric})",
                                   figsize=figsize)

def plot_parallel_coordinates(trials: List[Dict[str, Any]],
                             metric: str = 'score',
                             top_n: int = 10,
                             figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Create a parallel coordinates plot of hyperparameters.

    Args:
        trials: List of trial dictionaries
        metric: The metric to use for coloring
        top_n: Number of top trials to highlight
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    analyzer = ParameterImportanceAnalyzer()
    return analyzer.plot_parallel_coordinates(trials,
                                            target_metric=metric,
                                            top_n=top_n,
                                            figsize=figsize)

def plot_correlation_heatmap(trials: List[Dict[str, Any]],
                           metric: str = 'score',
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create a correlation heatmap of hyperparameters and the target metric.

    Args:
        trials: List of trial dictionaries
        metric: The target metric
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    analyzer = ParameterImportanceAnalyzer()
    return analyzer.plot_correlation_heatmap(trials,
                                          target_metric=metric,
                                          figsize=figsize)

def plot_contour(trials: List[Dict[str, Any]],
                param_x: str,
                param_y: str,
                metric: str = 'score',
                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create a contour plot of two hyperparameters.

    Args:
        trials: List of trial dictionaries
        param_x: Parameter for x-axis
        param_y: Parameter for y-axis
        metric: The metric to use for contour levels
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if not trials:
        logger.warning("No trials provided for contour plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No trials available", ha='center', va='center')
        return fig

    # Extract parameter values and scores
    x_values = []
    y_values = []
    scores = []

    for trial in trials:
        params = trial.get('parameters', {})
        score = trial.get(metric, trial.get('score', None))

        if param_x not in params or param_y not in params or score is None:
            continue

        x_values.append(params[param_x])
        y_values.append(params[param_y])
        scores.append(score)

    if len(x_values) < 5:
        logger.warning("Insufficient data for contour plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        return fig

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    try:
        # Convert to numpy arrays
        x = np.array(x_values)
        y = np.array(y_values)
        z = np.array(scores)

        # Check if parameters are numeric
        if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            logger.warning("Parameters must be numeric for contour plot")
            ax.text(0.5, 0.5, "Parameters must be numeric", ha='center', va='center')
            return fig

        # Create a grid for contour plot
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate z values on the grid
        from scipy.interpolate import griddata
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        # Plot contour
        contour = ax.contourf(xi, yi, zi, 15, cmap='viridis')
        ax.scatter(x, y, c=z, cmap='viridis', edgecolor='k', s=100, alpha=0.6)

        # Add colorbar
        plt.colorbar(contour, ax=ax, label=metric)

        # Add labels
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_title(f"Contour Plot of {param_x} vs {param_y} ({metric})")

    except Exception as e:
        logger.error(f"Error creating contour plot: {str(e)}")
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}", ha='center', va='center')

    plt.tight_layout()
    return fig

def plot_slice(trials: List[Dict[str, Any]],
              param: str,
              metric: str = 'score',
              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create a slice plot for a single parameter.

    Args:
        trials: List of trial dictionaries
        param: Parameter to plot
        metric: The metric to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if not trials:
        logger.warning("No trials provided for slice plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No trials available", ha='center', va='center')
        return fig

    # Extract parameter values and scores
    param_values = []
    scores = []

    for trial in trials:
        params = trial.get('parameters', {})
        score = trial.get(metric, trial.get('score', None))

        if param not in params or score is None:
            continue

        param_values.append(params[param])
        scores.append(score)

    if not param_values:
        logger.warning(f"No {param} values found in trials")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No {param} values found", ha='center', va='center')
        return fig

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    try:
        # Convert to numpy arrays
        x = np.array(param_values)
        y = np.array(scores)

        # Check if parameter is numeric
        if np.issubdtype(x.dtype, np.number):
            # For numeric parameters, sort by parameter value
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            y = y[sort_idx]

            # Plot scatter and line
            ax.scatter(x, y, alpha=0.6, s=50)

            # Try to fit a smoothing curve if enough points
            if len(x) >= 5:
                try:
                    from scipy.interpolate import make_interp_spline
                    x_smooth = np.linspace(min(x), max(x), 100)
                    spl = make_interp_spline(x, y, k=min(3, len(x)-1))
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2)
                except Exception:
                    # If smoothing fails, just connect the dots
                    ax.plot(x, y, 'r-', alpha=0.5)
            else:
                # Just connect the dots for few points
                ax.plot(x, y, 'r-', alpha=0.5)

            ax.set_xlabel(param)

        else:
            # For categorical parameters, use a box plot
            df = pd.DataFrame({param: x, metric: y})
            sns.boxplot(x=param, y=metric, data=df, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # Add labels
        ax.set_ylabel(metric)
        ax.set_title(f"Effect of {param} on {metric}")

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

    except Exception as e:
        logger.error(f"Error creating slice plot: {str(e)}")
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}", ha='center', va='center')

    plt.tight_layout()
    return fig

def plot_multi_objective_pareto(trials: List[Dict[str, Any]],
                                obj_x: str,
                                obj_y: str,
                                highlight_pareto: bool = True,
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot Pareto front for multi-objective optimization.
    
    Args:
        trials: List of trial dictionaries
        obj_x: First objective name
        obj_y: Second objective name
        highlight_pareto: Whether to highlight Pareto optimal points
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not trials:
        logger.warning("No trials for Pareto plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No trials available", ha='center', va='center')
        return fig
    
    # Extract objectives
    x_values = []
    y_values = []
    
    for trial in trials:
        x_val = trial.get(obj_x, None)
        y_val = trial.get(obj_y, None)
        
        if x_val is not None and y_val is not None:
            x_values.append(x_val)
            y_values.append(y_val)
    
    if len(x_values) < 2:
        logger.warning("Insufficient data for Pareto plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        return fig
    
    x = np.array(x_values)
    y = np.array(y_values)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all points
    ax.scatter(x, y, alpha=0.5, s=50, label='All trials')
    
    # Identify and plot Pareto front if requested
    if highlight_pareto:
        pareto_mask = np.ones(len(x), dtype=bool)
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j:
                    # Check if point j dominates point i
                    # Assuming both objectives are to be maximized
                    # Adjust logic based on actual optimization direction
                    if x[j] >= x[i] and y[j] >= y[i] and (x[j] > x[i] or y[j] > y[i]):
                        pareto_mask[i] = False
                        break
        
        pareto_x = x[pareto_mask]
        pareto_y = y[pareto_mask]
        
        # Sort for line plotting
        sort_idx = np.argsort(pareto_x)
        pareto_x = pareto_x[sort_idx]
        pareto_y = pareto_y[sort_idx]
        
        ax.scatter(pareto_x, pareto_y, color='red', s=100, 
                  label='Pareto front', zorder=10)
        ax.plot(pareto_x, pareto_y, 'r--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel(obj_x)
    ax.set_ylabel(obj_y)
    ax.set_title('Multi-objective Optimization: Pareto Front')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_3d_pareto(trials: List[Dict[str, Any]],
                   obj_x: str,
                   obj_y: str,
                   obj_z: str,
                   figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot 3D Pareto front for three objectives.
    
    Args:
        trials: List of trial dictionaries
        obj_x: First objective name
        obj_y: Second objective name
        obj_z: Third objective name
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not trials:
        logger.warning("No trials for 3D Pareto plot")
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.text(0.5, 0.5, 0.5, "No trials available", ha='center', va='center')
        return fig
    
    # Extract objectives
    x_values = []
    y_values = []
    z_values = []
    
    for trial in trials:
        x_val = trial.get(obj_x, None)
        y_val = trial.get(obj_y, None)
        z_val = trial.get(obj_z, None)
        
        if x_val is not None and y_val is not None and z_val is not None:
            x_values.append(x_val)
            y_values.append(y_val)
            z_values.append(z_val)
    
    if len(x_values) < 2:
        logger.warning("Insufficient data for 3D Pareto plot")
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.text(0.5, 0.5, 0.5, "Insufficient data", ha='center', va='center')
        return fig
    
    x = np.array(x_values)
    y = np.array(y_values)
    z = np.array(z_values)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.6, s=50)
    
    ax.set_xlabel(obj_x)
    ax.set_ylabel(obj_y)
    ax.set_zlabel(obj_z)
    ax.set_title('3D Multi-objective Optimization')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label=obj_z)
    
    plt.tight_layout()
    return fig

def plot_convergence_comparison(trials_dict: Dict[str, List[Dict[str, Any]]],
                                metric: str = 'score',
                                figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Compare convergence of multiple optimization runs.
    
    Args:
        trials_dict: Dictionary mapping run names to trial lists
        metric: Metric to compare
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not trials_dict:
        logger.warning("No trials for convergence comparison")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No trials available", ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for run_name, trials in trials_dict.items():
        if not trials:
            continue
        
        # Extract scores and compute best so far
        trial_numbers = []
        best_so_far = []
        current_best = float('-inf')
        
        for i, trial in enumerate(trials):
            trial_numbers.append(i + 1)
            score = trial.get(metric, trial.get('score', None))
            
            if score is not None:
                current_best = max(current_best, score)
            best_so_far.append(current_best)
        
        # Plot best so far for this run
        ax.plot(trial_numbers, best_so_far, label=run_name, linewidth=2)
    
    ax.set_xlabel("Trial number")
    ax.set_ylabel(f"Best {metric}")
    ax.set_title(f"Convergence Comparison ({metric})")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_optimization_landscape(trials: List[Dict[str, Any]],
                                param: str,
                                metric: str = 'score',
                                n_bins: int = 20,
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot optimization landscape for a single parameter.
    
    Args:
        trials: List of trial dictionaries
        param: Parameter name
        metric: Metric name
        n_bins: Number of bins for histogram
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not trials:
        logger.warning("No trials for landscape plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No trials available", ha='center', va='center')
        return fig
    
    # Extract parameter values and scores
    param_values = []
    scores = []
    
    for trial in trials:
        params = trial.get('parameters', {})
        score = trial.get(metric, trial.get('score', None))
        
        if param in params and score is not None:
            param_values.append(params[param])
            scores.append(score)
    
    if not param_values:
        logger.warning(f"No {param} values found")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No {param} values found", ha='center', va='center')
        return fig
    
    x = np.array(param_values)
    y = np.array(scores)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Top plot: scatter with smoothing
    ax1.scatter(x, y, alpha=0.5, s=30)
    
    # Add smoothing curve if enough points
    if len(x) >= 5 and np.issubdtype(x.dtype, np.number):
        try:
            from scipy.interpolate import make_interp_spline
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            
            x_smooth = np.linspace(min(x), max(x), 100)
            spl = make_interp_spline(x_sorted, y_sorted, k=min(3, len(x)-1))
            y_smooth = spl(x_smooth)
            ax1.plot(x_smooth, y_smooth, 'r-', linewidth=2, alpha=0.7, label='Trend')
        except Exception:
            pass
    
    ax1.set_ylabel(metric)
    ax1.set_title(f'Optimization Landscape: {param}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bottom plot: histogram of parameter values
    if np.issubdtype(x.dtype, np.number):
        ax2.hist(x, bins=n_bins, alpha=0.7, color='steelblue')
    else:
        # For categorical, use value counts
        from collections import Counter
        counts = Counter(x)
        ax2.bar(range(len(counts)), list(counts.values()), alpha=0.7, color='steelblue')
        ax2.set_xticks(range(len(counts)))
        ax2.set_xticklabels(list(counts.keys()), rotation=45)
    
    ax2.set_xlabel(param)
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def create_optimization_report(trials: List[Dict[str, Any]],
                              metric: str = 'score',
                              output_path: str = 'hpo_report.html') -> str:
    """
    Create a comprehensive HTML report of the optimization results.

    Args:
        trials: List of trial dictionaries
        metric: The metric to use for analysis
        output_path: Path to save the HTML report

    Returns:
        Path to the saved report
    """
    try:
        import base64
        from io import BytesIO

        # Create all plots
        history_fig = plot_optimization_history(trials, metric)
        importance_fig = plot_param_importance(trials, metric)
        parallel_fig = plot_parallel_coordinates(trials, metric)
        heatmap_fig = plot_correlation_heatmap(trials, metric)

        # Convert figures to base64 for embedding in HTML
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_str

        history_img = fig_to_base64(history_fig)
        importance_img = fig_to_base64(importance_fig)
        parallel_img = fig_to_base64(parallel_fig)
        heatmap_img = fig_to_base64(heatmap_fig)

        # Get best trial
        best_trial = max(trials, key=lambda t: t.get(metric, t.get('score', float('-inf'))))
        best_params = best_trial.get('parameters', {})
        best_score = best_trial.get(metric, best_trial.get('score', 'N/A'))

        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hyperparameter Optimization Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                    background-color: #f5f5f5;
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .plot {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .plot img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .summary {{
                    background-color: #e8f5e9;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Hyperparameter Optimization Report</h1>

                <div class="summary">
                    <h2>Summary</h2>
                    <p><strong>Number of trials:</strong> {len(trials)}</p>
                    <p><strong>Best {metric}:</strong> {best_score}</p>
                </div>

                <h2>Best Parameters</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
        """

        # Add best parameters to the table
        for param, value in best_params.items():
            html += f"""
                    <tr>
                        <td>{param}</td>
                        <td>{value}</td>
                    </tr>
            """

        html += """
                </table>

                <h2>Optimization History</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{}" alt="Optimization History">
                </div>

                <h2>Parameter Importance</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{}" alt="Parameter Importance">
                </div>

                <h2>Parallel Coordinates Plot</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{}" alt="Parallel Coordinates">
                </div>

                <h2>Parameter Correlation Heatmap</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{}" alt="Correlation Heatmap">
                </div>

                <h2>All Trials</h2>
                <table>
                    <tr>
                        <th>Trial</th>
                        <th>{metric}</th>
        """.format(history_img, importance_img, parallel_img, heatmap_img)

        # Add parameter columns to the header
        all_params = set()
        for trial in trials:
            all_params.update(trial.get('parameters', {}).keys())
        param_names = sorted(list(all_params))

        for param in param_names:
            html += f"""
                        <th>{param}</th>
            """

        html += """
                    </tr>
        """

        # Add all trials to the table
        for i, trial in enumerate(trials):
            score = trial.get(metric, trial.get('score', 'N/A'))
            params = trial.get('parameters', {})

            html += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{score}</td>
            """

            for param in param_names:
                value = params.get(param, 'N/A')
                html += f"""
                        <td>{value}</td>
                """

            html += """
                    </tr>
            """

        html += """
                </table>
            </div>
        </body>
        </html>
        """

        # Save the report
        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"Optimization report saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error creating optimization report: {str(e)}")
        return ""
