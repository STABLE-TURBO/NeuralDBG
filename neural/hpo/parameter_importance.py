"""
Parameter importance analysis for hyperparameter optimization.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ParameterImportanceAnalyzer:
    """Analyzes the importance of hyperparameters in optimization trials."""

    def __init__(self, method: str = 'random_forest'):
        """
        Initialize the parameter importance analyzer.
        
        Args:
            method: Method for importance analysis ('random_forest', 'gradient_boosting', 'permutation')
        """
        self.encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.method = method

    def analyze(self, trials: List[Dict[str, Any]], target_metric: str = 'score') -> Dict[str, float]:
        """
        Analyze parameter importance from a list of trials.

        Args:
            trials: List of trial dictionaries with parameters and scores
            target_metric: The metric to use for importance analysis

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if not trials:
            logger.warning("No trials provided for parameter importance analysis")
            return {}

        # Extract parameters and target values
        X, y, param_names = self._prepare_data(trials, target_metric)

        if X.shape[0] < 5:
            logger.warning("Too few trials for reliable parameter importance analysis")
            return {param: 1.0 / len(param_names) for param in param_names}

        # Train a model to estimate parameter importance
        try:
            if self.method == 'gradient_boosting':
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:  # Default to random forest
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            self.model.fit(X, y)

            # Get feature importances
            if self.method == 'permutation':
                perm_importance = permutation_importance(
                    self.model, X, y, n_repeats=10, random_state=42
                )
                importances = perm_importance.importances_mean
            else:
                importances = self.model.feature_importances_

            # Create a dictionary mapping parameter names to importance scores
            importance_dict = {param: float(imp) for param, imp in zip(param_names, importances)}

            return importance_dict

        except Exception as e:
            logger.error(f"Error in parameter importance analysis: {str(e)}")
            return {param: 1.0 / len(param_names) for param in param_names}

    def analyze_with_fanova(self, trials: List[Dict[str, Any]], 
                           target_metric: str = 'score') -> Dict[str, float]:
        """
        Analyze parameter importance using fANOVA (functional ANOVA).
        
        Args:
            trials: List of trial dictionaries
            target_metric: Target metric name
            
        Returns:
            Dictionary of parameter importances
        """
        try:
            import optuna
            from optuna.importance import FanovaImportanceEvaluator
            
            # Create a temporary study from trials
            study = optuna.create_study()
            
            for trial_data in trials:
                trial = optuna.trial.create_trial(
                    params=trial_data.get('parameters', {}),
                    distributions={},
                    values=[trial_data.get(target_metric, 0)]
                )
                study.add_trial(trial)
            
            # Calculate fANOVA importances
            evaluator = FanovaImportanceEvaluator()
            importance = evaluator.evaluate(study)
            
            return importance
            
        except ImportError:
            logger.warning("fANOVA requires optuna. Falling back to standard method.")
            return self.analyze(trials, target_metric)
        except Exception as e:
            logger.error(f"Error in fANOVA analysis: {str(e)}")
            return self.analyze(trials, target_metric)

    def _prepare_data(self, trials: List[Dict[str, Any]], target_metric: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for parameter importance analysis.

        Args:
            trials: List of trial dictionaries
            target_metric: The metric to use for importance analysis

        Returns:
            X: Feature matrix
            y: Target values
            param_names: List of parameter names
        """
        # Extract all parameter names from trials
        all_params = set()
        for trial in trials:
            all_params.update(trial.get('parameters', {}).keys())
        param_names = sorted(list(all_params))

        # Create feature matrix and target vector
        X_data = []
        y_data = []

        for trial in trials:
            params = trial.get('parameters', {})
            score = trial.get(target_metric, trial.get('score', None))

            if score is None:
                continue

            # Extract parameter values
            row = []
            for param in param_names:
                value = params.get(param, None)
                row.append(value)

            X_data.append(row)
            y_data.append(score)

        # Convert to numpy arrays
        X_raw = np.array(X_data)
        y = np.array(y_data)

        # Handle categorical parameters and missing values
        X = self._encode_and_scale(X_raw, param_names)

        return X, y, param_names

    def _encode_and_scale(self, X_raw: np.ndarray, param_names: List[str]) -> np.ndarray:
        """
        Encode categorical parameters and scale numerical parameters.

        Args:
            X_raw: Raw feature matrix
            param_names: List of parameter names

        Returns:
            Encoded and scaled feature matrix
        """
        X_encoded = np.zeros_like(X_raw, dtype=float)

        # Encode each column
        for i, param in enumerate(param_names):
            column = X_raw[:, i]

            # Check if column contains non-numeric values
            try:
                X_encoded[:, i] = column.astype(float)
            except (ValueError, TypeError):
                # Categorical parameter, use label encoding
                if param not in self.encoders:
                    self.encoders[param] = LabelEncoder()
                    # Fit the encoder on non-None values
                    non_none_values = [v for v in column if v is not None]
                    if non_none_values:
                        self.encoders[param].fit(non_none_values)

                # Replace None with a placeholder value
                column_with_placeholder = np.array([v if v is not None else 'NONE_PLACEHOLDER' for v in column])

                # Ensure all values are in the encoder's classes
                for value in np.unique(column_with_placeholder):
                    if value != 'NONE_PLACEHOLDER' and value not in self.encoders[param].classes_:
                        # Refit the encoder with the new value
                        self.encoders[param].fit(np.append(self.encoders[param].classes_, [value]))

                # Transform the column
                try:
                    X_encoded[:, i] = self.encoders[param].transform(column_with_placeholder)
                except Exception as e:
                    logger.error(f"Error encoding parameter {param}: {str(e)}")
                    # Use zeros as a fallback
                    X_encoded[:, i] = 0

        # Replace NaN values with column means
        for i in range(X_encoded.shape[1]):
            col = X_encoded[:, i]
            nan_mask = np.isnan(col)
            if np.any(nan_mask):
                col[nan_mask] = np.nanmean(col) if np.any(~nan_mask) else 0
                X_encoded[:, i] = col

        # Scale the features
        try:
            X_scaled = self.scaler.fit_transform(X_encoded)
            return X_scaled
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return X_encoded

    def plot_importance(self, importance_dict: Dict[str, float],
                       title: str = "Hyperparameter Importance",
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot parameter importance as a bar chart.

        Args:
            importance_dict: Dictionary mapping parameter names to importance scores
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Sort parameters by importance
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        params = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot horizontal bars
        y_pos = np.arange(len(params))
        bars = ax.barh(y_pos, scores, align='center')
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(params)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_importance_with_std(self, trials: List[Dict[str, Any]],
                                 target_metric: str = 'score',
                                 n_iterations: int = 10,
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot parameter importance with standard deviation (bootstrap).
        
        Args:
            trials: List of trial dictionaries
            target_metric: Target metric name
            n_iterations: Number of bootstrap iterations
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not trials or len(trials) < 5:
            logger.warning("Insufficient trials for bootstrap analysis")
            return self.plot_importance({}, figsize=figsize)
        
        # Perform bootstrap sampling
        all_importances = []
        for _ in range(n_iterations):
            # Sample with replacement
            sampled_trials = np.random.choice(trials, size=len(trials), replace=True).tolist()
            importance = self.analyze(sampled_trials, target_metric)
            all_importances.append(importance)
        
        # Calculate mean and std
        param_names = list(all_importances[0].keys())
        mean_importance = {}
        std_importance = {}
        
        for param in param_names:
            values = [imp.get(param, 0) for imp in all_importances]
            mean_importance[param] = np.mean(values)
            std_importance[param] = np.std(values)
        
        # Sort by mean importance
        sorted_params = sorted(param_names, key=lambda x: mean_importance[x], reverse=True)
        means = [mean_importance[p] for p in sorted_params]
        stds = [std_importance[p] for p in sorted_params]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(sorted_params))
        
        ax.barh(y_pos, means, xerr=stds, align='center', alpha=0.7, capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_params)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (mean ± std)')
        ax.set_title('Hyperparameter Importance with Uncertainty')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_parallel_coordinates(self, trials: List[Dict[str, Any]],
                                 target_metric: str = 'score',
                                 top_n: int = 10,
                                 figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Create a parallel coordinates plot of hyperparameters.

        Args:
            trials: List of trial dictionaries
            target_metric: The metric to use for coloring
            top_n: Number of top trials to highlight
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not trials:
            logger.warning("No trials provided for parallel coordinates plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No trials available", ha='center', va='center')
            return fig

        # Convert trials to DataFrame
        df_list = []
        for trial in trials:
            row = trial.get('parameters', {}).copy()
            row[target_metric] = trial.get(target_metric, trial.get('score', 0))
            df_list.append(row)

        df = pd.DataFrame(df_list)

        if df.empty or len(df.columns) <= 1:
            logger.warning("Insufficient data for parallel coordinates plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
            return fig

        # Sort by target metric and mark top_n trials
        df = df.sort_values(by=target_metric, ascending=False)
        df['is_top'] = False
        df.iloc[:top_n, df.columns.get_loc('is_top')] = True

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use Seaborn's parallel coordinates plot
        # First, normalize all columns for better visualization
        cols_to_plot = [col for col in df.columns if col not in [target_metric, 'is_top']]

        if not cols_to_plot:
            logger.warning("No parameters to plot")
            ax.text(0.5, 0.5, "No parameters to plot", ha='center', va='center')
            return fig

        df_norm = df.copy()
        for col in cols_to_plot:
            if df[col].dtype in [np.float64, np.int64]:
                df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)

        # Plot non-top trials with low alpha
        for i, (_, row) in enumerate(df_norm[~df['is_top']].iterrows()):
            ax.plot(cols_to_plot, row[cols_to_plot], color='gray', alpha=0.3)

        # Plot top trials with high alpha and different color
        for i, (_, row) in enumerate(df_norm[df['is_top']].iterrows()):
            ax.plot(cols_to_plot, row[cols_to_plot], color='red', alpha=0.7, linewidth=2)

        # Set the axes
        ax.set_xticks(range(len(cols_to_plot)))
        ax.set_xticklabels(cols_to_plot, rotation=45)
        ax.set_title(f"Parallel Coordinates Plot (Top {top_n} trials highlighted)")

        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, trials: List[Dict[str, Any]],
                               target_metric: str = 'score',
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a correlation heatmap of hyperparameters and the target metric.

        Args:
            trials: List of trial dictionaries
            target_metric: The target metric
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not trials:
            logger.warning("No trials provided for correlation heatmap")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No trials available", ha='center', va='center')
            return fig

        # Convert trials to DataFrame
        df_list = []
        for trial in trials:
            row = trial.get('parameters', {}).copy()
            row[target_metric] = trial.get(target_metric, trial.get('score', 0))
            df_list.append(row)

        df = pd.DataFrame(df_list)

        if df.empty or len(df.columns) <= 1:
            logger.warning("Insufficient data for correlation heatmap")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
            return fig

        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            logger.warning("Insufficient numeric data for correlation heatmap")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Insufficient numeric data", ha='center', va='center')
            return fig

        # Calculate correlation matrix
        corr = df[numeric_cols].corr()

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1, 
                   fmt='.2f', square=True, linewidths=0.5)
        ax.set_title("Parameter Correlation Heatmap")

        plt.tight_layout()
        return fig

    def plot_interaction_heatmap(self, trials: List[Dict[str, Any]],
                                 target_metric: str = 'score',
                                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot pairwise parameter interaction heatmap.
        
        Args:
            trials: List of trial dictionaries
            target_metric: Target metric name
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not trials or len(trials) < 10:
            logger.warning("Insufficient trials for interaction analysis")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Insufficient trials", ha='center', va='center')
            return fig
        
        # Prepare data
        X, y, param_names = self._prepare_data(trials, target_metric)
        
        if len(param_names) < 2:
            logger.warning("Need at least 2 parameters for interaction analysis")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Need at least 2 parameters", ha='center', va='center')
            return fig
        
        # Calculate interaction strengths using RF feature importance on cross-products
        try:
            from sklearn.preprocessing import PolynomialFeatures
            
            # Create interaction features
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_interactions = poly.fit_transform(X)
            
            # Get interaction feature names
            feature_names = poly.get_feature_names_out(param_names)
            
            # Train model on interactions
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_interactions, y)
            
            # Extract interaction importances
            interaction_matrix = np.zeros((len(param_names), len(param_names)))
            
            for i, feat_name in enumerate(feature_names):
                if ' ' in feat_name:  # Interaction term
                    parts = feat_name.split(' ')
                    if len(parts) == 2:
                        idx1 = param_names.index(parts[0])
                        idx2 = param_names.index(parts[1])
                        interaction_matrix[idx1, idx2] = model.feature_importances_[i]
                        interaction_matrix[idx2, idx1] = model.feature_importances_[i]
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(interaction_matrix, annot=True, cmap='YlOrRd', 
                       xticklabels=param_names, yticklabels=param_names,
                       ax=ax, fmt='.3f', square=True, linewidths=0.5)
            ax.set_title("Parameter Interaction Strength")
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error in interaction analysis: {str(e)}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return fig

    def plot_marginal_effects(self, trials: List[Dict[str, Any]],
                             target_metric: str = 'score',
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot marginal effects of each parameter on the target metric.
        
        Args:
            trials: List of trial dictionaries
            target_metric: Target metric name
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not trials:
            logger.warning("No trials for marginal effects plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No trials available", ha='center', va='center')
            return fig
        
        # Prepare data
        X, y, param_names = self._prepare_data(trials, target_metric)
        
        if len(param_names) == 0:
            logger.warning("No parameters to plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No parameters available", ha='center', va='center')
            return fig
        
        # Create subplots
        n_params = len(param_names)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        for idx, param in enumerate(param_names):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Get parameter values
            param_values = X[:, idx]
            
            # Create scatter plot with smoothing
            ax.scatter(param_values, y, alpha=0.5)
            
            # Add smoothing curve
            try:
                from scipy.interpolate import make_interp_spline
                if len(np.unique(param_values)) > 3:
                    x_sorted = np.sort(param_values)
                    indices = np.argsort(param_values)
                    y_sorted = y[indices]
                    
                    # Use moving average for smoothing
                    window = max(3, len(x_sorted) // 10)
                    y_smooth = np.convolve(y_sorted, np.ones(window)/window, mode='valid')
                    x_smooth = x_sorted[window//2:-(window//2)+1] if window > 1 else x_sorted
                    
                    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, alpha=0.7)
            except Exception:
                pass
            
            ax.set_xlabel(param)
            ax.set_ylabel(target_metric)
            ax.set_title(f'{param} effect')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_params, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('Marginal Effects of Parameters', fontsize=14, y=1.00)
        plt.tight_layout()
        return fig
