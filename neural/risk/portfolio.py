"""
Portfolio Optimization and Correlation Analysis

This module provides sophisticated portfolio construction and optimization
capabilities for multi-strategy and multi-asset trading:

- Modern Portfolio Theory (MPT) optimization
- Risk parity allocation methods
- Correlation analysis and clustering
- Factor exposure analysis
- Dynamic rebalancing algorithms
- Portfolio risk attribution

The optimization considers both return expectations and risk characteristics
to construct portfolios that maximize risk-adjusted returns while
managing correlation and concentration risk.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import warnings

from neural.strategy.base import Signal

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Portfolio allocation methods."""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    MAX_DIVERSIFICATION = "max_diversification"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    SIGNAL_WEIGHTED = "signal_weighted"


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""
    method: AllocationMethod
    weights: Dict[str, float]  # asset_id -> weight
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown_estimate: float
    diversification_ratio: float
    concentration_risk: float  # Herfindahl index
    correlation_risk: float
    optimization_success: bool
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationAnalysis:
    """Correlation analysis results."""
    correlation_matrix: pd.DataFrame
    distance_matrix: pd.DataFrame
    clusters: Dict[int, List[str]]  # cluster_id -> [asset_ids]
    cluster_correlations: Dict[int, float]  # avg intra-cluster correlation
    diversification_potential: float  # 0-1 score
    concentration_assets: List[str]  # highly correlated assets
    independent_assets: List[str]  # low correlation assets
    risk_factors: Dict[str, float]  # factor loadings if available


class CorrelationAnalyzer:
    """
    Analyzes correlations and clustering in returns/signals.
    
    This class helps identify correlation patterns that impact
    portfolio diversification and risk management decisions.
    """
    
    def __init__(
        self,
        lookback_period: int = 60,
        correlation_threshold: float = 0.7,
        min_cluster_size: int = 2
    ):
        """
        Initialize correlation analyzer.
        
        Args:
            lookback_period: Days of history to analyze
            correlation_threshold: High correlation threshold
            min_cluster_size: Minimum assets per cluster
        """
        self.lookback_period = lookback_period
        self.correlation_threshold = correlation_threshold
        self.min_cluster_size = min_cluster_size
        
    def analyze_correlations(
        self,
        returns_data: pd.DataFrame,
        method: str = "pearson"
    ) -> CorrelationAnalysis:
        """
        Analyze correlation structure of returns.
        
        Args:
            returns_data: DataFrame with returns (assets as columns)
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            CorrelationAnalysis with comprehensive results
        """
        if returns_data.empty or returns_data.shape[1] < 2:
            raise ValueError("Need at least 2 assets with returns data")
        
        # Calculate correlation matrix
        correlation_matrix = returns_data.corr(method=method).fillna(0)
        
        # Convert to distance matrix for clustering
        distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
        
        # Perform hierarchical clustering
        clusters = self._perform_clustering(correlation_matrix, distance_matrix)
        
        # Analyze cluster characteristics
        cluster_correlations = self._analyze_cluster_correlations(
            correlation_matrix, clusters
        )
        
        # Identify concentration and independent assets
        concentration_assets = self._identify_concentration_assets(correlation_matrix)
        independent_assets = self._identify_independent_assets(correlation_matrix)
        
        # Calculate diversification potential
        diversification_potential = self._calculate_diversification_potential(
            correlation_matrix
        )
        
        return CorrelationAnalysis(
            correlation_matrix=correlation_matrix,
            distance_matrix=pd.DataFrame(distance_matrix, 
                                       index=correlation_matrix.index,
                                       columns=correlation_matrix.columns),
            clusters=clusters,
            cluster_correlations=cluster_correlations,
            diversification_potential=diversification_potential,
            concentration_assets=concentration_assets,
            independent_assets=independent_assets,
            risk_factors={}  # Could be extended with factor analysis
        )
    
    def _perform_clustering(
        self, 
        correlation_matrix: pd.DataFrame, 
        distance_matrix: np.ndarray
    ) -> Dict[int, List[str]]:
        """Perform hierarchical clustering on correlation matrix."""
        if correlation_matrix.shape[0] < self.min_cluster_size:
            return {0: list(correlation_matrix.index)}
        
        try:
            # Convert to condensed distance matrix
            condensed_distances = squareform(distance_matrix, checks=False)
            
            # Perform linkage
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Form clusters
            n_clusters = max(2, min(correlation_matrix.shape[0] // self.min_cluster_size, 5))
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Group assets by cluster
            clusters = {}
            for i, asset in enumerate(correlation_matrix.index):
                cluster_id = cluster_labels[i] - 1  # Convert to 0-based
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(asset)
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, using single cluster")
            return {0: list(correlation_matrix.index)}
    
    def _analyze_cluster_correlations(
        self,
        correlation_matrix: pd.DataFrame,
        clusters: Dict[int, List[str]]
    ) -> Dict[int, float]:
        """Calculate average intra-cluster correlations."""
        cluster_correlations = {}
        
        for cluster_id, assets in clusters.items():
            if len(assets) < 2:
                cluster_correlations[cluster_id] = 0.0
                continue
            
            # Get correlations within cluster
            cluster_corr = correlation_matrix.loc[assets, assets]
            
            # Calculate average correlation (excluding diagonal)
            mask = ~np.eye(cluster_corr.shape[0], dtype=bool)
            avg_correlation = cluster_corr.values[mask].mean()
            
            cluster_correlations[cluster_id] = avg_correlation
        
        return cluster_correlations
    
    def _identify_concentration_assets(
        self, 
        correlation_matrix: pd.DataFrame
    ) -> List[str]:
        """Identify assets with high average correlations."""
        avg_correlations = correlation_matrix.abs().mean()
        return avg_correlations[avg_correlations > self.correlation_threshold].index.tolist()
    
    def _identify_independent_assets(
        self, 
        correlation_matrix: pd.DataFrame
    ) -> List[str]:
        """Identify assets with low average correlations."""
        avg_correlations = correlation_matrix.abs().mean()
        independence_threshold = min(0.3, self.correlation_threshold * 0.5)
        return avg_correlations[avg_correlations < independence_threshold].index.tolist()
    
    def _calculate_diversification_potential(
        self, 
        correlation_matrix: pd.DataFrame
    ) -> float:
        """Calculate diversification potential (0-1 scale)."""
        avg_correlation = correlation_matrix.abs().values[
            ~np.eye(correlation_matrix.shape[0], dtype=bool)
        ].mean()
        
        # Higher average correlation = lower diversification potential
        return max(0.0, 1.0 - avg_correlation)


class RiskParityOptimizer:
    """
    Risk parity optimization for equal risk contribution.
    
    Constructs portfolios where each asset contributes equally
    to portfolio risk, leading to better diversification.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,
        max_weight: float = 0.4,
        min_weight: float = 0.01
    ):
        """
        Initialize risk parity optimizer.
        
        Args:
            target_volatility: Target portfolio volatility
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
        """
        self.target_volatility = target_volatility
        self.max_weight = max_weight
        self.min_weight = min_weight
    
    def optimize(
        self,
        covariance_matrix: pd.DataFrame,
        assets: List[str]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize for risk parity allocation.
        
        Args:
            covariance_matrix: Asset covariance matrix
            assets: List of asset identifiers
            
        Returns:
            Tuple of (weights, optimization_info)
        """
        n_assets = len(assets)
        
        if n_assets < 2:
            return np.array([1.0]), {'success': True, 'method': 'single_asset'}
        
        # Initial guess: equal weights
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        try:
            # Optimize
            result = optimize.minimize(
                fun=self._risk_parity_objective,
                x0=x0,
                args=(covariance_matrix.values,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if result.success:
                return result.x, {
                    'success': True,
                    'risk_contributions': self._calculate_risk_contributions(
                        result.x, covariance_matrix.values
                    ),
                    'iterations': result.nit,
                    'objective_value': result.fun
                }
            else:
                logger.warning("Risk parity optimization failed, using equal weights")
                return x0, {'success': False, 'fallback': 'equal_weight'}
                
        except Exception as e:
            logger.error(f"Risk parity optimization error: {e}")
            return x0, {'success': False, 'error': str(e)}
    
    def _risk_parity_objective(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Objective function for risk parity: minimize sum of squared risk contribution differences."""
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        if portfolio_volatility == 0:
            return 1e6
        
        # Calculate risk contributions
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility
        risk_contrib = weights * marginal_contrib / portfolio_volatility
        
        # Target equal risk contributions
        target_contrib = 1.0 / len(weights)
        
        # Minimize sum of squared deviations from target
        return np.sum((risk_contrib - target_contrib) ** 2)
    
    def _calculate_risk_contributions(
        self, 
        weights: np.ndarray, 
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate risk contributions for each asset."""
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        if portfolio_volatility == 0:
            return weights
        
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility
        return weights * marginal_contrib / portfolio_volatility


class PortfolioOptimizer:
    """
    Main portfolio optimization orchestrator.
    
    Provides multiple optimization methods and handles the complexities
    of real-world portfolio construction with practical constraints.
    """
    
    def __init__(
        self,
        lookback_period: int = 60,
        min_weight: float = 0.01,
        max_weight: float = 0.3,
        max_concentration: float = 0.6,  # Max sum of top 3 weights
        transaction_costs: float = 0.001,  # 0.1% transaction cost
        rebalance_threshold: float = 0.05   # 5% weight drift to trigger rebalance
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            lookback_period: Historical data period for estimation
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset  
            max_concentration: Maximum concentration in top holdings
            transaction_costs: Estimated transaction costs
            rebalance_threshold: Threshold for rebalancing
        """
        self.lookback_period = lookback_period
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_concentration = max_concentration
        self.transaction_costs = transaction_costs
        self.rebalance_threshold = rebalance_threshold
        
        self.correlation_analyzer = CorrelationAnalyzer(lookback_period)
        self.risk_parity_optimizer = RiskParityOptimizer(
            max_weight=max_weight, min_weight=min_weight
        )
        
        logger.info("Initialized PortfolioOptimizer")
    
    def optimize_portfolio(
        self,
        method: AllocationMethod,
        signals: List[Signal],
        returns_data: pd.DataFrame = None,
        current_weights: Dict[str, float] = None
    ) -> PortfolioAllocation:
        """
        Optimize portfolio allocation using specified method.
        
        Args:
            method: Allocation method to use
            signals: List of trading signals
            returns_data: Historical returns for optimization
            current_weights: Current portfolio weights
            
        Returns:
            PortfolioAllocation with optimal weights and metrics
        """
        if not signals:
            raise ValueError("No signals provided for optimization")
        
        asset_ids = [signal.market_id for signal in signals]
        
        try:
            if method == AllocationMethod.EQUAL_WEIGHT:
                return self._equal_weight_allocation(signals)
            
            elif method == AllocationMethod.SIGNAL_WEIGHTED:
                return self._signal_weighted_allocation(signals)
            
            elif method == AllocationMethod.RISK_PARITY:
                return self._risk_parity_allocation(signals, returns_data)
            
            elif method == AllocationMethod.MIN_VARIANCE:
                return self._min_variance_allocation(signals, returns_data)
            
            elif method == AllocationMethod.MAX_SHARPE:
                return self._max_sharpe_allocation(signals, returns_data)
            
            elif method == AllocationMethod.HIERARCHICAL_RISK_PARITY:
                return self._hierarchical_risk_parity_allocation(signals, returns_data)
            
            else:
                logger.warning(f"Method {method} not implemented, using equal weight")
                return self._equal_weight_allocation(signals)
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return self._fallback_allocation(signals)
    
    def _equal_weight_allocation(self, signals: List[Signal]) -> PortfolioAllocation:
        """Simple equal weight allocation."""
        n_assets = len(signals)
        weight = 1.0 / n_assets
        
        weights = {signal.market_id: weight for signal in signals}
        
        return PortfolioAllocation(
            method=AllocationMethod.EQUAL_WEIGHT,
            weights=weights,
            expected_return=0.0,  # Would need expected returns
            expected_volatility=0.0,  # Would need covariance matrix
            sharpe_ratio=0.0,
            max_drawdown_estimate=0.0,
            diversification_ratio=1.0,
            concentration_risk=1.0 / n_assets,  # Herfindahl index
            correlation_risk=0.0,
            optimization_success=True,
            metadata={'n_assets': n_assets}
        )
    
    def _signal_weighted_allocation(self, signals: List[Signal]) -> PortfolioAllocation:
        """Weight by signal strength/confidence."""
        # Calculate weights based on signal confidence and edge
        weights_raw = []
        asset_ids = []
        
        for signal in signals:
            confidence = signal.confidence
            edge = getattr(signal, 'edge', 0.0)
            
            # Combine confidence and edge for weight
            weight = confidence * max(0, 1 + edge)  # Boost for positive edge
            weights_raw.append(weight)
            asset_ids.append(signal.market_id)
        
        # Normalize weights
        total_weight = sum(weights_raw)
        if total_weight == 0:
            return self._equal_weight_allocation(signals)
        
        weights = {
            asset_id: w / total_weight 
            for asset_id, w in zip(asset_ids, weights_raw)
        }
        
        # Apply constraints
        weights = self._apply_weight_constraints(weights)
        
        # Calculate concentration risk
        concentration_risk = sum(w**2 for w in weights.values())
        
        return PortfolioAllocation(
            method=AllocationMethod.SIGNAL_WEIGHTED,
            weights=weights,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown_estimate=0.0,
            diversification_ratio=0.0,
            concentration_risk=concentration_risk,
            correlation_risk=0.0,
            optimization_success=True,
            metadata={'total_raw_weight': total_weight}
        )
    
    def _risk_parity_allocation(
        self, 
        signals: List[Signal], 
        returns_data: pd.DataFrame = None
    ) -> PortfolioAllocation:
        """Risk parity allocation."""
        if returns_data is None or returns_data.empty:
            logger.warning("No returns data for risk parity, using signal weighting")
            return self._signal_weighted_allocation(signals)
        
        asset_ids = [signal.market_id for signal in signals]
        
        # Filter returns data to available assets
        available_assets = [aid for aid in asset_ids if aid in returns_data.columns]
        if len(available_assets) < 2:
            return self._signal_weighted_allocation(signals)
        
        # Calculate covariance matrix
        returns_subset = returns_data[available_assets].dropna()
        if len(returns_subset) < 10:  # Need minimum observations
            return self._signal_weighted_allocation(signals)
        
        cov_matrix = returns_subset.cov()
        
        # Optimize
        optimal_weights, opt_info = self.risk_parity_optimizer.optimize(
            cov_matrix, available_assets
        )
        
        # Create weights dictionary
        weights = {asset: 0.0 for asset in asset_ids}
        for i, asset in enumerate(available_assets):
            weights[asset] = optimal_weights[i]
        
        # Calculate portfolio metrics
        portfolio_vol = np.sqrt(
            np.dot(optimal_weights, np.dot(cov_matrix.values, optimal_weights))
        )
        
        return PortfolioAllocation(
            method=AllocationMethod.RISK_PARITY,
            weights=weights,
            expected_return=0.0,
            expected_volatility=portfolio_vol,
            sharpe_ratio=0.0,
            max_drawdown_estimate=0.0,
            diversification_ratio=self._calculate_diversification_ratio(
                optimal_weights, cov_matrix.values
            ),
            concentration_risk=sum(w**2 for w in optimal_weights),
            correlation_risk=0.0,
            optimization_success=opt_info.get('success', False),
            metadata=opt_info
        )
    
    def _min_variance_allocation(
        self, 
        signals: List[Signal], 
        returns_data: pd.DataFrame = None
    ) -> PortfolioAllocation:
        """Minimum variance portfolio allocation."""
        # Placeholder implementation
        logger.warning("Minimum variance optimization not fully implemented")
        return self._risk_parity_allocation(signals, returns_data)
    
    def _max_sharpe_allocation(
        self, 
        signals: List[Signal], 
        returns_data: pd.DataFrame = None
    ) -> PortfolioAllocation:
        """Maximum Sharpe ratio allocation."""
        # Placeholder implementation  
        logger.warning("Maximum Sharpe optimization not fully implemented")
        return self._signal_weighted_allocation(signals)
    
    def _hierarchical_risk_parity_allocation(
        self, 
        signals: List[Signal], 
        returns_data: pd.DataFrame = None
    ) -> PortfolioAllocation:
        """Hierarchical risk parity allocation."""
        if returns_data is None or returns_data.empty:
            return self._signal_weighted_allocation(signals)
        
        asset_ids = [signal.market_id for signal in signals]
        available_assets = [aid for aid in asset_ids if aid in returns_data.columns]
        
        if len(available_assets) < 3:
            return self._risk_parity_allocation(signals, returns_data)
        
        try:
            # Perform correlation analysis and clustering
            returns_subset = returns_data[available_assets].dropna()
            correlation_analysis = self.correlation_analyzer.analyze_correlations(returns_subset)
            
            # Allocate within and across clusters
            weights = self._allocate_hierarchically(
                correlation_analysis, available_assets, returns_subset
            )
            
            # Extend to all assets
            full_weights = {asset: 0.0 for asset in asset_ids}
            full_weights.update(weights)
            
            return PortfolioAllocation(
                method=AllocationMethod.HIERARCHICAL_RISK_PARITY,
                weights=full_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown_estimate=0.0,
                diversification_ratio=correlation_analysis.diversification_potential,
                concentration_risk=sum(w**2 for w in weights.values()),
                correlation_risk=1.0 - correlation_analysis.diversification_potential,
                optimization_success=True,
                metadata={'n_clusters': len(correlation_analysis.clusters)}
            )
            
        except Exception as e:
            logger.warning(f"Hierarchical risk parity failed: {e}")
            return self._risk_parity_allocation(signals, returns_data)
    
    def _allocate_hierarchically(
        self,
        correlation_analysis: CorrelationAnalysis,
        available_assets: List[str],
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Allocate weights hierarchically based on clusters."""
        clusters = correlation_analysis.clusters
        n_clusters = len(clusters)
        
        if n_clusters <= 1:
            # Single cluster, use risk parity
            cov_matrix = returns_data.cov()
            weights, _ = self.risk_parity_optimizer.optimize(cov_matrix, available_assets)
            return {asset: weights[i] for i, asset in enumerate(available_assets)}
        
        # Equal weight across clusters initially
        cluster_weights = {cluster_id: 1.0 / n_clusters for cluster_id in clusters.keys()}
        
        # Within each cluster, use risk parity
        final_weights = {}
        
        for cluster_id, assets_in_cluster in clusters.items():
            if len(assets_in_cluster) == 1:
                # Single asset cluster
                final_weights[assets_in_cluster[0]] = cluster_weights[cluster_id]
            else:
                # Multiple assets, optimize within cluster
                cluster_returns = returns_data[assets_in_cluster]
                cluster_cov = cluster_returns.cov()
                
                cluster_opt_weights, _ = self.risk_parity_optimizer.optimize(
                    cluster_cov, assets_in_cluster
                )
                
                # Scale by cluster weight
                for i, asset in enumerate(assets_in_cluster):
                    final_weights[asset] = cluster_opt_weights[i] * cluster_weights[cluster_id]
        
        return final_weights
    
    def _apply_weight_constraints(
        self, 
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply weight constraints (min/max/concentration)."""
        # Apply min/max weight constraints
        constrained_weights = {}
        total_adjustment = 0.0
        
        for asset, weight in weights.items():
            if weight < self.min_weight:
                constrained_weights[asset] = self.min_weight
                total_adjustment += self.min_weight - weight
            elif weight > self.max_weight:
                constrained_weights[asset] = self.max_weight
                total_adjustment += self.max_weight - weight
            else:
                constrained_weights[asset] = weight
        
        # Renormalize if needed
        total_weight = sum(constrained_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            for asset in constrained_weights:
                constrained_weights[asset] /= total_weight
        
        return constrained_weights
    
    def _calculate_diversification_ratio(
        self, 
        weights: np.ndarray, 
        cov_matrix: np.ndarray
    ) -> float:
        """Calculate diversification ratio (weighted avg vol / portfolio vol)."""
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = np.dot(weights, individual_vols)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        if portfolio_vol == 0:
            return 1.0
        
        return weighted_avg_vol / portfolio_vol
    
    def _fallback_allocation(self, signals: List[Signal]) -> PortfolioAllocation:
        """Fallback allocation when optimization fails."""
        return self._equal_weight_allocation(signals)
    
    def should_rebalance(
        self, 
        target_weights: Dict[str, float], 
        current_weights: Dict[str, float]
    ) -> bool:
        """Determine if portfolio should be rebalanced."""
        max_drift = 0.0
        
        for asset in target_weights:
            target = target_weights.get(asset, 0.0)
            current = current_weights.get(asset, 0.0)
            drift = abs(target - current)
            max_drift = max(max_drift, drift)
        
        return max_drift > self.rebalance_threshold
