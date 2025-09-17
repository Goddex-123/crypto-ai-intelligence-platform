"""
Advanced Portfolio Optimization for Cryptocurrency Trading
Implements Modern Portfolio Theory, Risk Parity, and Dynamic Rebalancing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
import cvxpy as cp
from scipy.optimize import minimize, Bounds
from sklearn.covariance import LedoitWolf, empirical_covariance

# Risk management
from scipy.stats import norm, jarque_bera, kurtosis, skew
import scipy.stats as stats

# Modern Portfolio Theory implementation
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation
    from pypfopt.cla import CLA
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    print("PyPortfolioOpt not available - using custom implementation")


class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple strategies and risk models
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.prices = None
        self.returns = None
        self.weights = None
        self.expected_returns = None
        self.cov_matrix = None
        self.risk_metrics = {}
        
    def _get_default_config(self) -> Dict:
        return {
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'max_weight': 0.4,  # Maximum weight per asset (40%)
            'min_weight': 0.01,  # Minimum weight per asset (1%)
            'rebalance_frequency': 'monthly',  # 'daily', 'weekly', 'monthly'
            'lookback_window': 252,  # Trading days for historical data
            'confidence_level': 0.05,  # 95% confidence for VaR/CVaR
            'transaction_cost': 0.001,  # 0.1% transaction cost
            'target_volatility': 0.15,  # 15% annual volatility target
            'black_litterman_tau': 0.025,  # Black-Litterman uncertainty parameter
        }
    
    def load_price_data(self, prices: pd.DataFrame) -> None:
        """
        Load price data and calculate returns
        """
        self.prices = prices.copy()
        self.returns = prices.pct_change().dropna()
        
        print(f"Loaded price data: {self.prices.shape}")
        print(f"Calculated returns: {self.returns.shape}")
        print(f"Assets: {list(self.prices.columns)}")
        
    def calculate_expected_returns(self, method: str = 'mean_historical') -> pd.Series:
        """
        Calculate expected returns using various methods
        """
        if self.returns is None:
            raise ValueError("Price data not loaded. Call load_price_data() first.")
        
        if method == 'mean_historical':
            # Simple historical mean
            expected_rets = self.returns.mean() * 252  # Annualized
            
        elif method == 'exponential_weighted':
            # Exponentially weighted moving average
            alpha = 0.94  # Decay factor
            expected_rets = self.returns.ewm(alpha=alpha).mean().iloc[-1] * 252
            
        elif method == 'capm':
            # CAPM expected returns (simplified - assuming market proxy)
            market_returns = self.returns.mean(axis=1)  # Equal-weighted market
            market_premium = market_returns.mean() * 252 - self.config['risk_free_rate']
            
            expected_rets = pd.Series(index=self.returns.columns)
            for asset in self.returns.columns:
                beta = self.returns[asset].cov(market_returns) / market_returns.var()
                expected_rets[asset] = self.config['risk_free_rate'] + beta * market_premium
                
        elif method == 'black_litterman':
            expected_rets = self._black_litterman_returns()
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.expected_returns = expected_rets
        return expected_rets
    
    def calculate_covariance_matrix(self, method: str = 'sample') -> pd.DataFrame:
        """
        Calculate covariance matrix using various estimators
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Call load_price_data() first.")
        
        returns_array = self.returns.values
        
        if method == 'sample':
            # Sample covariance matrix
            cov_matrix = self.returns.cov() * 252  # Annualized
            
        elif method == 'ledoit_wolf':
            # Ledoit-Wolf shrinkage estimator
            lw = LedoitWolf()
            cov_array, _ = lw.fit(returns_array).covariance_, lw.shrinkage_
            cov_matrix = pd.DataFrame(cov_array * 252, 
                                    index=self.returns.columns,
                                    columns=self.returns.columns)
            
        elif method == 'exponential_weighted':
            # Exponentially weighted covariance
            alpha = 0.94
            cov_matrix = self.returns.ewm(alpha=alpha).cov().iloc[-len(self.returns.columns):] * 252
            
        elif method == 'robust':
            # Robust covariance estimator
            from sklearn.covariance import MinCovDet
            robust_cov = MinCovDet().fit(returns_array)
            cov_array = robust_cov.covariance_
            cov_matrix = pd.DataFrame(cov_array * 252,
                                    index=self.returns.columns,
                                    columns=self.returns.columns)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.cov_matrix = cov_matrix
        return cov_matrix
    
    def optimize_portfolio(self, objective: str = 'max_sharpe',
                          constraints: List[Dict] = None,
                          expected_returns: pd.Series = None,
                          cov_matrix: pd.DataFrame = None) -> Dict:
        """
        Optimize portfolio using various objectives
        """
        # Use provided or stored expected returns and covariance
        if expected_returns is None:
            if self.expected_returns is None:
                expected_returns = self.calculate_expected_returns()
            else:
                expected_returns = self.expected_returns
                
        if cov_matrix is None:
            if self.cov_matrix is None:
                cov_matrix = self.calculate_covariance_matrix()
            else:
                cov_matrix = self.cov_matrix
        
        n_assets = len(expected_returns)
        
        # Decision variables (portfolio weights)
        weights = cp.Variable(n_assets)
        
        # Objective functions
        portfolio_return = expected_returns.values @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        portfolio_risk = cp.sqrt(portfolio_variance)
        
        # Basic constraints
        constraints_list = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= self.config['min_weight'],  # Minimum weight
            weights <= self.config['max_weight']   # Maximum weight
        ]
        
        # Additional constraints
        if constraints:
            for constraint in constraints:
                if constraint['type'] == 'sector_limit':
                    # Sector exposure limits
                    sector_weights = constraint['sector_weights']
                    sector_limit = constraint['limit']
                    constraints_list.append(
                        cp.sum(cp.multiply(sector_weights, weights)) <= sector_limit
                    )
                elif constraint['type'] == 'turnover':
                    # Portfolio turnover constraint
                    if hasattr(self, 'previous_weights'):
                        turnover = cp.sum(cp.abs(weights - self.previous_weights))
                        constraints_list.append(turnover <= constraint['max_turnover'])
        
        # Solve optimization problem
        if objective == 'max_sharpe':
            # Maximize Sharpe ratio (equivalent to maximizing return/risk)
            # We use the transformation: max(μ'w / sqrt(w'Σw)) = min(w'Σw) s.t. μ'w = 1
            prob = cp.Problem(
                cp.Minimize(portfolio_variance),
                constraints_list + [portfolio_return == 1]
            )
            
        elif objective == 'min_volatility':
            # Minimize portfolio volatility
            prob = cp.Problem(cp.Minimize(portfolio_risk), constraints_list)
            
        elif objective == 'max_return':
            # Maximize expected return (with volatility constraint)
            target_vol = self.config.get('target_volatility', 0.15)
            prob = cp.Problem(
                cp.Maximize(portfolio_return),
                constraints_list + [portfolio_risk <= target_vol]
            )
            
        elif objective == 'risk_parity':
            # Risk parity optimization
            return self._optimize_risk_parity(cov_matrix)
            
        elif objective == 'max_diversification':
            # Maximum diversification ratio
            asset_vols = np.sqrt(np.diag(cov_matrix.values))
            weighted_avg_vol = asset_vols @ weights
            prob = cp.Problem(
                cp.Maximize(weighted_avg_vol / portfolio_risk),
                constraints_list
            )
            
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Solve the problem
        prob.solve(verbose=False)
        
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"Optimization failed: {prob.status}")
        
        # Extract optimal weights
        optimal_weights = weights.value
        
        # Normalize weights (handle numerical errors)
        optimal_weights = np.maximum(optimal_weights, 0)  # Remove negative weights
        optimal_weights = optimal_weights / np.sum(optimal_weights)  # Normalize
        
        # Handle special case for max_sharpe (need to normalize)
        if objective == 'max_sharpe':
            optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Create results dictionary
        weights_dict = dict(zip(expected_returns.index, optimal_weights))
        
        # Calculate portfolio metrics
        port_return = np.dot(optimal_weights, expected_returns.values)
        port_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix.values, optimal_weights)))
        sharpe_ratio = (port_return - self.config['risk_free_rate']) / port_vol if port_vol > 0 else 0
        
        result = {
            'weights': weights_dict,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe_ratio,
            'objective': objective,
            'status': prob.status
        }
        
        self.weights = weights_dict
        return result
    
    def _optimize_risk_parity(self, cov_matrix: pd.DataFrame) -> Dict:
        """
        Risk parity optimization using scipy
        """
        n_assets = len(cov_matrix)
        
        def risk_parity_objective(weights, cov_matrix):
            """
            Risk parity objective function - minimize sum of squared risk contribution differences
            """
            weights = np.array(weights)
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = np.multiply(weights, marginal_contrib)
            contrib = contrib / portfolio_var
            
            # Target: equal risk contribution (1/n each)
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints and bounds
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        bounds = [(self.config['min_weight'], self.config['max_weight']) 
                 for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            args=(cov_matrix.values,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            raise ValueError(f"Risk parity optimization failed: {result.message}")
        
        optimal_weights = result.x
        weights_dict = dict(zip(cov_matrix.index, optimal_weights))
        
        # Calculate metrics
        port_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix.values, optimal_weights)))
        
        return {
            'weights': weights_dict,
            'expected_return': np.dot(optimal_weights, self.expected_returns.values) if self.expected_returns is not None else 0,
            'volatility': port_vol,
            'sharpe_ratio': 0,  # Not applicable for risk parity
            'objective': 'risk_parity',
            'status': 'optimal'
        }
    
    def _black_litterman_returns(self) -> pd.Series:
        """
        Calculate Black-Litterman expected returns
        """
        if PYPFOPT_AVAILABLE:
            # Use PyPortfolioOpt implementation if available
            market_caps = pd.Series(
                np.random.exponential(1e9, len(self.returns.columns)),
                index=self.returns.columns
            )  # Mock market caps
            
            S = risk_models.CovarianceShrinkage(self.returns).ledoit_wolf()
            delta = expected_returns.market_implied_risk_aversion(
                market_caps.values, risk_aversion=1
            )
            
            market_prior = expected_returns.market_implied_prior_returns(
                market_caps, delta, S
            )
            
            return market_prior
        else:
            # Simple implementation
            return self.returns.mean() * 252
    
    def calculate_risk_metrics(self, weights: Dict = None, 
                              returns: pd.DataFrame = None) -> Dict:
        """
        Calculate comprehensive risk metrics for the portfolio
        """
        if weights is None:
            if self.weights is None:
                raise ValueError("No weights provided or stored")
            weights = self.weights
            
        if returns is None:
            returns = self.returns
            
        # Convert weights to array
        weight_array = np.array([weights[asset] for asset in returns.columns])
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weight_array).sum(axis=1)
        
        # Basic statistics
        metrics = {
            'annual_return': portfolio_returns.mean() * 252,
            'annual_volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252 - self.config['risk_free_rate']) / 
                          (portfolio_returns.std() * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'var_95': np.percentile(portfolio_returns, 5),
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
            'skewness': skew(portfolio_returns),
            'kurtosis': kurtosis(portfolio_returns),
            'downside_deviation': self._calculate_downside_deviation(portfolio_returns),
            'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns),
            'calmar_ratio': self._calculate_calmar_ratio(portfolio_returns),
        }
        
        # Advanced risk metrics
        metrics.update({
            'tracking_error': self._calculate_tracking_error(portfolio_returns),
            'information_ratio': self._calculate_information_ratio(portfolio_returns),
            'omega_ratio': self._calculate_omega_ratio(portfolio_returns),
            'tail_ratio': self._calculate_tail_ratio(portfolio_returns),
        })
        
        self.risk_metrics = metrics
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_downside_deviation(self, returns: pd.Series, 
                                    target: float = 0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target]
        return downside_returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, 
                               target: float = 0) -> float:
        """Calculate Sortino ratio"""
        excess_return = returns.mean() * 252 - target
        downside_dev = self._calculate_downside_deviation(returns, target)
        return excess_return / downside_dev if downside_dev > 0 else 0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = returns.mean() * 252
        max_dd = abs(self._calculate_max_drawdown(returns))
        return annual_return / max_dd if max_dd > 0 else 0
    
    def _calculate_tracking_error(self, portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series = None) -> float:
        """Calculate tracking error vs benchmark"""
        if benchmark_returns is None:
            # Use equal-weighted benchmark
            benchmark_returns = self.returns.mean(axis=1)
        
        active_returns = portfolio_returns - benchmark_returns
        return active_returns.std() * np.sqrt(252)
    
    def _calculate_information_ratio(self, portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series = None) -> float:
        """Calculate information ratio"""
        if benchmark_returns is None:
            benchmark_returns = self.returns.mean(axis=1)
        
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        active_return = active_returns.mean() * 252
        
        return active_return / tracking_error if tracking_error > 0 else 0
    
    def _calculate_omega_ratio(self, returns: pd.Series, 
                             threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        returns_above = returns[returns > threshold]
        returns_below = returns[returns <= threshold]
        
        if len(returns_below) == 0:
            return np.inf
        
        gains = (returns_above - threshold).sum()
        losses = (threshold - returns_below).sum()
        
        return gains / losses if losses > 0 else np.inf
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return abs(p95 / p5) if p5 != 0 else np.inf
    
    def create_efficient_frontier(self, num_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier
        """
        if self.expected_returns is None:
            self.calculate_expected_returns()
        if self.cov_matrix is None:
            self.calculate_covariance_matrix()
        
        # Range of target returns
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            try:
                # Optimize for minimum volatility at target return
                n_assets = len(self.expected_returns)
                weights = cp.Variable(n_assets)
                
                portfolio_return = self.expected_returns.values @ weights
                portfolio_variance = cp.quad_form(weights, self.cov_matrix.values)
                
                constraints = [
                    cp.sum(weights) == 1,
                    weights >= self.config['min_weight'],
                    weights <= self.config['max_weight'],
                    portfolio_return == target_ret
                ]
                
                prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)
                prob.solve(verbose=False)
                
                if prob.status == 'optimal':
                    opt_weights = weights.value
                    opt_weights = np.maximum(opt_weights, 0)
                    opt_weights = opt_weights / np.sum(opt_weights)
                    
                    port_vol = np.sqrt(np.dot(opt_weights, 
                                            np.dot(self.cov_matrix.values, opt_weights)))
                    sharpe = (target_ret - self.config['risk_free_rate']) / port_vol
                    
                    efficient_portfolios.append({
                        'return': target_ret,
                        'volatility': port_vol,
                        'sharpe_ratio': sharpe,
                        'weights': dict(zip(self.expected_returns.index, opt_weights))
                    })
                    
            except Exception as e:
                continue
        
        return pd.DataFrame(efficient_portfolios)
    
    def backtest_strategy(self, rebalance_frequency: str = 'monthly',
                         start_date: str = None, end_date: str = None,
                         initial_capital: float = 100000) -> Dict:
        """
        Backtest the portfolio optimization strategy
        """
        if self.prices is None:
            raise ValueError("Price data not loaded")
        
        # Filter data by date range if provided
        if start_date or end_date:
            mask = pd.Series(True, index=self.prices.index)
            if start_date:
                mask &= (self.prices.index >= start_date)
            if end_date:
                mask &= (self.prices.index <= end_date)
            prices = self.prices[mask]
            returns = prices.pct_change().dropna()
        else:
            prices = self.prices
            returns = self.returns
        
        # Rebalancing dates
        if rebalance_frequency == 'daily':
            rebal_dates = prices.index[::1]
        elif rebalance_frequency == 'weekly':
            rebal_dates = prices.index[::5]  # Every 5 trading days
        elif rebalance_frequency == 'monthly':
            rebal_dates = prices.index[::21]  # Every 21 trading days
        else:
            rebal_dates = [prices.index[0], prices.index[-1]]
        
        # Backtest results
        portfolio_values = [initial_capital]
        portfolio_weights_history = []
        transaction_costs = []
        
        current_weights = None
        
        for i, date in enumerate(rebal_dates[:-1]):
            # Get historical data up to rebalancing date
            hist_prices = prices.loc[:date]
            hist_returns = hist_prices.pct_change().dropna()
            
            if len(hist_returns) < 60:  # Need minimum history
                continue
            
            # Use only recent data for optimization
            lookback = min(self.config['lookback_window'], len(hist_returns))
            recent_returns = hist_returns.iloc[-lookback:]
            
            # Update data
            self.returns = recent_returns
            
            # Calculate expected returns and covariance
            expected_rets = self.calculate_expected_returns()
            cov_matrix = self.calculate_covariance_matrix()
            
            # Optimize portfolio
            try:
                opt_result = self.optimize_portfolio(
                    objective='max_sharpe',
                    expected_returns=expected_rets,
                    cov_matrix=cov_matrix
                )
                new_weights = opt_result['weights']
                
                # Calculate transaction costs if we have previous weights
                if current_weights is not None:
                    turnover = sum(abs(new_weights[asset] - current_weights.get(asset, 0)) 
                                 for asset in new_weights.keys())
                    transaction_cost = turnover * self.config['transaction_cost'] * portfolio_values[-1]
                    transaction_costs.append(transaction_cost)
                else:
                    transaction_costs.append(0)
                
                current_weights = new_weights
                portfolio_weights_history.append({
                    'date': date,
                    'weights': new_weights.copy()
                })
                
            except Exception as e:
                print(f"Optimization failed at {date}: {e}")
                continue
        
        # Calculate portfolio performance
        if current_weights and portfolio_weights_history:
            # Simulate portfolio returns
            portfolio_returns = []
            
            for i in range(1, len(prices)):
                date = prices.index[i]
                prev_date = prices.index[i-1]
                
                # Find applicable weights
                applicable_weights = None
                for weight_data in reversed(portfolio_weights_history):
                    if weight_data['date'] <= prev_date:
                        applicable_weights = weight_data['weights']
                        break
                
                if applicable_weights:
                    # Calculate portfolio return for this period
                    period_returns = returns.loc[date]
                    portfolio_return = sum(applicable_weights.get(asset, 0) * period_returns.get(asset, 0)
                                         for asset in applicable_weights.keys())
                    portfolio_returns.append(portfolio_return)
                else:
                    portfolio_returns.append(0)
            
            portfolio_returns = pd.Series(portfolio_returns, index=prices.index[1:])
            
            # Calculate cumulative portfolio value
            portfolio_cumulative = (1 + portfolio_returns).cumprod() * initial_capital
            
            # Performance metrics
            total_return = (portfolio_cumulative.iloc[-1] - initial_capital) / initial_capital
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe = (annual_return - self.config['risk_free_rate']) / annual_vol
            max_dd = self._calculate_max_drawdown(portfolio_returns)
            
            # Benchmark performance (equal-weighted)
            equal_weights = {asset: 1/len(prices.columns) for asset in prices.columns}
            benchmark_returns = (returns * pd.Series(equal_weights)).sum(axis=1)
            benchmark_cumulative = (1 + benchmark_returns).cumprod() * initial_capital
            benchmark_total_return = (benchmark_cumulative.iloc[-1] - initial_capital) / initial_capital
            
            backtest_results = {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'benchmark_return': benchmark_total_return,
                'excess_return': total_return - benchmark_total_return,
                'total_transaction_costs': sum(transaction_costs),
                'number_of_rebalances': len(portfolio_weights_history),
                'portfolio_values': portfolio_cumulative,
                'portfolio_returns': portfolio_returns,
                'weights_history': portfolio_weights_history
            }
            
            return backtest_results
        
        else:
            raise ValueError("Backtest failed - no valid optimization results")


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic price data for testing
    np.random.seed(42)
    
    # Create synthetic cryptocurrency prices
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    n_assets = 5
    assets = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT']
    
    # Generate correlated random walks
    n_days = len(dates)
    returns = np.random.multivariate_normal(
        mean=[0.001, 0.0008, 0.0005, 0.0012, 0.0007],  # Different expected returns
        cov=[[0.0004, 0.0002, 0.0001, 0.0002, 0.0001],  # Correlation structure
             [0.0002, 0.0006, 0.0002, 0.0003, 0.0002],
             [0.0001, 0.0002, 0.0005, 0.0002, 0.0002],
             [0.0002, 0.0003, 0.0002, 0.0008, 0.0003],
             [0.0001, 0.0002, 0.0002, 0.0003, 0.0006]],
        size=n_days
    )
    
    # Convert to prices
    prices_data = {}
    initial_prices = [50000, 3000, 1, 100, 20]  # Different starting prices
    
    for i, asset in enumerate(assets):
        cumulative_returns = np.cumprod(1 + returns[:, i])
        prices_data[asset] = initial_prices[i] * cumulative_returns
    
    prices_df = pd.DataFrame(prices_data, index=dates)
    
    print("=== Cryptocurrency Portfolio Optimization ===")
    print(f"Generated synthetic price data: {prices_df.shape}")
    print(f"Date range: {prices_df.index[0]} to {prices_df.index[-1]}")
    print(f"Assets: {assets}")
    
    # Initialize portfolio optimizer
    optimizer = AdvancedPortfolioOptimizer()
    optimizer.load_price_data(prices_df)
    
    # Test different optimization objectives
    objectives = ['max_sharpe', 'min_volatility', 'risk_parity', 'max_diversification']
    
    results = {}
    for objective in objectives:
        print(f"\n--- Optimizing for: {objective} ---")
        try:
            result = optimizer.optimize_portfolio(objective=objective)
            results[objective] = result
            
            print(f"Expected Return: {result['expected_return']:.4f}")
            print(f"Volatility: {result['volatility']:.4f}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            print("Weights:")
            for asset, weight in result['weights'].items():
                print(f"  {asset}: {weight:.4f}")
                
        except Exception as e:
            print(f"Optimization failed: {e}")
    
    # Risk metrics analysis
    if 'max_sharpe' in results:
        print(f"\n--- Risk Metrics (Max Sharpe Portfolio) ---")
        risk_metrics = optimizer.calculate_risk_metrics(results['max_sharpe']['weights'])
        
        for metric, value in risk_metrics.items():
            print(f"{metric}: {value:.6f}")
    
    # Backtest the strategy
    print(f"\n--- Backtesting Strategy ---")
    try:
        backtest_results = optimizer.backtest_strategy(
            rebalance_frequency='monthly',
            initial_capital=100000
        )
        
        print(f"Total Return: {backtest_results['total_return']:.4f}")
        print(f"Annual Return: {backtest_results['annual_return']:.4f}")
        print(f"Annual Volatility: {backtest_results['annual_volatility']:.4f}")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.4f}")
        print(f"Benchmark Return: {backtest_results['benchmark_return']:.4f}")
        print(f"Excess Return: {backtest_results['excess_return']:.4f}")
        print(f"Number of Rebalances: {backtest_results['number_of_rebalances']}")
        print(f"Total Transaction Costs: ${backtest_results['total_transaction_costs']:.2f}")
        
    except Exception as e:
        print(f"Backtesting failed: {e}")
    
    print("\nPortfolio optimization completed successfully!")
