"""
Advanced Backtesting Engine for Cryptocurrency Trading Strategies
Comprehensive performance evaluation and risk analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Statistics and metrics
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


class Position:
    """
    Represents a trading position
    """
    
    def __init__(self, symbol: str, side: str, size: float, 
                 entry_price: float, entry_time: datetime,
                 stop_loss: float = None, take_profit: float = None):
        self.symbol = symbol
        self.side = side  # 'long' or 'short'
        self.size = size
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.is_open = True
        self.pnl = 0.0
        self.pnl_pct = 0.0
        
    def update_position(self, current_price: float, current_time: datetime):
        """Update position with current market price"""
        if self.side == 'long':
            self.pnl = (current_price - self.entry_price) * self.size
            self.pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - current_price) * self.size
            self.pnl_pct = (self.entry_price - current_price) / self.entry_price
            
        # Check stop loss and take profit
        should_exit = False
        exit_reason = None
        
        if self.side == 'long':
            if self.stop_loss and current_price <= self.stop_loss:
                should_exit = True
                exit_reason = 'stop_loss'
            elif self.take_profit and current_price >= self.take_profit:
                should_exit = True
                exit_reason = 'take_profit'
        else:  # short
            if self.stop_loss and current_price >= self.stop_loss:
                should_exit = True
                exit_reason = 'stop_loss'
            elif self.take_profit and current_price <= self.take_profit:
                should_exit = True
                exit_reason = 'take_profit'
                
        if should_exit:
            self.close_position(current_price, current_time, exit_reason)
            
        return should_exit
    
    def close_position(self, exit_price: float, exit_time: datetime, reason: str = 'manual'):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.is_open = False
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.size
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.size
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'duration': (self.exit_time - self.entry_time) if self.exit_time else None,
            'is_open': self.is_open
        }


class TradingAccount:
    """
    Simulates a trading account with balance, positions, and transaction costs
    """
    
    def __init__(self, initial_balance: float, transaction_cost: float = 0.001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.transaction_cost = transaction_cost
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.transaction_log = []
        
    def place_order(self, symbol: str, side: str, size: float, 
                   price: float, timestamp: datetime,
                   stop_loss: float = None, take_profit: float = None) -> bool:
        """
        Place a trading order
        """
        # Calculate order value and transaction cost
        order_value = size * price
        cost = order_value * self.transaction_cost
        
        # Check if sufficient balance for long positions
        if side == 'long' and (order_value + cost) > self.balance:
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions.append(position)
        
        # Update balance
        if side == 'long':
            self.balance -= (order_value + cost)
        else:  # For short positions, we typically need margin
            self.balance -= cost
            
        # Log transaction
        self.transaction_log.append({
            'timestamp': timestamp,
            'action': 'open',
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'cost': cost,
            'balance': self.balance
        })
        
        return True
    
    def close_position(self, position: Position, price: float, timestamp: datetime) -> float:
        """
        Close a specific position
        """
        if position.is_open:
            position.close_position(price, timestamp)
            
            # Calculate proceeds
            if position.side == 'long':
                proceeds = position.size * price
            else:  # short
                proceeds = position.size * position.entry_price + position.pnl
                
            cost = proceeds * self.transaction_cost
            net_proceeds = proceeds - cost
            
            self.balance += net_proceeds
            
            # Move to closed positions
            self.positions.remove(position)
            self.closed_positions.append(position)
            
            # Log transaction
            self.transaction_log.append({
                'timestamp': timestamp,
                'action': 'close',
                'symbol': position.symbol,
                'side': position.side,
                'size': position.size,
                'price': price,
                'pnl': position.pnl,
                'cost': cost,
                'balance': self.balance
            })
            
            return position.pnl
        
        return 0.0
    
    def update_positions(self, market_data: Dict[str, float], timestamp: datetime):
        """
        Update all open positions with current market prices
        """
        positions_to_close = []
        
        for position in self.positions:
            if position.symbol in market_data:
                current_price = market_data[position.symbol]
                should_exit = position.update_position(current_price, timestamp)
                
                if should_exit:
                    positions_to_close.append(position)
        
        # Close positions that hit stop loss or take profit
        for position in positions_to_close:
            current_price = market_data[position.symbol]
            self.close_position(position, current_price, timestamp)
    
    def get_equity(self, market_data: Dict[str, float]) -> float:
        """
        Calculate current account equity
        """
        equity = self.balance
        
        for position in self.positions:
            if position.symbol in market_data:
                current_price = market_data[position.symbol]
                position.update_position(current_price, datetime.now())
                
                if position.side == 'long':
                    equity += position.size * current_price
                else:  # short
                    equity += position.size * position.entry_price + position.pnl
                    
        return equity
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary statistics"""
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p.pnl > 0])
        losing_trades = len([p for p in self.closed_positions if p.pnl < 0])
        
        if total_trades == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_win': 0,
                'max_loss': 0
            }
        
        wins = [p.pnl for p in self.closed_positions if p.pnl > 0]
        losses = [p.pnl for p in self.closed_positions if p.pnl < 0]
        
        total_pnl = sum([p.pnl for p in self.closed_positions])
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': max(wins) if wins else 0,
            'max_loss': min(losses) if losses else 0
        }


class AdvancedBacktestEngine:
    """
    Advanced backtesting engine with comprehensive analytics
    """
    
    def __init__(self, initial_balance: float = 100000, 
                 transaction_cost: float = 0.001):
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.account = TradingAccount(initial_balance, transaction_cost)
        self.data = None
        self.results = {}
        self.performance_metrics = {}
        
    def load_data(self, data: pd.DataFrame):
        """
        Load market data for backtesting
        Data should have columns: timestamp, symbol, open, high, low, close, volume
        """
        self.data = data.copy()
        self.data = self.data.sort_values('timestamp')
        
    def run_backtest(self, strategy_func: Callable, 
                    data: pd.DataFrame = None,
                    start_date: str = None,
                    end_date: str = None,
                    **strategy_params) -> Dict:
        """
        Run backtest with given strategy function
        
        Strategy function should take parameters:
        - data: DataFrame with current and historical market data
        - account: TradingAccount instance
        - timestamp: current timestamp
        - **strategy_params: additional strategy parameters
        """
        
        if data is not None:
            self.load_data(data)
        
        if self.data is None:
            raise ValueError("No data loaded for backtesting")
        
        # Filter data by date range
        test_data = self.data.copy()
        if start_date:
            test_data = test_data[test_data['timestamp'] >= start_date]
        if end_date:
            test_data = test_data[test_data['timestamp'] <= end_date]
        
        # Initialize tracking variables
        equity_curve = []
        timestamps = []
        
        # Group data by timestamp for multi-symbol support
        for timestamp, group_data in test_data.groupby('timestamp'):
            
            # Create market data dictionary
            market_data = {}
            for _, row in group_data.iterrows():
                market_data[row['symbol']] = {
                    'open': row['open'],
                    'high': row['high'], 
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
            
            # Update existing positions
            current_prices = {symbol: data['close'] for symbol, data in market_data.items()}
            self.account.update_positions(current_prices, timestamp)
            
            # Run strategy
            strategy_func(
                data=test_data[test_data['timestamp'] <= timestamp],
                account=self.account,
                timestamp=timestamp,
                market_data=market_data,
                **strategy_params
            )
            
            # Record equity
            current_equity = self.account.get_equity(current_prices)
            equity_curve.append(current_equity)
            timestamps.append(timestamp)
        
        # Store results
        self.equity_curve = pd.Series(equity_curve, index=timestamps)
        self.results = {
            'equity_curve': self.equity_curve,
            'positions': self.account.closed_positions,
            'transactions': self.account.transaction_log,
            'portfolio_summary': self.account.get_portfolio_summary()
        }
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics()
        
        return self.results
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if len(self.equity_curve) == 0:
            return {}
        
        # Basic returns calculation
        returns = self.equity_curve.pct_change().dropna()
        
        # Total and annualized returns
        total_return = (self.equity_curve.iloc[-1] - self.initial_balance) / self.initial_balance
        
        # Assume daily data for annualization (252 trading days)
        periods_per_year = 252
        if len(self.equity_curve) > 1:
            days_elapsed = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
            if days_elapsed > 0:
                periods_per_year = len(self.equity_curve) * 365.25 / days_elapsed
        
        annual_return = (1 + total_return) ** (periods_per_year / len(self.equity_curve)) - 1
        
        # Volatility metrics
        volatility = returns.std() * np.sqrt(periods_per_year)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        
        # Risk-adjusted returns
        risk_free_rate = 0.02  # 2% annual risk-free rate
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_duration = self._calculate_drawdown_duration(drawdown)
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (VaR) and Expected Shortfall
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        expected_shortfall_95 = returns[returns <= var_95].mean()
        
        # Statistical measures
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Portfolio-specific metrics
        portfolio_summary = self.account.get_portfolio_summary()
        
        # Beta calculation (assuming market return is average of all returns)
        market_returns = returns.mean()  # Simplified market proxy
        if returns.var() > 0:
            beta = returns.cov(pd.Series([market_returns] * len(returns))) / pd.Series([market_returns] * len(returns)).var()
        else:
            beta = 0
        
        # Information ratio (assuming benchmark is risk-free rate)
        benchmark_return = risk_free_rate / periods_per_year
        active_returns = returns - benchmark_return
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)
        information_ratio = active_returns.mean() * np.sqrt(periods_per_year) / tracking_error if tracking_error > 0 else 0
        
        # Omega ratio
        omega_ratio = self._calculate_omega_ratio(returns)
        
        return {
            # Return metrics
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'downside_deviation': downside_deviation,
            
            # Risk-adjusted metrics
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'omega_ratio': omega_ratio,
            
            # Drawdown metrics
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': drawdown_duration,
            
            # Risk metrics
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': expected_shortfall_95,
            'beta': beta,
            
            # Statistical metrics
            'skewness': skewness,
            'kurtosis': kurtosis,
            
            # Trading metrics
            'total_trades': portfolio_summary['total_trades'],
            'win_rate': portfolio_summary['win_rate'],
            'profit_factor': portfolio_summary['profit_factor'],
            'avg_win': portfolio_summary['avg_win'],
            'avg_loss': portfolio_summary['avg_loss'],
            
            # Additional metrics
            'final_equity': self.equity_curve.iloc[-1],
            'peak_equity': self.equity_curve.max(),
            'equity_curve_length': len(self.equity_curve)
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration"""
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for in_drawdown in is_drawdown:
            if in_drawdown:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                    current_period = 0
        
        # Add final period if still in drawdown
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        above_threshold = returns[returns > threshold]
        below_threshold = returns[returns <= threshold]
        
        if len(below_threshold) == 0:
            return float('inf')
        
        gains = (above_threshold - threshold).sum()
        losses = (threshold - below_threshold).sum()
        
        return gains / losses if losses > 0 else float('inf')
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive backtest report
        """
        if not self.performance_metrics:
            return {"error": "No backtest results available"}
        
        report = {
            'summary': {
                'initial_balance': self.initial_balance,
                'final_equity': self.performance_metrics['final_equity'],
                'total_return': f"{self.performance_metrics['total_return']*100:.2f}%",
                'annual_return': f"{self.performance_metrics['annual_return']*100:.2f}%",
                'max_drawdown': f"{self.performance_metrics['max_drawdown']*100:.2f}%",
                'sharpe_ratio': f"{self.performance_metrics['sharpe_ratio']:.2f}",
                'total_trades': self.performance_metrics['total_trades']
            },
            
            'performance_metrics': self.performance_metrics,
            
            'trade_analysis': self._analyze_trades(),
            
            'risk_analysis': {
                'volatility': f"{self.performance_metrics['volatility']*100:.2f}%",
                'var_95': f"{self.performance_metrics['var_95']*100:.2f}%",
                'expected_shortfall': f"{self.performance_metrics['expected_shortfall_95']*100:.2f}%",
                'max_drawdown': f"{self.performance_metrics['max_drawdown']*100:.2f}%",
                'beta': f"{self.performance_metrics['beta']:.2f}"
            },
            
            'monthly_returns': self._calculate_monthly_returns() if len(self.equity_curve) > 30 else {}
        }
        
        return report
    
    def _analyze_trades(self) -> Dict:
        """Analyze trading patterns"""
        if not self.account.closed_positions:
            return {}
        
        positions = [p.to_dict() for p in self.account.closed_positions]
        trades_df = pd.DataFrame(positions)
        
        # Trade duration analysis
        if 'duration' in trades_df.columns:
            trades_df['duration_hours'] = trades_df['duration'].dt.total_seconds() / 3600
            avg_duration = trades_df['duration_hours'].mean()
            median_duration = trades_df['duration_hours'].median()
        else:
            avg_duration = median_duration = 0
        
        # PnL distribution
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        return {
            'avg_trade_duration_hours': avg_duration,
            'median_trade_duration_hours': median_duration,
            'best_trade': trades_df['pnl'].max(),
            'worst_trade': trades_df['pnl'].min(),
            'avg_winning_trade': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_losing_trade': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'largest_winning_streak': self._calculate_winning_streak(trades_df, True),
            'largest_losing_streak': self._calculate_winning_streak(trades_df, False)
        }
    
    def _calculate_winning_streak(self, trades_df: pd.DataFrame, winning: bool = True) -> int:
        """Calculate longest winning or losing streak"""
        if len(trades_df) == 0:
            return 0
        
        condition = trades_df['pnl'] > 0 if winning else trades_df['pnl'] < 0
        streaks = []
        current_streak = 0
        
        for is_win in condition:
            if is_win:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        # Add final streak
        if current_streak > 0:
            streaks.append(current_streak)
        
        return max(streaks) if streaks else 0
    
    def _calculate_monthly_returns(self) -> Dict:
        """Calculate monthly returns breakdown"""
        monthly_equity = self.equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        return {
            'monthly_returns': monthly_returns.to_dict(),
            'best_month': f"{monthly_returns.max()*100:.2f}%",
            'worst_month': f"{monthly_returns.min()*100:.2f}%",
            'positive_months': len(monthly_returns[monthly_returns > 0]),
            'negative_months': len(monthly_returns[monthly_returns < 0])
        }
    
    def plot_results(self, show_trades: bool = True, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot backtest results
        """
        if PLOTLY_AVAILABLE:
            self._plot_plotly(show_trades)
        else:
            self._plot_matplotlib(show_trades, figsize)
    
    def _plot_plotly(self, show_trades: bool = True):
        """Create interactive plots using Plotly"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Portfolio Equity', 'Drawdown', 'Trade P&L'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=self.equity_curve.index,
                y=self.equity_curve.values,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add trade markers
        if show_trades and self.account.closed_positions:
            entry_times = [p.entry_time for p in self.account.closed_positions]
            exit_times = [p.exit_time for p in self.account.closed_positions]
            entry_prices = [p.entry_price for p in self.account.closed_positions]
            exit_prices = [p.exit_price for p in self.account.closed_positions]
            
            # Entry points
            fig.add_trace(
                go.Scatter(
                    x=entry_times,
                    y=entry_prices,
                    mode='markers',
                    name='Trade Entry',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                ),
                row=1, col=1
            )
            
            # Exit points
            fig.add_trace(
                go.Scatter(
                    x=exit_times,
                    y=exit_prices,
                    mode='markers',
                    name='Trade Exit',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                ),
                row=1, col=1
            )
        
        # Drawdown
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode='lines',
                name='Drawdown %',
                fill='tonexty',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Trade P&L
        if self.account.closed_positions:
            trade_pnl = [p.pnl for p in self.account.closed_positions]
            trade_times = [p.exit_time for p in self.account.closed_positions]
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnl]
            
            fig.add_trace(
                go.Bar(
                    x=trade_times,
                    y=trade_pnl,
                    name='Trade P&L',
                    marker_color=colors
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title="Backtest Results",
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=3, col=1)
        
        fig.show()
    
    def _plot_matplotlib(self, show_trades: bool = True, figsize: Tuple[int, int] = (15, 10)):
        """Create plots using Matplotlib"""
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Equity curve
        axes[0].plot(self.equity_curve.index, self.equity_curve.values, 'b-', linewidth=2)
        axes[0].set_title('Portfolio Equity')
        axes[0].set_ylabel('Equity ($)')
        axes[0].grid(True)
        
        # Trade markers
        if show_trades and self.account.closed_positions:
            entry_times = [p.entry_time for p in self.account.closed_positions]
            exit_times = [p.exit_time for p in self.account.closed_positions]
            
            # Find equity values at trade times
            entry_equity = [self.equity_curve.loc[self.equity_curve.index <= t].iloc[-1] 
                          if len(self.equity_curve.loc[self.equity_curve.index <= t]) > 0 
                          else self.equity_curve.iloc[0] for t in entry_times]
            exit_equity = [self.equity_curve.loc[self.equity_curve.index <= t].iloc[-1] 
                         if len(self.equity_curve.loc[self.equity_curve.index <= t]) > 0 
                         else self.equity_curve.iloc[-1] for t in exit_times]
            
            axes[0].scatter(entry_times, entry_equity, c='green', marker='^', s=50, label='Entry')
            axes[0].scatter(exit_times, exit_equity, c='red', marker='v', s=50, label='Exit')
            axes[0].legend()
        
        # Drawdown
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak * 100
        
        axes[1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[1].plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True)
        
        # Trade P&L
        if self.account.closed_positions:
            trade_pnl = [p.pnl for p in self.account.closed_positions]
            trade_times = [p.exit_time for p in self.account.closed_positions]
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnl]
            
            axes[2].bar(trade_times, trade_pnl, color=colors, alpha=0.7)
            axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        axes[2].set_title('Trade P&L')
        axes[2].set_ylabel('P&L ($)')
        axes[2].set_xlabel('Time')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()


# Example strategies for testing
def buy_and_hold_strategy(data, account, timestamp, market_data, symbol='BTC', **kwargs):
    """
    Simple buy and hold strategy
    """
    # Only buy on first day
    if len(account.positions) == 0 and len(account.closed_positions) == 0:
        if symbol in market_data:
            price = market_data[symbol]['close']
            # Use 95% of balance
            size = (account.balance * 0.95) / price
            account.place_order(symbol, 'long', size, price, timestamp)


def moving_average_crossover_strategy(data, account, timestamp, market_data, 
                                    symbol='BTC', fast_period=10, slow_period=30, **kwargs):
    """
    Moving average crossover strategy
    """
    if symbol not in market_data:
        return
    
    # Get historical data for the symbol
    symbol_data = data[data['symbol'] == symbol].copy()
    
    if len(symbol_data) < slow_period:
        return
    
    # Calculate moving averages
    symbol_data['fast_ma'] = symbol_data['close'].rolling(window=fast_period).mean()
    symbol_data['slow_ma'] = symbol_data['close'].rolling(window=slow_period).mean()
    
    if len(symbol_data) < 2:
        return
    
    current_price = market_data[symbol]['close']
    
    # Get current and previous MA values
    current_fast = symbol_data['fast_ma'].iloc[-1]
    current_slow = symbol_data['slow_ma'].iloc[-1]
    prev_fast = symbol_data['fast_ma'].iloc[-2]
    prev_slow = symbol_data['slow_ma'].iloc[-2]
    
    # Check for crossover
    current_position = None
    for pos in account.positions:
        if pos.symbol == symbol:
            current_position = pos
            break
    
    # Buy signal: fast MA crosses above slow MA
    if (prev_fast <= prev_slow and current_fast > current_slow and 
        current_position is None and not pd.isna(current_fast) and not pd.isna(current_slow)):
        
        # Use 95% of available balance
        size = (account.balance * 0.95) / current_price
        if size > 0:
            account.place_order(symbol, 'long', size, current_price, timestamp)
    
    # Sell signal: fast MA crosses below slow MA
    elif (prev_fast >= prev_slow and current_fast < current_slow and 
          current_position is not None and not pd.isna(current_fast) and not pd.isna(current_slow)):
        
        account.close_position(current_position, current_price, timestamp)


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    import random
    
    print("=== Advanced Backtesting Engine Demo ===")
    
    # Generate synthetic market data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create synthetic BTC price data
    np.random.seed(42)
    initial_price = 30000
    
    price_data = []
    current_price = initial_price
    
    for date in dates:
        # Add some realistic price movement
        daily_return = np.random.normal(0.001, 0.03)  # Average 0.1% daily return with 3% volatility
        current_price *= (1 + daily_return)
        
        # Generate OHLCV data
        open_price = current_price * (1 + np.random.uniform(-0.01, 0.01))
        high_price = max(open_price, current_price) * (1 + np.random.uniform(0, 0.02))
        low_price = min(open_price, current_price) * (1 - np.random.uniform(0, 0.02))
        volume = np.random.exponential(1000000)
        
        price_data.append({
            'timestamp': date,
            'symbol': 'BTC',
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': current_price,
            'volume': volume
        })
    
    # Create DataFrame
    market_data = pd.DataFrame(price_data)
    
    print(f"Generated {len(market_data)} days of synthetic BTC data")
    print(f"Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")
    
    # Initialize backtest engine
    engine = AdvancedBacktestEngine(initial_balance=100000, transaction_cost=0.001)
    
    # Test buy and hold strategy
    print("\n--- Testing Buy & Hold Strategy ---")
    results_bh = engine.run_backtest(
        strategy_func=buy_and_hold_strategy,
        data=market_data,
        symbol='BTC'
    )
    
    # Generate report
    report_bh = engine.generate_report()
    print(f"Buy & Hold Results:")
    print(f"  Total Return: {report_bh['summary']['total_return']}")
    print(f"  Annual Return: {report_bh['summary']['annual_return']}")
    print(f"  Max Drawdown: {report_bh['summary']['max_drawdown']}")
    print(f"  Sharpe Ratio: {report_bh['summary']['sharpe_ratio']}")
    
    # Test moving average crossover strategy
    print("\n--- Testing MA Crossover Strategy ---")
    engine2 = AdvancedBacktestEngine(initial_balance=100000, transaction_cost=0.001)
    
    results_ma = engine2.run_backtest(
        strategy_func=moving_average_crossover_strategy,
        data=market_data,
        symbol='BTC',
        fast_period=10,
        slow_period=30
    )
    
    report_ma = engine2.generate_report()
    print(f"MA Crossover Results:")
    print(f"  Total Return: {report_ma['summary']['total_return']}")
    print(f"  Annual Return: {report_ma['summary']['annual_return']}")
    print(f"  Max Drawdown: {report_ma['summary']['max_drawdown']}")
    print(f"  Sharpe Ratio: {report_ma['summary']['sharpe_ratio']}")
    print(f"  Total Trades: {report_ma['summary']['total_trades']}")
    
    # Plot results (if matplotlib is available)
    try:
        print("\nGenerating plots...")
        engine2.plot_results()
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\nBacktesting demo completed successfully!")
