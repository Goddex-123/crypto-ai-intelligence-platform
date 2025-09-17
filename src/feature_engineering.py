"""
Advanced Feature Engineering for Cryptocurrency Prediction
Includes technical, fundamental, and alternative data features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for cryptocurrency time series prediction
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_importance = {}
        self.selected_features = []
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp index
        """
        df = df.copy()
        
        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Market session indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)
        df['is_asian_session'] = ((df['hour'] >= 23) | (df['hour'] <= 8)).astype(int)
        df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
        df['is_us_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
        
        # Holiday indicators (simplified)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        df['is_quarter_end'] = (df['month'].isin([3, 6, 9, 12]) & 
                               (df['day_of_month'] >= 28)).astype(int)
        
        return df
    
    def create_lagged_features(self, df: pd.DataFrame, 
                              target_col: str = 'close',
                              lags: List[int] = None) -> pd.DataFrame:
        """
        Create lagged features for time series prediction
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10, 15, 30, 60, 120]  # Various lag periods
            
        df = df.copy()
        
        # Price lags
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
        # Return lags
        returns = df[target_col].pct_change()
        for lag in lags:
            df[f'return_lag_{lag}'] = returns.shift(lag)
            
        # Volume lags
        if 'volume' in df.columns:
            for lag in lags[:5]:  # Only short-term volume lags
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Volatility lags (using rolling standard deviation)
        volatility = df[target_col].rolling(window=24).std()
        for lag in [1, 2, 3, 5, 10]:
            df[f'volatility_lag_{lag}'] = volatility.shift(lag)
            
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                               target_col: str = 'close',
                               windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling window statistical features
        """
        if windows is None:
            windows = [5, 10, 20, 50, 100, 200]
            
        df = df.copy()
        
        for window in windows:
            # Rolling statistics for price
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window).std()
            df[f'rolling_min_{window}'] = df[target_col].rolling(window).min()
            df[f'rolling_max_{window}'] = df[target_col].rolling(window).max()
            df[f'rolling_median_{window}'] = df[target_col].rolling(window).median()
            df[f'rolling_skew_{window}'] = df[target_col].rolling(window).skew()
            df[f'rolling_kurtosis_{window}'] = df[target_col].rolling(window).kurtosis()
            
            # Price position within rolling window
            df[f'price_position_{window}'] = (
                (df[target_col] - df[f'rolling_min_{window}']) /
                (df[f'rolling_max_{window}'] - df[f'rolling_min_{window}'])
            )
            
            # Distance from moving averages
            df[f'price_ma_ratio_{window}'] = df[target_col] / df[f'rolling_mean_{window}']
            df[f'price_ma_diff_{window}'] = df[target_col] - df[f'rolling_mean_{window}']
            
            # Bollinger Band features
            rolling_mean = df[f'rolling_mean_{window}']
            rolling_std = df[f'rolling_std_{window}']
            df[f'bb_upper_{window}'] = rolling_mean + (2 * rolling_std)
            df[f'bb_lower_{window}'] = rolling_mean - (2 * rolling_std)
            df[f'bb_position_{window}'] = (
                (df[target_col] - df[f'bb_lower_{window}']) /
                (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
            )
            
            # Volume features (if available)
            if 'volume' in df.columns:
                df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
                df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_mean_{window}']
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame,
                                target_col: str = 'close') -> pd.DataFrame:
        """
        Create momentum and trend features
        """
        df = df.copy()
        
        # Rate of Change (ROC)
        for period in [1, 3, 5, 10, 20]:
            df[f'roc_{period}'] = df[target_col].pct_change(periods=period)
            
        # Momentum oscillators
        for window in [10, 20, 50]:
            df[f'momentum_{window}'] = df[target_col] / df[target_col].shift(window)
            
        # Price acceleration (second derivative)
        df['price_acceleration'] = df[target_col].diff().diff()
        
        # Trend strength indicators
        for window in [20, 50, 100]:
            # Linear regression slope
            df[f'trend_slope_{window}'] = df[target_col].rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
            )
            
            # R-squared of linear regression
            def calculate_r2(y):
                if len(y) != window:
                    return np.nan
                x = np.arange(len(y))
                try:
                    slope, intercept = np.polyfit(x, y, 1)
                    y_pred = slope * x + intercept
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                except:
                    return np.nan
                    
            df[f'trend_strength_{window}'] = df[target_col].rolling(window).apply(calculate_r2)
        
        # Higher timeframe trends
        if len(df) > 200:
            # Weekly trend (7 days)
            df['weekly_trend'] = df[target_col].rolling(7*24).mean().pct_change(7*24)
            # Monthly trend (30 days)  
            df['monthly_trend'] = df[target_col].rolling(30*24).mean().pct_change(30*24)
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame,
                                  target_col: str = 'close') -> pd.DataFrame:
        """
        Create volatility and risk features
        """
        df = df.copy()
        
        # Simple volatility measures
        returns = df[target_col].pct_change()
        
        for window in [10, 20, 50, 100]:
            # Historical volatility
            df[f'volatility_{window}'] = returns.rolling(window).std()
            
            # Parkinson volatility (using OHLC)
            if all(col in df.columns for col in ['high', 'low', 'open']):
                df[f'parkinson_vol_{window}'] = np.sqrt(
                    (1/(4*np.log(2))) * 
                    (np.log(df['high']/df['low'])**2).rolling(window).mean()
                )
                
                # Garman-Klass volatility
                df[f'gk_vol_{window}'] = np.sqrt(
                    ((np.log(df['high']/df['close'])*np.log(df['high']/df['open'])) +
                     (np.log(df['low']/df['close'])*np.log(df['low']/df['open']))
                    ).rolling(window).mean()
                )
            
            # Volatility of volatility
            vol_col = f'volatility_{window}'
            if vol_col in df.columns:
                df[f'vol_of_vol_{window}'] = df[vol_col].rolling(window//2).std()
        
        # GARCH-like features
        df['volatility_regime'] = (returns.rolling(50).std() > 
                                  returns.rolling(200).std()).astype(int)
        
        # Volatility clustering
        abs_returns = np.abs(returns)
        df['vol_clustering'] = abs_returns.rolling(20).corr(abs_returns.shift(1))
        
        return df
    
    def create_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market microstructure features
        """
        df = df.copy()
        
        if all(col in df.columns for col in ['high', 'low', 'open', 'close', 'volume']):
            # Price spreads
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
            df['oc_spread'] = np.abs(df['open'] - df['close']) / df['close']
            
            # Intrabar price movements
            df['upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
            df['lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
            df['body_size'] = np.abs(df['close'] - df['open']) / df['close']
            
            # Volume-price relationships
            df['volume_price_trend'] = (df['close'] - df['open']) * df['volume']
            df['price_volume_correlation'] = df['close'].rolling(50).corr(df['volume'])
            
            # Order flow approximation
            df['buying_pressure'] = ((df['close'] - df['low']) - 
                                   (df['high'] - df['close'])) / (df['high'] - df['low'])
            df['buying_pressure'] = df['buying_pressure'].fillna(0)
            
            # Tick direction approximation
            df['tick_rule'] = np.sign(df['close'].diff()).fillna(0)
            df['tick_momentum'] = df['tick_rule'].rolling(10).sum()
        
        return df
    
    def create_alternative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create alternative and experimental features
        """
        df = df.copy()
        
        # Fractal and chaos theory features
        if 'close' in df.columns:
            # Hurst exponent approximation
            def hurst_approx(ts, max_lag=20):
                if len(ts) < max_lag * 2:
                    return 0.5
                lags = range(2, max_lag)
                tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
                try:
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    return poly[0] * 2.0
                except:
                    return 0.5
            
            df['hurst_exp'] = df['close'].rolling(100).apply(hurst_approx)
            
            # Fractal dimension approximation
            df['fractal_dim'] = 2 - df['hurst_exp']
            
            # Approximate entropy
            def approx_entropy(ts, m=2, r=0.2):
                if len(ts) < 10:
                    return 0
                
                def _maxdist(xi, xj, m):
                    return max([abs(ua - va) for ua, va in zip(xi, xj)])
                
                def _phi(m):
                    patterns = []
                    for i in range(len(ts) - m + 1):
                        patterns.append(ts[i:i+m])
                    
                    C = []
                    for i in range(len(patterns)):
                        template_i = patterns[i]
                        matches = 0
                        for j in range(len(patterns)):
                            if _maxdist(template_i, patterns[j], m) <= r * np.std(ts):
                                matches += 1
                        C.append(matches / len(patterns))
                    
                    phi = np.mean([np.log(c) for c in C if c > 0])
                    return phi
                
                try:
                    return _phi(m) - _phi(m + 1)
                except:
                    return 0
            
            df['approx_entropy'] = df['close'].rolling(50).apply(
                lambda x: approx_entropy(x.values)
            )
        
        # Fourier transform features
        if len(df) > 100:
            price_fft = np.fft.fft(df['close'].fillna(df['close'].mean()).values)
            freqs = np.fft.fftfreq(len(price_fft))
            
            # Dominant frequencies
            magnitude = np.abs(price_fft)
            dominant_freq_idx = np.argsort(magnitude)[-5:]  # Top 5 frequencies
            
            for i, idx in enumerate(dominant_freq_idx):
                df[f'dominant_freq_{i+1}'] = freqs[idx]
                df[f'dominant_mag_{i+1}'] = magnitude[idx]
        
        # Seasonality features
        if len(df) > 24*7:  # At least a week of hourly data
            # Hour of day seasonality
            hourly_means = df.groupby(df.index.hour)['close'].mean()
            df['hourly_seasonal'] = df.index.hour.map(hourly_means)
            
            # Day of week seasonality
            daily_means = df.groupby(df.index.dayofweek)['close'].mean()
            df['daily_seasonal'] = df.index.dayofweek.map(daily_means)
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str,
                       method: str = 'mutual_info', k: int = 50) -> List[str]:
        """
        Feature selection using various methods
        """
        # Prepare data
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]
        
        # Remove features with too many NaNs
        nan_threshold = 0.5
        valid_features = []
        for col in X.columns:
            if X[col].notna().sum() / len(X) >= (1 - nan_threshold):
                valid_features.append(col)
        
        X = X[valid_features]
        
        # Handle remaining NaNs
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Remove rows where target is NaN
        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        if len(X) == 0 or len(y) == 0:
            return []
        
        # Feature selection
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(X.columns)))
        else:  # f_regression
            selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
        
        try:
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Store feature scores
            feature_scores = dict(zip(X.columns, selector.scores_))
            self.feature_importance = {k: v for k, v in sorted(
                feature_scores.items(), key=lambda x: x[1], reverse=True
            )}
            
            self.selected_features = selected_features
            return selected_features
            
        except Exception as e:
            print(f"Error in feature selection: {e}")
            return X.columns.tolist()[:k]
    
    def scale_features(self, df: pd.DataFrame, 
                      feature_cols: List[str],
                      method: str = 'robust') -> pd.DataFrame:
        """
        Scale features using various methods
        """
        df = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return df
        
        # Fit scaler only on numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Handle NaNs before scaling
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Fit and transform
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            # Store scaler for future use
            self.scalers[method] = scaler
        
        return df
    
    def create_target_features(self, df: pd.DataFrame,
                              target_col: str = 'close',
                              prediction_horizons: List[int] = None) -> pd.DataFrame:
        """
        Create target variables for different prediction horizons
        """
        if prediction_horizons is None:
            prediction_horizons = [1, 6, 12, 24, 48]  # Hours ahead
        
        df = df.copy()
        
        for horizon in prediction_horizons:
            # Price target
            df[f'target_price_{horizon}h'] = df[target_col].shift(-horizon)
            
            # Return target
            df[f'target_return_{horizon}h'] = (
                df[f'target_price_{horizon}h'] / df[target_col] - 1
            )
            
            # Direction target (classification)
            df[f'target_direction_{horizon}h'] = (
                df[f'target_return_{horizon}h'] > 0
            ).astype(int)
            
            # Volatility target
            future_returns = df[target_col].pct_change().shift(-horizon).rolling(horizon).std()
            df[f'target_volatility_{horizon}h'] = future_returns
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame,
                             target_col: str = 'close') -> pd.DataFrame:
        """
        Apply all feature engineering steps
        """
        print("Starting feature engineering...")
        
        # Create time features
        print("Creating time features...")
        df = self.create_time_features(df)
        
        # Create lagged features
        print("Creating lagged features...")
        df = self.create_lagged_features(df, target_col)
        
        # Create rolling features
        print("Creating rolling features...")
        df = self.create_rolling_features(df, target_col)
        
        # Create momentum features
        print("Creating momentum features...")
        df = self.create_momentum_features(df, target_col)
        
        # Create volatility features
        print("Creating volatility features...")
        df = self.create_volatility_features(df, target_col)
        
        # Create microstructure features
        print("Creating microstructure features...")
        df = self.create_market_microstructure_features(df)
        
        # Create alternative features
        print("Creating alternative features...")
        df = self.create_alternative_features(df)
        
        # Create target features
        print("Creating target features...")
        df = self.create_target_features(df, target_col)
        
        print(f"Feature engineering complete. Created {len(df.columns)} features.")
        return df


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Generate synthetic OHLCV data
    close_prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
    sample_data = pd.DataFrame({
        'open': close_prices + np.random.randn(1000) * 50,
        'high': close_prices + np.abs(np.random.randn(1000)) * 100,
        'low': close_prices - np.abs(np.random.randn(1000)) * 100,
        'close': close_prices,
        'volume': np.random.exponential(1000, 1000)
    }, index=dates)
    
    # Initialize feature engineer
    fe = AdvancedFeatureEngineer()
    
    # Apply feature engineering
    engineered_df = fe.engineer_all_features(sample_data)
    
    print(f"Original features: {sample_data.shape[1]}")
    print(f"Engineered features: {engineered_df.shape[1]}")
    print(f"Sample features: {list(engineered_df.columns[:20])}")
    
    # Feature selection
    selected_features = fe.select_features(engineered_df, 'close', k=30)
    print(f"Selected {len(selected_features)} most important features")
    
    # Show top features
    print("Top 10 features by importance:")
    for i, (feature, score) in enumerate(list(fe.feature_importance.items())[:10]):
        print(f"{i+1}. {feature}: {score:.4f}")
