"""
Advanced Cryptocurrency Data Collection System
Integrates multiple data sources for comprehensive market analysis
"""

import os
import time
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import ta
from alpha_vantage.timeseries import TimeSeries
import cryptocompare
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataCollector:
    """
    Advanced cryptocurrency data collector with multiple sources
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.exchanges = self._initialize_exchanges()
        self.alpha_vantage = TimeSeries(key=config.get('alpha_vantage_key', ''))
        
        # Popular cryptocurrencies for analysis
        self.top_cryptos = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'MATIC',
            'AVAX', 'LINK', 'UNI', 'ATOM', 'XLM', 'ALGO', 'VET'
        ]
        
    def _initialize_exchanges(self) -> Dict:
        """Initialize cryptocurrency exchanges"""
        exchanges = {}
        
        try:
            exchanges['binance'] = ccxt.binance({
                'apiKey': self.config.get('binance_api_key', ''),
                'secret': self.config.get('binance_secret', ''),
                'sandbox': False,
                'rateLimit': 1000,
                'enableRateLimit': True,
            })
            
            exchanges['coinbase'] = ccxt.coinbasepro({
                'apiKey': self.config.get('coinbase_api_key', ''),
                'secret': self.config.get('coinbase_secret', ''),
                'password': self.config.get('coinbase_password', ''),
                'sandbox': False,
            })
            
            exchanges['kraken'] = ccxt.kraken({
                'apiKey': self.config.get('kraken_api_key', ''),
                'secret': self.config.get('kraken_secret', ''),
            })
            
        except Exception as e:
            logger.warning(f"Exchange initialization warning: {e}")
            
        return exchanges
    
    def collect_historical_data(self, 
                              symbol: str, 
                              timeframe: str = '1d', 
                              limit: int = 1000,
                              exchange: str = 'binance') -> pd.DataFrame:
        """
        Collect historical OHLCV data from exchange
        """
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not available")
                
            exchange_client = self.exchanges[exchange]
            
            # Fetch OHLCV data
            ohlcv = exchange_client.fetch_ohlcv(
                symbol=f"{symbol}/USDT",
                timeframe=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Add market sentiment indicators
            df = self._add_sentiment_indicators(df)
            
            logger.info(f"Collected {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def collect_realtime_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Collect real-time data for multiple symbols
        """
        realtime_data = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            
            for symbol in symbols:
                future = executor.submit(self._fetch_ticker_data, symbol)
                futures[future] = symbol
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    realtime_data[symbol] = data
                except Exception as e:
                    logger.error(f"Error fetching real-time data for {symbol}: {e}")
        
        return realtime_data
    
    def _fetch_ticker_data(self, symbol: str) -> Dict:
        """Fetch ticker data for a single symbol"""
        try:
            exchange = self.exchanges.get('binance')
            if not exchange:
                return {}
                
            ticker = exchange.fetch_ticker(f"{symbol}/USDT")
            
            # Add additional metrics
            ticker['fear_greed_index'] = self._get_fear_greed_index()
            ticker['social_sentiment'] = self._get_social_sentiment(symbol)
            
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators
        """
        try:
            # Moving Averages
            df['sma_7'] = ta.trend.sma_indicator(df['close'], window=7)
            df['sma_25'] = ta.trend.sma_indicator(df['close'], window=25)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # RSI and Stochastic
            df['rsi'] = ta.momentum.rsi(df['close'])
            df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            
            # MACD
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            df['macd_histogram'] = ta.trend.macd_diff(df['close'])
            
            # Bollinger Bands
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Volume indicators
            df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
            df['vwap'] = ta.volume.volume_weighted_average_price(
                df['high'], df['low'], df['close'], df['volume']
            )
            
            # Volatility
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Price action features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Support and Resistance levels
            df['support'] = df['low'].rolling(window=20).min()
            df['resistance'] = df['high'].rolling(window=20).max()
            
            # Fibonacci retracement levels
            high_max = df['high'].rolling(window=50).max()
            low_min = df['low'].rolling(window=50).min()
            diff = high_max - low_min
            
            df['fib_23.6'] = high_max - 0.236 * diff
            df['fib_38.2'] = high_max - 0.382 * diff
            df['fib_50.0'] = high_max - 0.5 * diff
            df['fib_61.8'] = high_max - 0.618 * diff
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _add_sentiment_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market sentiment indicators"""
        try:
            # Fear and Greed Index (mock implementation)
            df['fear_greed'] = np.random.randint(0, 100, len(df))
            
            # Social sentiment score (mock)
            df['social_sentiment'] = np.random.uniform(-1, 1, len(df))
            
            # Market cap dominance (mock)
            df['btc_dominance'] = np.random.uniform(40, 60, len(df))
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding sentiment indicators: {e}")
            return df
    
    def _get_fear_greed_index(self) -> float:
        """Get current Fear and Greed Index"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return float(data['data'][0]['value'])
        except Exception as e:
            logger.error(f"Error fetching fear and greed index: {e}")
        
        return 50.0  # Default neutral value
    
    def _get_social_sentiment(self, symbol: str) -> float:
        """Get social sentiment for symbol (mock implementation)"""
        # In real implementation, this would connect to Twitter API, Reddit API, etc.
        return np.random.uniform(-1, 1)
    
    def collect_macro_economic_data(self) -> pd.DataFrame:
        """
        Collect macroeconomic indicators that affect crypto markets
        """
        macro_data = {}
        
        try:
            # Gold, Oil, USD Index from Yahoo Finance
            tickers = ['^GSPC', '^VIX', 'GLD', 'CL=F', 'DX-Y.NYB']
            
            for ticker in tickers:
                try:
                    data = yf.download(ticker, period='1y', interval='1d', progress=False)
                    if not data.empty:
                        macro_data[ticker] = data['Close'].iloc[-1]
                except:
                    continue
            
            # Create DataFrame with macro indicators
            macro_df = pd.DataFrame([macro_data], index=[datetime.now()])
            
            return macro_df
            
        except Exception as e:
            logger.error(f"Error collecting macro data: {e}")
            return pd.DataFrame()
    
    def get_market_overview(self) -> Dict:
        """Get comprehensive market overview"""
        try:
            overview = {
                'total_market_cap': 0,
                'total_volume_24h': 0,
                'btc_dominance': 0,
                'active_cryptos': 0,
                'fear_greed_index': self._get_fear_greed_index(),
                'top_gainers': [],
                'top_losers': []
            }
            
            # Collect data for top cryptocurrencies
            crypto_data = self.collect_realtime_data(self.top_cryptos[:10])
            
            if crypto_data:
                # Calculate market metrics
                total_volume = sum([
                    float(data.get('quoteVolume', 0)) 
                    for data in crypto_data.values()
                ])
                
                price_changes = [
                    float(data.get('percentage', 0))
                    for data in crypto_data.values()
                    if data.get('percentage')
                ]
                
                overview.update({
                    'total_volume_24h': total_volume,
                    'avg_price_change': np.mean(price_changes) if price_changes else 0,
                    'market_volatility': np.std(price_changes) if price_changes else 0
                })
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save data to CSV file"""
        try:
            filepath = os.path.join('data', filename)
            os.makedirs('data', exist_ok=True)
            df.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")


# Example usage and configuration
if __name__ == "__main__":
    # Configuration (in production, use environment variables)
    config = {
        'binance_api_key': '',  # Add your API keys
        'binance_secret': '',
        'alpha_vantage_key': '',
        'coinbase_api_key': '',
        'coinbase_secret': '',
        'coinbase_password': '',
    }
    
    # Initialize collector
    collector = CryptoDataCollector(config)
    
    # Collect historical data for Bitcoin
    btc_data = collector.collect_historical_data('BTC', timeframe='1h', limit=500)
    
    if not btc_data.empty:
        print(f"Collected Bitcoin data: {btc_data.shape}")
        print(f"Features: {list(btc_data.columns)}")
        
        # Save the data
        collector.save_data(btc_data, 'btc_historical.csv')
        
        # Display sample
        print("\nSample data:")
        print(btc_data.tail())
    
    # Get market overview
    market_overview = collector.get_market_overview()
    print(f"\nMarket Overview: {market_overview}")
