#!/usr/bin/env python3
"""
Cryptocurrency Intelligence Platform - Main Launcher
====================================================

This script provides multiple ways to run and interact with the platform:
1. Launch the Streamlit dashboard
2. Run example analyses
3. Execute backtests
4. Train ML models

Usage:
    python run.py --dashboard          # Launch web dashboard
    python run.py --example           # Run example analysis
    python run.py --backtest          # Run example backtesting
    python run.py --train-models      # Train ML models
    python run.py --help              # Show help
"""

import argparse
import sys
import os
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Cryptocurrency Intelligence Platform Dashboard...")
    print("📊 The dashboard will open in your browser at http://localhost:8501")
    print("💡 Use Ctrl+C to stop the server\n")
    
    dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'crypto_dashboard.py')
    
    try:
        subprocess.run(['streamlit', 'run', dashboard_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")

def run_example_analysis():
    """Run example data analysis and ML prediction"""
    print("🔍 Running Example Cryptocurrency Analysis...\n")
    
    try:
        from data_collector import CryptoDataCollector
        from feature_engineering import AdvancedFeatureEngineer
        from ml_models import EnsembleModel, ModelEvaluator
        
        print("1️⃣  Initializing data collector...")
        config = {}  # Using demo mode without API keys
        collector = CryptoDataCollector(config)
        
        print("2️⃣  Generating synthetic cryptocurrency data...")
        # Generate synthetic BTC data for demonstration
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='1H')
        
        base_price = 50000
        returns = np.random.normal(0.001, 0.02, 1000)
        prices = base_price * np.cumprod(1 + returns)
        
        # Create OHLCV data
        btc_data = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, 1000)),
            'high': prices * (1 + np.random.uniform(0, 0.02, 1000)),
            'low': prices * (1 - np.random.uniform(0, 0.02, 1000)),
            'close': prices,
            'volume': np.random.exponential(1000000, 1000)
        }, index=dates)
        
        print(f"   ✅ Generated {len(btc_data)} data points")
        print(f"   📈 Price range: ${btc_data['close'].min():,.2f} - ${btc_data['close'].max():,.2f}")
        
        print("3️⃣  Engineering features...")
        feature_engineer = AdvancedFeatureEngineer()
        featured_data = feature_engineer.engineer_all_features(btc_data)
        
        print(f"   ✅ Created {featured_data.shape[1]} features from {btc_data.shape[1]} original columns")
        
        print("4️⃣  Selecting top features...")
        selected_features = feature_engineer.select_features(featured_data, 'close', k=30)
        print(f"   ✅ Selected {len(selected_features)} most important features")
        
        print("5️⃣  Training ensemble model...")
        # Prepare data for traditional ML
        valid_data = featured_data.dropna()
        if len(valid_data) < 100:
            print("   ⚠️  Insufficient data for training")
            return
        
        # Simple train-test split
        train_size = int(0.8 * len(valid_data))
        train_data = valid_data[:train_size]
        test_data = valid_data[train_size:]
        
        if len(selected_features) > 0:
            X_train = train_data[selected_features].values
            X_test = test_data[selected_features].values
            y_train = train_data['close'].values[1:]  # Predict next close
            y_test = test_data['close'].values[1:]
            
            # Adjust X to match y length
            X_train = X_train[:-1]
            X_test = X_test[:-1]
            
            if len(X_train) > 50:  # Minimum data for meaningful training
                ensemble = EnsembleModel()
                ensemble.train(X_train, y_train)
                
                print("6️⃣  Evaluating model performance...")
                predictions = ensemble.predict(X_test)
                metrics = ModelEvaluator.evaluate_model(y_test, predictions)
                
                print("   📊 Model Performance:")
                print(f"      • RMSE: ${metrics['rmse']:,.2f}")
                print(f"      • MAE:  ${metrics['mae']:,.2f}")
                print(f"      • R²:   {metrics['r2']:.4f}")
                print(f"      • Directional Accuracy: {metrics['directional_accuracy']:.1%}")
                
                print("\n🎉 Example analysis completed successfully!")
                print("💡 Launch the dashboard to explore interactive features: python run.py --dashboard")
            else:
                print("   ⚠️  Insufficient training data")
        else:
            print("   ⚠️  No features selected")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

def run_example_backtest():
    """Run example backtesting"""
    print("📊 Running Example Backtesting...\n")
    
    try:
        from backtesting_engine import AdvancedBacktestEngine, buy_and_hold_strategy, moving_average_crossover_strategy
        
        print("1️⃣  Generating synthetic market data...")
        # Generate synthetic price data
        np.random.seed(42)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        initial_price = 30000
        price_data = []
        current_price = initial_price
        
        for date in dates:
            daily_return = np.random.normal(0.001, 0.03)
            current_price *= (1 + daily_return)
            
            price_data.append({
                'timestamp': date,
                'symbol': 'BTC',
                'open': current_price * (1 + np.random.uniform(-0.01, 0.01)),
                'high': current_price * (1 + np.random.uniform(0, 0.02)),
                'low': current_price * (1 - np.random.uniform(0, 0.02)),
                'close': current_price,
                'volume': np.random.exponential(1000000)
            })
        
        market_data = pd.DataFrame(price_data)
        print(f"   ✅ Generated {len(market_data)} days of data")
        print(f"   📈 Price range: ${market_data['close'].min():,.2f} - ${market_data['close'].max():,.2f}")
        
        print("2️⃣  Testing Buy & Hold strategy...")
        engine1 = AdvancedBacktestEngine(initial_balance=100000)
        engine1.run_backtest(buy_and_hold_strategy, market_data, symbol='BTC')
        report1 = engine1.generate_report()
        
        print("   📊 Buy & Hold Results:")
        print(f"      • Total Return: {report1['summary']['total_return']}")
        print(f"      • Annual Return: {report1['summary']['annual_return']}")
        print(f"      • Max Drawdown: {report1['summary']['max_drawdown']}")
        print(f"      • Sharpe Ratio: {report1['summary']['sharpe_ratio']}")
        
        print("3️⃣  Testing Moving Average Crossover strategy...")
        engine2 = AdvancedBacktestEngine(initial_balance=100000)
        engine2.run_backtest(
            moving_average_crossover_strategy, 
            market_data, 
            symbol='BTC',
            fast_period=10, 
            slow_period=30
        )
        report2 = engine2.generate_report()
        
        print("   📊 MA Crossover Results:")
        print(f"      • Total Return: {report2['summary']['total_return']}")
        print(f"      • Annual Return: {report2['summary']['annual_return']}")
        print(f"      • Max Drawdown: {report2['summary']['max_drawdown']}")
        print(f"      • Sharpe Ratio: {report2['summary']['sharpe_ratio']}")
        print(f"      • Total Trades: {report2['summary']['total_trades']}")
        
        print("\n🎉 Backtesting completed successfully!")
        print("💡 Launch the dashboard for interactive backtesting: python run.py --dashboard")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during backtesting: {e}")

def run_portfolio_optimization():
    """Run example portfolio optimization"""
    print("💼 Running Example Portfolio Optimization...\n")
    
    try:
        from portfolio_optimizer import AdvancedPortfolioOptimizer
        
        print("1️⃣  Generating synthetic portfolio data...")
        # Generate data for multiple assets
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')  # 1 year daily
        assets = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT']
        
        # Correlated returns
        mean_returns = [0.001, 0.0008, 0.0005, 0.0012, 0.0007]
        cov_matrix = np.array([
            [0.0004, 0.0002, 0.0001, 0.0002, 0.0001],
            [0.0002, 0.0006, 0.0002, 0.0003, 0.0002],
            [0.0001, 0.0002, 0.0005, 0.0002, 0.0002],
            [0.0002, 0.0003, 0.0002, 0.0008, 0.0003],
            [0.0001, 0.0002, 0.0002, 0.0003, 0.0006]
        ])
        
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, len(dates))
        
        # Convert to prices
        initial_prices = [50000, 3000, 1, 100, 20]
        prices_data = {}
        
        for i, asset in enumerate(assets):
            cumulative_returns = np.cumprod(1 + returns[:, i])
            prices_data[asset] = initial_prices[i] * cumulative_returns
        
        prices_df = pd.DataFrame(prices_data, index=dates)
        print(f"   ✅ Generated data for {len(assets)} assets over {len(dates)} days")
        
        print("2️⃣  Optimizing portfolio...")
        optimizer = AdvancedPortfolioOptimizer()
        optimizer.load_price_data(prices_df)
        
        # Test different optimization objectives
        objectives = ['max_sharpe', 'min_volatility', 'risk_parity']
        
        for obj in objectives:
            print(f"\n   📊 {obj.replace('_', ' ').title()} Optimization:")
            try:
                result = optimizer.optimize_portfolio(objective=obj)
                
                print(f"      • Expected Return: {result['expected_return']*100:.2f}%")
                print(f"      • Volatility: {result['volatility']*100:.2f}%")
                print(f"      • Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                print("      • Weights:")
                for asset, weight in result['weights'].items():
                    print(f"        - {asset}: {weight*100:.1f}%")
            except Exception as e:
                print(f"      ❌ Optimization failed: {e}")
        
        print("\n🎉 Portfolio optimization completed successfully!")
        print("💡 Launch the dashboard for interactive portfolio optimization: python run.py --dashboard")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during portfolio optimization: {e}")

def train_models():
    """Train ML models with synthetic data"""
    print("🤖 Training Machine Learning Models...\n")
    
    try:
        from ml_models import AdvancedLSTM, EnsembleModel, SequencePreparator
        from feature_engineering import AdvancedFeatureEngineer
        
        print("1️⃣  Preparing training data...")
        # Generate comprehensive training data
        np.random.seed(42)
        n_samples = 2000
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='1H')
        
        # Generate realistic price series
        base_price = 50000
        returns = np.random.normal(0.0005, 0.02, n_samples)
        prices = base_price * np.cumprod(1 + returns)
        
        # Create full OHLCV dataset
        training_data = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_samples)),
            'high': prices * (1 + np.random.uniform(0, 0.025, n_samples)),
            'low': prices * (1 - np.random.uniform(0, 0.025, n_samples)),
            'close': prices,
            'volume': np.random.exponential(1000000, n_samples)
        }, index=dates)
        
        print(f"   ✅ Generated {len(training_data)} samples")
        
        print("2️⃣  Engineering features...")
        feature_engineer = AdvancedFeatureEngineer()
        featured_data = feature_engineer.engineer_all_features(training_data)
        
        # Clean data
        clean_data = featured_data.dropna()
        print(f"   ✅ Created {clean_data.shape[1]} features")
        
        if len(clean_data) < 500:
            print("   ⚠️  Insufficient clean data for training")
            return
        
        print("3️⃣  Training LSTM model...")
        # Prepare sequence data for LSTM
        sequence_length = 60
        seq_prep = SequencePreparator(sequence_length=sequence_length)
        
        # Use a subset of features for LSTM
        feature_cols = [col for col in clean_data.columns 
                       if col not in ['close', 'target_price_1h', 'target_return_1h', 'target_direction_1h']][:20]
        
        if len(feature_cols) > 0:
            feature_data = clean_data[feature_cols + ['close']].values
            target_data = clean_data['close'].values
            
            X_seq, y_seq = seq_prep.create_sequences(feature_data, target_data)
            
            if len(X_seq) > 100:
                # Train-test split
                split_idx = int(0.8 * len(X_seq))
                X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
                y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
                
                # Further split for validation
                val_split = int(0.8 * len(X_train))
                X_train_final, X_val = X_train[:val_split], X_train[val_split:]
                y_train_final, y_val = y_train[:val_split], y_train[val_split:]
                
                # Train LSTM
                lstm_config = {
                    'sequence_length': sequence_length,
                    'lstm_units': [64, 32],
                    'dense_units': [32, 16],
                    'epochs': 20,  # Reduced for demo
                    'batch_size': 32
                }
                
                lstm_model = AdvancedLSTM(lstm_config)
                lstm_model.train(X_train_final, y_train_final, X_val, y_val)
                
                # Evaluate
                lstm_pred = lstm_model.predict(X_test)
                lstm_mse = np.mean((y_test - lstm_pred.flatten()) ** 2)
                print(f"   ✅ LSTM trained - MSE: {lstm_mse:,.2f}")
            else:
                print("   ⚠️  Insufficient sequence data for LSTM training")
        
        print("4️⃣  Training ensemble model...")
        # Prepare data for ensemble
        feature_cols = feature_engineer.select_features(clean_data, 'close', k=20)
        
        if len(feature_cols) > 0:
            X_ensemble = clean_data[feature_cols].values[:-1]  # Remove last row to align with targets
            y_ensemble = clean_data['close'].values[1:]  # Predict next close
            
            if len(X_ensemble) > 100:
                split_idx = int(0.8 * len(X_ensemble))
                X_train_ens = X_ensemble[:split_idx]
                X_test_ens = X_ensemble[split_idx:]
                y_train_ens = y_ensemble[:split_idx]
                y_test_ens = y_ensemble[split_idx:]
                
                ensemble = EnsembleModel()
                ensemble.train(X_train_ens, y_train_ens)
                
                ensemble_pred = ensemble.predict(X_test_ens)
                ensemble_mse = np.mean((y_test_ens - ensemble_pred) ** 2)
                print(f"   ✅ Ensemble trained - MSE: {ensemble_mse:,.2f}")
                
                # Feature importance
                importance = ensemble.get_feature_importance()
                if importance:
                    print("   📊 Top 5 Important Features:")
                    for model_name, importances in importance.items():
                        if hasattr(importances, '__len__') and len(importances) > 0:
                            top_features = sorted(zip(feature_cols, importances), 
                                                key=lambda x: x[1], reverse=True)[:5]
                            print(f"      {model_name}:")
                            for feature, imp in top_features:
                                print(f"        • {feature}: {imp:.4f}")
                            break
            else:
                print("   ⚠️  Insufficient data for ensemble training")
        
        print("\n🎉 Model training completed successfully!")
        print("💡 Models are trained on synthetic data for demonstration purposes")
        print("💡 Use real market data and longer training for production models")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during model training: {e}")

def check_requirements():
    """Check if all required packages are installed"""
    print("🔍 Checking Requirements...\n")
    
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if not os.path.exists(requirements_file):
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open(requirements_file, 'r') as f:
            requirements = [line.strip().split('==')[0] for line in f.readlines() 
                          if line.strip() and not line.startswith('#')]
        
        missing_packages = []
        
        for package in requirements[:10]:  # Check first 10 core packages
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package}")
        
        if missing_packages:
            print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
            print("💡 Run: pip install -r requirements.txt")
            return False
        else:
            print("\n🎉 All core requirements satisfied!")
            return True
            
    except Exception as e:
        print(f"❌ Error checking requirements: {e}")
        return False

def show_platform_info():
    """Show platform information and capabilities"""
    print("""
🚀 Cryptocurrency Intelligence Platform
======================================

🌟 FEATURES:
  🤖 AI-Powered Price Predictions    - LSTM, Transformer, Ensemble models
  💼 Portfolio Optimization          - Modern Portfolio Theory, Risk Parity
  📊 Advanced Technical Analysis     - 50+ indicators, Interactive charts  
  ⚡ Real-time Trading Dashboard     - Live data, Paper trading, Signals
  🎯 Comprehensive Backtesting       - Strategy testing, Performance analytics

📊 CAPABILITIES:
  • Multi-timeframe analysis (1H, 4H, 1D, 1W)
  • 200+ engineered features
  • Risk management and portfolio construction
  • Real-time market data integration
  • Professional-grade backtesting engine

💡 GET STARTED:
  python run.py --dashboard     # Launch interactive web dashboard
  python run.py --example      # Run example analysis
  python run.py --backtest     # Test trading strategies
  python run.py --train        # Train ML models

📖 For detailed documentation, see README.md
""")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Cryptocurrency Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --dashboard          Launch the web dashboard
  python run.py --example           Run example analysis
  python run.py --backtest          Run backtesting demo
  python run.py --train-models      Train ML models
  python run.py --portfolio         Test portfolio optimization
  python run.py --check             Check requirements
        """
    )
    
    parser.add_argument('--dashboard', action='store_true', 
                       help='Launch Streamlit dashboard')
    parser.add_argument('--example', action='store_true',
                       help='Run example analysis')
    parser.add_argument('--backtest', action='store_true',
                       help='Run example backtesting')
    parser.add_argument('--train-models', action='store_true',
                       help='Train ML models')
    parser.add_argument('--portfolio', action='store_true',
                       help='Run portfolio optimization example')
    parser.add_argument('--check', action='store_true',
                       help='Check requirements')
    
    args = parser.parse_args()
    
    # Show info if no arguments
    if not any(vars(args).values()):
        show_platform_info()
        return
    
    # Execute requested action
    if args.check:
        check_requirements()
    elif args.dashboard:
        launch_dashboard()
    elif args.example:
        run_example_analysis()
    elif args.backtest:
        run_example_backtest()
    elif args.train_models:
        train_models()
    elif args.portfolio:
        run_portfolio_optimization()

if __name__ == '__main__':
    main()
