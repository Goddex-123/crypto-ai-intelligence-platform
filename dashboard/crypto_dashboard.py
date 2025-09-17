"""
Advanced Cryptocurrency Intelligence Dashboard
Real-time market analysis, price predictions, and portfolio optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_collector import CryptoDataCollector
    from feature_engineering import AdvancedFeatureEngineer
    from ml_models import AdvancedLSTM, EnsembleModel, TransformerModel, SequencePreparator, ModelEvaluator
    from portfolio_optimizer import AdvancedPortfolioOptimizer
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Crypto Intelligence Platform",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B35;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF6B35;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_collector' not in st.session_state:
    st.session_state.data_collector = None
if 'portfolio_optimizer' not in st.session_state:
    st.session_state.portfolio_optimizer = None
if 'prediction_models' not in st.session_state:
    st.session_state.prediction_models = {}

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0; text-align: center;">
            ₿ Cryptocurrency Intelligence Platform
        </h1>
        <p style="color: white; margin: 0; text-align: center; opacity: 0.9;">
            Advanced Analytics • AI Predictions • Portfolio Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        st.subheader("🔑 API Keys")
        with st.expander("Configure API Keys"):
            binance_api = st.text_input("Binance API Key", type="password")
            binance_secret = st.text_input("Binance Secret", type="password")
            alpha_vantage_key = st.text_input("Alpha Vantage Key", type="password")
        
        # Data Configuration
        st.subheader("📊 Data Settings")
        selected_cryptos = st.multiselect(
            "Select Cryptocurrencies",
            ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'MATIC', 'AVAX', 'LINK', 'UNI'],
            default=['BTC', 'ETH', 'ADA']
        )
        
        data_timeframe = st.selectbox(
            "Data Timeframe",
            ['1h', '4h', '1d', '1w'],
            index=2
        )
        
        data_limit = st.slider("Historical Data Points", 100, 2000, 500)
        
        # Initialize data collector
        if st.button("🚀 Initialize Data Collector"):
            config = {
                'binance_api_key': binance_api,
                'binance_secret': binance_secret,
                'alpha_vantage_key': alpha_vantage_key
            }
            
            with st.spinner("Initializing data collector..."):
                st.session_state.data_collector = CryptoDataCollector(config)
            
            st.success("Data collector initialized!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Market Overview", 
        "🤖 AI Predictions", 
        "💼 Portfolio Optimizer",
        "📊 Technical Analysis",
        "🎯 Backtesting",
        "📱 Real-time Trading"
    ])
    
    with tab1:
        market_overview_tab()
    
    with tab2:
        ai_predictions_tab(selected_cryptos, data_timeframe, data_limit)
    
    with tab3:
        portfolio_optimizer_tab(selected_cryptos)
    
    with tab4:
        technical_analysis_tab(selected_cryptos, data_timeframe)
    
    with tab5:
        backtesting_tab(selected_cryptos)
    
    with tab6:
        realtime_trading_tab(selected_cryptos)

def market_overview_tab():
    """Market overview and sentiment analysis"""
    
    st.header("🌍 Global Cryptocurrency Market Overview")
    
    # Market metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Mock market data (in real implementation, fetch from APIs)
    with col1:
        st.metric("Total Market Cap", "$2.1T", "↗️ 3.2%")
    
    with col2:
        st.metric("24h Volume", "$89.5B", "↘️ -1.8%")
    
    with col3:
        st.metric("BTC Dominance", "42.8%", "↗️ 0.5%")
    
    with col4:
        st.metric("Fear & Greed Index", "72", "↗️ Greed")
    
    # Market data visualization
    st.subheader("📊 Market Performance")
    
    # Generate sample market data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    market_data = pd.DataFrame({
        'Date': dates,
        'BTC': 50000 + np.cumsum(np.random.randn(30) * 1000),
        'ETH': 3000 + np.cumsum(np.random.randn(30) * 100),
        'BNB': 300 + np.cumsum(np.random.randn(30) * 15),
        'ADA': 1.0 + np.cumsum(np.random.randn(30) * 0.05),
        'SOL': 100 + np.cumsum(np.random.randn(30) * 5)
    })
    
    # Interactive price chart
    fig = go.Figure()
    
    for crypto in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']:
        fig.add_trace(go.Scatter(
            x=market_data['Date'],
            y=market_data[crypto],
            mode='lines',
            name=crypto,
            hovertemplate=f'{crypto}: $%{{y:,.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Cryptocurrency Price Trends (30 Days)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top gainers and losers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Top Gainers (24h)")
        gainers_data = pd.DataFrame({
            'Symbol': ['MATIC', 'AVAX', 'LINK', 'UNI', 'ATOM'],
            'Price': ['$1.23', '$18.45', '$14.67', '$6.89', '$11.22'],
            'Change': ['+15.4%', '+12.8%', '+9.3%', '+7.1%', '+5.9%']
        })
        st.dataframe(gainers_data, hide_index=True)
    
    with col2:
        st.subheader("📉 Top Losers (24h)")
        losers_data = pd.DataFrame({
            'Symbol': ['XRP', 'DOT', 'LTC', 'BCH', 'ETC'],
            'Price': ['$0.52', '$7.89', '$145.23', '$267.45', '$23.56'],
            'Change': ['-8.2%', '-6.7%', '-4.3%', '-3.9%', '-2.1%']
        })
        st.dataframe(losers_data, hide_index=True)
    
    # Market sentiment indicators
    st.subheader("🧠 Market Sentiment Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Fear & Greed Index gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=72,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fear & Greed Index"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 25], 'color': "red"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Social sentiment
        sentiment_data = pd.DataFrame({
            'Platform': ['Twitter', 'Reddit', 'News', 'YouTube'],
            'Sentiment': [0.65, 0.72, 0.58, 0.69],
            'Volume': [15420, 8932, 2341, 5678]
        })
        
        fig_sentiment = px.bar(
            sentiment_data, 
            x='Platform', 
            y='Sentiment',
            title="Social Media Sentiment",
            color='Sentiment',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col3:
        # Market indicators
        st.write("**Technical Indicators**")
        
        indicators = pd.DataFrame({
            'Indicator': ['RSI (14)', 'MACD', 'Bollinger %B', 'Stoch RSI'],
            'Value': [67.3, 'Bullish', 0.73, 82.1],
            'Signal': ['Neutral', '🟢 Buy', '🟡 Neutral', '🔴 Overbought']
        })
        
        st.dataframe(indicators, hide_index=True)

def ai_predictions_tab(selected_cryptos, data_timeframe, data_limit):
    """AI-powered price predictions"""
    
    st.header("🤖 AI-Powered Price Predictions")
    
    if not selected_cryptos:
        st.warning("Please select cryptocurrencies in the sidebar.")
        return
    
    # Model selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_model = st.selectbox(
            "Select Prediction Model",
            ["LSTM", "Transformer", "Ensemble", "All Models"]
        )
    
    with col2:
        prediction_horizon = st.selectbox(
            "Prediction Horizon",
            ["1 hour", "6 hours", "24 hours", "7 days"],
            index=2
        )
    
    with col3:
        confidence_interval = st.slider("Confidence Interval", 0.80, 0.99, 0.95)
    
    # Data collection and prediction
    if st.button("🔮 Generate Predictions"):
        
        for crypto in selected_cryptos:
            st.subheader(f"📊 {crypto} Price Prediction")
            
            with st.spinner(f"Collecting data and generating predictions for {crypto}..."):
                
                # Generate synthetic prediction data (in real implementation, use actual models)
                current_price = np.random.uniform(1, 70000)  # Mock current price
                
                # Create prediction timeline
                if prediction_horizon == "1 hour":
                    timeline = pd.date_range(start=datetime.now(), periods=60, freq='1T')
                    horizon_periods = 1
                elif prediction_horizon == "6 hours":
                    timeline = pd.date_range(start=datetime.now(), periods=360, freq='1T')
                    horizon_periods = 6
                elif prediction_horizon == "24 hours":
                    timeline = pd.date_range(start=datetime.now(), periods=24, freq='1H')
                    horizon_periods = 24
                else:  # 7 days
                    timeline = pd.date_range(start=datetime.now(), periods=7, freq='1D')
                    horizon_periods = 7
                
                # Generate synthetic predictions
                price_trend = np.cumsum(np.random.randn(len(timeline)) * 0.02) + current_price
                
                # Add confidence intervals
                upper_bound = price_trend * (1 + np.random.uniform(0.05, 0.15, len(timeline)))
                lower_bound = price_trend * (1 - np.random.uniform(0.05, 0.15, len(timeline)))
                
                # Create prediction dataframe
                predictions_df = pd.DataFrame({
                    'Timestamp': timeline,
                    'Predicted_Price': price_trend,
                    'Upper_Bound': upper_bound,
                    'Lower_Bound': lower_bound
                })
                
                # Display prediction metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${current_price:,.2f}")
                
                with col2:
                    predicted_price = predictions_df['Predicted_Price'].iloc[-1]
                    change = ((predicted_price - current_price) / current_price) * 100
                    st.metric(
                        f"Predicted ({prediction_horizon})",
                        f"${predicted_price:,.2f}",
                        f"{change:+.2f}%"
                    )
                
                with col3:
                    volatility = np.std(predictions_df['Predicted_Price']) / np.mean(predictions_df['Predicted_Price'])
                    st.metric("Predicted Volatility", f"{volatility*100:.1f}%")
                
                with col4:
                    confidence_score = np.random.uniform(0.7, 0.95)
                    st.metric("Model Confidence", f"{confidence_score*100:.1f}%")
                
                # Prediction chart
                fig = go.Figure()
                
                # Add prediction line
                fig.add_trace(go.Scatter(
                    x=predictions_df['Timestamp'],
                    y=predictions_df['Predicted_Price'],
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=predictions_df['Timestamp'],
                    y=predictions_df['Upper_Bound'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='Upper Bound'
                ))
                
                fig.add_trace(go.Scatter(
                    x=predictions_df['Timestamp'],
                    y=predictions_df['Lower_Bound'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name=f'{confidence_interval*100:.0f}% Confidence Interval',
                    fillcolor='rgba(68, 68, 68, 0.2)'
                ))
                
                # Add current price line
                fig.add_hline(
                    y=current_price,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Current Price"
                )
                
                fig.update_layout(
                    title=f"{crypto} Price Prediction - {prediction_horizon}",
                    xaxis_title="Time",
                    yaxis_title="Price ($)",
                    hovermode='x'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction accuracy metrics (mock)
                with st.expander(f"📈 {crypto} Model Performance Metrics"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Regression Metrics**")
                        metrics_df = pd.DataFrame({
                            'Metric': ['RMSE', 'MAE', 'R²', 'MAPE'],
                            'Value': [f'{np.random.uniform(100, 500):.1f}', 
                                     f'{np.random.uniform(50, 200):.1f}',
                                     f'{np.random.uniform(0.85, 0.95):.3f}',
                                     f'{np.random.uniform(2, 8):.1f}%']
                        })
                        st.dataframe(metrics_df, hide_index=True)
                    
                    with col2:
                        st.write("**Directional Accuracy**")
                        direction_metrics = pd.DataFrame({
                            'Timeframe': ['1H', '4H', '1D', '1W'],
                            'Accuracy': ['73.2%', '68.9%', '71.5%', '65.3%']
                        })
                        st.dataframe(direction_metrics, hide_index=True)
                    
                    with col3:
                        st.write("**Risk Metrics**")
                        risk_metrics = pd.DataFrame({
                            'Metric': ['Sharpe Ratio', 'Max Drawdown', 'VaR (95%)', 'Hit Rate'],
                            'Value': [f'{np.random.uniform(1.2, 2.5):.2f}',
                                     f'{np.random.uniform(-15, -5):.1f}%',
                                     f'{np.random.uniform(-8, -3):.1f}%',
                                     f'{np.random.uniform(55, 75):.1f}%']
                        })
                        st.dataframe(risk_metrics, hide_index=True)

def portfolio_optimizer_tab(selected_cryptos):
    """Portfolio optimization interface"""
    
    st.header("💼 Advanced Portfolio Optimization")
    
    if len(selected_cryptos) < 2:
        st.warning("Please select at least 2 cryptocurrencies for portfolio optimization.")
        return
    
    # Portfolio configuration
    col1, col2 = st.columns(2)
    
    with col1:
        optimization_objective = st.selectbox(
            "Optimization Objective",
            ["Maximum Sharpe Ratio", "Minimum Volatility", "Risk Parity", "Maximum Diversification"]
        )
        
        investment_amount = st.number_input("Investment Amount ($)", 
                                          min_value=1000, 
                                          max_value=1000000, 
                                          value=100000,
                                          step=1000)
    
    with col2:
        rebalance_frequency = st.selectbox(
            "Rebalancing Frequency",
            ["Daily", "Weekly", "Monthly", "Quarterly"]
        )
        
        max_weight = st.slider("Maximum Asset Weight", 0.1, 1.0, 0.4, 0.05)
    
    # Risk management settings
    with st.expander("🛡️ Risk Management Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_free_rate = st.number_input("Risk-free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
            target_volatility = st.number_input("Target Volatility (%)", 5.0, 50.0, 15.0, 1.0) / 100
        
        with col2:
            transaction_cost = st.number_input("Transaction Cost (%)", 0.0, 2.0, 0.1, 0.05) / 100
            min_weight = st.number_input("Minimum Asset Weight (%)", 0.0, 10.0, 1.0, 0.5) / 100
        
        with col3:
            lookback_window = st.number_input("Lookback Window (days)", 30, 365, 252, 10)
            confidence_level = st.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
    
    # Generate portfolio optimization
    if st.button("🎯 Optimize Portfolio"):
        
        with st.spinner("Optimizing portfolio..."):
            
            # Generate synthetic price data for selected cryptos
            dates = pd.date_range(end=datetime.now(), periods=lookback_window, freq='D')
            
            price_data = {}
            for crypto in selected_cryptos:
                # Generate realistic price series
                initial_price = np.random.uniform(0.1, 70000)
                returns = np.random.normal(0.001, 0.03, lookback_window)  # Daily returns
                prices = initial_price * np.cumprod(1 + returns)
                price_data[crypto] = prices
            
            prices_df = pd.DataFrame(price_data, index=dates)
            
            # Initialize portfolio optimizer
            if st.session_state.portfolio_optimizer is None:
                st.session_state.portfolio_optimizer = AdvancedPortfolioOptimizer()
            
            optimizer = st.session_state.portfolio_optimizer
            optimizer.load_price_data(prices_df)
            
            # Map objectives
            objective_map = {
                "Maximum Sharpe Ratio": "max_sharpe",
                "Minimum Volatility": "min_volatility", 
                "Risk Parity": "risk_parity",
                "Maximum Diversification": "max_diversification"
            }
            
            # Optimize portfolio
            try:
                result = optimizer.optimize_portfolio(
                    objective=objective_map[optimization_objective]
                )
                
                # Display results
                st.success("✅ Portfolio optimization completed!")
                
                # Portfolio metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Expected Annual Return", f"{result['expected_return']*100:.2f}%")
                
                with col2:
                    st.metric("Annual Volatility", f"{result['volatility']*100:.2f}%")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                
                with col4:
                    total_assets = len([w for w in result['weights'].values() if w > 0.01])
                    st.metric("Active Assets", total_assets)
                
                # Portfolio allocation chart
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart of allocations
                    weights = result['weights']
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(weights.keys()),
                        values=list(weights.values()),
                        hole=.3,
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    
                    fig_pie.update_layout(
                        title="Portfolio Allocation",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart of weights
                    weights_df = pd.DataFrame({
                        'Asset': list(weights.keys()),
                        'Weight': list(weights.values()),
                        'Amount': [w * investment_amount for w in weights.values()]
                    })
                    
                    fig_bar = px.bar(
                        weights_df,
                        x='Asset',
                        y='Weight',
                        title="Asset Weights",
                        text='Weight',
                        hover_data=['Amount']
                    )
                    
                    fig_bar.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                    fig_bar.update_layout(yaxis_tickformat='.1%')
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Detailed allocation table
                st.subheader("📊 Detailed Portfolio Allocation")
                
                allocation_df = pd.DataFrame({
                    'Asset': list(weights.keys()),
                    'Weight (%)': [f"{w*100:.2f}%" for w in weights.values()],
                    'Amount ($)': [f"${w * investment_amount:,.2f}" for w in weights.values()],
                    'Current Price': [f"${prices_df[asset].iloc[-1]:,.2f}" for asset in weights.keys()],
                    'Shares': [f"{(w * investment_amount) / prices_df[asset].iloc[-1]:.4f}" for asset, w in weights.items()]
                })
                
                st.dataframe(allocation_df, hide_index=True)
                
                # Risk metrics
                risk_metrics = optimizer.calculate_risk_metrics(weights)
                
                st.subheader("⚖️ Risk Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Return Metrics**")
                    return_metrics = pd.DataFrame({
                        'Metric': ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio'],
                        'Value': [f"{risk_metrics['annual_return']*100:.2f}%",
                                 f"{risk_metrics['annual_volatility']*100:.2f}%", 
                                 f"{risk_metrics['sharpe_ratio']:.2f}",
                                 f"{risk_metrics['sortino_ratio']:.2f}"]
                    })
                    st.dataframe(return_metrics, hide_index=True)
                
                with col2:
                    st.write("**Risk Metrics**")
                    risk_df = pd.DataFrame({
                        'Metric': ['Max Drawdown', 'VaR (95%)', 'CVaR (95%)', 'Calmar Ratio'],
                        'Value': [f"{risk_metrics['max_drawdown']*100:.2f}%",
                                 f"{risk_metrics['var_95']*100:.2f}%",
                                 f"{risk_metrics['cvar_95']*100:.2f}%",
                                 f"{risk_metrics['calmar_ratio']:.2f}"]
                    })
                    st.dataframe(risk_df, hide_index=True)
                
                with col3:
                    st.write("**Advanced Metrics**")
                    advanced_df = pd.DataFrame({
                        'Metric': ['Skewness', 'Kurtosis', 'Omega Ratio', 'Tail Ratio'],
                        'Value': [f"{risk_metrics['skewness']:.2f}",
                                 f"{risk_metrics['kurtosis']:.2f}",
                                 f"{risk_metrics['omega_ratio']:.2f}",
                                 f"{risk_metrics['tail_ratio']:.2f}"]
                    })
                    st.dataframe(advanced_df, hide_index=True)
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")

def technical_analysis_tab(selected_cryptos, data_timeframe):
    """Technical analysis and charting"""
    
    st.header("📊 Advanced Technical Analysis")
    
    if not selected_cryptos:
        st.warning("Please select cryptocurrencies in the sidebar.")
        return
    
    # Asset selection for detailed analysis
    selected_asset = st.selectbox("Select Asset for Detailed Analysis", selected_cryptos)
    
    # Technical indicators configuration
    with st.expander("🔧 Technical Indicators Settings"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ma_periods = st.multiselect("Moving Averages", [7, 25, 50, 200], default=[25, 50])
            rsi_period = st.number_input("RSI Period", 5, 30, 14)
        
        with col2:
            bb_period = st.number_input("Bollinger Bands Period", 10, 50, 20)
            bb_std = st.number_input("BB Standard Deviations", 1.0, 3.0, 2.0, 0.1)
        
        with col3:
            macd_fast = st.number_input("MACD Fast Period", 5, 20, 12)
            macd_slow = st.number_input("MACD Slow Period", 20, 40, 26)
        
        with col4:
            stoch_k = st.number_input("Stochastic %K", 5, 20, 14)
            stoch_d = st.number_input("Stochastic %D", 2, 10, 3)
    
    # Generate synthetic OHLCV data
    if st.button("📈 Generate Technical Analysis"):
        
        with st.spinner(f"Generating technical analysis for {selected_asset}..."):
            
            # Create synthetic OHLCV data
            periods = 200
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
            
            # Generate realistic price movement
            base_price = np.random.uniform(0.1, 70000)
            returns = np.random.normal(0, 0.02, periods)
            prices = base_price * np.cumprod(1 + returns)
            
            # Generate OHLCV data
            ohlcv_data = pd.DataFrame({
                'Open': prices * (1 + np.random.uniform(-0.02, 0.02, periods)),
                'High': prices * (1 + np.random.uniform(0, 0.05, periods)),
                'Low': prices * (1 - np.random.uniform(0, 0.05, periods)),
                'Close': prices,
                'Volume': np.random.exponential(1000000, periods)
            }, index=dates)
            
            # Ensure OHLC logic
            ohlcv_data['High'] = ohlcv_data[['Open', 'High', 'Close']].max(axis=1)
            ohlcv_data['Low'] = ohlcv_data[['Open', 'Low', 'Close']].min(axis=1)
            
            # Calculate technical indicators
            df = ohlcv_data.copy()
            
            # Moving averages
            for period in ma_periods:
                df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
            bb_std_dev = df['Close'].rolling(window=bb_period).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
            
            # MACD
            ema_fast = df['Close'].ewm(span=macd_fast).mean()
            ema_slow = df['Close'].ewm(span=macd_slow).mean()
            df['MACD'] = ema_fast - ema_slow
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Stochastic Oscillator
            lowest_low = df['Low'].rolling(window=stoch_k).min()
            highest_high = df['High'].rolling(window=stoch_k).max()
            df['Stoch_K'] = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
            df['Stoch_D'] = df['Stoch_K'].rolling(window=stoch_d).mean()
            
            # Create multi-panel chart
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(f'{selected_asset} Price & Volume', 'RSI', 'MACD', 'Stochastic'),
                row_heights=[0.5, 0.15, 0.15, 0.15]
            )
            
            # Price and volume subplot
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            colors = ['blue', 'orange', 'green', 'red']
            for i, period in enumerate(ma_periods):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[f'MA_{period}'],
                        mode='lines',
                        name=f'MA {period}',
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=1, col=1
                )
            
            # Add Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)'
                ),
                row=1, col=1
            )
            
            # Volume bars
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    yaxis='y2',
                    opacity=0.3
                ),
                row=1, col=1
            )
            
            # RSI
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # MACD
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD_Signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='red')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color='green'
                ),
                row=3, col=1
            )
            
            # Stochastic
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Stoch_K'],
                    mode='lines',
                    name='%K',
                    line=dict(color='blue')
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Stoch_D'],
                    mode='lines',
                    name='%D',
                    line=dict(color='red')
                ),
                row=4, col=1
            )
            
            # Stochastic levels
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
            
            # Update layout
            fig.update_layout(
                title=f"{selected_asset} Technical Analysis",
                height=800,
                xaxis_rangeslider_visible=False,
                showlegend=True
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="Stochastic", row=4, col=1, range=[0, 100])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical analysis summary
            st.subheader("📋 Technical Analysis Summary")
            
            # Current indicator values
            current_rsi = df['RSI'].iloc[-1]
            current_macd = df['MACD'].iloc[-1]
            current_stoch_k = df['Stoch_K'].iloc[-1]
            
            # Generate signals
            signals = []
            
            # RSI signals
            if current_rsi > 70:
                signals.append(("RSI", "🔴 OVERBOUGHT", f"{current_rsi:.1f}"))
            elif current_rsi < 30:
                signals.append(("RSI", "🟢 OVERSOLD", f"{current_rsi:.1f}"))
            else:
                signals.append(("RSI", "🟡 NEUTRAL", f"{current_rsi:.1f}"))
            
            # MACD signals
            if current_macd > df['MACD_Signal'].iloc[-1]:
                signals.append(("MACD", "🟢 BULLISH", f"{current_macd:.4f}"))
            else:
                signals.append(("MACD", "🔴 BEARISH", f"{current_macd:.4f}"))
            
            # Stochastic signals
            if current_stoch_k > 80:
                signals.append(("Stochastic", "🔴 OVERBOUGHT", f"{current_stoch_k:.1f}%"))
            elif current_stoch_k < 20:
                signals.append(("Stochastic", "🟢 OVERSOLD", f"{current_stoch_k:.1f}%"))
            else:
                signals.append(("Stochastic", "🟡 NEUTRAL", f"{current_stoch_k:.1f}%"))
            
            signals_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Value'])
            st.dataframe(signals_df, hide_index=True)

def backtesting_tab(selected_cryptos):
    """Backtesting interface"""
    
    st.header("🎯 Strategy Backtesting")
    
    if not selected_cryptos:
        st.warning("Please select cryptocurrencies in the sidebar.")
        return
    
    # Backtesting configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Backtest Period")
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())
        
        st.subheader("💰 Trading Parameters")
        initial_capital = st.number_input("Initial Capital ($)", 1000, 1000000, 100000)
        transaction_cost = st.number_input("Transaction Cost (%)", 0.0, 2.0, 0.1) / 100
    
    with col2:
        st.subheader("📈 Strategy Configuration")
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Buy & Hold", "Moving Average Crossover", "RSI Mean Reversion", "Portfolio Rebalancing"]
        )
        
        if strategy_type == "Moving Average Crossover":
            fast_ma = st.number_input("Fast MA Period", 5, 50, 10)
            slow_ma = st.number_input("Slow MA Period", 20, 200, 50)
        elif strategy_type == "RSI Mean Reversion":
            rsi_period = st.number_input("RSI Period", 5, 30, 14)
            rsi_oversold = st.number_input("RSI Oversold Level", 10, 40, 30)
            rsi_overbought = st.number_input("RSI Overbought Level", 60, 90, 70)
        elif strategy_type == "Portfolio Rebalancing":
            rebalance_freq = st.selectbox("Rebalancing Frequency", ["Weekly", "Monthly", "Quarterly"])
    
    # Run backtest
    if st.button("🚀 Run Backtest"):
        
        with st.spinner("Running backtest..."):
            
            # Generate synthetic historical data
            days = (end_date - start_date).days
            if days < 30:
                st.error("Please select a longer backtest period (minimum 30 days)")
                return
            
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate price data for selected assets
            portfolio_data = {}
            for crypto in selected_cryptos:
                initial_price = np.random.uniform(0.1, 70000)
                returns = np.random.normal(0.001, 0.03, len(dates))  # Daily returns
                prices = initial_price * np.cumprod(1 + returns)
                portfolio_data[crypto] = prices
            
            prices_df = pd.DataFrame(portfolio_data, index=dates)
            
            # Simulate strategy performance
            if strategy_type == "Buy & Hold":
                # Equal weight buy and hold
                weights = {crypto: 1/len(selected_cryptos) for crypto in selected_cryptos}
                portfolio_returns = (prices_df.pct_change() * pd.Series(weights)).sum(axis=1)
                
            elif strategy_type == "Moving Average Crossover":
                # Simple MA crossover for first asset
                main_asset = selected_cryptos[0]
                prices = prices_df[main_asset]
                
                fast_ma_series = prices.rolling(window=fast_ma).mean()
                slow_ma_series = prices.rolling(window=slow_ma).mean()
                
                # Generate signals
                signals = np.where(fast_ma_series > slow_ma_series, 1, 0)  # 1 = long, 0 = cash
                positions = pd.Series(signals, index=prices.index).diff()
                
                # Calculate returns
                asset_returns = prices.pct_change()
                strategy_returns = signals[:-1] * asset_returns[1:]  # Apply signals to returns
                portfolio_returns = pd.Series(strategy_returns, index=prices.index[1:])
                
            elif strategy_type == "RSI Mean Reversion":
                # RSI strategy for first asset
                main_asset = selected_cryptos[0]
                prices = prices_df[main_asset]
                
                # Calculate RSI
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                # Generate signals
                signals = np.where(rsi < rsi_oversold, 1,  # Buy when oversold
                         np.where(rsi > rsi_overbought, -1, 0))  # Sell when overbought
                
                asset_returns = prices.pct_change()
                strategy_returns = signals[:-1] * asset_returns[1:]
                portfolio_returns = pd.Series(strategy_returns, index=prices.index[1:])
                
            else:  # Portfolio Rebalancing
                # Monthly rebalancing to equal weights
                equal_weights = {crypto: 1/len(selected_cryptos) for crypto in selected_cryptos}
                portfolio_returns = (prices_df.pct_change() * pd.Series(equal_weights)).sum(axis=1)
            
            # Remove NaN values
            portfolio_returns = portfolio_returns.dropna()
            
            # Calculate cumulative performance
            portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
            
            # Calculate benchmark (equal-weighted buy & hold)
            benchmark_weights = {crypto: 1/len(selected_cryptos) for crypto in selected_cryptos}
            benchmark_returns = (prices_df.pct_change() * pd.Series(benchmark_weights)).sum(axis=1).dropna()
            benchmark_value = initial_capital * (1 + benchmark_returns).cumprod()
            
            # Performance metrics
            total_return = (portfolio_value.iloc[-1] - initial_capital) / initial_capital
            benchmark_return = (benchmark_value.iloc[-1] - initial_capital) / initial_capital
            
            annual_return = (1 + total_return) ** (365 / len(portfolio_returns)) - 1
            annual_vol = portfolio_returns.std() * np.sqrt(365)
            sharpe_ratio = (annual_return - 0.02) / annual_vol  # Assuming 2% risk-free rate
            
            max_drawdown = ((portfolio_value / portfolio_value.cummax()) - 1).min()
            
            # Display results
            st.success("✅ Backtest completed!")
            
            # Performance summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{total_return*100:.2f}%")
            
            with col2:
                st.metric("Annual Return", f"{annual_return*100:.2f}%")
            
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            with col4:
                st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
            
            # Performance comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Strategy Return", f"{total_return*100:.2f}%")
                st.metric("Strategy Volatility", f"{annual_vol*100:.2f}%")
            
            with col2:
                st.metric("Benchmark Return", f"{benchmark_return*100:.2f}%", 
                         f"{(total_return - benchmark_return)*100:+.2f}%")
                benchmark_vol = benchmark_returns.std() * np.sqrt(365)
                st.metric("Benchmark Volatility", f"{benchmark_vol*100:.2f}%")
            
            # Performance chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode='lines',
                name='Strategy',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=benchmark_value.index,
                y=benchmark_value.values,
                mode='lines',
                name='Benchmark (Buy & Hold)',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Performance vs Benchmark",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed performance metrics
            st.subheader("📊 Detailed Performance Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Return Metrics**")
                return_metrics = pd.DataFrame({
                    'Metric': ['Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio'],
                    'Strategy': [f"{total_return*100:.2f}%", f"{annual_return*100:.2f}%", 
                               f"{annual_vol*100:.2f}%", f"{sharpe_ratio:.2f}"],
                    'Benchmark': [f"{benchmark_return*100:.2f}%", 
                                f"{(1 + benchmark_return)**(365/len(benchmark_returns)) - 1:.2%}",
                                f"{benchmark_vol*100:.2f}%", 
                                f"{((1 + benchmark_return)**(365/len(benchmark_returns)) - 1 - 0.02) / benchmark_vol:.2f}"]
                })
                st.dataframe(return_metrics, hide_index=True)
            
            with col2:
                st.write("**Risk Metrics**")
                
                # Calculate additional risk metrics
                portfolio_dd = ((portfolio_value / portfolio_value.cummax()) - 1)
                benchmark_dd = ((benchmark_value / benchmark_value.cummax()) - 1)
                
                risk_metrics = pd.DataFrame({
                    'Metric': ['Max Drawdown', 'VaR (95%)', 'Skewness', 'Kurtosis'],
                    'Strategy': [f"{max_drawdown*100:.2f}%", 
                               f"{np.percentile(portfolio_returns, 5)*100:.2f}%",
                               f"{portfolio_returns.skew():.2f}",
                               f"{portfolio_returns.kurtosis():.2f}"],
                    'Benchmark': [f"{benchmark_dd.min()*100:.2f}%",
                                f"{np.percentile(benchmark_returns, 5)*100:.2f}%", 
                                f"{benchmark_returns.skew():.2f}",
                                f"{benchmark_returns.kurtosis():.2f}"]
                })
                st.dataframe(risk_metrics, hide_index=True)
            
            with col3:
                st.write("**Trading Statistics**")
                
                # Calculate number of trades (simplified)
                n_trades = len(portfolio_returns) // 10  # Mock calculation
                win_rate = np.random.uniform(0.45, 0.65)  # Mock win rate
                avg_win = portfolio_returns[portfolio_returns > 0].mean()
                avg_loss = portfolio_returns[portfolio_returns < 0].mean()
                
                trading_stats = pd.DataFrame({
                    'Metric': ['Number of Trades', 'Win Rate', 'Avg Win', 'Avg Loss'],
                    'Value': [f"{n_trades}", f"{win_rate*100:.1f}%", 
                            f"{avg_win*100:.2f}%", f"{avg_loss*100:.2f}%"]
                })
                st.dataframe(trading_stats, hide_index=True)

def realtime_trading_tab(selected_cryptos):
    """Real-time trading interface"""
    
    st.header("📱 Real-time Trading Dashboard")
    
    if not selected_cryptos:
        st.warning("Please select cryptocurrencies in the sidebar.")
        return
    
    # Trading configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Trading Configuration")
        trading_mode = st.selectbox("Trading Mode", ["Paper Trading", "Live Trading"])
        if trading_mode == "Live Trading":
            st.warning("⚠️ Live trading mode requires proper API keys and risk management!")
        
        portfolio_size = st.number_input("Portfolio Size ($)", 1000, 1000000, 10000)
        max_position_size = st.slider("Max Position Size (%)", 1, 50, 20)
        
        risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0)
        stop_loss = st.slider("Stop Loss (%)", 1, 10, 5)
        take_profit = st.slider("Take Profit (%)", 2, 20, 10)
    
    with col2:
        st.subheader("🚨 Risk Management")
        daily_loss_limit = st.number_input("Daily Loss Limit ($)", 100, 10000, 1000)
        max_open_positions = st.number_input("Max Open Positions", 1, 10, 3)
        
        auto_trading = st.checkbox("Enable Auto Trading")
        if auto_trading:
            st.info("⚡ Auto trading is enabled. Positions will be managed automatically.")
    
    # Current positions
    st.subheader("📊 Current Positions")
    
    # Mock current positions
    positions_data = pd.DataFrame({
        'Symbol': ['BTC-USDT', 'ETH-USDT'],
        'Side': ['Long', 'Short'],
        'Size': [0.5, 2.0],
        'Entry Price': [45000, 2800],
        'Current Price': [46200, 2750],
        'P&L': ['+$600', '-$100'],
        'P&L %': ['+2.67%', '-1.79%']
    })
    
    if not positions_data.empty:
        st.dataframe(positions_data, hide_index=True)
        
        # Position management buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔴 Close All Positions"):
                st.success("All positions closed!")
        
        with col2:
            if st.button("⏸️ Pause Auto Trading"):
                st.info("Auto trading paused.")
        
        with col3:
            if st.button("📊 Update Positions"):
                st.success("Positions updated!")
    else:
        st.info("No open positions.")
    
    # Trading signals
    st.subheader("⚡ Real-time Trading Signals")
    
    # Generate mock signals
    signals_data = []
    for crypto in selected_cryptos[:3]:  # Show signals for first 3 assets
        signal_strength = np.random.choice(['Strong', 'Medium', 'Weak'])
        signal_type = np.random.choice(['Buy', 'Sell', 'Hold'])
        confidence = np.random.uniform(0.6, 0.95)
        price = np.random.uniform(0.1, 70000)
        
        color = "🟢" if signal_type == "Buy" else "🔴" if signal_type == "Sell" else "🟡"
        
        signals_data.append({
            'Asset': crypto,
            'Signal': f"{color} {signal_type}",
            'Strength': signal_strength,
            'Confidence': f"{confidence*100:.1f}%",
            'Price': f"${price:,.2f}",
            'Timestamp': datetime.now().strftime("%H:%M:%S")
        })
    
    signals_df = pd.DataFrame(signals_data)
    st.dataframe(signals_df, hide_index=True)
    
    # Order placement
    st.subheader("📝 Place Order")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        order_asset = st.selectbox("Select Asset", selected_cryptos)
        order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop"])
        order_side = st.selectbox("Side", ["Buy", "Sell"])
    
    with col2:
        order_quantity = st.number_input("Quantity", min_value=0.001, value=1.0, step=0.001)
        if order_type == "Limit":
            limit_price = st.number_input("Limit Price ($)", min_value=0.01, value=50000.0)
        elif order_type == "Stop":
            stop_price = st.number_input("Stop Price ($)", min_value=0.01, value=50000.0)
    
    with col3:
        st.write("**Order Summary**")
        estimated_cost = order_quantity * 50000  # Mock price
        st.write(f"Estimated Cost: ${estimated_cost:,.2f}")
        st.write(f"Trading Fee: ${estimated_cost * 0.001:.2f}")
        
        if st.button("🚀 Place Order", type="primary"):
            if trading_mode == "Paper Trading":
                st.success(f"Paper order placed: {order_side} {order_quantity} {order_asset}")
            else:
                st.warning("Live trading not implemented in demo")
    
    # Account summary
    st.subheader("💰 Account Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Account Balance", f"${portfolio_size:,.2f}")
        
    with col2:
        available_balance = portfolio_size * 0.8  # Mock available balance
        st.metric("Available Balance", f"${available_balance:,.2f}")
    
    with col3:
        unrealized_pnl = 500  # Mock P&L
        st.metric("Unrealized P&L", f"${unrealized_pnl:+,.2f}")
    
    with col4:
        daily_pnl = 250  # Mock daily P&L
        st.metric("Today's P&L", f"${daily_pnl:+,.2f}")
    
    # Real-time price feed
    st.subheader("📈 Real-time Price Feed")
    
    # Create real-time price chart placeholder
    price_placeholder = st.empty()
    
    if st.button("📊 Start Price Feed"):
        # In a real implementation, this would connect to a WebSocket feed
        price_data = []
        timestamps = []
        
        for i in range(50):
            timestamp = datetime.now() - timedelta(minutes=49-i)
            price = 50000 + np.random.randn() * 100
            price_data.append(price)
            timestamps.append(timestamp)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=price_data,
            mode='lines',
            name='BTC-USDT',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title="Real-time BTC Price",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=400
        )
        
        price_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Trading alerts
    st.subheader("🔔 Trading Alerts")
    
    alerts_data = pd.DataFrame({
        'Time': ['14:30:15', '14:25:32', '14:20:18'],
        'Type': ['Price Alert', 'Position Closed', 'Signal Generated'],
        'Message': [
            'BTC reached target price of $46,000',
            'ETH short position closed with +$150 profit', 
            'Strong buy signal detected for ADA'
        ],
        'Status': ['🟢 Active', '✅ Completed', '🟡 Pending']
    })
    
    st.dataframe(alerts_data, hide_index=True)

# Run the app
if __name__ == "__main__":
    main()
