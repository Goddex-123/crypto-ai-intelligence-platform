"""
Cryptocurrency Intelligence Platform
Streamlit Cloud Deployment
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Crypto Intelligence Platform",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

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

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    selected_cryptos = st.multiselect(
        "Select Cryptocurrencies",
        ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'MATIC', 'AVAX'],
        default=['BTC', 'ETH', 'ADA']
    )
    
    data_timeframe = st.selectbox("Timeframe", ['1h', '4h', '1d', '1w'], index=2)

# Main content
tab1, tab2, tab3 = st.tabs(["📈 Market Overview", "🤖 AI Predictions", "💼 Portfolio"])

with tab1:
    st.header("🌍 Global Cryptocurrency Market")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Market Cap", "$2.1T", "↗️ 3.2%")
    with col2:
        st.metric("24h Volume", "$89.5B", "↘️ -1.8%")
    with col3:
        st.metric("BTC Dominance", "42.8%", "↗️ 0.5%")
    with col4:
        st.metric("Fear & Greed", "72", "Greed")
    
    # Sample chart
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    market_data = pd.DataFrame({
        'Date': dates,
        'BTC': 50000 + np.cumsum(np.random.randn(30) * 1000),
        'ETH': 3000 + np.cumsum(np.random.randn(30) * 100),
    })
    
    fig = go.Figure()
    for crypto in ['BTC', 'ETH']:
        fig.add_trace(go.Scatter(x=market_data['Date'], y=market_data[crypto], 
                                  mode='lines', name=crypto))
    fig.update_layout(title="Price Trends (30 Days)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🤖 AI Price Predictions")
    st.info("Select cryptocurrencies in sidebar and click Generate Predictions")
    
    if st.button("🔮 Generate Predictions"):
        for crypto in selected_cryptos:
            current_price = np.random.uniform(1, 70000)
            predicted = current_price * (1 + np.random.randn() * 0.1)
            change = ((predicted - current_price) / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{crypto} Current", f"${current_price:,.2f}")
            with col2:
                st.metric(f"{crypto} Predicted (24h)", f"${predicted:,.2f}", f"{change:+.2f}%")
            with col3:
                st.metric("Confidence", f"{np.random.uniform(70, 95):.1f}%")

with tab3:
    st.header("💼 Portfolio Optimizer")
    
    investment = st.number_input("Investment Amount ($)", 1000, 1000000, 100000)
    
    if st.button("🎯 Optimize Portfolio"):
        weights = np.random.dirichlet(np.ones(len(selected_cryptos)))
        
        fig = go.Figure(data=[go.Pie(labels=selected_cryptos, values=weights, hole=.3)])
        fig.update_layout(title="Optimal Portfolio Allocation")
        st.plotly_chart(fig, use_container_width=True)
        
        for crypto, weight in zip(selected_cryptos, weights):
            st.write(f"**{crypto}**: {weight*100:.1f}% (${investment*weight:,.2f})")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>₿ Crypto Intelligence Platform | Educational Demo</p>", 
            unsafe_allow_html=True)
