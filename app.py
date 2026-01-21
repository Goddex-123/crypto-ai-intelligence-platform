"""
Cryptocurrency Intelligence Platform
Root app.py for Streamlit Cloud deployment
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import and run the dashboard
from dashboard.crypto_dashboard import main

# Run main directly (Streamlit runs top-level code)
main()

