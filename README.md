BTC/USDT Trading Dashboard
Streamlit app for live BTC/USDT trading signals, aiming for >80% accuracy using XGBoost, Supertrend, and dynamic thresholds.
Setup

Install: pip install -r requirements.txt
Add API key to Streamlit secrets:CRYPTOPANIC_API_KEY = "your_cryptopanic_api_key"


Configure proxies in app.py (replace us-proxy:8080, eu-proxy:8080 with real proxy URLs).
Run: streamlit run app.py
Deploy to Streamlit Cloud via GitHub.

Notes

Binance API may return 451 errors if hosted in restricted regions (e.g., US). Use proxies to bypass.
Dune Analytics integration pending; add query ID to fetch_onchain_data later.
Backtest for >80% accuracy using backtrader.

