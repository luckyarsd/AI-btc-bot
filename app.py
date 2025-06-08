import streamlit as st
import pandas as pd
import numpy as np
import requests
import aiohttp
import asyncio
import ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from transformers import pipeline
import datetime
import pytz

# Configuration
SYMBOL = "BTCUSDT"
CRYPTOPANIC_API_KEY = "YOUR_CRYPTOPANIC_API_KEY"  # Replace
GLASSNODE_API_KEY = "YOUR_GLASSNODE_API_KEY"      # Replace or set to None

# Fetch Binance OHLCV
@st.cache_data(ttl=300)
def fetch_binance_ohlcv(symbol, interval, limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Apply indicators
def apply_indicators(df):
    df['ema20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'], df['macd_signal'] = macd.macd(), macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    # True Supertrend
    df['upperband'] = ((df['high'] + df['low']) / 2) + (3 * df['atr'])
    df['lowerband'] = ((df['high'] + df['low']) / 2) - (3 * df['atr'])
    df['supertrend'] = df['close'].copy()
    for i in range(1, len(df)):
        df['supertrend'].iloc[i] = df['lowerband'].iloc[i] if df['close'].iloc[i-1] > df['upperband'].iloc[i-1] else df['upperband'].iloc[i]
    return df

# Detect engulfing
def detect_engulfing(df):
    if len(df) < 2:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    if (last['close'] > last['open'] and prev['close'] < prev['open'] and 
        last['close'] > prev['open'] and last['open'] < prev['close']):
        return 'bullish'
    if (last['close'] < last['open'] and prev['close'] > prev['open'] and 
        last['open'] > prev['close'] and last['close'] < prev['open']):
        return 'bearish'
    return None

# News sentiment
async def fetch_news_sentiment():
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}¤cies=BTC&public=true"
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
                headlines = [post['title'] for post in data['results'][:10]]
                score = 0
                for h in headlines:
                    result = sentiment_analyzer(h)[0]
                    score += result['score'] if result['label'] == 'POSITIVE' else -result['score']
                return score, headlines
    except Exception as e:
        st.warning(f"News sentiment failed: {e}. Using neutral score.")
        return 0, []

# On-chain data (placeholder)
async def fetch_onchain_data():
    return 0  # Replace with Glassnode API call if key available

# Prepare ML features
def prepare_features(df_5m, df_1h, df_4h, news_score, onchain_score):
    engulf = 1 if detect_engulfing(df_5m) == 'bullish' else -1 if detect_engulfing(df_5m) == 'bearish' else 0
    features = {
        'rsi': df_5m['rsi'].iloc[-1],
        'macd': df_5m['macd'].iloc[-1],
        'macd_signal': df_5m['macd_signal'].iloc[-1],
        'atr': df_5m['atr'].iloc[-1],
        'adx': df_5m['adx'].iloc[-1],
        'supertrend': 1 if df_5m['close'].iloc[-1] < df_5m['supertrend'].iloc[-1] else -1,
        'engulfing': engulf,
        'ema_diff_1h': df_1h['ema20'].iloc[-1] - df_1h['ema50'].iloc[-1],
        'ema_diff_4h': df_4h['ema20'].iloc[-1] - df_4h['ema50'].iloc[-1],
        'news_score': news_score,
        'onchain_score': onchain_score
    }
    return pd.DataFrame([features])

# Train ML model
@st.cache_resource
def train_ml_model(df_5m, df_1h, df_4h, news_score):
    X = pd.DataFrame()
    for i in range(20, len(df_5m)-3):
        row = prepare_features(df_5m.iloc[:i+1], df_1h, df_4h, news_score, 0)
        X = pd.concat([X, row], ignore_index=True)
    y = (df_5m['close'].shift(-3) > df_5m['close']).iloc[20:-3].astype(int)
    if len(X) > 0 and len(X) == len(y):
        model = XGBClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model
    return None

# Dynamic threshold
def dynamic_threshold(df_5m):
    adx = df_5m['adx'].iloc[-1]
    atr = df_5m['atr'].iloc[-1]
    base_threshold = 0.8
    if adx < 20:
        return base_threshold + 0.1
    if atr > df_5m['atr'].mean():
        return base_threshold - 0.1
    return base_threshold

# Generate signal
def ml_signal(df_5m, df_1h, df_4h, news_score, onchain_score, model):
    if model is None:
        return 'WAIT', 0.5
    features = prepare_features(df_5m, df_1h, df_4h, news_score, onchain_score)
    prob = model.predict_proba(features)[0][1]
    return 'BUY' if prob > 0.8 else 'SELL' if prob < 0.2 else 'WAIT', prob

# Streamlit app
async def main():
    st.set_page_config(page_title="BTC/USDT Trading Dashboard", layout="wide")
    st.title("🌟 BTC/USDT Trading Dashboard 🌟")
    
    # Fetch data
    with st.spinner("Fetching market data..."):
        df_5m = apply_indicators(fetch_binance_ohlcv(SYMBOL, "5m"))
        df_1h = apply_indicators(fetch_binance_ohlcv(SYMBOL, "1h"))
        df_4h = apply_indicators(fetch_binance_ohlcv(SYMBOL, "4h"))
        news_score, headlines = await fetch_news_sentiment()
        onchain_score = await fetch_onchain_data()
    
    # Train model
    model = train_ml_model(df_5m, df_1h, df_4h, news_score)
    
    # Generate signal
    signal, prob = ml_signal(df_5m, df_1h, df_4h, news_score, onchain_score, model)
    threshold = dynamic_threshold(df_5m)
    if signal != 'WAIT' and prob < threshold:
        signal = 'WAIT'
    
    # Risk management
    last = df_5m.iloc[-1]
    sl = last['close'] - 2 * last['atr'] if signal == 'BUY' else last['close'] + 2 * last['atr'] if signal == 'SELL' else None
    tp = last['close'] + 4 * last['atr'] if signal == 'BUY' else last['close'] - 4 * last['atr'] if signal == 'SELL' else None
    
    # Layout
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Candlestick Chart")
        fig = go.Figure(data=[
            go.Candlestick(x=df_5m['timestamp'], open=df_5m['open'], high=df_5m['high'], low=df_5m['low'], close=df_5m['close']),
            go.Scatter(x=df_5m['timestamp'], y=df_5m['supertrend'], name="Supertrend", line=dict(color='purple'))
        ])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📢 Trading Signal")
        st.markdown(f"**Signal**: {signal} (Confidence: {prob:.2%})")
        st.markdown(f"**Price**: ${last['close']:.2f}")
        if signal != 'WAIT':
            st.markdown(f"**Stop Loss**: ${sl:.2f}")
            st.markdown(f"**Take Profit**: ${tp:.2f}")
        st.markdown(f"**RSI**: {last['rsi']:.2f}")
        st.markdown(f"**MACD**: {last['macd']:.4f}")
        st.markdown(f"**News Sentiment**: {news_score:.2f}")
        st.markdown(f"**On-Chain Score**: {onchain_score}")
        st.markdown("**Top News**:")
        for h in headlines[:5]:
            st.markdown(f"- {h}")
    
    st.markdown(f"**Last Updated**: {datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')}")
    st.button("🔄 Refresh")
    st.info("Note: ML model is trained on limited data. For >80% accuracy, backtest with historical data.")

if __name__ == "__main__":
    asyncio.run(main())
