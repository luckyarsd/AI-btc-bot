# BTC/USDT Trading Dashboard

Streamlit app for live BTC/USDT trading signals, aiming for >80% accuracy using XGBoost, Supertrend, and dynamic thresholds.

## Setup
1. Install: `pip install -r requirements.txt`
2. Add API key to Streamlit Cloud secrets (Management > Settings > Secrets):
   ```toml
   CRYPTOPANIC_API_KEY = "your-secret-api-key"
   ```
   For local testing, create `.secrets.toml` in project root:
   ```toml
   CRYPTOPANIC_API_KEY = "your-secret-api-key"
   ```
3. Configure proxies in `app.py` (replace `us-proxy:8080`, `eu-proxy:8080` with real proxy URLs, e.g., from free-proxy-list.net).
4. Run: `streamlit run app.py`
5. Deploy to Streamlit Cloud via GitHub.

## Notes
- **Security**: Never hardcode API keys in `app.py`. Use Streamlit secrets to protect sensitive data.
- **Binance API**: May return 451 errors if hosted in restricted regions (e.g., US). Use proxies to bypass restrictions.
- **Dune Analytics**: Integration pending; add query ID to `fetch_onchain_data` later.
- **Backtesting**: Use `backtrader` with historical data for >80% accuracy.

## Troubleshooting
- If CryptoPanic fails, check `CRYPTOPANIC_API_KEY` in secrets.
- If Binance 451 persists, try alternative proxies or use historical data.
</xai-readme>
```

### GitHub Repo Structure
```
ai-btc-bot/
├── app.py
├── requirements.txt
├── README.md
```

### Deployment Steps
**Time**: ~20 minutes (10:10 AM – 10:30 AM IST).

1. **Configure Streamlit Secrets**:
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud), select your app.
   - Click “Manage app” → “Settings” → “Secrets”.
   - Add:
     ```toml
     CRYPTOPANIC_API_KEY = "062951aa1cbb5e40ed898a90a48b6eb1bcf3f8a7"
     ```
   - Save your CryptoPanic API key securely (replace with your actual key if different).
2. **Configure Proxies**:
   - Find free proxies (e.g., [free-proxy-list.net](http://free-proxy-list.net)).
   - Update `proxies` list in `app.py` with real proxy URLs (e.g., `http://123.45.67.89:8080`).
3. **Update GitHub Repo** (`ai-btc-bot`):
   ```bash
   echo -e "$(cat <xaiArtifact id=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx>)" > app.py
   echo -e "$(cat <xaiArtifact id=yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy>)" > requirements.txt
   echo -e "$(cat <xaiArtifact id=zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz>)" > README.md
   git add .
   git commit -m "Remove hardcoded CryptoPanic key, secure secrets"
   git push
   ```
4. **Redeploy on Streamlit Cloud**:
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud), select your app.
   - Click “Manage app” → “Reboot”.
   - Check logs for errors.
   - Verify live URL (`ai-btc-bot.streamlit.app`).
5. **Check Output**:
   - Confirm candlestick chart, signals, RSI, MACD, and CryptoPanic headlines.
   - If Binance 451 errors persist, check logs for proxy issues.

### Accuracy
- **Current**: 65–70% (no Binance data, no NLP/Dune).
- **With Fix**: 75–80% if Binance data resolves via proxies.
- **Target**: >80% with NLP (re-add `transformers` later) and Dune query.

### Fallbacks
- **Binance 451 Persists**:
  - Implement historical data or CoinGecko API (see previous response, ID: f4b7a9b3-…).
  - Download BTC/USDT klines from [data.binance.com](https://data.binance.com).
- **CryptoPanic Fails**:
  - Ensure `CRYPTOPANIC_API_KEY` is set in secrets.
  - Set `fetch_news_sentiment` to `return 0, []` if issues continue.

### Dune Integration
- Re-post the Dune prompt (ID: d1619977-748b-…) to Dune’s Discord (#query-questions).
- Update `fetch_onchain_data` with query ID later.

### If Issues
- **Check Logs**: Reply with new errors or proxy failures.
- **Local Test**: Run `streamlit run app.py` locally with `.secrets.toml` to debug.

**Timeline**:
- **10:10 AM–10:30 AM**: Configure secrets, proxies, redeploy.
- **Evening**: Monitor app, post Dune prompt.

**Next**: Configure secrets, update proxies, redeploy, and verify. Reply with errors or logs if issues arise!
