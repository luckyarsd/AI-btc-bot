import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import pandas as pd
import numpy as np
import pandas_ta as ta # Import pandas_ta for technical analysis
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="BTC/USD Trading Signal Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced aesthetics (Tailwind-like styling) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    body {
        font-family: 'Inter', sans-serif;
        background-color: #0f172a; /* Dark Slate Blue */
        color: #e2e8f0; /* Light Gray */
    }
    .stApp {
        background-color: #0f172a;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc; /* White */
        font-weight: 700;
    }
    .stButton>button {
        background-color: #2563eb; /* Blue 600 */
        color: white;
        border-radius: 0.5rem; /* rounded-lg */
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stButton>button:hover {
        background-color: #1d4ed8; /* Blue 700 */
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .stTextInput>div>div>input, .stSelectbox>div>div {
        background-color: #1e293b; /* Darker Slate Blue */
        color: #f8fafc;
        border-radius: 0.375rem; /* rounded-md */
        border: 1px solid #334155; /* Slate 700 */
        padding: 0.5rem 1rem;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div:focus-visible {
        border-color: #3b82f6; /* Blue 500 */
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.5);
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .css-1d391kg { /* Target Streamlit's main block container for rounded corners */
        background-color: #1a202c; /* Darker background for content cards */
        border-radius: 0.75rem; /* rounded-xl */
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .css-q8spsw { /* Adjust sidebar width if needed */
        width: 300px;
    }
    .css-1y48vrf { /* Adjust sidebar background color */
        background-color: #1e293b; /* Darker Slate Blue for sidebar */
        border-right: 1px solid #334155;
    }
    .stSpinner > div {
        color: #3b82f6; /* Blue 500 */
    }
    </style>
""", unsafe_allow_html=True)

# --- Title and Description ---
st.title("💰 BTC/USD Trading Signal Predictor")
st.markdown("""
    This application predicts Bitcoin (BTC) prices against the US Dollar (USD) using the Prophet forecasting model
    and generates trading signals (Buy/Sell/Wait) based on technical indicators.
    Please remember this tool is for educational purposes only and not financial advice.
""")

# --- Sidebar for user inputs ---
st.sidebar.header("Prediction & Signal Settings")

# Number of days to predict
n_days = st.sidebar.slider(
    "Prediction Horizon (days)",
    min_value=7,
    max_value=180, # Max 6 months for reasonable short-term prediction
    value=30,
    help="Number of days into the future for price prediction and signal generation."
)

# Model configuration options (Prophet specific)
st.sidebar.subheader("Prophet Model Configuration")
seasonality_mode = st.sidebar.selectbox(
    "Seasonality Mode",
    ["additive", "multiplicative"],
    index=0,
    help="Additive: seasonality is constant. Multiplicative: seasonality scales with the trend."
)
daily_seasonality = st.sidebar.checkbox("Include Daily Seasonality", value=True)
weekly_seasonality = st.sidebar.checkbox("Include Weekly Seasonality", value=True)
yearly_seasonality = st.sidebar.checkbox("Include Yearly Seasonality", value=True)
changepoint_prior_scale = st.sidebar.slider(
    "Changepoint Prior Scale",
    min_value=0.01,
    max_value=0.5,
    value=0.05,
    step=0.01,
    help="Adjusts the flexibility of the trend. Higher values allow more flexible trends."
)

# Technical Indicator Parameters
st.sidebar.subheader("Technical Indicator Parameters")
sma_short_period = st.sidebar.number_input("SMA Short Period", min_value=5, max_value=50, value=20, step=1)
sma_long_period = st.sidebar.number_input("SMA Long Period", min_value=20, max_value=200, value=50, step=5)
rsi_period = st.sidebar.number_input("RSI Period", min_value=7, max_value=30, value=14, step=1)
macd_fast_period = st.sidebar.number_input("MACD Fast Period", min_value=5, max_value=20, value=12, step=1)
macd_slow_period = st.sidebar.number_input("MACD Slow Period", min_value=20, max_value=40, value=26, step=1)
macd_signal_period = st.sidebar.number_input("MACD Signal Period", min_value=5, max_value=15, value=9, step=1)

# Stop Loss / Take Profit Parameters
st.sidebar.subheader("Risk Management Parameters")
stop_loss_pct = st.sidebar.number_input(
    "Stop Loss Percentage (%)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.1,
    format="%.1f",
    help="Percentage below (for Buy) or above (for Sell) entry price for stop loss."
) / 100.0 # Convert to decimal

take_profit_pct = st.sidebar.number_input(
    "Take Profit Percentage (%)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.1,
    format="%.1f",
    help="Percentage above (for Buy) or below (for Sell) entry price for take profit."
) / 100.0 # Convert to decimal


# --- Data Fetching Function ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_historical_data(ticker):
    """Fetches historical cryptocurrency data using yfinance."""
    try:
        # Automatically fetch all available data from a very early date
        # BTC-USD is generally more reliable for long history on Yahoo Finance
        data = yf.download(ticker, start="2014-01-01", end=datetime.now().strftime('%Y-%m-%d'))
        if data.empty:
            st.warning(f"No data found for {ticker}. Please check the ticker symbol or try again later.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --- Technical Indicator Calculation ---
def calculate_technical_indicators(df):
    """Calculates SMAs, RSI, and MACD using pandas_ta."""
    # Ensure 'Close' column exists
    if 'Close' not in df.columns:
        st.error("Missing 'Close' price column in data for technical analysis.")
        return df

    # SMAs
    df['SMA_Short'] = ta.sma(df['Close'], length=sma_short_period)
    df['SMA_Long'] = ta.sma(df['Close'], length=sma_long_period)

    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=rsi_period)

    # MACD
    macd = ta.macd(df['Close'], fast=macd_fast_period, slow=macd_slow_period, signal=macd_signal_period)
    if macd is not None and not macd.empty:
        df['MACD'] = macd[f'MACD_{macd_fast_period}_{macd_slow_period}_{macd_signal_period}']
        df['MACD_Signal'] = macd[f'MACDH_{macd_fast_period}_{macd_slow_period}_{macd_signal_period}'] # Using MACDH for signal line
        df['MACD_Hist'] = macd[f'MACDS_{macd_fast_period}_{macd_slow_period}_{macd_signal_period}'] # Using MACDS for histogram
    else:
        # Handle cases where MACD calculation might fail (e.g., insufficient data)
        df['MACD'] = np.nan
        df['MACD_Signal'] = np.nan
        df['MACD_Hist'] = np.nan
        st.warning("MACD could not be calculated. Not enough data or invalid parameters.")

    return df

# --- Signal Generation Logic ---
def generate_signals(df_with_indicators, forecast_trend, current_price, prediction_horizon_days):
    """Generates Buy/Sell/Wait signals with Stop Loss and Target."""
    signal = "WAIT"
    stop_loss = None
    target_price = None
    estimated_time = f"within {prediction_horizon_days} days"

    # Get the latest indicator values
    if df_with_indicators.empty:
        return "WAIT", None, None, estimated_time

    latest_data = df_with_indicators.iloc[-1]
    
    # Check for NaN values in critical indicators before using them
    if (pd.isna(latest_data['SMA_Short']) or pd.isna(latest_data['SMA_Long']) or
        pd.isna(latest_data['RSI']) or pd.isna(latest_data['MACD']) or 
        pd.isna(latest_data['MACD_Signal'])):
        return "WAIT", None, None, estimated_time


    # --- Signal Rules ---
    # Rule 1: SMA Crossover (Bullish/Bearish)
    # Check if SMA_Short and SMA_Long are available from previous data point for crossover detection
    sma_crossover_buy = False
    sma_crossover_sell = False
    if len(df_with_indicators) >= 2:
        prev_data = df_with_indicators.iloc[-2]
        if (latest_data['SMA_Short'] > latest_data['SMA_Long'] and
            prev_data['SMA_Short'] <= prev_data['SMA_Long']):
            sma_crossover_buy = True
        elif (latest_data['SMA_Short'] < latest_data['SMA_Long'] and
              prev_data['SMA_Short'] >= prev_data['SMA_Long']):
            sma_crossover_sell = True

    # Rule 2: RSI (Oversold/Overbought)
    rsi_oversold = latest_data['RSI'] < 30
    rsi_overbought = latest_data['RSI'] > 70

    # Rule 3: MACD Crossover (Bullish/Bearish)
    macd_crossover_buy = False
    macd_crossover_sell = False
    if len(df_with_indicators) >= 2:
        prev_macd = df_with_indicators['MACD'].iloc[-2]
        prev_macd_signal = df_with_indicators['MACD_Signal'].iloc[-2]
        if (latest_data['MACD'] > latest_data['MACD_Signal'] and
            prev_macd <= prev_macd_signal):
            macd_crossover_buy = True
        elif (latest_data['MACD'] < latest_data['MACD_Signal'] and
              prev_macd >= prev_macd_signal):
            macd_crossover_sell = True
    
    # --- Combine Signals based on Prophet's forecast direction ---
    # Prophet's forecast trend (forecast_trend: 'Upward' or 'Downward' or 'Neutral')
    
    buy_score = 0
    sell_score = 0

    if sma_crossover_buy: buy_score += 1
    if rsi_oversold: buy_score += 1
    if macd_crossover_buy: buy_score += 1

    if sma_crossover_sell: sell_score += 1
    if rsi_overbought: sell_score += 1
    if macd_crossover_sell: sell_score += 1

    # Prioritize strong signals, then consider Prophet's trend
    if buy_score >= 2: # At least 2 bullish indicators
        signal = "BUY"
    elif sell_score >= 2: # At least 2 bearish indicators
        signal = "SELL"
    elif forecast_trend == "Upward" and buy_score >=1: # If Prophet is upward and at least one bullish indicator
        signal = "BUY"
    elif forecast_trend == "Downward" and sell_score >=1: # If Prophet is downward and at least one bearish indicator
        signal = "SELL"
    else:
        signal = "WAIT"

    # Calculate Stop Loss and Target Price if a signal is generated
    if signal == "BUY":
        stop_loss = current_price * (1 - stop_loss_pct)
        target_price = current_price * (1 + take_profit_pct)
    elif signal == "SELL": # For short selling, SL is above, TP is below
        stop_loss = current_price * (1 + stop_loss_pct)
        target_price = current_price * (1 - take_profit_pct)

    return signal, stop_loss, target_price, estimated_time

# --- Main app logic ---
if st.sidebar.button("Predict Price & Generate Signals"):
    st.info("Fetching historical data and training model...")

    # Fetch data
    df = get_historical_data("BTC-USD") 

    if df is not None:
        # Add technical indicators to the DataFrame
        df = calculate_technical_indicators(df)

        # Prepare data for Prophet
        df_prophet = df[['Close']].reset_index()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

        # --- Model Training ---
        st.subheader("Prophet Model Training")
        st.write("Training the Prophet model on historical data...")

        # Initialize Prophet model with user-selected parameters
        model = Prophet(
            seasonality_mode=seasonality_mode,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=changepoint_prior_scale
        )

        # Fit the model
        try:
            model.fit(df_prophet)
            st.success("Prophet model trained successfully!")

            # --- Create future dataframe for prediction ---
            future = model.make_future_dataframe(periods=n_days, freq='D')

            # --- Make predictions ---
            forecast = model.predict(future)

            # Determine Prophet's future trend (simple check on the predicted price over the horizon)
            last_historical_price = df_prophet['y'].iloc[-1]
            predicted_price_at_horizon = forecast['yhat'].iloc[-1]
            
            forecast_trend = "Neutral"
            if predicted_price_at_horizon > last_historical_price * (1 + 0.01): # > 1% increase
                forecast_trend = "Upward"
            elif predicted_price_at_horizon < last_historical_price * (1 - 0.01): # > 1% decrease
                forecast_trend = "Downward"

            # --- Generate Trading Signals ---
            st.subheader("Current Trading Signal")
            
            current_price = df['Close'].iloc[-1]
            signal, stop_loss, target_price, estimated_time = generate_signals(
                df, forecast_trend, current_price, n_days
            )

            # Display Signal prominently
            if signal == "BUY":
                st.markdown(f"<h3 style='color:#10b981;'>Signal: {signal} 🟢</h3>", unsafe_allow_html=True)
                st.write(f"**Current Price:** `{current_price:.2f} USD`")
                st.write(f"**Recommended Stop Loss:** `{stop_loss:.2f} USD`")
                st.write(f"**Recommended Target Price:** `{target_price:.2f} USD`")
                st.write(f"**Estimated Timeframe:** `{estimated_time}`")
            elif signal == "SELL":
                st.markdown(f"<h3 style='color:#ef4444;'>Signal: {signal} 🔴</h3>", unsafe_allow_html=True)
                st.write(f"**Current Price:** `{current_price:.2f} USD`")
                st.write(f"**Recommended Stop Loss:** `{stop_loss:.2f} USD`")
                st.write(f"**Recommended Target Price:** `{target_price:.2f} USD`")
                st.write(f"**Estimated Timeframe:** `{estimated_time}`")
            else:
                st.markdown(f"<h3 style='color:#facc15;'>Signal: {signal} 🟡</h3>", unsafe_allow_html=True)
                st.write(f"**Current Price:** `{current_price:.2f} USD`")
                st.write(f"No specific stop loss or target for 'WAIT' signal. Re-evaluate market conditions.")
                st.write(f"**Estimated Timeframe:** `{estimated_time}`")
            
            st.markdown("---")

            # --- Visualization ---
            st.subheader("BTC/USD Historical, Predicted Prices & Indicators")

            fig = go.Figure()

            # Add actual closing price
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Close Price', line=dict(color='#858585')))

            # Add Prophet's forecast
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet Forecast', line=dict(color='#3b82f6', width=2)))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', fillcolor='rgba(59,130,246,0.1)', mode='lines', line=dict(width=0), name='Forecast Lower Bound', showlegend=False))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', fillcolor='rgba(59,130,246,0.1)', mode='lines', line=dict(width=0), name='Forecast Upper Bound', showlegend=False))

            # Add Technical Indicators to the plot
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Short'], mode='lines', name=f'SMA {sma_short_period}', line=dict(color='#facc15', dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Long'], mode='lines', name=f'SMA {sma_long_period}', line=dict(color='#ef4444', dash='dash')))

            # Add signal markers (example for historical signals if we were to backtest)
            # For this version, we'll indicate historical SMA crossovers on the chart
            df['Buy_SMA_Crossover'] = (df['SMA_Short'] > df['SMA_Long']) & (df['SMA_Short'].shift(1) <= df['SMA_Long'].shift(1))
            df['Sell_SMA_Crossover'] = (df['SMA_Short'] < df['SMA_Long']) & (df['SMA_Short'].shift(1) >= df['SMA_Long'].shift(1))

            fig.add_trace(go.Scatter(
                x=df[df['Buy_SMA_Crossover']].index,
                y=df['Close'][df['Buy_SMA_Crossover']],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Historical Buy Signal (SMA)'
            ))
            fig.add_trace(go.Scatter(
                x=df[df['Sell_SMA_Crossover']].index,
                y=df['Close'][df['Sell_SMA_Crossover']],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Historical Sell Signal (SMA)'
            ))


            fig.update_layout(
                title='BTC/USD Price with Forecast & Moving Averages',
                xaxis_title='Date',
                yaxis_title='Close Price (USD)',
                hovermode='x unified',
                template='plotly_dark',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Subplot for RSI
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='#10b981')))
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Oversold (30)", annotation_position="bottom right")
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="top right")
            fig_rsi.update_layout(
                title=f'Relative Strength Index (RSI - {rsi_period} periods)',
                xaxis_title='Date',
                yaxis_title='RSI Value',
                template='plotly_dark',
                height=300
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

            # Subplot for MACD
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD Line', line=dict(color='#3b82f6')))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='#ef4444', dash='dash')))
            fig_macd.add_bar(x=df.index, y=df['MACD_Hist'], name='MACD Histogram', marker_color='#facc15')
            fig_macd.update_layout(
                title=f'MACD ({macd_fast_period},{macd_slow_period},{macd_signal_period})',
                xaxis_title='Date',
                yaxis_title='Value',
                template='plotly_dark',
                height=300
            )
            st.plotly_chart(fig_macd, use_container_width=True)


            # --- Display Forecast Data ---
            st.subheader(f"Future Price Predictions for next {n_days} days")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days).rename(
                columns={'ds': 'Date', 'yhat': 'Predicted Price', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
            ).style.format({
                'Predicted Price': '{:.2f}',
                'Lower Bound': '{:.2f}',
                'Upper Bound': '{:.2f}'
            }))

            # --- Model Evaluation (on historical data) ---
            st.subheader("Model Evaluation on Historical Data")

            evaluation_df = forecast.set_index('ds').join(df_prophet.set_index('ds')['y']).reset_index()
            evaluation_df = evaluation_df.dropna() 

            if not evaluation_df.empty:
                actual_prices = evaluation_df['y']
                predicted_prices = evaluation_df['yhat']

                mse = mean_squared_error(actual_prices, predicted_prices)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual_prices, predicted_prices)
                
                ss_total = ((actual_prices - actual_prices.mean()) ** 2).sum()
                ss_res = ((actual_prices - predicted_prices) ** 2).sum()
                r_squared = 1 - (ss_res / ss_total) if ss_total > 0 else 0

                st.markdown(f"""
                    - **Mean Squared Error (MSE):** `{mse:.2f}`
                    - **Root Mean Squared Error (RMSE):** `{rmse:.2f}` (Lower is better)
                    - **Mean Absolute Error (MAE):** `{mae:.2f}` (Lower is better)
                    - **R-squared (Coefficient of Determination):** `{r_squared:.2f}` (Closer to 1 is better)
                """)

                st.markdown(
                    """
                    **Note on Accuracy:** The term "accuracy" in price prediction is complex.
                    For time series, metrics like RMSE and MAE give a sense of prediction error.
                    R-squared indicates how much of the variance in the actual prices is
                    explained by the model. **Achieving over 80% R-squared for volatile
                    cryptocurrency prices on *future* predictions is generally not feasible
                    with any model, as prices are influenced by unpredictable external factors.**
                    This model aims to capture historical patterns and provide a reasonable forecast.
                    """
                )
            else:
                st.warning("Could not perform historical evaluation. Data mismatch or insufficient data.")

        except Exception as e:
            st.error(f"An error occurred during model training or prediction: {e}")
            st.warning("Please try adjusting the historical data range or model parameters.")

# --- About the Model and Limitations ---
st.markdown("---")
st.subheader("About the Model and Limitations")
st.markdown("""
    This application combines **Facebook Prophet** for trend and seasonality analysis with **traditional technical indicators**
    (Moving Averages, RSI, MACD) to generate trading signals.

    **How Signals are Generated:**
    The system looks for alignment between the Prophet's predicted trend (upward/downward) and signals from the technical indicators:
    -   **SMA Crossover:** Short-term moving average crossing above (bullish) or below (bearish) a long-term moving average.
    -   **RSI (Relative Strength Index):** Indicates overbought (>70) or oversold (<30) conditions.
    -   **MACD (Moving Average Convergence Divergence):** Identifies trend, momentum, and potential reversals through crossovers of its lines.

    A "Buy" or "Sell" signal is generated when at least two bullish/bearish technical indicators align, or when Prophet's forecast aligns with at least one indicator. Otherwise, a "Wait" signal is issued.

    **Limitations and Important Disclaimer:**
    -   **No Financial Advice:** This tool is purely for educational and experimental purposes. It is **not** financial advice, and you should **not** use it for actual trading decisions.
    -   **Accuracy Disclaimer:** While the model strives to capture historical patterns and provides an R-squared metric for its fit to *past* data, **achieving a consistent "over 80% accuracy" for *future* cryptocurrency price predictions is extremely challenging and generally unrealistic.** Cryptocurrency markets are highly volatile and influenced by a multitude of unpredictable external factors (news, regulations, global events, social sentiment) that are not incorporated into this model.
    -   **Rule-Based Signals:** The trading signals are based on simplified rules using common technical analysis principles. Real-world trading strategies are far more complex and involve deep fundamental analysis, sentiment analysis, macroeconomic factors, and sophisticated risk management.
    -   **Market Volatility:** Prices can change rapidly, and stop-loss/target prices are illustrative.
""")

# --- Footer ---
st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #64748b;">
        Developed with ❤️ using Streamlit, yfinance, Prophet, and pandas_ta.
    </div>
""", unsafe_allow_html=True)
