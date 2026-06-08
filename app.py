import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.data_loader import (
    search_ticker, fetch_stock_data, fetch_news_sentiment
)
from utils.preprocessor import (
    calculate_technical_indicators, prepare_prophet_df
)
from models.arima_model import run_arima_pipeline
from models.prophet_model import run_prophet_pipeline
from models.lstm_model import run_lstm_pipeline

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Dynamic Stock Forecasting Dashboard")
st.caption(
    "🇺🇸 US Stocks: Finnhub (Real-time, 60 req/min) | "
    "🇮🇳 Indian + US Historical: yfinance (Free) | "
    "📰 News Sentiment: Finnhub"
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    stock_query = st.text_input(
        "🔍 Search Any Stock",
        placeholder="AAPL, MSFT, TCS, INFY, RELIANCE...",
        help="US stocks via Finnhub | Indian stocks via yfinance"
    )

    period = st.selectbox(
        "📅 Historical Period",
        ["6mo", "1y", "2y", "5y"],
        index=2
    )

    forecast_days = st.slider(
        "🔮 Forecast Days",
        min_value=7, max_value=60, value=30
    )

    models_selected = st.multiselect(
        "🤖 Forecast Models",
        ["ARIMA", "Prophet", "LSTM"],
        default=["ARIMA", "Prophet", "LSTM"]
    )

    show_news = st.checkbox("📰 Show News Sentiment", value=True)

    run_btn = st.button(
        "🚀 Run Forecast",
        type="primary",
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("**💡 Examples:**")
    st.markdown("🇺🇸 `AAPL` `MSFT` `TSLA` `NVDA`")
    st.markdown("🇮🇳 `TCS` `INFY` `RELIANCE` `WIPRO`")

# ── Main ──────────────────────────────────────────────────────
if run_btn:
    if not stock_query:
        st.error("❌ Please enter a stock ticker!")
        st.stop()

    # Step 1: Search ticker — auto routes to right API
    with st.spinner(f"🔍 Searching '{stock_query}'..."):
        ticker, company_name, source = search_ticker(stock_query)

    if not ticker:
        st.error(
            f"❌ Could not find '{stock_query}'. "
            "Check the ticker symbol and try again."
        )
        st.stop()

    # Show which API is being used
    api_badge = "🔴 Finnhub (Real-time)" if source == "finnhub" \
                else "🔵 yfinance"
    st.info(f"📡 Data source: {api_badge} | Symbol: `{ticker}`")

    # Step 2: Fetch data
    with st.spinner(f"📡 Fetching data for {ticker}..."):
        df, info, realtime = fetch_stock_data(ticker, period, source)

    if df is None or df.empty:
        st.error(f"❌ No data found for {ticker}.")
        st.stop()

    # Step 3: Calculate indicators
    df = calculate_technical_indicators(df)

    # ── Stock Overview ─────────────────────────────────────
    currency = info.get('currency', 'USD')
    curr_sym = '₹' if currency == 'INR' else '$'

    # Use real-time price if available (Finnhub)
    if realtime and realtime.get("current"):
        current_price = realtime["current"]
        price_change = realtime.get("pct_change", 0)
        price_delta = realtime.get("change", 0)
        is_realtime = True
    else:
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        price_delta = current_price - prev_price
        is_realtime = False

    st.subheader(f"🏢 {company_name} ({ticker})")
    if is_realtime:
        st.caption("⚡ Real-time price from Finnhub")

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Current Price",
        f"{curr_sym}{current_price:.2f}",
        f"{price_change:+.2f}%"
    )
    c2.metric(
        "30D Volatility",
        f"{df['Volatility_30d'].iloc[-1]*100:.1f}%"
    )
    c3.metric(
        "52W High",
        f"{curr_sym}{df['Close'].rolling(min(252, len(df))).max().iloc[-1]:.2f}"
    )
    c4.metric(
        "52W Low",
        f"{curr_sym}{df['Close'].rolling(min(252, len(df))).min().iloc[-1]:.2f}"
    )
    c5.metric(
        "Avg Volume",
        f"{int(df['Volume'].mean()):,}"
    )

    # Real-time extra metrics (Finnhub only)
    if is_realtime and realtime:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Day Open", f"${realtime.get('open', 0):.2f}")
        r2.metric("Day High", f"${realtime.get('high', 0):.2f}")
        r3.metric("Day Low", f"${realtime.get('low', 0):.2f}")
        r4.metric("Prev Close", f"${realtime.get('prev_close', 0):.2f}")

    # ── Price Chart ────────────────────────────────────────
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='OHLC'
    ))
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df['MA_20'],
        name='MA 20',
        line=dict(color='orange', width=1)
    ))
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df['MA_50'],
        name='MA 50',
        line=dict(color='cyan', width=1)
    ))
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df['Upper_Band'],
        name='Upper Bollinger',
        line=dict(color='rgba(255,255,255,0.3)', dash='dash')
    ))
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df['Lower_Band'],
        name='Lower Bollinger',
        line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(255,255,255,0.05)'
    ))
    fig_price.update_layout(
        title=f"{company_name} — Price + Bollinger Bands ({period})",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        height=450, template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ── News Sentiment ─────────────────────────────────────
    if show_news and source == "finnhub":
        st.subheader("📰 Latest News & Sentiment")

        with st.spinner("Fetching news..."):
            news_list, sentiment_data = fetch_news_sentiment(ticker)

        if sentiment_data:
            bull = sentiment_data.get("buzz", {}).get("bullishPercent", 0)
            bear = sentiment_data.get("buzz", {}).get("bearishPercent", 0)
            score = sentiment_data.get("companyNewsScore", 0)
            articles = sentiment_data.get("buzz", {}).get("articlesInLastWeek", 0)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("News Score", f"{score:.2f}/1.0")
            s2.metric("Bullish", f"{bull*100:.1f}%", "Positive news")
            s3.metric("Bearish", f"{bear*100:.1f}%", "Negative news")
            s4.metric("Articles (7d)", str(articles))

            # Sentiment gauge
            sentiment_label = (
                "🟢 Bullish" if bull > 0.6
                else "🔴 Bearish" if bear > 0.6
                else "🟡 Neutral"
            )
            st.markdown(f"**Overall Sentiment: {sentiment_label}**")

        # News articles
        if news_list:
            with st.expander("📄 Latest News Articles", expanded=True):
                for article in news_list[:5]:
                    headline = article.get("headline", "")
                    source_name = article.get("source", "")
                    url = article.get("url", "")
                    date = article.get("datetime", 0)
                    date_str = pd.to_datetime(date, unit='s').strftime(
                        "%b %d, %Y"
                    )
                    summary = article.get("summary", "")[:200]

                    st.markdown(f"**[{headline}]({url})**")
                    st.caption(f"{source_name} · {date_str}")
                    if summary:
                        st.markdown(f"_{summary}..._")
                    st.divider()
        else:
            st.info("No recent news found for this stock.")

    elif show_news and source == "yfinance":
        st.info(
            "📰 News sentiment available for US stocks only. "
            "Try AAPL or MSFT for news features."
        )

    # ── Forecasting ────────────────────────────────────────
    st.subheader(f"🔮 {forecast_days}-Day Price Forecast")

    close_data = df['Close']
    future_dates = pd.bdate_range(
        start=df.index[-1] + timedelta(days=1),
        periods=forecast_days
    )

    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(
        x=df.index[-90:],
        y=df['Close'].iloc[-90:],
        name='Historical (Last 90 Days)',
        line=dict(color='white', width=2)
    ))

    summary_results = []

    # ARIMA
    if "ARIMA" in models_selected:
        with st.spinner("⚙️ Training ARIMA..."):
            try:
                arima_fc, conf_int, order = run_arima_pipeline(
                    close_data, forecast_days
                )
                forecast_fig.add_trace(go.Scatter(
                    x=future_dates, y=arima_fc,
                    name=f'ARIMA{order}',
                    line=dict(color='#FF6B6B', width=2, dash='dash')
                ))
                forecast_fig.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates[::-1]),
                    y=list(conf_int[:, 1]) +
                      list(conf_int[:, 0][::-1]),
                    fill='toself',
                    fillcolor='rgba(255,107,107,0.1)',
                    line=dict(color='rgba(255,107,107,0)'),
                    name='ARIMA Confidence'
                ))
                summary_results.append({
                    "Model": f"ARIMA{order}",
                    f"Day 7": f"{curr_sym}{arima_fc[6]:.2f}",
                    f"Day 14": f"{curr_sym}{arima_fc[13]:.2f}" if forecast_days >= 14 else "N/A",
                    f"Day 30": f"{curr_sym}{arima_fc[29]:.2f}" if forecast_days >= 30 else "N/A",
                    "Trend": "📈 Up" if arima_fc[-1] > current_price else "📉 Down"
                })
                st.success(f"✅ ARIMA{order} complete")
            except Exception as e:
                st.warning(f"⚠️ ARIMA: {e}")

    # Prophet
    if "Prophet" in models_selected:
        with st.spinner("⚙️ Training Prophet..."):
            try:
                prophet_df = prepare_prophet_df(df)
                prophet_fc = run_prophet_pipeline(
                    prophet_df, forecast_days
                )
                forecast_fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=prophet_fc['yhat'].values,
                    name='Prophet',
                    line=dict(color='#4CAF50', width=2, dash='dash')
                ))
                forecast_fig.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates[::-1]),
                    y=list(prophet_fc['yhat_upper'].values) +
                      list(prophet_fc['yhat_lower'].values[::-1]),
                    fill='toself',
                    fillcolor='rgba(76,175,80,0.1)',
                    line=dict(color='rgba(76,175,80,0)'),
                    name='Prophet Confidence'
                ))
                summary_results.append({
                    "Model": "Prophet",
                    "Day 7": f"{curr_sym}{prophet_fc['yhat'].values[6]:.2f}",
                    "Day 14": f"{curr_sym}{prophet_fc['yhat'].values[13]:.2f}" if forecast_days >= 14 else "N/A",
                    "Day 30": f"{curr_sym}{prophet_fc['yhat'].values[29]:.2f}" if forecast_days >= 30 else "N/A",
                    "Trend": "📈 Up" if prophet_fc['yhat'].values[-1] > current_price else "📉 Down"
                })
                st.success("✅ Prophet complete")
            except Exception as e:
                st.warning(f"⚠️ Prophet: {e}")

    # LSTM
    if "LSTM" in models_selected:
        with st.spinner("⚙️ Training LSTM (1-2 mins)..."):
            try:
                lstm_fc = run_lstm_pipeline(close_data, forecast_days)
                forecast_fig.add_trace(go.Scatter(
                    x=future_dates, y=lstm_fc,
                    name='LSTM',
                    line=dict(color='#FF9800', width=2, dash='dash')
                ))
                summary_results.append({
                    "Model": "LSTM",
                    "Day 7": f"{curr_sym}{lstm_fc[6]:.2f}",
                    "Day 14": f"{curr_sym}{lstm_fc[13]:.2f}" if forecast_days >= 14 else "N/A",
                    "Day 30": f"{curr_sym}{lstm_fc[29]:.2f}" if forecast_days >= 30 else "N/A",
                    "Trend": "📈 Up" if lstm_fc[-1] > current_price else "📉 Down"
                })
                st.success("✅ LSTM complete")
            except Exception as e:
                st.warning(f"⚠️ LSTM: {e}")

    # Forecast chart
    forecast_fig.update_layout(
        title=f"{company_name} — {forecast_days}-Day Forecast",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        height=500, template="plotly_dark"
    )
    st.plotly_chart(forecast_fig, use_container_width=True)

    # Summary table
    if summary_results:
        st.subheader("📋 Forecast Summary")
        st.dataframe(
            pd.DataFrame(summary_results),
            use_container_width=True
        )

    # ── Volume + Volatility ────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        vfig = go.Figure()
        vfig.add_trace(go.Bar(
            x=df.index[-60:],
            y=df['Volume'].iloc[-60:],
            marker_color='#2196F3'
        ))
        vfig.update_layout(
            title="Volume (Last 60 Days)",
            height=300, template="plotly_dark"
        )
        st.plotly_chart(vfig, use_container_width=True)

    with col2:
        vofig = go.Figure()
        vofig.add_trace(go.Scatter(
            x=df.index,
            y=df['Volatility_30d'] * 100,
            line=dict(color='#9C27B0', width=2),
            fill='tozeroy',
            fillcolor='rgba(156,39,176,0.1)'
        ))
        vofig.update_layout(
            title="30D Rolling Volatility (%)",
            height=300, template="plotly_dark"
        )
        st.plotly_chart(vofig, use_container_width=True)

    # Returns distribution
    returns = df['Daily_Return'].dropna() * 100
    dist_fig = ff.create_distplot(
        [returns.tolist()],
        ['Daily Returns (%)'],
        bin_size=0.3,
        colors=['#00BCD4']
    )
    dist_fig.update_layout(
        title="Daily Returns Distribution",
        height=300, template="plotly_dark"
    )
    st.plotly_chart(dist_fig, use_container_width=True)

    st.warning(
        "⚠️ For educational purposes only. "
        "Not financial advice."
    )

else:
    st.info("👈 Search any stock and click **Run Forecast**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🇺🇸 US Stocks", "Finnhub", "Real-time prices")
    c2.metric("🇮🇳 Indian Stocks", "yfinance", "NSE/BSE")
    c3.metric("📰 News", "Sentiment", "Bullish/Bearish")
    c4.metric("🤖 Models", "3 ML Models", "ARIMA+Prophet+LSTM")