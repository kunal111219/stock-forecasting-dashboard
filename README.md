# 📈 Dynamic Stock Price Forecasting Dashboard

## ARIMA + Prophet + LSTM | Real-time Data | News Sentiment

---

## 🎯 Overview

An AI-powered interactive stock forecasting dashboard that predicts future stock prices using three ML models simultaneously. Supports any stock worldwide — Indian (NSE/BSE) and US (NYSE/NASDAQ).

🔗 **Live Demo:** [Click here to try it](https://stock-forecasting-dashboard-ixirgtukrseknahg58hvn6.streamlit.app/)

---

## ✨ Features

- 🔍 **Dynamic stock search** — type any ticker worldwide
- 📡 **Real-time prices** via Finnhub (US stocks)
- 🤖 **3 ML models** — ARIMA, Facebook Prophet, LSTM
- 📰 **News sentiment** — Bullish/Bearish score + latest articles
- 📊 **Interactive charts** — Candlestick, Bollinger Bands, Volatility
- 🇮🇳 Indian stocks (NSE) + 🇺🇸 US stocks supported

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Historical Data | yfinance (free, no API key) |
| Real-time + News | Finnhub API (60 req/min free) |
| ML Models | ARIMA (statsmodels), Prophet, LSTM (TensorFlow/Keras) |
| Visualization | Plotly |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
stock-forecasting-dashboard/
│
├── models/
│   ├── arima_model.py      # ARIMA pipeline
│   ├── prophet_model.py    # Prophet pipeline
│   └── lstm_model.py       # LSTM deep learning pipeline
│
├── utils/
│   ├── data_loader.py      # yfinance + Finnhub data fetching
│   ├── preprocessor.py     # Data scaling & sequence generation
│   └── evaluator.py        # RMSE, MAE, MAPE metrics
│
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env                    # API keys (not pushed to GitHub)
├── .gitignore
└── README.md
```

---

## 🚀 Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/kunal111219/stock-forecasting-dashboard.git
cd stock-forecasting-dashboard
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API key

Create `.env` file:

```
FINNHUB_KEY=your_finnhub_api_key_here
```

Get free key at 👉 [finnhub.io](https://finnhub.io)

### 5. Run

```bash
streamlit run app.py
```

---

## 💬 Example Stocks to Try

| Stock | Ticker |
|-------|--------|
| TCS | `TCS` or `TCS.NS` |
| Infosys | `INFY` |
| Reliance | `RELIANCE.NS` |
| Apple | `AAPL` |
| Tesla | `TSLA` |
| Nvidia | `NVDA` |

---

## ⚠️ Disclaimer

For educational purposes only. Not financial advice.

---

## 🤝 Connect

**Kunal Rastogi**

- 🔗 [LinkedIn](https://linkedin.com/in/kunal-rastogi)
- 💻 [GitHub](https://github.com/kunal111219)
- 📧 <rastogikunal191@gmail.com>

---

*If you found this helpful, please ⭐ star the repo!*
