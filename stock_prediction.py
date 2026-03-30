import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import datetime
import feedparser
from urllib.parse import quote_plus
import pytz
import os
import matplotlib.pyplot as plt
from forex_python.converter import CurrencyRates
from pathlib import Path

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers.utils import logging

logging.set_verbosity_error()

########### KEYWORDS
# Company sentiment keywords
keywords = ["earnings", "revenue", "profit", "loss", "forecast", "beat", "miss", "upgrade", "lawsuit", "investigation",
            "acquisition", "merger", "bankruptcy", "default", "shares", "stock", "technology", "rally", "insolvency", "jump",
            "bear", "bull", "bearish", "bullish", "down"]

suffixes = [".", ",", "Inc.", "Incorporated", "Corp.", "Corporation", "Pvt.", "Private", "Ltd.", "Limited", "Co.", "Company", "LLC", "PLC", "SA", "S.A.",
            "Pte.", "AB", "AG", "GmbH"]

# Market sentiment keywords
m_ind_keywords = ["RBI", "economy", "inflation", "interest", "rates", "GDP", "Nifty", "Sensex", "Dalal", "rupee", "war"]
m_us_keywords = ["inflation", "interest", "Federal Reserve", "GDP", "Bonds", "Nasdaq", "war", "dollar", "oil"]
###########

model_finbert = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(model_finbert)
model = TFAutoModelForSequenceClassification.from_pretrained(model_finbert)

labels = ["negative", "neutral", "positive"]
def sentiment_prediction(sentence):
    inputs = tokenizer(sentence, return_tensors="tf", truncation=True, padding=True, max_length=512)

    logits = model(**inputs).logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]

    sentiment = labels[np.argmax(probs)]
    if probs[0] > probs[2]:
        sentiment_score = -probs[0]
    else:
        sentiment_score = probs[2]

    return sentiment, sentiment_score, probs

def stock_info(ticker):
    stock = yf.Ticker(ticker)
    start_date = "2024-10-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    df = stock.history(start=start_date, end=end_date, auto_adjust=True)
    df = df.reset_index()
    company = stock.info.get("longName", ticker)
    for suffix in suffixes:
        company = company.replace(suffix, "").strip()
    return df, company

def news_query(company, ticker):
    keyword = " OR ".join(keywords)
    query = (
        f'("{company}" OR {ticker}) '
        f'AND ({keyword})'
    )
    print(f"Collecting historical news for {company}")
    query = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(url)
    return feed.entries

def market_news_query(country):
    if country == 1:
        keywords = m_ind_keywords
        country_name = "India"
    else:
        keywords = m_us_keywords
        country_name = "US"

    keyword_query = " OR ".join([f'"{k}"' for k in keywords])
    query = (
        f'("{country_name}") '
        f'AND ({keyword_query})'
    )

    print(f"Collecting historical market news for {country_name}")
    query = quote_plus(query)
    today = datetime.today().strftime("%Y-%m-%d")
    url = f"https://news.google.com/rss/search?q={query}+after:2025-01-01+before:{today}"
    feed = feedparser.parse(url)
    return feed.entries

def news_window(feed, date):
    filtered_news = []
    for article in feed:
        if hasattr(article, "published"):
            published = pd.to_datetime(article.published)
            if published.tzinfo is None:
                published = published.tz_localize(pytz.UTC)
            else:
                published = published.tz_convert(pytz.UTC)
            if published.date() == date.date():
                filtered_news.append(article)
    return filtered_news

def average_sentiment(articles):
    scores = []
    for a in articles:
        sentence = a.title
        scores.append(sentiment_prediction(sentence)[1])
    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))

old_sentiment = 0.0
old_market_sentiment = 0.0
def ticker_sentiment_dataframe(ticker, c):
    global old_sentiment
    global old_market_sentiment
    news_size = 0
    market_news_size = 0
    df, company = stock_info(ticker)

    df["sentiment"] = 0.0
    df["market_sentiment"] = 0.0

    feed = news_query(company, ticker)
    market_feed = market_news_query(c)

    for idx, row in df.iterrows():
        date = row["Date"]
        company_news = news_window(feed, date=date)
        market_news = news_window(market_feed, date=date)
        news_size += len(company_news)
        market_news_size += len(market_news)
        sentiment = average_sentiment(company_news)
        market_sentiment = average_sentiment(market_news)

        if sentiment == 0.0:
            sentiment = old_sentiment
        df.at[idx, "sentiment"] = sentiment
        if not sentiment == 0.0:
            old_sentiment = sentiment

        if market_sentiment == 0.0:
            market_sentiment = old_market_sentiment
        df.at[idx, "market_sentiment"] = market_sentiment
        if not market_sentiment == 0.0:
            old_market_sentiment = market_sentiment

    print(f"{news_size} published entries found for company")
    print(f"{market_news_size} published found for country")
    print(f"Analysing sentiments...")
    return df

def lstm_sequences(df, feature_info, target_info, window=30):
    df = df.loc[df["Date"] >= '2025-01-01']
    X = []
    y = []
    for i in range(window, len(df)):
        X.append(df[feature_info].iloc[i-window:i].values)
        y.append(df[target_info].iloc[i])

    X = np.array(X)
    y = np.array(y)

    return X, y

def lstm_model(window, feature_len):
    ip_lstm = Input(shape=(window, feature_len))
    op_lstm = LSTM(64, return_sequences=False)(ip_lstm)
    op_lstm = Dropout(0.3)(op_lstm)
    op_target = Dense(1, activation="linear")(op_lstm)

    model_lstm = Model(ip_lstm, op_target)

    model_lstm.compile(optimizer=Adam(learning_rate=1e-4), loss="huber", metrics=["mae"])

    return model_lstm

def main():
    current_dir = Path(__file__).parent

    t = input("Enter a valid ticker (within quotes), E.g.: hdb: ")
    stock = yf.Ticker(t)
    company = stock.info.get("longName", t)
    country = stock.info.get("country")
    if country == "India":
        c = 1
    else:
        c = 2

    stock_store_path = Path(os.path.join(current_dir, t))
    stock_store_path.mkdir(exist_ok=True)

    print(f"Stock prediction analysis for {company} running...")
    print(f"Country: ", country)

    company_df = ticker_sentiment_dataframe(t, c)
    close = company_df["Close"]

    features = ["Volume", "return", "sentiment", "market_sentiment", "has_news", "has_market_news", "trend", "rsi", "adx", "resistance", "support"]
    target7 = ["future_7"]
    target3 = ["future_3"]
    target1 = ["future_1"]

    company_df["return"] = company_df["Close"].pct_change()
    company_df["future_7"] = company_df['Close'].shift(-7) / company_df['Close'] - 1
    company_df["future_3"] = company_df['Close'].shift(-3) / company_df['Close'] - 1
    company_df["future_1"] = company_df['Close'].shift(-1) / company_df['Close'] - 1
    company_df["has_news"] = (company_df["sentiment"] != 0).astype(int)
    company_df["has_market_news"] = (company_df["market_sentiment"] != 0).astype(int)
    company_df["hl_range"] = (company_df["High"] - company_df["Low"]) / company_df["Close"]     # Not used right now
    company_df["HH"] = company_df["High"] > company_df["High"].shift(1)
    company_df["HL"] = company_df["Low"] > company_df["Low"].shift(1)
    company_df["LH"] = company_df["High"] < company_df["High"].shift(1)
    company_df["LL"] = company_df["Low"] < company_df["Low"].shift(1)
    company_df["trend"] = 0
    company_df.loc[company_df["HH"] & company_df["HL"], "trend"] = 1
    company_df.loc[company_df["LH"] & company_df["LL"], "trend"] = -1
    company_df["rsi"] = ta.rsi(company_df["Close"], length=14)
    adx_data = ta.adx(company_df["High"], company_df["Low"], company_df["Close"], length=14)
    company_df["adx"] = adx_data["ADX_14"]
    res_sup_window = 50
    company_df["resistance"] = company_df["High"].rolling(res_sup_window).max()
    company_df["support"] = company_df["Low"].rolling(res_sup_window).min()

    window = 60
    target = 7
    epochs = 50

    price_scaler = MinMaxScaler()
    volume_scaler = MinMaxScaler()
    sentiment_scaler = MinMaxScaler()
    market_sentiment_scalar = MinMaxScaler()
    return_scaler = StandardScaler()
    rsi_scaler = MinMaxScaler()
    adx_scaler = MinMaxScaler()
    resistance_scaler = StandardScaler()
    support_scaler = StandardScaler()

    company_df["Close"] = price_scaler.fit_transform(company_df[["Close"]])
    company_df["Volume"] = volume_scaler.fit_transform(company_df[["Volume"]])
    company_df["sentiment"] = sentiment_scaler.fit_transform(company_df[["sentiment"]])
    company_df["market_sentiment"] = market_sentiment_scalar.fit_transform(company_df[["market_sentiment"]])
    company_df["return"] = return_scaler.fit_transform(company_df[["return"]])
    company_df["rsi"] = rsi_scaler.fit_transform(company_df[["rsi"]])
    company_df["adx"] = adx_scaler.fit_transform(company_df[["adx"]])
    company_df["resistance"] = resistance_scaler.fit_transform(company_df[["resistance"]])
    company_df["support"] = support_scaler.fit_transform(company_df[["support"]])

    if target == 7:
        X, y = lstm_sequences(company_df, features, target7, window=window)
    elif target == 3:
        X, y = lstm_sequences(company_df, features, target3, window=window)
    elif target == 1:
        X, y = lstm_sequences(company_df, features, target1, window=window)
    else:
        print("Target not set ")
        X, y = [], []
    X_test, y_test = X[-7:], y[-7:]
    X_train, y_train = X[:-7], y[:-7]

    print("Shapes ", np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test))
    model = lstm_model(window, len(features))
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1, shuffle=False)

    y_pred = model.predict(X_test).flatten()

    y_pred_percentage = y_pred * 100
    print("Predicted returns in % over the next 7 days", y_pred_percentage)

    close = close.tolist()
    for i in range(target):
        close.append((close[-target] * y_pred[i]) + close[-target])

    stock_store_date_path = os.path.join(stock_store_path, f"{str(company_df["Date"].iloc[-1]).split(" ")[0]}.txt")
    if os.path.exists(stock_store_date_path):
        os.remove(stock_store_date_path)
    with open(stock_store_date_path, 'w') as f:
            f.write(f"{country}\n")
            for val in close[-target:]:
                f.write(f"{val}\n")

    # Plotting
    close_plot = close[-30 - target:]
    x_plot = range(1, len(close_plot) + 1)
    plt.figure(figsize=(16, 9))
    plt.title(f"{company} stock closing price prediction for the next 7 days")
    plt.plot(x_plot, close_plot, color="grey", label="Closing prices (last 30 days)", alpha=0.5)
    plt.plot(x_plot[-8:], close_plot[-8:], color="black", label="Predicted closing prices", alpha=0.5)
    if c == 1:
        rate = CurrencyRates().get_rate("USD", "INR")
        currency = "₹"
    else:
        rate = 1
        currency = "$"
    for i in range(target, 0, -1):
        value = close_plot[-i]
        prev_value = close_plot[-i-1]
        if value > prev_value:
            plt.scatter(x_plot[-i], value, marker="^", color="green", label=f"{currency}{(value * rate):.2f}", s=100)
        elif value < prev_value:
            plt.scatter(x_plot[-i], value, marker="v", color="red", label=f"{currency}{(value * rate):.2f}", s=100)
        else:
            plt.scatter(x_plot[-i], value, color="blue", label=f"{currency}{(value * rate):.2f}", s=100)
    plt.legend()
    stock_store_plot_path = os.path.join(stock_store_path, f"{str(company_df["Date"].iloc[-1]).split(" ")[0]}.png")
    if os.path.exists(stock_store_plot_path):
        os.remove(stock_store_plot_path)
    plt.savefig(stock_store_plot_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()







