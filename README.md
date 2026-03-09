# stock_predictor

# Stock Prediction using Market Sentiment and Technical Indicators



## Overview



This project predicts stock movement for 7 days using financial and global news sentiments and technical indicators.



Stock price data is downloaded automatically using yfinance.



The model uses technical indicators such as:



- RSI

- ADX

- Support

- Resistance

- Trend State



These features are combined with news sentiment to predict future stock movement.



---



## Features



- Automatic stock data download

- News sentiment analysis using ProsusAI FinBERT

- Technical indicator calculation

- LSTM prediction model using tensorflow



---



## News Sentiment Analysis



Financial news sentiment is analyzed using the FinBERT model developed by ProsusAI.



The model is accessed through the HuggingFace transformers library and is specifically trained on financial text to classify sentiment as:



- Positive

- Negative

- Neutral



---



## Installation



Clone the repository:



```

git clone https://github.com/Rohit-741/stock_predictor.git

```



Install dependencies:



```

pip install -r requirements.txt

```



---



## Usage



Run the main script:



```

python stock_prediction.py

```



The script will:



1. Download stock data

2. Perform sentiment analysis

3. Compute technical indicators

4. Train the prediction model



---



## Technologies Used



- Python

- yfinance

- pandas

- scikit-learn

- numpy

- tensorflow

- transformers

- FinBERT (ProsusAI financial sentiment model)



---



## Project Status



This project is currently a work in progress.



The model and feature engineering are still being developed and evaluated. 

Results and accuracy are subject to change as improvements are made.



---



## Future Improvements



- Additional technical indicators



---



## License



MIT License

