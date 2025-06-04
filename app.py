#################################################################################################
# File Name - app.py
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This file contains the code for the Flask application that serves the frontend
#               with stock and index prediction using yfinance data
#################################################################################################

from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import click
import time
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from bs4 import BeautifulSoup
import concurrent.futures
from flask_cors import CORS

# Initialize Flask app and set up logging
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define indices for yfinance
INDICES = {
    'NIFTY50': {'yf': '^NSEI', 'name': 'NIFTY 50'},
    'BSE': {'yf': '^BSESN', 'name': 'SENSEX'},
    'BANKNIFTY': {'yf': '^NSEBANK', 'name': 'NIFTY BANK'},
    'NIFTYAUTO': {'yf': 'NIFTY-AUTO.NS', 'name': 'NIFTY AUTO'},
    'NIFTYFINSERV': {'yf': 'NIFTY-FIN-SERVICE.NS', 'name': 'NIFTY FINANCIAL SERVICES'},
    'NIFTYFMCG': {'yf': 'NIFTY-FMCG.NS', 'name': 'NIFTY FMCG'},
    'NIFTYIT': {'yf': 'NIFTY-IT.NS', 'name': 'NIFTY IT'},
    'NIFTYMETAL': {'yf': 'NIFTY-METAL.NS', 'name': 'NIFTY METAL'},
    'NIFTYPHARMA': {'yf': 'NIFTY-PHARMA.NS', 'name': 'NIFTY PHARMA'},
}

#######################################################################################################################
# Function: initialize_app
# Input: app
# Output: None
# Description: Initialize the application state by checking the data sources
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
def initialize_app(app):
    """Initialize application state"""
    try:
        with app.app_context():
            status = check_data_sources()
            logging.info(f"Initial data source status: {status.get_json()}")
    except Exception as e:
        logging.error(f"Application initialization failed: {str(e)}")
        raise

#######################################################################################################################
# Class: AttentionLayer
# Input: nn.Module
# Output: None
# Description: Attention layer for the BiLSTM model
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * hidden_states, dim=1)
        return attended

#######################################################################################################################
# Class: EnhancedStockBiLSTM
# Input: nn.Module
# Output: None
# Description: Enhanced BiLSTM model for stock price prediction
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
class EnhancedStockBiLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, output_size=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size * 2)
        
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm1(x, (h0, c0))
        out = self.norm1(out)
        
        attended = self.attention(out)
        attended = self.dropout(attended)
        
        out = self.fc1(attended)
        out = self.relu(out)
        out = self.norm2(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        return out

#######################################################################################################################
# Class: StockDataset
# Input: Dataset
# Output: None
# Description: Custom PyTorch dataset for stock data
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#######################################################################################################################
# Class: DataFetchError
# Input: Exception
# Output: None
# Description: Custom exception for data fetching errors
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
class DataFetchError(Exception):
    """Custom exception for data fetching errors"""
    pass

#######################################################################################################################
# Function: calculate_rsi
# Input: prices, period
# Output: 100 - (100 / (1 + rs))
# Description: Calculate the Relative Strength Index (RSI) of a stock
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

#######################################################################################################################
# Function: calculate_macd
# Input: prices, fast, slow
# Output: exp1 - exp2
# Description: Calculate the Moving Average Convergence Divergence (MACD) of a stock
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
def calculate_macd(prices, fast=12, slow=26):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

#######################################################################################################################
# Function: prepare_data
# Input: data, sequence_length, prediction_days
# Output: X_train, X_val, y_train, y_val, scaler
# Description: Prepare the data for training the model
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
def prepare_data(data, sequence_length=30, prediction_days=7):
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - sequence_length - prediction_days + 1):
            seq = scaled_data[i:i + sequence_length]
            target = scaled_data[i + sequence_length:i + sequence_length + prediction_days, 0]
            sequences.append(seq)
            targets.append(target)
        
        if not sequences or not targets:
            raise ValueError("No valid sequences could be created from the data")
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        split = int(len(sequences) * 0.8)
        X_train, X_val = sequences[:split], sequences[split:]
        y_train, y_val = targets[:split], targets[split:]
        
        return X_train, X_val, y_train, y_val, scaler
        
    except Exception as e:
        logging.error(f"Data preparation error: {str(e)}")
        raise

#######################################################################################################################
# Function: prepare_features
# Input: df
# Output: df[['Close', 'Volume', 'MA20', 'RSI', 'MACD']].values
# Description: Prepare features from DataFrame
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
def prepare_features(df):
    """Prepare features from DataFrame"""
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    
    df = df.dropna()
    
    if len(df) < 50:
        raise ValueError("Insufficient data points after processing")
    
    return df[['Close', 'Volume', 'MA20', 'RSI', 'MACD']].values

#######################################################################################################################
# Function: get_stock_data_yf
# Input: symbol, period, max_retries
# Output: features, data['Close'].values
# Description: Fetch stock or index data from Yahoo Finance with retries
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
def get_stock_data_yf(symbol, period='10y', max_retries=3):
    """Fetch data from yfinance with retries"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2)
            
            # Determine if it's an index or stock
            if symbol in INDICES:
                yf_symbol = INDICES[symbol]['yf']
            else:
                yf_symbol = f"{symbol}.NS"  # Assume NSE stock
            
            # Calculate date range
            stock = yf.Ticker(yf_symbol)
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=10*365)).strftime('%Y-%m-%d')
            data = stock.history(start=start_date, end=end_date, interval='1d')
            print(data)
            # data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data available for {yf_symbol}")
            
            if 'Volume' not in data.columns:
                data['Volume'] = 0
            
            features = prepare_features(data)
            return features, data['Close'].values
            
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"All yfinance attempts failed for {symbol}: {str(e)}")
                raise
            logging.warning(f"Yfinance attempt {attempt + 1} failed: {str(e)}")

#######################################################################################################################
# Function: get_data_with_fallbacks
# Input: symbol, period, max_retries
# Output: features, close_prices
# Description: Fetch data using yfinance as the primary source
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
def get_data_with_fallbacks(symbol, period='1y', max_retries=3):
    errors = []
    
    try:
        return get_stock_data_yf(symbol, period, max_retries)
    except Exception as e:
        errors.append(f"YFinance error: {str(e)}")
        logging.warning(f"YFinance failed for {symbol}: {str(e)}")
    
    error_msg = f"All data sources failed for {symbol}. Errors: {'; '.join(errors)}"
    logging.error(error_msg)
    raise DataFetchError(error_msg)

#######################################################################################################################
# Function: calculate_metrics
# Input: model, X_val, y_val, scaler
# Output: {'mape': float(mape), 'r2': float(r2), 'accuracy': float(100 - mape)}
# Description: Calculate the model metrics
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
def calculate_metrics(model, X_val, y_val, scaler):
    try:
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val)
            predictions = model(X_val_tensor)
            
            pred_reshaped = np.zeros((len(predictions), scaler.n_features_in_))
            pred_reshaped[:, 0] = predictions.numpy()[:, 0]
            y_reshaped = np.zeros((len(y_val), scaler.n_features_in_))
            y_reshaped[:, 0] = y_val[:, 0]
            
            pred_actual = scaler.inverse_transform(pred_reshaped)[:, 0]
            y_actual = scaler.inverse_transform(y_reshaped)[:, 0]
            
            mape = mean_absolute_percentage_error(y_actual, pred_actual)
            r2 = r2_score(y_actual, pred_actual)
            
            return {
                'mape': float(mape),
                'r2': float(r2),
                'accuracy': float(100 - mape)
            }
    except Exception as e:
        logging.error(f"Metrics calculation error: {str(e)}")
        raise

#################################################################################################
# Class Name - IndianStockAnalyzer
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This class contains the logic for analyzing Indian stocks using technical indicators
#################################################################################################
class IndianStockAnalyzer:
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 risk_free_rate: float = 0.05):
        """
        Initialize the Indian Stock Analyzer.
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.risk_free_rate = risk_free_rate
        self.stock_data = None
        self.stock_info = None
        self.recommendationTopGainer = {}

    def fetch_stock_data(self, symbol: str, exchange: str = "NS", 
                        start_date: str = None,
                        interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock or index data from Yahoo Finance.
        """
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
            if symbol in INDICES:
                stock_symbol = INDICES[symbol]['yf']
            else:
                stock_symbol = f"{symbol}.{exchange}"
                
            stock = yf.Ticker(stock_symbol)
            data = stock.history(start=start_date, interval=interval)

            print("intervl data : == ", data)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {stock_symbol}")
                
            self.stock_data = data
            self.stock_info = stock.info
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def fetch_fundamental_data(self, symbol: str) -> dict:
        """
        Fetch fundamental data for a given stock symbol using yfinance.
        """
        try:
            full_symbol = f"{symbol}.NS" if symbol not in INDICES else INDICES[symbol]['yf']
            stock = yf.Ticker(full_symbol)
            info = stock.info
            website = info.get('website', '')
            domain = website.split('//')[-1].split('/')[0] if website else ''
            logoURL = f"https://logo.clearbit.com/{domain}" if domain else ''

            income_stmt = stock.income_stmt
            gross_profit = income_stmt.loc['Gross Profit'].iloc[0]

            stock_historical_price_and_date = self.get_indian_stock_closing_price(symbol)
            

            fundamental_data = {
                'historical_date_price': stock_historical_price_and_date,
                'basic_info': {
                    'company_name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'companyBusinessSummary': info.get('longBusinessSummary', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'company_logo': info.get('logo_url', 'N/A'),
                    'logoURL': logoURL
                },
                'valuation_metrics': {
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                    'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                },
                'financial_health': {
                    'total_revenue': info.get('totalRevenue', 0),
                    'gross_profit': gross_profit,
                    'net_income': info.get('netIncomeToCommon', 0),
                    'total_debt': info.get('totalDebt', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'return_on_equity': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
                },
                'growth_metrics': {
                    'revenue_growth': info.get('revenueGrowth', 0) * 100,
                    'earnings_growth': info.get('earningsGrowth', 0) * 100,
                    'profit_margins': info.get('profitMargins', 0) * 100
                }
            }
        
            return fundamental_data
        
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {}

    def perform_fundamental_analysis(self, symbol: str) -> dict:
        """
        Perform comprehensive fundamental analysis.
        """
        try:
            fundamental_data = self.fetch_fundamental_data(symbol)
            
            def score_fundamental_metrics(data):
                score = 0
                pe_ratio = data['valuation_metrics']['pe_ratio']
                if 0 < pe_ratio < 15:
                    score += 2
                elif 15 <= pe_ratio <= 25:
                    score += 1
                else:
                    score -= 1
                
                debt_to_equity = data['financial_health']['debt_to_equity']
                if debt_to_equity < 0.5:
                    score += 2
                elif debt_to_equity < 1:
                    score += 1
                else:
                    score -= 1
                
                revenue_growth = data['growth_metrics']['revenue_growth']
                earnings_growth = data['growth_metrics']['earnings_growth']
                if revenue_growth > 10 and earnings_growth > 10:
                    score += 2
                elif revenue_growth > 5 and earnings_growth > 5:
                    score += 1
                else:
                    score -= 1
                
                profit_margins = data['growth_metrics']['profit_margins']
                if profit_margins > 15:
                    score += 2
                elif profit_margins > 10:
                    score += 1
                else:
                    score -= 1
                
                dividend_yield = data['valuation_metrics']['dividend_yield']
                if dividend_yield > 3:
                    score += 1
                
                return score
            
            global fundamental_score
            fundamental_score = score_fundamental_metrics(fundamental_data)

            if fundamental_score >= 4:
                recommendation = "Strong Buy"
            elif fundamental_score >= 2:
                recommendation = "Buy"
            elif fundamental_score >= 0:
                recommendation = "Hold"
            elif fundamental_score >= -2:
                recommendation = "Sell"
            else:
                recommendation = "Strong Sell"
            
            fundamental_data['recommendation'] = recommendation
            fundamental_data['score'] = fundamental_score
            
            return fundamental_data
        
        except Exception as e:
            logger.error(f"Fundamental analysis error for {symbol}: {str(e)}")
            return {}

    def calculate_rsi(self, data: pd.Series) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and Histogram."""
        exp1 = data.ewm(span=12, adjust=False).mean()
        exp2 = data.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle_band = data.rolling(window=window).mean()
        std_dev = data.rolling(window=window).std()
        upper_band = middle_band + (std_dev * 2)
        lower_band = middle_band - (std_dev * 2)
        return upper_band, middle_band, lower_band

    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators for price action analysis."""
        df = data.copy()
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_mf_14 = positive_flow.rolling(window=14).sum()
        negative_mf_14 = negative_flow.rolling(window=14).sum()
        money_ratio = positive_mf_14 / negative_mf_14
        df['MFI'] = 100 - (100 / (1 + money_ratio))
        
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        clv = clv.fillna(0)
        ad = clv * df['Volume']
        df['ADL'] = ad.cumsum()
        df['CMF'] = (ad.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum())
        df['Price_ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        return df

    def identify_support_resistance(self, data: pd.DataFrame, window: int = 20, 
                                   num_levels: int = 3) -> tuple[list, list]:
        """
        Identify support and resistance levels using various methods.
        """
        df = data.copy()
        highs = df['High'].rolling(window=window, center=True).max()
        lows = df['Low'].rolling(window=window, center=True).min()
        
        resistance_levels = []
        for i in range(window, len(df) - window):
            if highs.iloc[i] == df['High'].iloc[i] and \
               all(highs.iloc[i] >= highs.iloc[i-window:i]) and \
               all(highs.iloc[i] >= highs.iloc[i+1:i+window+1]):
                resistance_levels.append(df['High'].iloc[i])
        
        support_levels = []
        for i in range(window, len(df) - window):
            if lows.iloc[i] == df['Low'].iloc[i] and \
               all(lows.iloc[i] <= lows.iloc[i-window:i]) and \
               all(lows.iloc[i] <= lows.iloc[i+1:i+window+1]):
                support_levels.append(df['Low'].iloc[i])
        
        last_day = df.iloc[-1]
        pivot = (last_day['High'] + last_day['Low'] + last_day['Close']) / 3
        r1 = 2 * pivot - last_day['Low']
        r2 = pivot + (last_day['High'] - last_day['Low'])
        s1 = 2 * pivot - last_day['High']
        s2 = pivot - (last_day['High'] - last_day['Low'])
        resistance_levels.extend([r1, r2])
        support_levels.extend([s1, s2])
        
        current_price = df['Close'].iloc[-1]
        magnitude = 10 ** (len(str(int(current_price))) - 1)
        for mult in range(1, 11):
            level = mult * magnitude
            if abs(current_price - level) / current_price < 0.15:
                if level < current_price:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)
        
        support_levels = sorted(list(set([round(x, 2) for x in support_levels])), reverse=True)
        resistance_levels = sorted(list(set([round(x, 2) for x in resistance_levels])))
        support_levels = sorted(support_levels, key=lambda x: abs(current_price - x))
        resistance_levels = sorted(resistance_levels, key=lambda x: abs(current_price - x))
        
        return support_levels[:num_levels], resistance_levels[:num_levels]

    def analyze_price_action(self, data: pd.DataFrame) -> dict:
        """
        Analyze price action patterns and market context.
        """
        df = data.copy()
        recent_data = df.tail(5)
        body_sizes = abs(recent_data['Close'] - recent_data['Open'])
        wicks_upper = recent_data['High'] - recent_data[['Open', 'Close']].max(axis=1)
        wicks_lower = recent_data[['Open', 'Close']].min(axis=1) - recent_data['Low']
        
        avg_body = body_sizes.mean()
        avg_upper_wick = wicks_upper.mean()
        avg_lower_wick = wicks_lower.mean()
        recent_volatility = df['Close'].pct_change().rolling(window=5).std().iloc[-1] * 100
        
        patterns = []
        last_day = df.iloc[-1]
        prev_day = df.iloc[-2]
        
        if abs(last_day['Open'] - last_day['Close']) / (last_day['High'] - last_day['Low']) < 0.1:
            patterns.append("Doji")
        
        if (last_day['Low'] < last_day[['Open', 'Close']].min()) and \
           (wicks_lower.iloc[-1] > 2 * body_sizes.iloc[-1]) and \
           (wicks_upper.iloc[-1] < 0.5 * body_sizes.iloc[-1]):
            patterns.append("Hammer")
        
        if (last_day['High'] > last_day[['Open', 'Close']].max()) and \
           (wicks_upper.iloc[-1] > 2 * body_sizes.iloc[-1]) and \
           (wicks_lower.iloc[-1] < 0.5 * body_sizes.iloc[-1]):
            patterns.append("Shooting Star")
        
        if (last_day['Open'] > prev_day['Close']) and (last_day['Close'] < prev_day['Open']):
            patterns.append("Bearish Engulfing")
        elif (last_day['Open'] < prev_day['Close']) and (last_day['Close'] > prev_day['Open']):
            patterns.append("Bullish Engulfing")
        
        sma20 = df['SMA_20'].iloc[-1]
        sma50 = df['SMA_50'].iloc[-1]
        
        if df['Close'].iloc[-1] > sma20 > sma50:
            trend = "Strong Uptrend"
        elif df['Close'].iloc[-1] > sma20 and sma20 < sma50:
            trend = "Possible Trend Reversal (Bullish)"
        elif df['Close'].iloc[-1] < sma20 < sma50:
            trend = "Strong Downtrend"
        elif df['Close'].iloc[-1] < sma20 and sma50 < sma20:
            trend = "Possible Trend Reversal (Bearish)"
        else:
            trend = "Sideways/Consolidation"
        
        volume_signal = ""
        if df['Volume'].iloc[-1] > df['Volume_SMA_20'].iloc[-1] * 1.5:
            if df['Close'].iloc[-1] > df['Open'].iloc[-1]:
                volume_signal = "High Volume Bullish Day"
            else:
                volume_signal = "High Volume Bearish Day"
        
        momentum = "Neutral"
        if df['MFI'].iloc[-1] < 20:
            momentum = "Oversold"
        elif df['MFI'].iloc[-1] > 80:
            momentum = "Overbought"
        
        result = {
            'trend': trend,
            'patterns': patterns,
            'momentum': momentum,
            'volume_signal': volume_signal,
            'recent_volatility': round(recent_volatility, 2),
            'avg_body_size': round(avg_body, 2),
            'avg_upper_wick': round(avg_upper_wick, 2),
            'avg_lower_wick': round(avg_lower_wick, 2)
        }
        
        return result

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """Calculate various technical indicators."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        df = self.stock_data.copy()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df = self.calculate_volume_indicators(df)
        return df

    def calculate_risk_metrics(self) -> dict:
        """Calculate various risk metrics."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        daily_returns = self.stock_data['Close'].pct_change().dropna()
        excess_returns = daily_returns - (self.risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        volatility = daily_returns.std() * np.sqrt(252) * 100
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        try:
            nifty = yf.download('^NSEI', start=self.stock_data.index[0])
            nifty_returns = nifty['Close'].pct_change().dropna()
            beta = np.cov(daily_returns, nifty_returns)[0][1] / np.var(nifty_returns)
        except:
            beta = None
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'beta': beta
        }

    def generate_trading_signals(self) -> tuple[str, list, list]:
        """Generate trading signals based on technical indicators."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        df = self.calculate_technical_indicators()
        current = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        global technical_Analysis_score
        technical_Analysis_score = 0
        predictTopgainer = []
        
        if current['RSI'] < self.rsi_oversold:
            signals.append(f"Oversold (RSI: {current['RSI']:.2f})")
            technical_Analysis_score += 1
        elif current['RSI'] > self.rsi_overbought:
            signals.append(f"Overbought (RSI: {current['RSI']:.2f})")
            technical_Analysis_score -= 1
            
        if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append("MACD Bullish Crossover")
            technical_Analysis_score += 1
        elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals.append("MACD Bearish Crossover")
            technical_Analysis_score -= 1
            
        if current['Close'] > current['SMA_200']:
            signals.append("Price above 200 SMA (Bullish)")
            technical_Analysis_score += 0.5
        else:
            signals.append("Price below 200 SMA (Bearish)")
            technical_Analysis_score -= 0.5
            
        if current['Close'] < current['BB_Lower']:
            signals.append("Price below lower Bollinger Band (Potential Buy)")
            technical_Analysis_score += 1
        elif current['Close'] > current['BB_Upper']:
            signals.append("Price above upper Bollinger Band (Potential Sell)")
            technical_Analysis_score -= 1
        
        if current['Volume'] > current['Volume_SMA_20'] * 1.5:
            if current['Close'] > current['Open']:
                signals.append("High Volume Bullish Day (Confirmation)")
                technical_Analysis_score += 1
            else:
                signals.append("High Volume Bearish Day (Confirmation)")
                technical_Analysis_score -= 1
        
        if current['MFI'] < 20:
            signals.append(f"MFI Oversold ({current['MFI']:.2f})")
            technical_Analysis_score += 1
        elif current['MFI'] > 80:
            signals.append(f"MFI Overbought ({current['MFI']:.2f})")
            technical_Analysis_score -= 1

        if current['CMF'] > 0.1:
            signals.append("Positive Chaikin Money Flow (Bullish)")
            technical_Analysis_score += 0.5
        elif current['CMF'] < -0.1:
            signals.append("Negative Chaikin Money Flow (Bearish)")
            technical_Analysis_score -= 0.5
            
        if current['Price_ROC'] > 5:
            signals.append(f"Strong Price Momentum (ROC: {current['Price_ROC']:.2f}%)")
            technical_Analysis_score += 0.5
        elif current['Price_ROC'] < -5:
            signals.append(f"Weak Price Momentum (ROC: {current['Price_ROC']:.2f}%)")
            technical_Analysis_score -= 0.5

        
        
            
        if technical_Analysis_score >= 2:
            recommendation = "Strong Buy"
            predictTopgainer.append(technical_Analysis_score)
        elif technical_Analysis_score > 0:
            recommendation = "Buy"
            predictTopgainer.append(technical_Analysis_score)
        elif technical_Analysis_score == 0:
            recommendation = "Hold"
            predictTopgainer.append(technical_Analysis_score)
        elif technical_Analysis_score > -2:
            recommendation = "Sell"
        else:
            recommendation = "Strong Sell"
            
        return recommendation, signals, predictTopgainer

    def fetch_top_gainers(self, limit: int = 5) -> list:
        """Fetch top gainers from NSE."""
        nse_url = "https://www.nseindia.com"
        api_url_topgainer = "https://www.nseindia.com/api/live-analysis-variations?index=gainers"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }
        session = requests.Session()
        self.recommendationTopGainer = {}
        
        try:
            session.get(nse_url, headers=headers)
            gainer_response = session.get(api_url_topgainer, headers=headers)
            gainer_response.raise_for_status()
            top_gainer_data = gainer_response.json()
            top_gainer_symbols = [item["symbol"] for item in top_gainer_data["NIFTY"].get("data", [])]
            logger.info("Symbols extracted from top gainer API: %s", top_gainer_symbols)
            
            for symbol in top_gainer_symbols[:limit]:
                try:
                    self.fetch_stock_data(symbol)
                    self.calculate_technical_indicators()
                    _, _, score_list = self.generate_trading_signals()
                    if score_list:
                        self.recommendationTopGainer[symbol] = score_list[0]
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    self.recommendationTopGainer[symbol] = None
            
            return [{'symbol': symbol, 'percentChange': item['var']} 
                    for item, symbol in zip(top_gainer_data["NIFTY"].get("data", []), top_gainer_symbols)]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching top gainers: {str(e)}")
            return []

    def predict_stock_prices(self, symbol: str, prediction_days: int = 30) -> dict:
        """
        Predict stock or index prices for the next 7, 15, or 30 days.
        """
        try:
            prediction_days = int(prediction_days)
            if prediction_days not in [7, 15, 30]:
                prediction_days = 30
            
            logging.info(f"Fetching data for {symbol} with {prediction_days} days prediction")
            
            features, close_prices = get_data_with_fallbacks(symbol)
            
            if len(features) < 30 + prediction_days:
                raise ValueError('Insufficient data points')
            
            X_train, X_val, y_train, y_val, scaler = prepare_data(features, prediction_days=prediction_days)
            
            model = EnhancedStockBiLSTM(input_size=features.shape[1], output_size=prediction_days)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            train_dataset = StockDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            model.train()
            for epoch in range(50):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
            
            metrics = calculate_metrics(model, X_val, y_val, scaler)
            
            last_sequence = features[-30:]
            X = torch.FloatTensor(scaler.transform(last_sequence)).unsqueeze(0)
            
            model.eval()
            with torch.no_grad():
                scaled_predictions = model(X)[0].numpy()
            
            pred_reshaped = np.zeros((len(scaled_predictions), scaler.n_features_in_))
            pred_reshaped[:, 0] = scaled_predictions
            predictions = scaler.inverse_transform(pred_reshaped)[:, 0]
            
            historical = close_prices[-30:].tolist()
            dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                     for i in range(0, prediction_days + 1)]
            
            return {
                'dates': dates,
                'values': historical + predictions.tolist(),
                'predictions': predictions.tolist(),
                'lastClose': historical[-1],
                'metrics': metrics
            }
        
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {str(e)}")
            raise

    def analyze_stock(self, symbol: str, exchange: str = "NS", start_date: str = None, clickPredict: bool = True) -> dict:
        """
        Perform a comprehensive analysis of a stock or index.
        """
        self.fetch_stock_data(symbol, exchange, start_date)
        tech_data = self.calculate_technical_indicators()
        recommendation, signals, predictTopgainer = self.generate_trading_signals()
        risk_metrics = self.calculate_risk_metrics()
        support_levels, resistance_levels = self.identify_support_resistance(tech_data)
        price_action = self.analyze_price_action(tech_data)
        fundamental_analysis = self.perform_fundamental_analysis(symbol)
        difference = self.calculate_year(start_date)
        average_recomendation = self.average_recomendation(difference)
        
        # if(clickPredict):
        #     try:
        #          prediction = self.predict_stock_prices(symbol)
                 
        #     except Exception as e:
        #         prediction = {'error': str(e)}
        
        chart_data = []
        for date, row in tech_data.iterrows():
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': round(float(row['Close']), 2),
                'sma20': round(float(row['SMA_20']), 2) if pd.notnull(row['SMA_20']) else None,
                'sma50': round(float(row['SMA_50']), 2) if pd.notnull(row['SMA_50']) else None,
                'rsi': round(float(row['RSI']), 2) if pd.notnull(row['RSI']) else None,
                'macd': round(float(row['MACD']), 2) if pd.notnull(row['MACD']) else None,
                'signal': round(float(row['MACD_Signal']), 2) if pd.notnull(row['MACD_Signal']) else None,
                'volume': int(row['Volume']),
                'volumeSMA20': int(row['Volume_SMA_20']) if pd.notnull(row['Volume_SMA_20']) else None
            })
        
        result = {
            'symbol': symbol,
            'exchange': exchange,
            'recommendation': recommendation,
            'averageRecomendation': average_recomendation,
            'fundamentalAnalysis': fundamental_analysis,
            'currentPrice': round(float(self.stock_data['Close'][-1]), 2),
            'signals': signals,
            'predictTopgainer': predictTopgainer,
            'riskMetrics': {
                'sharpeRatio': round(float(risk_metrics['sharpe_ratio']), 2),
                'volatility': round(float(risk_metrics['volatility']), 2),
                'maxDrawdown': round(float(risk_metrics['max_drawdown']), 2),
                'beta': round(float(risk_metrics['beta']), 2) if risk_metrics['beta'] else None
            },
            'supportResistance': {
                'support': [round(float(level), 2) for level in support_levels],
                'resistance': [round(float(level), 2) for level in resistance_levels]
            },
            'priceAction': {
                'trend': price_action['trend'],
                'patterns': price_action['patterns'],
                'momentum': price_action['momentum'],
                'volumeSignal': price_action['volume_signal'],
                'recentVolatility': price_action['recent_volatility']
            },
            'chartData': chart_data,
            # 'prediction': prediction
        }

           
        
        
        return result
    
    def clickPredictStock(self, symbol: str):
        try:
            prediction = self.predict_stock_prices(symbol)
            result = {'prediction': prediction}
                 
        except Exception as e:
            prediction = {'error': str(e)}
        print(result)
        return result

    def analyze_top_gainers(self, limit: int = 5) -> list:
        """
        Fetch and analyze top gainers stocks with enhanced analysis.
        """
        try:
            top_gainers = self.fetch_top_gainers(limit)
            results = {}
            
            for gainer in top_gainers:
                symbol = gainer['symbol']
                logger.info(f"Analyzing top gainer: {symbol}")
                
                try:
                    analysis = self.analyze_stock(symbol)
                    analysis['percentChange'] = gainer['percentChange']
                    results.update({symbol : analysis})
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing top gainers: {str(e)}")
            raise

    # average recomendation of stocks for buying, selling and holding
    def average_recomendation (self, difference):
            if difference < 1:
                if technical_Analysis_score >= 2:
                    return "Strong Buy"
                elif technical_Analysis_score > 0:
                    return "Buy"
           
                elif technical_Analysis_score == 0:
                    return "Hold"
                elif technical_Analysis_score > -2:
                    return "Sell"
                else:
                    return "Strong Sell"
            else:
                average_tech_and_fundamental_score = (technical_Analysis_score*0.4) + (fundamental_score*0.6)
                if average_tech_and_fundamental_score >=4:
                    return "Strong Buy"
                elif average_tech_and_fundamental_score >= 2:
                    return "Buy"
                elif average_tech_and_fundamental_score >= 0:
                    return "Hold"
                elif average_tech_and_fundamental_score >= -2:
                    return "Sell"
                else:
                    return "Strong Sell"
         
    

    # year difference function: it give the entered date is < 1 year > 1 year
    def calculate_year(self,input_date_str):
        """Categorize date into >2y, >1y, or <1y"""
    # Parse input date
        input_date = datetime.strptime(input_date_str, "%Y-%m-%d").date()
        today = datetime.today().date()
    
    # Calculate time difference
        delta = today - input_date
    
        if delta.days > 730:  # 365*2 = 730
            return 2
        elif delta.days > 365:
            return 1
        else:
            return -1
        

    def get_indian_stock_closing_price(self,symbol, period='15y'):
        if not symbol.endswith(".NS"):
            symbol += ".NS"

        stock = yf.Ticker(symbol)

        historical = stock.history(period=period)

        if(historical.empty):
            logger.error(f"No data found for {symbol}")
            return None
        
        # Extract closing prices and dates
        closing_data = historical[['Close']].reset_index()
        closing_data['Date'] = closing_data['Date'].dt.strftime('%Y-%m-%d')
        closing_data['Close'] = closing_data['Close'].round(2)  # Round to 2 decimal places

        closing_price_and_date = {
            'Date': closing_data['Date'],
            'close' : closing_data['Close']
        }

        df = pd.DataFrame(closing_price_and_date)
        chart_data = df.to_dict(orient='records')


        return chart_data
#######################################################################################################################
# Function: check_data_sources
# Input: None
# Output: jsonify({'status': 'operational' if any(v == 'available' for v in results.values()) else 'down', 'sources': results})
# Description: Check the availability of data sources
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
@app.route('/api/data/status')
def check_data_sources():
    try:
        results = {}
        
        def check_source(name, symbol='^NSEI'):
            try:
                if name == 'yfinance':
                    yf.Ticker(symbol).history(period='1d')
                return 'available'
            except:
                return 'unavailable'
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(check_source, 'yfinance'): 'yfinance'
            }
            for future in concurrent.futures.as_completed(futures):
                source = futures[future]
                results[source] = future.result()
        
        return jsonify({
            'status': 'operational' if any(v == 'available' for v in results.values()) else 'down',
            'sources': results
        })
    except Exception as e:
        logging.error(f"Error checking data sources: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

#######################################################################################################################
# Function: get_indices
# Input: None
# Output: jsonify(indices)
# Description: Return the list of available indices
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
@app.route('/api/indices')
def get_indices():
    try:
        indices = list(INDICES.keys())
        logger.info(f"API call to /api/indices - Returning: {indices}")
        return jsonify(indices)
    except Exception as e:
        logger.error(f"Error in /api/indices: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

#######################################################################################################################
# Function: predict
# Input: symbol
# Output: jsonify(response)
# Description: Predict stock or index prices for the next 7, 15, or 30 days
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
@app.route('/api/predict/<symbol>')
def predict(symbol):
    try:
        prediction_days = int(request.args.get('days', 7))
        if prediction_days not in [7, 15, 30]:
            prediction_days = 7
        
        analyzer = IndianStockAnalyzer()
        prediction = analyzer.predict_stock_prices(symbol, prediction_days)
        
        return jsonify(prediction)
        
    except DataFetchError as e:
        logger.error(f"Data fetch error for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

#################################################################################################
# Function Name - index
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - Render the index.html template
#################################################################################################
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

#################################################################################################
# Function Name - stocks
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - Render the stock analyzer template
#################################################################################################
@app.route('/stocks')
def stocks():
    try:
        return render_template('stockAnalyzerWithInput.html')
    except Exception as e:
        logger.error(f"Error rendering stocks page: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

#################################################################################################
# Function Name - analyze
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - Analyze stock data and return results
#################################################################################################
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        exchange = data.get('exchange', 'NS')
        start_date = data.get('startDate')

        print(f"Start date {start_date}")

        analyzer = IndianStockAnalyzer()
        analysis = analyzer.analyze_stock(symbol, exchange, start_date)
        
        
        # difference = analyzer.calculate_year(start_date)
              
        # avg_recomendation = analyzer.average_recomendation(difference)

        

        response = {
            'success': True,
            'data': analysis,
            'technical_score': technical_Analysis_score,
            'fundamental_score': fundamental_score
            # "topGaineernew": tg
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

#######################################################################################################################
# Function: health_check
# Input: None
# Output: jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat(), 'version': '1.0.0'})
# Description: Check the health of the application
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
@app.route('/api/health')
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


# @####################################################################################################################################
# Router to handle the toogle buttion
# input : just click on the mouse buttion
# output: success
@app.route('/api/check', methods=['POST'])  # Add POST method explicitly
def check():
    try:
        # Get data from request (optional)
        data = request.get_json()
        symbol = data["symbol"]
        print("Received data:", data["success"])  # This will show the incoming JSON
        analyzer = IndianStockAnalyzer()
        prediction = analyzer.clickPredictStock(symbol)

        response = {
            'success': True,
            'data': prediction
        }
        return jsonify(response)
        
    
    except Exception as e:
        print(f"Toggle check error: {str(e)}")  # Simplified error logging
        return jsonify({'error': 'Internal server error'}), 500


#######################################################################################################################
# Function: not_found_error
# Input: error
# Output: jsonify({'error': 'Not found'}), 404
# Description: Handle 404 errors
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

#######################################################################################################################
# Function: internal_error
# Input: error
# Output: jsonify({'error': 'Internal server error'}), 500
# Description: Handle 500 errors
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

#######################################################################################################################
# Function: service_unavailable_error
# Input: error
# Output: jsonify({'error': 'Service temporarily unavailable'}), 503
# Description: Handle 503 errors
# Author: Ojas Ulhas Dighe
# Date: 24-Feb-2025
#######################################################################################################################
@app.errorhandler(503)
def service_unavailable_error(error):
    return jsonify({'error': 'Service temporarily unavailable'}), 503

@click.command()
@click.option('--limit', default=5, help='Number of top gainers to analyze')
def analyze_top_gainers_cli(limit):
    """Command line function to analyze top gainers and print results."""
    try:
        print(f"Fetching and analyzing top {limit} gainers from NSE...")
        
        analyzer = IndianStockAnalyzer()
        results = analyzer.analyze_top_gainers(limit)
        
        print("\n=================== TOP GAINERS ANALYSIS ===================\n")
        
        for idx, result in enumerate(results, 1):
            print(f"#{idx}: {result['symbol']} - ₹{result['currentPrice']} ({result['percentChange']}%)")
            print(f"Recommendation: {result['recommendation']}")
            
            print("Price Action:")
            print(f"  - Trend: {result['priceAction']['trend']}")
            if result['priceAction']['patterns']:
                print(f"  - Patterns: {', '.join(result['priceAction']['patterns'])}")
            print(f"  - Momentum: {result['priceAction']['momentum']}")
            if result['priceAction']['volumeSignal']:
                print(f"  - Volume: {result['priceAction']['volumeSignal']}")
            
            print("Support & Resistance:")
            print(f"  - Support Levels: {', '.join([f'₹{level}' for level in result['supportResistance']['support']])}")
            print(f"  - Resistance Levels: {', '.join([f'₹{level}' for level in result['supportResistance']['resistance']])}")
            
            print("Signals:")
            for signal in result['signals']:
                print(f"  - {signal}")
                
            print("Risk Metrics:")
            print(f"  - Sharpe Ratio: {result['riskMetrics']['sharpeRatio']}")
            print(f"  - Volatility: {result['riskMetrics']['volatility']}%")
            print(f"  - Max Drawdown: {result['riskMetrics']['maxDrawdown']}%")
            if result['riskMetrics']['beta']:
                print(f"  - Beta: {result['riskMetrics']['beta']}")
            
            print("Prediction:")
            if 'error' not in result['prediction']:
                print(f"  - Predicted prices for next {len(result['prediction']['predictions'])} days: "
                      f"{', '.join([f'₹{round(p, 2)}' for p in result['prediction']['predictions']])}")
            else:
                print(f"  - Prediction error: {result['prediction']['error']}")
                
            print("\n" + "-" * 60 + "\n")
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    initialize_app(app)
    app.run(debug=True, host='0.0.0.0', port=5000)
