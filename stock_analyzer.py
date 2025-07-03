import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta
from models import EnhancedStockBiLSTM, StockDataset
from data_utils import prepare_data, prepare_features, calculate_metrics
import torch
from torch.utils.data import DataLoader

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

class DataFetchError(Exception):
    """Custom exception for data fetching errors"""
    pass

class IndianStockAnalyzer:
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 risk_free_rate: float = 0.05):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.risk_free_rate = risk_free_rate
        self.stock_data = None
        self.stock_info = None
        self.recommendationTopGainer = {}
        self.fundamental_score = 0
        self.technical_Analysis_score = 0

    def fetch_stock_data(self, symbol: str, exchange: str = "NS", 
                        start_date: str = None,
                        interval: str = "1d") -> pd.DataFrame:
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
            if symbol in INDICES:
                stock_symbol = INDICES[symbol]['yf']
            else:
                stock_symbol = f"{symbol}.{exchange}"
                
            stock = yf.Ticker(stock_symbol)
            data = stock.history(start=start_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {stock_symbol}")
                
            self.stock_data = data
            self.stock_info = stock.info
            
            return data
            
        except Exception as e:
            logging.error(f"Error fetching stock data: {str(e)}")
            raise

    def get_data_with_fallbacks(self, symbol, period='1y', max_retries=3):
        errors = []
        
        try:
            return self.get_stock_data_yf(symbol, period, max_retries)
        except Exception as e:
            errors.append(f"YFinance error: {str(e)}")
            logging.warning(f"YFinance failed for {symbol}: {str(e)}")
        
        error_msg = f"All data sources failed for {symbol}. Errors: {'; '.join(errors)}"
        logging.error(error_msg)
        raise DataFetchError(error_msg)

    def get_stock_data_yf(self, symbol, period='10y', max_retries=3):
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    import time
                    time.sleep(2)
                
                if symbol in INDICES:
                    yf_symbol = INDICES[symbol]['yf']
                else:
                    yf_symbol = f"{symbol}.NS"
                
                stock = yf.Ticker(yf_symbol)
                end_date = datetime.today().strftime('%Y-%m-%d')
                start_date = (datetime.today() - timedelta(days=10*365)).strftime('%Y-%m-%d')
                data = stock.history(start=start_date, end=end_date, interval='1d')
                
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

    def predict_stock_prices(self, symbol: str, prediction_days: int = 30) -> dict:
        try:
            prediction_days = int(prediction_days)
            if prediction_days not in [7, 15, 30]:
                prediction_days = 30
            
            logging.info(f"Fetching data for {symbol} with {prediction_days} days prediction")
            
            features, close_prices = self.get_data_with_fallbacks(symbol)
            
            if len(features) < 30 + prediction_days:
                raise ValueError('Insufficient data points')
            
            X_train, X_val, y_train, y_val, scaler = prepare_data(features, prediction_days=prediction_days)
            
            model = EnhancedStockBiLSTM(input_size=features.shape[1], output_size=prediction_days)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            train_dataset = StockDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            model.train()
            for epoch in range(33):
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
            logging.error(f"Prediction error for {symbol}: {str(e)}")
            raise

    def fetch_top_gainers(self, limit: int = 5) -> list:
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
            logging.info("Symbols extracted from top gainer API: %s", top_gainer_symbols)
            
            # Use multithreading to fetch and analyze top gainers concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.fetch_and_score_gainer, symbol): symbol for symbol in top_gainer_symbols[:limit]}
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        score = future.result()
                        self.recommendationTopGainer[symbol] = score
                    except Exception as e:
                        logging.error(f"Error processing {symbol}: {str(e)}")
                        self.recommendationTopGainer[symbol] = None
            
            return [{'symbol': symbol, 'percentChange': item['var']} 
                    for item, symbol in zip(top_gainer_data["NIFTY"].get("data", []), top_gainer_symbols)]
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching top gainers: {str(e)}")
            return []

    def fetch_and_score_gainer(self, symbol):
        self.fetch_stock_data(symbol)
        self.calculate_technical_indicators()
        _, _, score_list = self.generate_trading_signals()
        if score_list:
            return score_list[0]
        return None

    def calculate_rsi(self, data):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data):
        exp1 = data.ewm(span=12, adjust=False).mean()
        exp2 = data.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def calculate_bollinger_bands(self, data, window=20):
        middle_band = data.rolling(window=window).mean()
        std_dev = data.rolling(window=window).std()
        upper_band = middle_band + (std_dev * 2)
        lower_band = middle_band - (std_dev * 2)
        return upper_band, middle_band, lower_band

    def calculate_volume_indicators(self, data):
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

    def identify_support_resistance(self, data, window=20, num_levels=3):
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

    def analyze_price_action(self, data):
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

    def calculate_technical_indicators(self):
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

    def calculate_risk_metrics(self):
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

    def generate_trading_signals(self):
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
