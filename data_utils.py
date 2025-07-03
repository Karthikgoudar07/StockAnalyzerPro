import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import torch

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

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

def prepare_features(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    
    df = df.dropna()
    
    if len(df) < 50:
        raise ValueError("Insufficient data points after processing")
    
    return df[['Close', 'Volume', 'MA20', 'RSI', 'MACD']].values

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
