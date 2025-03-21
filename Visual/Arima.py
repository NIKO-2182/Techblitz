import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def train_arima(series, order=(1,1,1)):
    """Train ARIMA model and make predictions"""
    # Split data into train and test
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(train, order=order)
    results = model.fit()
    
    # Make predictions
    predictions = results.forecast(steps=len(test))
    
    # Calculate MSE
    mse = mean_squared_error(test, predictions)
    
    return predictions, mse, results

def train_all_metrics(data, targets):
    """Train ARIMA models for all target metrics"""
    results_dict = {}
    
    print("Training ARIMA models...")
    print("=" * 50)
    
    for target in targets:
        predictions, mse, model = train_arima(data[target])
        results_dict[target] = {
            'predictions': predictions,
            'mse': mse,
            'model': model
        }
        print(f"\n{target}:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Next Month Prediction: {predictions[-1]:.2f}")
        print("-" * 30)
    
    return results_dict