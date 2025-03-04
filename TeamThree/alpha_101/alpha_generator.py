# generate WorldQuant Alpha101 
import pandas as pd
from utils import Alphas

def data_preprocessing(df):
    df['Date'] = pd.to_datetime(df['Date'])
    # Compute Typical Price
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = df.groupby('Ticker', group_keys=False).apply(
    lambda x: (x['Typical_Price'] * x['Volume']).rolling(window=5).sum() / x['Volume'].rolling(window=5).sum()
    )
    return df
    
if __name__ == 'main':
    df = pd.read_csv('stock_data.csv')
    df = data_preprocessing(df)
    
    
    stock = Alphas(df)

# List of available alpha factors
alpha_indices = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
    41, 42, 43, 44, 45, 46, 47, 49, 
    50, 51, 52, 53, 54, 55, 57, 60, 
    61, 62, 64, 65, 66, 68, 
    71, 72, 73, 74, 75, 77, 78, 
    81, 83, 84, 85, 86, 88, 
    92, 94, 95, 96, 98, 99, 101
]

# Loop through the indices and dynamically assign the alpha values
for i in alpha_indices:
    alpha_name = f'alpha{i:03d}'  # Ensure consistent naming (e.g., 'alpha001')
    df[alpha_name] = getattr(stock, alpha_name)()
    df.to_csv('data.csv', index=False)