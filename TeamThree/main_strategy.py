from master import MASTERModel
import pickle
import numpy as np
import time

# Please install qlib first before load the data.

universe = 'csi300' # ['csi300','csi800']
prefix = 'opensource' # ['original','opensource'], which training data are you using
train_data_dir = f'data'
with open(f'{train_data_dir}\{prefix}\{universe}_dl_train.pkl', 'rb') as f:
    dl_train = pickle.load(f)

predict_data_dir = f'data\opensource'
with open(f'{predict_data_dir}\{universe}_dl_valid.pkl', 'rb') as f:
    dl_valid = pickle.load(f)
with open(f'{predict_data_dir}\{universe}_dl_test.pkl', 'rb') as f:
    dl_test = pickle.load(f)

print("Data Loaded.")


d_feat = 158
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 158
gate_input_end_index = 221

if universe == 'csi300':
    beta = 5
elif universe == 'csi800':
    beta = 2

n_epoch = 1
lr = 1e-5
GPU = 0
train_stop_loss_thred = 0.95


ic = []
icir = []
ric = []
ricir = []

# Training
######################################################################################
'''for seed in [0, 1, 2, 3, 4]:
    model = MASTERModel(
        d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
        save_path='model', save_prefix=f'{universe}_{prefix}'
    )

    start = time.time()
    # Train
    model.fit(dl_train, dl_valid)

    print("Model Trained.")

    # Test
    predictions, metrics = model.predict(dl_test)
    
    running_time = time.time()-start
    
    print('Seed: {:d} time cost : {:.2f} sec'.format(seed, running_time))
    print(metrics)

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])'''
######################################################################################

# Load and Test
######################################################################################
for seed in [0]:
    param_path = f'model\{universe}_{prefix}_{seed}.pkl'

    print(f'Model Loaded from {param_path}')
    model = MASTERModel(
            d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
            save_path='model/', save_prefix=universe
        )
    model.load_param(param_path)
    ###### 1
    predictions, metrics, real_returns, real_prices, market_cap = model.predict(dl_test)
    ######
    print(metrics)

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])
    
    ####### 2
    # From Predictions to DataFrame
    df_predictions = predictions.reset_index()  # 让 MultiIndex 变成普通列
    df_predictions.columns = ["Tiker", "Date", "Predicted_Return"]  # Rename Column
    
    df_real_returns = real_returns.reset_index(drop=True)
    df_real_prices = real_prices.reset_index(drop=True)
    df_market_cap = market_cap.reset_index(drop=True)

    # 合并到表格右侧
    df_predictions["Actual_Return"] = df_real_returns
    df_predictions["Price"] = df_real_prices
    df_predictions["Market_Cap"] = df_market_cap


    # to CSV
    csv_path = "predictions.csv"
    df_predictions.to_csv(csv_path, index=False)
    ######

######################################################################################

# Compared to Real Return
print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))

###### 3
print(df_predictions.head())

# Strategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Copy the dataset to avoid modifying the original data
df = df_predictions.copy()

# Market Cap Weighted Trading Strategy
#
# This strategy selects the top N stocks each day based on the highest Predicted_Return.
# The selected stocks are weighted by their Market_Cap to determine portfolio allocation.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Copy the dataset to avoid modifying the original data
df = df_predictions.copy()

# Selection parameter
TOP_N = 5  # Number of top stocks selected daily
INITIAL_CAPITAL = 1_000_000  # Initial portfolio value in cash
TRADING_COST_RATE = 0.001  # 0.1% trading cost per transaction

def TopN(df):
    '''
    Strategy Overview:
        1. Daily rebalancing: Selects top N stocks (configurable TOP_N) each day based on highest predicted returns
        2. Market-cap weighting: Allocates capital proportionally to selected stocks' market capitalizations
        3. Realistic execution: 
            - Calculates position sizes using daily closing prices
            - Applies trading costs (0.1% per trade) for position changes
            - Tracks portfolio value with compounding returns
        4. Key mechanics:
            - Maintains daily position tracking to calculate turnover
            - Uses actual returns (not predictions) for performance calculation
            - Automatically handles missing positions as zero-weight allocations
        5. Output: Calculates cumulative returns, Sharpe ratio, and max drawdown
    '''
    # Ensure data is sorted by date
    df = df.sort_values(by=['Date'])
    unique_dates = df['Date'].unique()
    portfolio_value = INITIAL_CAPITAL  # Track portfolio value
    portfolio_returns = []
    prev_positions = {}  # Store previous day's holdings
    
    for date in unique_dates:
        daily_data = df[df['Date'] == date]
        
        # Select top N stocks based on Predicted_Return
        top_stocks = daily_data.nlargest(TOP_N, 'Predicted_Return')
        tickers = top_stocks['Tiker'].values
        prices = top_stocks['Price'].values
        market_caps = top_stocks['Market_Cap'].values
        
        # Compute market cap weighted allocation
        weights = market_caps / np.sum(market_caps)
        allocation = portfolio_value * weights  # Allocate capital based on weight
        new_positions = {ticker: alloc / price for ticker, alloc, price in zip(tickers, allocation, prices)}
        
        # Compute trading cost
        trading_cost = 0
        for ticker, new_shares in new_positions.items():
            prev_shares = prev_positions.get(ticker, 0)
            trading_cost += abs(new_shares - prev_shares) * prices[list(tickers).index(ticker)] * TRADING_COST_RATE
        
        # Compute portfolio return based on price changes
        actual_returns = top_stocks['Actual_Return'].values
        portfolio_return = np.sum(weights * actual_returns)
        
        # Update portfolio value
        portfolio_value *= (1 + portfolio_return)
        portfolio_value -= trading_cost  # Deduct trading cost
        prev_positions = new_positions  # Update holdings
        
        portfolio_returns.append(portfolio_value)
    
    # Convert results to DataFrame
    results = pd.DataFrame({'Date': unique_dates, 'Portfolio_Value': portfolio_returns})
    results['Cumulative_Return'] = results['Portfolio_Value'] / INITIAL_CAPITAL
    
    return results

# Run backtest
results = TopN(df)

# Plot cumulative return
plt.figure(figsize=(12, 6))
plt.plot(results['Date'], results['Cumulative_Return'], label='Market Cap Weighted Portfolio', color='b')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Market Cap Weighted Trading Strategy')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Compute performance metrics
portfolio_daily_returns = results['Cumulative_Return'].pct_change().dropna()
sharpe_ratio = portfolio_daily_returns.mean() / portfolio_daily_returns.std() * np.sqrt(252)
max_drawdown = (results['Cumulative_Return'] / results['Cumulative_Return'].cummax() - 1).min()

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

######