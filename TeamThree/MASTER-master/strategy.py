# Strategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Copy the dataset to avoid modifying the original data
df = pd.read_csv("predictions_1.csv")

# Market Cap Weighted Trading Strategy
#
# This strategy selects the top N stocks each day based on the highest Predicted_Return.
# The selected stocks are weighted by their Market_Cap to determine portfolio allocation.

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
    portfolio_values = []
    prev_positions = {}  # Store previous day's holdings
    
    for date in unique_dates:
        daily_data = df[df['Date'] == date]
        
        # Select top N stocks based on Predicted_Return
        top_stocks = daily_data.nlargest(TOP_N, 'Predicted_Return')
        tickers = top_stocks['Ticker'].values
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
            
        for ticker, prev_shares in prev_positions.items():
            if ticker not in new_positions:
                last_price = df[(df['Date'] == date) & (df['Ticker'] == ticker)]['Price'].values
                if len(last_price) > 0: 
                    trading_cost += abs(prev_shares) * last_price[0] * TRADING_COST_RATE
                    
        # Compute portfolio return based on price changes
        actual_returns = top_stocks['Actual_Return'].values
        portfolio_return = np.sum(weights * actual_returns)
        
        # Update portfolio value
        portfolio_value *= (1 + portfolio_return)
        portfolio_value -= trading_cost  # Deduct trading cost
        prev_positions = new_positions  # Update holdings
        
        portfolio_returns.append(portfolio_return)
        portfolio_values.append(portfolio_value)
    
    # Convert results to DataFrame
    results = pd.DataFrame({'Date': unique_dates, 'Portfolio_Value': portfolio_values, 'Portfolio_Return': portfolio_returns})
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
risk_free_rate = 0.04 / 252
epsilon = 1e-8  # Small value to avoid division by zero
portfolio_daily_returns = results['Cumulative_Return'].pct_change().dropna() # results['Portfolio_Return']
excess_returns = portfolio_daily_returns - risk_free_rate
sharpe_ratio = (excess_returns.mean() / (excess_returns.std() + epsilon)) * np.sqrt(252)

max_drawdown = (results['Cumulative_Return'] / results['Cumulative_Return'].cummax() - 1).min()

print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Max Drawdown: {max_drawdown:.2%}")

######

plt.figure(figsize=(12, 6))
plt.plot(results['Date'][1:], sharpe_ratio, label='Market Cap Weighted Portfolio', color='b')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Market Cap Weighted Trading Strategy')
plt.legend()
plt.xticks(rotation=90)
plt.grid()
plt.show()
