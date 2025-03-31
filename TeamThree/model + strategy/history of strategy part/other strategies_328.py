import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Copy the dataset to avoid modifying the original data
# df = df_predictions.copy()
df = pd.read_csv("predictions.csv")


# Strategy 1 equal weight strategy
# Selection parameter
TOP_N = 5  # Number of top stocks selected daily
INITIAL_CAPITAL = 1_000_000  # Initial portfolio value in cash
TRADING_COST_RATE = 0.001  # 0.1% trading cost per transaction

def TopN_EqualWeight(df):
    df = df.sort_values(by=['Date'])
    unique_dates = df['Date'].unique()
    portfolio_value = INITIAL_CAPITAL
    portfolio_returns = []
    portfolio_values = []
    prev_positions = {}

    positions_list = []  

    for date in unique_dates:
        daily_data = df[df['Date'] == date]
        top_stocks = daily_data.nlargest(TOP_N, 'Predicted_Return')
        tickers = top_stocks['Ticker'].values
        prices = top_stocks['Price'].values
        
        # Equal weight allocation
        weights = np.ones(TOP_N) / TOP_N  # Equal allocation
        allocation = portfolio_value * weights
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
        portfolio_value -= trading_cost
        prev_positions = new_positions
        
        portfolio_returns.append(portfolio_return)
        portfolio_values.append(portfolio_value)

        for ticker, alloc in zip(tickers, allocation):
            weight = np.ones(TOP_N) / TOP_N # alloc / portfolio_value  
            positions_list.append({'Date': date, 'Ticker': ticker, 'Allocation': alloc, 'Weight': weight})

    positions_df = pd.DataFrame(positions_list)
    
    # store required data
    positions_df.to_csv('TopN_EqualWeight_position.csv', index=False)

    # backtest result
    results = pd.DataFrame({'Date': unique_dates, 'Portfolio_Value': portfolio_values, 'Portfolio_Return': portfolio_returns})
    results['Cumulative_Return'] = results['Portfolio_Value'] / INITIAL_CAPITAL
    return results


results = TopN_EqualWeight(df)

plt.figure(figsize=(12, 6))
plt.plot(results['Date'], results['Cumulative_Return'], label='Equal Weight Portfolio', color='b')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Equal Weight Trading Strategy')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()


# Compute performance metrics
risk_free_rate = 0.04 / 252
epsilon = 1e-8
portfolio_daily_returns = results['Cumulative_Return'].pct_change().dropna()
excess_returns = portfolio_daily_returns - risk_free_rate
sharpe_ratio = (excess_returns.mean() / (excess_returns.std() + epsilon)) * np.sqrt(252)

max_drawdown = (results['Cumulative_Return'] / results['Cumulative_Return'].cummax() - 1).min()

print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Max Drawdown: {max_drawdown:.2%}")


### Strategy 2 - Market Cap Weighted Trading Strategy
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

    positions_list = []  

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
        
        total_allocation = np.sum(allocation)
        for ticker, alloc in zip(tickers, allocation):
            weight = weight = alloc / total_allocation # alloc / portfolio_value  
            positions_list.append({'Date': date, 'Ticker': ticker, 'Allocation': alloc, 'Weight': weight})

        positions_df = pd.DataFrame(positions_list)
    
    # store required data
    positions_df.to_csv('TopN_marketcap_position.csv', index=False)
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


# Strategy 3 long short strategy - equal weight

# Selection parameter
TOP_N = 5  # Number of top stocks selected daily for long and short
INITIAL_CAPITAL = 1_000_000  # Initial portfolio value
TRADING_COST_RATE = 0.001  # 0.1% trading cost per transaction

def TopN_LongShort(df):
    df = df.sort_values(by=['Date'])
    unique_dates = df['Date'].unique()
    portfolio_returns = []
    portfolio_values = []
    prev_positions = {}
    cumulative_return = 1  # Start at 1 for cumulative return calculation
    position_list = []

    for date in unique_dates:
        daily_data = df[df['Date'] == date]
        top_stocks = daily_data.nlargest(TOP_N, 'Predicted_Return')
        bottom_stocks = daily_data.nsmallest(TOP_N, 'Predicted_Return')
        
        long_tickers = top_stocks['Ticker'].values
        short_tickers = bottom_stocks['Ticker'].values
        long_prices = top_stocks['Price'].values
        short_prices = bottom_stocks['Price'].values
        
        # Equal weight allocation for long and short
        long_weights = np.ones(TOP_N) / TOP_N
        short_weights = np.ones(TOP_N) / TOP_N
        long_allocation = (INITIAL_CAPITAL*cumulative_return / 2) * long_weights  # 50% capital to long
        short_allocation = (INITIAL_CAPITAL*cumulative_return / 2) * short_weights  # 50% capital to short
        
        long_positions = {ticker: alloc / price for ticker, alloc, price in zip(long_tickers, long_allocation, long_prices)}
        short_positions = {ticker: alloc / price for ticker, alloc, price in zip(short_tickers, short_allocation, short_prices)}
        
        # Compute trading cost (including closing positions)
        trading_cost = 0
        for ticker, new_shares in {**long_positions, **short_positions}.items():
            prev_shares = prev_positions.get(ticker, 0)
            price = df[(df['Date'] == date) & (df['Ticker'] == ticker)]['Price'].values
            if len(price) > 0:
                trading_cost += abs(new_shares - prev_shares) * price[0] * TRADING_COST_RATE
        
        for ticker, prev_shares in prev_positions.items():
            if ticker not in long_positions and ticker not in short_positions:
                last_price = df[(df['Date'] == date) & (df['Ticker'] == ticker)]['Price'].values
                if len(last_price) > 0:
                    trading_cost += abs(prev_shares) * last_price[0] * TRADING_COST_RATE
        
        # Compute portfolio return
        long_returns = top_stocks['Actual_Return'].values
        short_returns = bottom_stocks['Actual_Return'].values
        portfolio_return = 0.5 * np.sum(long_weights * long_returns) - 0.5 * np.sum(short_weights * short_returns)

        # Update cumulative return (starting from 1)
        cumulative_return *= (1 + portfolio_return)

        # Deduct trading cost in percentage terms (relative to initial capital)
        cumulative_return *= (1 - trading_cost / INITIAL_CAPITAL)

        portfolio_returns.append(portfolio_return)
        portfolio_values.append(cumulative_return)
                
        # Store position records
        for ticker, weight, position in zip(long_tickers, long_weights, long_positions.values()):
            position_list.append([date, ticker, 'Long', position, weight])
        for ticker, weight, position in zip(short_tickers, short_weights, short_positions.values()):
            position_list.append([date, ticker, 'Short', position, weight])
        
        # Update previous positions
        prev_positions = {**long_positions, **short_positions}
    
    results = pd.DataFrame({'Date': unique_dates, 'Cumulative_Return': portfolio_values, 'Portfolio_Return': portfolio_returns})
    position_df = pd.DataFrame(position_list, columns=['Date', 'Ticker', 'Position_Type', 'Shares', 'Weight'])
    
    # Save results to CSV
    # results.to_csv(output_csv, index=False)
    position_df.to_csv('TopN_LongShort_position.csv', index=False)
    
    return results


# Plot cumulative return
plt.figure(figsize=(12, 6))
plt.plot(results['Date'], results['Cumulative_Return'], label='Long-Short Portfolio', color='b')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Long-Short Trading Strategy')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Compute performance metrics
risk_free_rate = 0.04 / 252
epsilon = 1e-8
portfolio_daily_returns = results['Portfolio_Return']
excess_returns = portfolio_daily_returns - risk_free_rate
sharpe_ratio = (excess_returns.mean() / (excess_returns.std() + epsilon)) * np.sqrt(252)

max_drawdown = (results['Cumulative_Return'] / results['Cumulative_Return'].cummax() - 1).min()

print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Selection parameter
TOP_N = 5  # Number of top stocks selected daily for long and short
INITIAL_CAPITAL = 1_000_000  # Initial portfolio value
TRADING_COST_RATE = 0.001  # 0.1% trading cost per transaction

def TopN_LongShort_Marketcap(df):
    df = df.sort_values(by=['Date'])
    unique_dates = df['Date'].unique()
    portfolio_returns = []
    portfolio_values = []
    prev_positions = {}
    cumulative_return = 1  # Start at 1 for cumulative return calculation
    position_list = []
    
    for date in unique_dates:
        daily_data = df[df['Date'] == date]

        # Select top & bottom N stocks by predicted return
        top_stocks = daily_data.nlargest(TOP_N, 'Predicted_Return')
        bottom_stocks = daily_data.nsmallest(TOP_N, 'Predicted_Return')

        # Extract tickers, prices, and market caps
        long_tickers = top_stocks['Ticker'].values
        short_tickers = bottom_stocks['Ticker'].values
        long_prices = top_stocks['Price'].values
        short_prices = bottom_stocks['Price'].values
        long_market_caps = top_stocks['Market_Cap'].values
        short_market_caps = bottom_stocks['Market_Cap'].values

        # Compute market cap weights
        long_weights = long_market_caps / np.sum(long_market_caps)
        short_weights = short_market_caps / np.sum(short_market_caps)

        # Capital allocation (50% long, 50% short)
        long_allocation = (INITIAL_CAPITAL*cumulative_return / 2) * long_weights  # 50% capital to long
        short_allocation = (INITIAL_CAPITAL*cumulative_return / 2) * short_weights  # 50% capital to short

        # Compute positions
        long_positions = {ticker: alloc / price for ticker, alloc, price in zip(long_tickers, long_allocation, long_prices)}
        short_positions = {ticker: alloc / price for ticker, alloc, price in zip(short_tickers, short_allocation, short_prices)}

        # Compute trading cost
        trading_cost = 0
        for ticker, new_shares in {**long_positions, **short_positions}.items():
            prev_shares = prev_positions.get(ticker, 0)
            price = df[(df['Date'] == date) & (df['Ticker'] == ticker)]['Price'].values
            if len(price) > 0:
                trading_cost += abs(new_shares - prev_shares) * price[0] * TRADING_COST_RATE

        # Closing positions cost
        for ticker, prev_shares in prev_positions.items():
            if ticker not in long_positions and ticker not in short_positions:
                last_price = df[(df['Date'] == date) & (df['Ticker'] == ticker)]['Price'].values
                if len(last_price) > 0:
                    trading_cost += abs(prev_shares) * last_price[0] * TRADING_COST_RATE

        # Compute portfolio return (market cap weighted)
        long_returns = top_stocks['Actual_Return'].values
        short_returns = bottom_stocks['Actual_Return'].values
        portfolio_return = 0.5 * np.sum(long_weights * long_returns) - 0.5 * np.sum(short_weights * short_returns)

        
        # Update cumulative return
        cumulative_return *= (1 + portfolio_return)

        # Deduct trading cost relative to current portfolio value
        cumulative_return *= (1 - trading_cost / INITIAL_CAPITAL) 

        portfolio_returns.append(portfolio_return)
        portfolio_values.append(cumulative_return)

        # Store position records
        for ticker, weight, position in zip(long_tickers, long_weights, long_positions.values()):
            position_list.append([date, ticker, 'Long', position, weight])
        for ticker, weight, position in zip(short_tickers, short_weights, short_positions.values()):
            position_list.append([date, ticker, 'Short', position, weight])
            
        # Update previous positions
        prev_positions = {**long_positions, **short_positions}

    results = pd.DataFrame({'Date': unique_dates, 'Cumulative_Return': portfolio_values, 'Portfolio_Return': portfolio_returns})
    position_df = pd.DataFrame(position_list, columns=['Date', 'Ticker', 'Position_Type', 'Shares', 'Weight'])
    
    # Save results to CSV
    # results.to_csv(output_csv, index=False)
    position_df.to_csv('TopN_LongShort_Marketcap_position.csv', index=False)
    
    return results



results = TopN_LongShort_Marketcap(df)

# Plot cumulative return
plt.figure(figsize=(12, 6))
plt.plot(results['Date'], results['Cumulative_Return'], label='Long-Short Portfolio', color='b')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Market Cap Weighted Long-Short Strategy')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Compute performance metrics
risk_free_rate = 0.04 / 252
epsilon = 1e-8
portfolio_daily_returns = results['Portfolio_Return']
excess_returns = portfolio_daily_returns - risk_free_rate
sharpe_ratio = (excess_returns.mean() / (excess_returns.std() + epsilon)) * np.sqrt(252)
max_drawdown = (results['Cumulative_Return'] / results['Cumulative_Return'].cummax() - 1).min()

print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Max Drawdown: {max_drawdown:.2%}")
