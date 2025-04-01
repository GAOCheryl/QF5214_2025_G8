from master_strategy import MASTERModel
import pickle
import numpy as np
import time
import pandas as pd
import sys, os
import qlib
from qlib.data.dataset import TSDataSampler
from bisect import bisect_right
import time
import optuna
import pandas as pd
from datetime import date
import pandas_market_calendars as mcal
from sqlalchemy import create_engine


# Add the parent directory to sys.path so that modules from one level up can be imported.
#parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
#sys.path.append(parent_dir)


# Import functions for alpha generation and sentiment extraction
from alpha_101.alpha_generator import get_alpha101_table_from_db
from alpha_101.alpha_generator import generate_alphas
from alpha_101.alpha_generator import get_updated_sentiment_table_from_db
from alpha_101.alpha_generator import get_sentiment_table_from_db


# =============================================================================
# Generate alpha factors from the stock_data table
# Returns: (df, final_df)
# =============================================================================
df, final_df = generate_alphas(input_schema = 'datacollection',
                    input_table_name = 'stock_data',
                    save = True, 
                    output_schema = 'datacollection',
                    output_table_name = 'alpha101',
                    if_return = True)


# =============================================================================
# Get sentiment data from the database
# =============================================================================
df_sentiment_updated, df_sentiment_newdata = get_updated_sentiment_table_from_db()
df_sentiment_history, df_sentiment_filter = get_sentiment_table_from_db()

# Combine updated sentiment data with history
df_sentiment_combined = pd.concat([df_sentiment_updated, df_sentiment_newdata], ignore_index=True)
df_sentiment_combined = pd.concat([df_sentiment_history, df_sentiment_combined], ignore_index=True)

# Drop duplicate sentiment rows based on Date and Ticker (history takes priority)
df_sentiment_combined = df_sentiment_combined.drop_duplicates(subset=["Date", "Ticker"], keep="first")

# sort the sentiment DataFrame by Date and Ticker
df_sentiment_combined = df_sentiment_combined.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

# Ensure the 'Date' column is in datetime format.
df_sentiment_combined['Date'] = pd.to_datetime(df_sentiment_combined['Date'])

# =============================================================================
# Create a full grid of Date and Ticker (fill missing rows with 0)
# =============================================================================
unique_dates = sorted(df_sentiment_combined['Date'].unique())
unique_tickers = sorted(df_sentiment_combined['Ticker'].unique())

# Create a complete MultiIndex with all combinations of Date and Ticker.
full_index = pd.MultiIndex.from_product([unique_dates, unique_tickers], names=['Date', 'Ticker'])

# Set the index of the sentiment DataFrame and reindex it to the full grid, filling missing rows with 0
df_sentiment_indexed = df_sentiment_combined.set_index(['Date', 'Ticker'])
df_sentiment_filled = df_sentiment_indexed.reindex(full_index, fill_value=0).reset_index()

# Remove any $ symbol from the Ticker names
df_sentiment_filled['Ticker'] = df_sentiment_filled['Ticker'].str.replace(r'\$', '', regex=True)

# =============================================================================
# Retrieve additional alpha data and index data from the database and filter by company_list
# =============================================================================
df_all, final_df_all, df_index = get_alpha101_table_from_db()

company_list = ["ADBE", "AMD", "ABNB", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", 
"ADI", "ANSS", "AAPL", "AMAT", "APP", "ASML", "AZN", 
"TEAM", "ADSK", "ADP", "AXON", "BKR", "BIIB", "BKNG", "AVGO", 
"CDNS", "CDW", "CHTR", "CTAS", "CSCO", "CCEP", "CTSH", "CMCSA", 
"CPRT", "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM","VRTX", "WBD", "WDAY", "XEL", "ZS","QCOM", 
"REGN", "ROP", "ROST", "FAST", "FTNT", "GILD" ,"ON", "PCAR", "PLTR", "PANW","PAYX", "PYPL", 
"PDD", "PEP","SBUX", "SNPS", "TTWO", "TMUS","TSLA", "TXN", "TTD", "VRSK"] 

filtered_df_all = df_all[df_all['Ticker'].isin(company_list)]
filtered_final_df_all = final_df_all[final_df_all['Ticker'].isin(company_list)]
filtered_df_sentiment = df_sentiment_filled[df_sentiment_filled['Ticker'].isin(company_list)]

# Save the filtered data into CSV files
filtered_df_all.to_csv("data/Input/updated_stock.csv", index=False)
filtered_final_df_all.to_csv("data/Input/updated_alpha.csv", index=False)
df_index.to_csv("data/Input/updated_index.csv", index=False)
filtered_df_sentiment.to_csv("data/Input/updated_sentiment.csv", index=False)

# =============================================================================
# Reload the CSVs and convert Date columns to datetime
# =============================================================================
df_all = pd.read_csv("data/Input/updated_stock.csv")
final_df_all = pd.read_csv("data/Input/updated_alpha.csv")
df_index = pd.read_csv("data/Input/updated_index.csv")
df_sentiment_filled = pd.read_csv("data/Input/updated_sentiment.csv")

df_all['Date'] = pd.to_datetime(df_all['Date'])
final_df_all['Date'] = pd.to_datetime(final_df_all['Date'])
df_index['Date'] = pd.to_datetime(df_index['Date'])
df_sentiment_filled['Date'] = pd.to_datetime(df_sentiment_filled['Date'])

# =============================================================================
# Filter DataFrames for a specific date range: from start_date until before today
# =============================================================================
start_date = pd.Timestamp('2024-07-12')
today = pd.Timestamp(date.today())

# Filter DataFrames: select rows where Date is on or after start_date and before today
filtered_df_all = df_all[(df_all['Date'] >= start_date) & (df_all['Date'] < today)]
filtered_final_df_all = final_df_all[(final_df_all['Date'] >= start_date) & (final_df_all['Date'] < today)]
filtered_df_index = df_index[(df_index['Date'] >= start_date) & (df_index['Date'] < today)]
filtered_df_sentiment = df_sentiment_filled[(df_sentiment_filled['Date'] >= start_date) & (df_sentiment_filled['Date'] < today)]


# =============================================================================
# Process index data: pivot, flatten columns, and reset the index
# =============================================================================
# 1) Drop the "IndexTicker" and "Volume" column if it exists
filtered_df_index = filtered_df_index.drop(columns="IndexTicker", errors="ignore")
filtered_df_index = filtered_df_index.drop(columns="Volume", errors="ignore")

# 2) Pivot the DataFrame
df_index_pivot = filtered_df_index.pivot(
    index="Date",                # Each Date becomes a row
    columns="IndexName",         # Each unique IndexName becomes columns
    values=["Open", "High", "Low", "Close", "Adj_Close"]
)

# 3) Flatten and rename columns to single-level format "<IndexName>_<Column>"
df_index_pivot.columns = [
    f"{col[1]}_{col[0]}"  # col is a tuple like ("Open", "S&P500")
    for col in df_index_pivot.columns
]

# 4) Move the 'Date' index back into a column
df_index_pivot.reset_index(inplace=True)

# =============================================================================
# Merge stock and alpha data
# =============================================================================
combined_df = pd.merge(filtered_df_all, filtered_final_df_all, on=["Date", "Ticker"], how="inner")

# Drop the "Intent Sentiment" column from sentiment data if it exists
filtered_df_sentiment = filtered_df_sentiment.drop(columns=["Intent Sentiment"])

# Convert remaining sentiment columns to numeric
for col in filtered_df_sentiment.columns:
    if col not in ['Date', 'Ticker']:
        filtered_df_sentiment[col] = pd.to_numeric(filtered_df_sentiment[col], errors='coerce')

# =============================================================================
# Aggregate sentiment data by trading date and ticker
# =============================================================================
trading_dates = sorted(combined_df['Date'].unique())
combined_sent_list = []
tickers = combined_df['Ticker'].unique()

# For each ticker and each trading date, average sentiment values over the period until the next trading day
for ticker in tickers:
    # Subset sentiment data for this ticker.
    df_sent_ticker = filtered_df_sentiment[filtered_df_sentiment['Ticker'] == ticker].copy()
    # Loop over each trading date.
    for i, current_date in enumerate(trading_dates):
        if i < len(trading_dates) - 1:
            next_date = trading_dates[i+1]
        else:
            next_date = None
        
        # Define the window: from current_date (inclusive) to next_date (exclusive).
        if next_date is not None:
            window = df_sent_ticker[(df_sent_ticker['Date'] >= current_date) & 
                                        (df_sent_ticker['Date'] < next_date)]
        else:
            window = df_sent_ticker[df_sent_ticker['Date'] >= current_date]
        
        # If there's sentiment data in the window, compute the average.
        if not window.empty:
            # Assume sentiment columns are all except 'Date' and 'Ticker'.
            sentiment_cols = [col for col in window.columns if col not in ['Date', 'Ticker']]
            avg_values = window[sentiment_cols].mean()
            
            # Create a new row with the current trading date, ticker, and averaged sentiment.
            row = {'Date': current_date, 'Ticker': ticker}
            row.update(avg_values.to_dict())
            combined_sent_list.append(row)


# Create a DataFrame from the combined sentiment data.
df_combined_sent = pd.DataFrame(combined_sent_list)

# Drop any rows whose dates are not in df_merged (should be none if we use trading_dates,
    # but we do this for safety).
df_combined_sent = df_combined_sent[df_combined_sent['Date'].isin(combined_df['Date'])]
    


# =============================================================================
# Merge combined sentiment data with the combined stock/alpha data, and then with index data
# =============================================================================
df_final = pd.merge(combined_df, df_combined_sent, on=['Date', 'Ticker'], how='left')
df_merged = pd.merge(
    df_final,
    df_index_pivot,
    on='Date',    # or how='left'/'right'/'outer' if needed
    how='left'    
)

# Drop non-numeric columns that may interfere with further processing
filtered_df_merged = df_merged.drop(columns=["IndClass_Sector", "IndClass_Industry"])
df_sorted = filtered_df_merged.sort_values(by='Date').reset_index(drop=True)


# =============================================================================
# Shift the Return column for prediction target: row T holds day T+1's label
# =============================================================================
df_sorted_shift = df_sorted.sort_values(["Ticker", "Date"])
df_sorted_shift["Return"] = df_sorted_shift.groupby("Ticker")["Return"].shift(-1)
df_sorted_shift.fillna(0, inplace=True)
df_sorted_shift['Date'] = pd.to_datetime(df_sorted_shift['Date'])
df_pred = df_sorted_shift.copy()

# =============================================================================
# Convert the DataFrame to Qlib format using a TSDataSampler
# =============================================================================
def convert_data_qlibformat(df):

    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # remove rows with missing "Return" values
    df.dropna(subset=['Return'], inplace=True)
    
    # 1) Ensure the "Date" column is a datetime type
    df["Date"] = pd.to_datetime(df["Date"])
    
    # 2) Separate label and features BEFORE setting the index
    #    Retain "Date" and "Ticker" for indexing, but remove them from features.
    df_label = df[["Return"]].copy()
    # Exclude "Date" and "Ticker" along with "Return" from features
    df_feature = df.drop(columns=["Return", "Date", "Ticker"], errors="ignore")
    
    # 3) Normalize feature columns (ensure they are numeric)
    df_feature = df_feature.apply(pd.to_numeric, errors='coerce')
    feature_mean = df_feature.mean()
    feature_std = df_feature.std()
    eps = 1e-8  # to avoid division by zero
    normalized_feature = (df_feature - feature_mean) / (feature_std + eps)
    df_feature = normalized_feature
    
    # 4) Now set the index of the original DataFrame using "Date" and "Ticker"
    #    These columns will be used as the index ("datetime" and "instrument")
    df_index = df.set_index(["Date", "Ticker"])
    
    # 5) Assign the index to both label and feature DataFrames
    df_feature.index = df_index.index
    df_label.index = df_index.index

    # 6) Construct MultiIndex for the columns:
    #    For features, use top-level "feature"; for the label, use "label"
    df_feature.columns = pd.MultiIndex.from_product([["feature"], df_feature.columns])
    df_label.columns = pd.MultiIndex.from_product([["label"], df_label.columns])
    
    # 7) Concatenate features and label columns, and fill missing values with 0
    df_qlib = pd.concat([df_feature, df_label], axis=1)
    df_qlib = df_qlib.fillna(0)
    
    # 8) Name the index levels and sort the DataFrame
    df_qlib.index.names = ["datetime", "instrument"]
    df_qlib = df_qlib.sort_index(level=["datetime", "instrument"])

    
    # 9) Determine start and end dates from the datetime index level
    start = df_qlib.index.get_level_values("datetime").min()
    end = df_qlib.index.get_level_values("datetime").max()
    
    # 10) Build TSDataSampler (using fillna_type='ffill+bfill' for reindexing)
    sampler = TSDataSampler(df_qlib, start, end, step_len=8, fillna_type='ffill+bfill')
    sampler.data_arr = np.nan_to_num(sampler.data_arr, nan=0.0)
    
    return sampler


dl_pred = convert_data_qlibformat(df_pred)


# Save the merged DataFrame tinpuo a pickle file
with open("data/input/updated_pred_input.pkl", "wb") as f:
    pickle.dump(dl_pred, f)

with open(f'data/input/updated_pred_input.pkl', 'rb') as f:
    dl_pred = pickle.load(f)

# =============================================================================
# Set model parameters and prepare to run predictions
# =============================================================================
d_feat = 106
d_model = 512
t_nhead = 4
s_nhead = 2
dropout = 0.7
gate_input_start_index = 106
gate_input_end_index = 121 + 9

beta = 5

n_epoch = 100
lr = 1e-4
GPU = 0
train_stop_loss_thred = 0.0007


ic = []
icir = []
ric = []
ricir = []

# =============================================================================
# Function to get the next trading day given a current date and sorted trading dates
# =============================================================================
def get_next_trading_day(current_date, trading_dates):
    """
    Given a current_date and a sorted list of trading_dates,
    returns the next trading day after current_date.
    If no next trading day exists, returns None.
    """
    pos = bisect_right(trading_dates, current_date)
    if pos < len(trading_dates):
        return trading_dates[pos]
    else:
        return None
    

# =============================================================================
# Run model predictions using MASTERModel
# =============================================================================
for seed in [0]:
    param_path = f'model/model_training_with_sentiment_0.pkl'

    print(f'Model Loaded from {param_path}')
    model = MASTERModel(
            d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
            save_path='model/Output', save_prefix=f'model_prediction'
        )
    model.load_param(param_path)
    ###### 1
    predictions, metrics, real_returns, real_prices, market_cap = model.predict(dl_pred, df_all)
    ###### 
    # predictions, metrics = model.predict(dl_test)
    print(metrics)

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])

    ####### 2
    # From Predictions to DataFrame
    df_predictions = predictions.reset_index()  # 让 MultiIndex 变成普通列
    df_predictions.columns = ["Date", "Ticker", "Predicted_Return"]  # Rename Column

    # Get a sorted list of unique trading dates.
    trading_dates = sorted(df_predictions['Date'].unique())

    # Get today's date (normalized to midnight)
    today = pd.Timestamp(date.today()).normalize()
    yesterday = (pd.Timestamp('today').normalize() - pd.Timedelta(days=1))
    trading_dates.append(today)
    trading_dates = sorted(trading_dates)

    # Now assume df_predictions has a column "Date" corresponding to the last day of the input window.
    # Create a new column "Predicted_Return_Date" which is the next trading day.
    df_predictions['Date'] = df_predictions['Date'].apply(
        lambda d: get_next_trading_day(d, trading_dates)
    )

    # Make sure real_returns, real_prices, market_cap are Series indexed by (datetime, instrument)
    df_real_returns = real_returns.rename("Actual_Return").reset_index()
    df_real_prices = real_prices.rename("Price").reset_index()
    df_market_cap = market_cap.rename("Market_Cap").reset_index()
    #df_predictions = df_predictions.reset_index()

    df_predictions = df_predictions.dropna(subset=["Date"])

    # 2) Rename columns in df_predictions so they match the merge keys.
    df_predictions.rename(columns={"Date": "datetime", "Ticker": "instrument"}, inplace=True)

    # 3) If df_predictions already has columns like "Actual_Return", "Price", "Market_Cap"
    #    (i.e. placeholders from earlier steps), remove them to avoid conflicts during merge.
    df_predictions.drop(columns=["Actual_Return", "Price", "Market_Cap"], errors="ignore", inplace=True)

    # Now df_predictions has columns: ["datetime", "instrument", "Predicted_Return", ...]

    df_merged_predictions = (
        df_predictions
        .merge(df_real_returns, on=["datetime", "instrument"], how="left")
        .merge(df_real_prices, on=["datetime", "instrument"], how="left")
        .merge(df_market_cap, on=["datetime", "instrument"], how="left")
    )



    df_merged_predictions.rename(columns={"datetime": "Date", "instrument": "Ticker"}, inplace=True)

    # Merge only the Date, Ticker, and Return columns from df_all.
    df_temp = df_all[['Date', 'Ticker', 'Return']]

    # Merge on 'Date' and 'Ticker' columns (using left join to keep all rows in df_merged_predictions)
    df_merged_predictions = df_merged_predictions.drop(columns=['Actual_Return'], errors='ignore') \
        .merge(df_temp, on=['Date', 'Ticker'], how='left')

    # Rename the merged column to 'Actual_Return'
    df_merged_predictions.rename(columns={'Return': 'Actual_Return'}, inplace=True)

    # To CSV
    csv_path = "data/Output/updated_predictions.csv"
    df_merged_predictions.to_csv(csv_path, index=False)
    ######
       

# Print summary metrics
print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))
print("\nLast 15 rows of df_merged_predictions:")
print(df_merged_predictions.tail(15))

# =============================================================================
# Allocation Update Section:
# =============================================================================
# Read the updated allocation file and merged predictions file

# PostgreSQL connection settings
db_user = "postgres"
db_password = "qf5214"
db_host = "134.122.167.14"
db_port = 5555
db_name = "QF5214"
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Define the SQL query to obtain the table from the specified schema and table name.
query = "SELECT * FROM tradingstrategy.dailytrading"

# Execute the query and load the result into a pandas DataFrame
df = pd.read_sql(query, engine)
df['Date'] = pd.to_datetime(df['Date'])

#df = pd.read_csv("data/Output/Updated_Allocation.csv")
df_merged_predictions = pd.read_csv("data/Output/updated_predictions.csv")
df_merged_predictions['Date'] = pd.to_datetime(df_merged_predictions['Date'])

# Merge df with df_merged_predictions to get the latest Price column; drop rows with missing Date
df = df.merge(
    df_merged_predictions[['Date', 'Ticker', 'Price']],
    on=['Date', 'Ticker'],
    how='left'
)
df = df.dropna(subset=['Date'])

# Prepare a DataFrame of predictions with renamed Price column (Trading_Price)
df_preds = df_merged_predictions[['Date', 'Ticker', 'Price']].rename(columns={'Price': 'Trading_Price'})
df_preds = df_preds.sort_values(['Ticker', 'Date'])

# Define a function to merge a single ticker group
def merge_group(group):
    group = group.sort_values('Date')
    ticker = group['Ticker'].iloc[0]
    preds_group = df_preds[df_preds['Ticker'] == ticker].sort_values('Date')
    # Perform merge_asof for this ticker group
    merged = pd.merge_asof(
        group,
        preds_group,
        on='Date',
        direction='forward',
        allow_exact_matches=False
    )
    return merged

# Apply the merge function groupwise
df_positions = df.groupby('Ticker', group_keys=False).apply(merge_group)

# If duplicate Ticker columns exist, keep only one
if 'Ticker_x' in df_positions.columns and 'Ticker_y' in df_positions.columns:
    df_positions['Ticker'] = df_positions['Ticker_x']
    df_positions = df_positions.drop(columns=['Ticker_x', 'Ticker_y'])

df_positions = df_positions.sort_values(['Date', 'Ticker'])
#csv_path = "data/Output/Updated_Allocation.csv"
#df_positions.to_csv(csv_path, index=False)


# =============================================================================
# 1) Compute Overall Allocation from the Penultimate Date:
# =============================================================================
# Selection parameter
TOP_N = 5  # Number of top stocks selected daily
TRADING_COST_RATE = 0.001  # 0.1% trading cost per transaction

df = df_positions.copy()
df_merged_predictions = pd.read_csv("data/Output/updated_predictions.csv")

# Convert Date columns using the appropriate format; here we use '%d/%m/%y'
df['Date'] = pd.to_datetime(df['Date'])
df_merged_predictions['Date'] = pd.to_datetime(df_merged_predictions['Date'])

# Sort dates to identify the penultimate and latest
unique_dates = df['Date'].unique()
unique_dates = np.sort(df['Date'].unique())
if len(unique_dates) < 2:
    raise ValueError("Not enough dates to perform the penultimate/ latest date logic.")

penultimate_date = unique_dates[-2]
latest_date = unique_dates[-1]

# Filter the DataFrame for the penultimate date
df_penultimate = df[df['Date'] == penultimate_date].copy()

# Calculate each position's value = old Shares × Trading_Price
df_penultimate['Position_Value'] = df_penultimate['Shares'] * df_penultimate['Trading_Price']

# Sum up to get total portfolio value
portfolio_value = df_penultimate['Position_Value'].sum()

# Subtract trading cost (if you consider closing or adjusting these positions)
overall_allocation = portfolio_value * (1 - TRADING_COST_RATE)
print("Overall allocation after penultimate date:", overall_allocation)


# =============================================================================
# 2) Update Shares for the Latest Date Using Price and Accounting for Trading Cost:
# =============================================================================
df_latest = df[df['Date'] == latest_date].copy()
df_long = df_latest[df_latest['Position_Type'] == 'Long'].copy()
df_short = df_latest[df_latest['Position_Type'] == 'Short'].copy()

# Split overall allocation equally between long and short positions
long_side_allocation = overall_allocation / 2
short_side_allocation = overall_allocation / 2

# Calculate allocated capital per stock on each side.
# Then, subtract the trading cost from that allocated capital before buying shares.
# New_Shares = (Allocation per stock * (1 - TRADING_COST_RATE)) / Price
allocation_per_stock_long = long_side_allocation * df_long['Weight']
allocation_per_stock_short = short_side_allocation * df_short['Weight']
df_long['New_Shares'] = (allocation_per_stock_long * (1 - TRADING_COST_RATE)) / df_long['Price']
df_short['New_Shares'] = (allocation_per_stock_short * (1 - TRADING_COST_RATE)) / df_short['Price']

# Combine updated positions and update Shares
df_updated_latest = pd.concat([df_long, df_short], ignore_index=True)
df_updated_latest['Shares'] = df_updated_latest['New_Shares']
df_updated_latest = df_updated_latest.drop(columns=['New_Shares'])
print("Updated positions on the latest date:")
print(df_updated_latest)

# Merge back into the main DataFrame 
df = df[df['Date'] != latest_date]
df = pd.concat([df, df_updated_latest], ignore_index=True)

# =============================================================================
# 3) Append New Positions from Latest Predictions:
# =============================================================================
# Identify latest date in df_merged_predictions and select top N and bottom N based on Predicted_Return
latest_pred_date = df_merged_predictions['Date'].max()
df_latest_pred = df_merged_predictions[df_merged_predictions['Date'] == latest_pred_date]
df_top = df_latest_pred.nlargest(TOP_N, 'Predicted_Return').copy()
df_top['Weight'] = 1 / TOP_N  # Equal weight for each long position
df_top['Position_Type'] = 'Long'
df_bottom = df_latest_pred.nsmallest(TOP_N, 'Predicted_Return').copy()
df_bottom['Weight'] = 1 / TOP_N  # Equal weight for each short position
df_bottom['Position_Type'] = 'Short'
df_new_positions = pd.concat([df_top, df_bottom], ignore_index=True)
df_new_positions = df_new_positions[['Date', 'Ticker', 'Weight', 'Position_Type']]

# --- Step 5: Append these new rows to your main DataFrame df ---
df = pd.concat([df, df_new_positions], ignore_index=True)

# Display the new rows that were added
print("New positions added based on the latest predictions:")
print(df)

# =============================================================================
# 4) Keep Only the Desired Columns in the Final DataFrame:
# =============================================================================
old_trading_updated = df[['Date', 'Ticker', 'Position_Type', 'Shares', 'Weight']]

# =============================================================================
# (Optional) Check if today is a trading day and update CSV file accordingly:
# =============================================================================
# Assume old_trading_updated is already defined and contains a 'Date' column.
# Ensure 'Date' column is in datetime format
# Define the SQL query to obtain the table from the specified schema and table name.
query = "SELECT * FROM tradingstrategy.dailytrading"
old_trading = pd.read_sql(query, engine)
old_trading['Date'] = pd.to_datetime(old_trading['Date'])

# Get today's date normalized to midnight
today = pd.Timestamp(date.today()).normalize()

# Check if today is a trading day
trading = False
if mcal is not None:
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=today, end_date=today)
    trading = not schedule.empty
else:
    # Fallback: assume trading day if weekday (Monday=0, Friday=4)
    trading = today.dayofweek < 5

# Get the latest date in the table
latest_date_in_table = old_trading['Date'].max()

# Condition: today is a trading day AND latest date in table is smaller than today
if trading and latest_date_in_table < today:
    csv_path = "data/Output/Updated_Allocation.csv"
    old_trading_updated.to_csv(csv_path, index=False)
    print(f"Today ({today.date()}) is a trading day and latest date in table ({latest_date_in_table.date()}) is before today.")
    print(f"Saved updated_predictions.csv to {csv_path}.")
else:
    print(f"Today ({today.date()}) is not a trading day or the table is already up-to-date (latest date: {latest_date_in_table.date()}). CSV not saved.")


# =============================================================================
# Store the final DataFrame into PostgreSQL:
# =============================================================================
# PostgreSQL connection settings
db_user = "postgres"
db_password = "qf5214"
db_host = "134.122.167.14"
db_port = 5555
db_name = "QF5214"
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
''''
file_name = "data/Output/Updated_Allocation.csv"

# Check if the file exists in the current folder
if os.path.exists(file_name):
    print(f"File '{file_name}' found in the current directory.")
    # Load and display the first few rows of the CSV file
    df = pd.read_csv(file_name)
    print(df.head())
else:
    print(f"File '{file_name}' not found in the current directory: {os.getcwd()}")
'''
# Define target table and schema
table_name = "dailytrading"
schema = "tradingstrategy"

# Insert DataFrame into PostgreSQL table using "replace"
try:
    old_trading_updated.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",  # Overwrite the table if it already exists
        index=False,
        schema=schema
    )
    print(f"Data successfully stored in {schema}.{table_name} (table replaced if it existed).")
except Exception as e:
    print(f"Error storing data: {e}")