#from master import MASTERModel
from master_strategy import MASTERModel
import pickle
import numpy as np
import time
import pandas as pd
import sys, os
import qlib
from qlib.data.dataset import TSDataSampler
from bisect import bisect_right

# Move up one directory from MASTER-master
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)


from alpha_101.alpha_generator import get_alpha101_table_from_db
from alpha_101.alpha_generator import generate_alphas
from alpha_101.alpha_generator import get_sentiment_table_from_db

'''
df_sentiment, df_sentiment_filter = get_sentiment_table_from_db()

# Ensure the 'Date' column is in datetime format.
df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])

# Get the sorted unique dates and tickers from df_sentiment.
unique_dates = sorted(df_sentiment['Date'].unique())
unique_tickers = sorted(df_sentiment['Ticker'].unique())

# Create a complete MultiIndex with all combinations of Date and Ticker.
full_index = pd.MultiIndex.from_product([unique_dates, unique_tickers], names=['Date', 'Ticker'])

# Set the DataFrame index to ['Date', 'Ticker'].
df_sentiment_indexed = df_sentiment.set_index(['Date', 'Ticker'])

# Reindex to the full index, filling missing rows with 0.
df_sentiment_filled = df_sentiment_indexed.reindex(full_index, fill_value=0).reset_index()

df_sentiment_filled['Ticker'] = df_sentiment_filled['Ticker'].str.replace(r'\$', '', regex=True)

# Call generate_alphas() which returns (df, final_df)
df, final_df = generate_alphas(input_schema = 'datacollection',
                    input_table_name = 'stock_data',
                    save = True, 
                    output_schema = 'datacollection',
                    output_table_name = 'alpha101',
                    if_return = True)
'''

df_all, final_df_all, df_index = get_alpha101_table_from_db()

company_list = company_list = ["ADBE", "AMD", "ABNB", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", 
"ADI", "ANSS", "AAPL", "AMAT", "APP", "ASML", "AZN", 
"TEAM", "ADSK", "ADP", "AXON", "BKR", "BIIB", "BKNG", "AVGO", 
"CDNS", "CDW", "CHTR", "CTAS", "CSCO", "CCEP", "CTSH", "CMCSA", 
"CPRT", "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM","VRTX", "WBD", "WDAY", "XEL", "ZS","QCOM", 
"REGN", "ROP", "ROST", "FAST", "FTNT", "GILD" ,"ON", "PCAR", "PLTR", "PANW","PAYX", "PYPL", 
"PDD", "PEP","SBUX", "SNPS", "TTWO", "TMUS","TSLA", "TXN", "TTD", "VRSK"] 

# Ensure the Date column is in datetime format.
df_all['Date'] = pd.to_datetime(df_all['Date'])
final_df_all['Date'] = pd.to_datetime(final_df_all['Date'])
df_index['Date'] = pd.to_datetime(df_index['Date'])
#df_sentiment_filled['Date'] = pd.to_datetime(df_sentiment_filled['Date'])

# Filter the DataFrame: select rows where Date is on or before 2025-02-28.
filtered_df_all = df_all[df_all['Date'] <= pd.Timestamp('2025-02-28')]
filtered_final_df_all = final_df_all[final_df_all['Date'] <= pd.Timestamp('2025-02-28')]
filtered_df_index = df_index[df_index['Date'] <= pd.Timestamp('2025-02-28')]
#filtered_df_sentiment = df_sentiment_filled[df_sentiment_filled['Date'] <= pd.Timestamp('2025-02-28')]

filtered_df_all = filtered_df_all[filtered_df_all['Ticker'].isin(company_list)]
filtered_final_df_all = filtered_final_df_all[filtered_final_df_all['Ticker'].isin(company_list)]
#filtered_df_sentiment = filtered_df_sentiment[filtered_df_sentiment['Ticker'].isin(company_list)]

filtered_df_all.to_csv("df_all.csv", index=False)
filtered_final_df_all.to_csv("final_df_all.csv", index=False)
filtered_df_index.to_csv("df_index.csv", index=False)
#filtered_df_sentiment.to_csv("filtered_df_sentiment.csv", index=False)

filtered_df_all = pd.read_csv("df_all.csv")
filtered_final_df_all = pd.read_csv("final_df_all.csv")
filtered_df_index = pd.read_csv("df_index.csv")
#filtered_df_sentiment = pd.read_csv("filtered_df_sentiment.csv")

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

# Now df_index_pivot has columns like:
# ['Date', 'S&P500_Open', 'S&P500_High', 'S&P500_Low', 'S&P500_Close', 'S&P500_Adj_Close', 'S&P500_Volume', ...]

# Merge on "Date" and "Ticker" (adjust join type if needed)
combined_df = pd.merge(filtered_df_all, filtered_final_df_all, on=["Date", "Ticker"], how="inner")

# 5) (Optional) Merge with your main DataFrame on 'Date'
df_merged = pd.merge(
    combined_df,
    df_index_pivot,
    on='Date',    # or how='left'/'right'/'outer' if needed
    how='left'    
)


df_merged.fillna(0, inplace=True)

# Example of dropping non-numeric columns:
filtered_df_merged = df_merged.drop(columns=["IndClass_Sector", "IndClass_Industry"])
'''
#filtered_df_sentiment = filtered_df_sentiment.drop(columns=["Intent Sentiment"])

for col in filtered_df_sentiment.columns:
    if col not in ['Date', 'Ticker']:
        filtered_df_sentiment[col] = pd.to_numeric(filtered_df_sentiment[col], errors='coerce')

# Get the sorted unique trading dates from df_merged.
trading_dates = sorted(df_merged['Date'].unique())
    
combined_sent_list = []

# Process each ticker separately.
tickers = filtered_df_merged['Ticker'].unique()
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
df_combined_sent = df_combined_sent[df_combined_sent['Date'].isin(filtered_df_merged['Date'])]
    
    

# Merge df_merged with the combined sentiment data.
# Use a left merge so that all trading dates and tickers in df_merged are preserved.
df_final = pd.merge(filtered_df_merged, df_combined_sent, on=['Date', 'Ticker'], how='left')
'''
df_final = filtered_df_merged.copy()
df_sorted = df_final.sort_values(by='Date').reset_index(drop=True)

df_sorted_shift = df_sorted.sort_values(["Ticker", "Date"])

# 2) Shift the Return column so that row T holds day T+1's label
df_sorted_shift["Return"] = df_sorted_shift.groupby("Ticker")["Return"].shift(-1)

# 3) Drop rows where Return became NaN after shifting
df_sorted_shift.dropna(subset=["Return"], inplace=True)

# Ensure the Date column is datetime type.
df_sorted_shift['Date'] = pd.to_datetime(df_sorted_shift['Date'])

# Get the sorted unique dates.
unique_dates = np.sort(df_sorted_shift['Date'].unique())

# Define split indices based on unique dates.
n_dates = len(unique_dates)
train_date_end = unique_dates[int(n_dates * 0.6)]
valid_date_end = unique_dates[int(n_dates * 0.8)]

# Now split the DataFrame based on the date thresholds.
df_train = df_sorted_shift[df_sorted_shift['Date'] <= train_date_end].copy()
df_valid = df_sorted_shift[(df_sorted_shift['Date'] > train_date_end) & (df_sorted_shift['Date'] <= valid_date_end)].copy()
df_test  = df_sorted_shift[(df_sorted_shift['Date'] > valid_date_end)].copy()



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

# Use the function for training, validation, and test sets:
dl_train = convert_data_qlibformat(df_train)
dl_valid = convert_data_qlibformat(df_valid)
dl_test = convert_data_qlibformat(df_test)


# Save the merged DataFrame tinpuo a pickle file
with open("training_input_without_sentiment.pkl", "wb") as f:
    pickle.dump(dl_train, f)
with open("valid_input_without_sentiment.pkl", "wb") as f:
    pickle.dump(dl_valid, f)
with open("testing_input_without_sentiment.pkl", "wb") as f:
    pickle.dump(dl_test, f)


with open(f'training_input_without_sentiment.pkl', 'rb') as f:
    dl_train = pickle.load(f)
with open(f'valid_input_without_sentiment.pkl', 'rb') as f:
    dl_valid = pickle.load(f)
with open(f'testing_input_without_sentiment.pkl', 'rb') as f:
    dl_test = pickle.load(f)

d_feat = 9
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 9
gate_input_end_index = 121 

beta = 5

n_epoch = 100
lr = 1e-4
GPU = 0
train_stop_loss_thred = 0.0007


ic = []
icir = []
ric = []
ricir = []

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
    
##Training
######################################################################################
for seed in [0]: # , 1, 2, 3, 4
    model = MASTERModel(
        d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
        save_path='model', save_prefix=f'model_training'
    )

    start = time.time()
    # Train
    model.fit(dl_train, df_all, dl_valid)

    print("Model Trained.")

    # Test
    # predictions, metrics = model.predict(dl_test)
    ###### 0
    print("Start Prediction")
    predictions, metrics, real_returns, real_prices, market_cap = model.predict(dl_test, df_all)
    ######       
    running_time = time.time()-start
    
    print('Seed: {:d} time cost : {:.2f} sec'.format(seed, running_time))
    print(metrics)

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])

    
######################################################################################


# Load and Test
######################################################################################
for seed in [0]:
    param_path = f'model/model_training_{seed}.pkl'

    print(f'Model Loaded from {param_path}')
    model = MASTERModel(
            d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
            save_path='model', save_prefix=f'model_prediction'
        )
    model.load_param(param_path)
    ###### 1
    predictions, metrics, real_returns, real_prices, market_cap = model.predict(dl_test, df_all)
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

    # 4) Merge on ["datetime", "instrument"] with how="inner" so we only keep rows that exist in both.
    df_merged_predictions = (
        df_predictions
        .merge(df_real_returns, on=["datetime", "instrument"], how="inner")
        .merge(df_real_prices, on=["datetime", "instrument"], how="inner")
        .merge(df_market_cap, on=["datetime", "instrument"], how="inner")
    )
    df_merged_predictions.rename(columns={"datetime": "Date", "instrument": "Ticker"}, inplace=True)

    '''
    df_real_returns = real_returns.reset_index(drop=True)
    df_real_prices = real_prices.reset_index(drop=True)
    df_market_cap = market_cap.reset_index(drop=True)

    # Merge the sheet to the right side.
    df_predictions["Actual_Return"] = df_real_returns
    df_predictions["Price"] = df_real_prices
    df_predictions["Market_Cap"] = df_market_cap
    '''

    # To CSV
    csv_path = "predictions_without_sentiment.csv"
    df_merged_predictions.to_csv(csv_path, index=False)
    ######
       
######################################################################################

print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))


###### 3
print(df_merged_predictions.head())

