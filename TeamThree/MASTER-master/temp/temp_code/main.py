
from master import MASTERModel
import pickle
import numpy as np
import time
import pandas as pd
import sys, os
import qlib
from qlib.data.dataset import TSDataSampler

# Move up one directory from MASTER-master
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
from alpha_101.alpha_generator import get_alpha101_table_from_db
from alpha_101.alpha_generator import generate_alphas

# Call generate_alphas() which returns (df, final_df)
'''
df, final_df = generate_alphas(input_schema = 'datacollection',
                    input_table_name = 'stock_data',
                    save = True, 
                    output_schema = 'datacollection',
                    output_table_name = 'alpha101',
                    if_return = True)
'''

# directly get data from db
df_all, final_df_all, df_index = get_alpha101_table_from_db()

# 1) Drop the "IndexTicker" column if it exists
df_index = df_index.drop(columns="IndexTicker", errors="ignore")

# 2) Pivot the DataFrame
df_index_pivot = df_index.pivot(
    index="Date",                # Each Date becomes a row
    columns="IndexName",         # Each unique IndexName becomes columns
    values=["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
)

# 3) Flatten and rename columns to single-level format "<IndexName>_<Column>"
df_index_pivot.columns = [
    f"{col[1]}_{col[0]}"  # col is a tuple like ("Open", "S&P500")
    for col in df_index_pivot.columns
]

df_index_pivot = df_index_pivot.drop(columns=["Russell 1000_Volume", 
                                              "Russell 3000_Volume", 
                                              "Wilshire 5000_Volume"])


# 4) Move the 'Date' index back into a column
df_index_pivot.reset_index(inplace=True)

# Now df_index_pivot has columns like:
# ['Date', 'S&P500_Open', 'S&P500_High', 'S&P500_Low', 'S&P500_Close', 'S&P500_Adj_Close', 'S&P500_Volume', ...]


# Merge on "Date" and "Ticker" (adjust join type if needed)
combined_df = pd.merge(df_all, final_df_all, on=["Date", "Ticker"], how="inner")   

# 5) (Optional) Merge with your main DataFrame on 'Date'
df_merged = pd.merge(
    combined_df,
    df_index_pivot,
    on='Date',    # or how='left'/'right'/'outer' if needed
    how='left'
)

df_merged.fillna(0, inplace=True)
# Example of dropping non-numeric columns:
df_merged = df_merged.drop(columns=["IndClass_Sector", "IndClass_Industry"])

df_sorted = df_merged.sort_values(by='Date').reset_index(drop=True)



# Compute indices for splits
n = len(df_sorted)
train_end = int(n * 0.6)  # 60%
valid_end = int(n * 0.8)  # 80%

# Slice the DataFrame
df_train = df_sorted.iloc[:train_end]
df_valid = df_sorted.iloc[train_end:valid_end]
df_test = df_sorted.iloc[valid_end:]
    

def convert_data_qlibformat(df):
    import pandas as pd
    import numpy as np
    from qlib.data.dataset import TSDataSampler  # adjust import as needed

    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
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

print("Data Loaded.")

# Save the merged DataFrame tinpuo a pickle file
with open("training_input.pkl", "wb") as f:
    pickle.dump(dl_train, f)
with open("valid_input.pkl", "wb") as f:
    pickle.dump(dl_valid, f)
with open("testing_input.pkl", "wb") as f:
    pickle.dump(dl_test, f)


with open(f'training_input.pkl', 'rb') as f:
    dl_train = pickle.load(f)
with open(f'valid_input.pkl', 'rb') as f:
    dl_valid = pickle.load(f)
with open(f'testing_input.pkl', 'rb') as f:
    dl_test = pickle.load(f)




d_feat = 9
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 9
gate_input_end_index = 123

beta = 5

n_epoch = 100
lr = 1e-4
GPU = 0
train_stop_loss_thred = 0.001


ic = []
icir = []
ric = []
ricir = []


##Training
######################################################################################
for seed in [0, 1, 2, 3, 4]:
    model = MASTERModel(
        d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
        save_path='model', save_prefix=f'model_training'
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
    predictions, metrics = model.predict(dl_test)
    print(metrics)

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])
    
######################################################################################

print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))