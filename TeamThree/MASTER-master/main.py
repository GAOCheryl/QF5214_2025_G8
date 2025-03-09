
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

# generate alphas, and store in db
df, final_df = generate_alphas(input_schema = 'datacollection',
                    input_table_name = 'stock_data',
                    save = True, 
                    output_schema = 'datacollection',
                    output_table_name = 'alpha101',
                    if_return = True)

# directly get data from db
df_all, final_df_all = get_alpha101_table_from_db()
    

#universe = 'csi300' # ['csi300','csi800']
#prefix = 'opensource' # ['original','opensource'], which training data are you using
#train_data_dir = f'data'
#with open(f'{train_data_dir}\{prefix}\{universe}_dl_train.pkl', 'rb') as f:
    #dl_train = pickle.load(f)
#predict_data_dir = f'data\opensource'
#with open(f'{predict_data_dir}\{universe}_dl_valid.pkl', 'rb') as f:
    #dl_valid = pickle.load(f)
#with open(f'{predict_data_dir}\{universe}_dl_test.pkl', 'rb') as f:
    #dl_test = pickle.load(f)
    

# Merge on "Date" and "Ticker" (adjust join type if needed)
combined_df = pd.merge(df_all, final_df_all, on=["Date", "Ticker"], how="inner")

# Example of dropping non-numeric columns:
combined_df = combined_df.drop(columns=["IndClass_Sector", "IndClass_Industry"])

df_sorted = combined_df.sort_values(by='Date').reset_index(drop=True)

# Compute indices for splits
n = len(df_sorted)
train_end = int(n * 0.6)  # 60%
valid_end = int(n * 0.8)  # 80%

# Slice the DataFrame
df_train = df_sorted.iloc[:train_end]
df_valid = df_sorted.iloc[train_end:valid_end]
df_test = df_sorted.iloc[valid_end:]

def convert_data_qlibformat(df):
    # 1) Ensure the "Date" column is a datetime type
    df["Date"] = pd.to_datetime(df["Date"])

    # 2) Set the index as [Date, Ticker] and rename to ["datetime", "instrument"]
    df = df.set_index(["Date", "Ticker"])
    df.index.names = ["datetime", "instrument"]

    # 3) Ensure that the "datetime" level is properly converted and sort the index
    dt_level = pd.to_datetime(df.index.get_level_values("datetime"))
    instrument_level = df.index.get_level_values("instrument")
    df.index = pd.MultiIndex.from_arrays(
        [dt_level, instrument_level],
        names=["datetime", "instrument"]
    )
    # Sort so that datetime is at position 0 and ordered ascendingly
    df = df.sort_index(level=["datetime", "instrument"])

    # 4) Separate the "Return" column as the label
    df_label = df[["Return"]].copy()
    df_label.columns = pd.MultiIndex.from_product([["label"], ["Return"]])

    # 5) The remaining columns are features
    df_feature = df.drop(columns=["Return"], errors="ignore")
    df_feature.columns = pd.MultiIndex.from_product([["feature"], df_feature.columns])

    # 6) Concatenate features and label columns
    df_qlib = pd.concat([df_feature, df_label], axis=1)

    # 7) Determine start and end dates from the datetime level
    start = df_qlib.index.get_level_values("datetime").min()
    end = df_qlib.index.get_level_values("datetime").max()

    # 8) Build TSDataSampler (note: no 'freq' parameter)
    sampler = TSDataSampler(df_qlib, start, end, step_len=8)
    
    return sampler

dl_train = convert_data_qlibformat(df_train)
dl_valid = convert_data_qlibformat(df_valid)
dl_test = convert_data_qlibformat(df_test)
print("Data Loaded.")


d_feat = 158
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 158
gate_input_end_index = 221

beta = 5

#if universe == 'csi300':
    #beta = 5
#elif universe == 'csi800':
    #beta = 2

n_epoch = 1
lr = 1e-5
GPU = 0
train_stop_loss_thred = 0.95


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
    param_path = f'model\model_training_{seed}.pkl'

    print(f'Model Loaded from {param_path}')
    model = MASTERModel(
            d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
            save_path='model/', save_prefix=f'model'
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