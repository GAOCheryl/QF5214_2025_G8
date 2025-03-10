# generate WorldQuant Alpha101 
import pandas as pd
from .utils import Alphas

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from database_utils import *

def data_preprocessing(df):
    df['Date'] = pd.to_datetime(df['Date'],utc=True)
    # Compute Typical Price
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Return'] = df.groupby('Ticker')['Close'].pct_change()
    df['VWAP_5'] = df.groupby('Ticker', group_keys=False).apply(
    lambda x: (x['Typical_Price'] * x['Volume']).rolling(window=5).sum() / x['Volume'].rolling(window=5).sum()
    ) # 5-period VWAP
    return df

def run_by_ticker(df, tickers, alpha_indices):
    final_df = pd.DataFrame()
    # Iterate over each stock
    for tik in tickers:
        df_tik = df[df['Ticker'] == tik]
        stock = Alphas(df_tik)

        result_df = pd.DataFrame()
        result_df['Date'] = df_tik['Date']
        
        for idx in alpha_indices:
            func_name = f"alpha{idx:03d}"  # Format as 'alpha001', 'alpha002', etc.
            print(f"Running function {func_name} for {tik}...")
            
            if hasattr(stock, func_name):
                func = getattr(stock, func_name)
                result_df[func_name] = func()
            else:
                print(f"Function {func_name} not found in Alphas class.")

        result_df['Ticker'] = tik
        final_df = pd.concat([final_df, result_df], ignore_index=True)
    return final_df
    
def generate_alphas(input_schema = 'datacollection',
                    input_table_name = 'stock_data',
                    save = False, 
                    output_schema = 'datacollection',
                    output_table_name = 'alpha101',
                    if_return = False):
    alpha_indices = [
        1, 2, 3, 4, 6, 7, 8, 9, 
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
        20, 21, 22, 23, 24, 25, 26, 28, 29, 
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
        50, 51, 52, 53, 54, 55, 56, 57, 58, 
        60, 61, 62, 63, 64, 65, 66,  
        71, 72, 73, 74, 75, 77, 78, 
        81, 83, 84, 85, 88, 
        92, 94, 96, 98, 99, 101
    ]
    
    # alpha_lst1 = [5, 27, 68, 86, 95] # potential error
    # alpha_dict = {'alpha019': 250, 'alpha039': 250,'alpha048': 250,} # with lots of NULL values
    # new_alpha_lst = [48, 56, 58, 59, 63]

    db = database_utils()
    db.connect()
    db.set_schema(input_schema)
    db.execute_query(f'''
        SELECT "Date", "Open", "High", "Low", "Close", 
               "Volume", "Market_Cap", "Ticker", 
               "IndClass_Sector", "IndClass_Industry" 
        FROM {input_table_name}
    ''')
    df = pd.DataFrame(db.fetch_results(), 
                      columns=["Date", "Open", "High", "Low", "Close", 
                               "Volume", "Market_Cap", "Ticker", 
                               "IndClass_Sector", "IndClass_Industry"])
    
    df = data_preprocessing(df)
    tickers = df['Ticker'].unique()
    final_df = run_by_ticker(df, tickers, alpha_indices)
    
    if save:
        new_columns = final_df.columns.to_list()
        db.df_to_sql_table(final_df, ['object', 'string'] + ['float64'] * (len(new_columns) - 2), output_schema, output_table_name)
        # save processed data
        col_types = [
            "object",    # Date (or "string", if you prefer to store as text)
            "float64",   # Open
            "float64",   # High
            "float64",   # Low
            "float64",   # Close
            "float64",   # Volume
            "float64",   # Market_Cap
            "string",    # Ticker
            "string",    # IndClass_Sector
            "string",    # IndClass_Industry
            "float64",   # Typical_Price
            "float64",   # Return
            "float64"    # VWAP_5
        ]
        db.df_to_sql_table(df, col_types, "datacollection", "processed_data")
    db.close_connection()
    
    if if_return:
        return df, final_df


def get_alpha101_table_from_db():

    db = database_utils()
    db.connect()
    # Fetch the input stock_data table.
    query_input = f'''
        SELECT *
        FROM datacollection.processed_data
    '''
    db.execute_query(query_input)
    df = pd.DataFrame(db.fetch_results(), 
                      columns=["Date", "Open", "High", "Low", "Close", "Volume", "Market_Cap", "Ticker", 
                               "IndClass_Sector", "IndClass_Industry", "Typical_Price", "Return", "VWAP_5"])

    df["Date"] = (pd.to_datetime(df["Date"]) .dt.strftime("%Y-%m-%d"))
    
    # Fetch the alpha101 table.
    query_alpha = f'''
        SELECT * 
        FROM datacollection.alpha101
    '''
    db.execute_query(query_alpha)

    final_df = pd.DataFrame(db.fetch_results())
    final_df.rename(columns={final_df.columns[0]: "Date", final_df.columns[1]: "Ticker"}, inplace=True)
    final_df["Date"] = pd.to_datetime(final_df["Date"]).dt.strftime("%Y-%m-%d")
    db.close_connection()
    
    return df, final_df



# if __name__ == '__main__':
#     generate_alphas()
#     df, final_df = generate_alphas()
#     print("Alpha DataFrame:\n", final_df.head())
#     final_df.to_csv("alpha_results.csv", index=False)
#     print("Done.")



#if __name__ == '__main__':
#    alpha_indices = [
 #       1, 2, 3, 4, 5, 6, 7, 8, 9, 
  #      10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
   #     20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
    #    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
     #   40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
      #  50, 51, 52, 53, 54, 55, 56, 57, 58, 
       # 60, 61, 62, 63, 64, 65, 66, 68, 
 #       71, 72, 73, 74, 75, 77, 78, 
  #      81, 83, 84, 85, 86, 88, 
   #     92, 94, 95, 96, 98, 99, 101,
    #]
    # for later usage
 #   db = database_utils()
  #  db.connect()
   # db.set_schema('datacollection')
    #db.execute_query('SELECT "Date", "Open", "High", "Low", "Close", "Volume", "Market_Cap", "Ticker", "IndClass_Sector", "IndClass_Industry" FROM stock_data')
   # df = pd.DataFrame(db.fetch_results(), columns=["Date", "Open", "High", "Low", "Close", "Volume", "Market_Cap", "Ticker", "IndClass_Sector", "IndClass_Industry"])
    # df = pd.read_csv('.\TeamOne\stock_data.csv')
    #df = data_preprocessing(df)
   # tickers = df['Ticker'].unique()
    #final_df = run_by_ticker(df, tickers, alpha_indices) #compute alpha for each stock
    #print(final_df)
    # final_df.to_csv("alpha_results.csv", index=False)
    #print("Done.")
    #db.close_connection()