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
        result_df['Ticker'] = tik
        
        for idx in alpha_indices:
            func_name = f"alpha{idx:03d}"  # Format as 'alpha001', 'alpha002', etc.
            print(f"Running function {func_name} for {tik}...")
            
            if hasattr(stock, func_name):
                func = getattr(stock, func_name)
                result_df[func_name] = func()
            else:
                print(f"Function {func_name} not found in Alphas class.")

        
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
        71, 72, 73, 74, 75, 76, 77, 78, 79,
        81, 82, 83, 84, 85, 87, 88, 89,
        92, 93, 94, 96, 97, 98, 99, 101
    ]
    
    # alpha_lst1 = [5, 27, 68, 86, 95] # potential error
    # alpha_dict = {'alpha019': 254, 'alpha039': 254, 'alpha048': 255,
    #               'alpha052': 243, 'alpha032': 237, 'alpha037': 203, 'alpha036': 202,
    #               'alpha094': 85,     
    #               'alpha078': 66, 
    #               'alpha043': 39, 'alpha084': 39, 'alpha085': 39,
    #               'alpha035': 33,
    #               'alpha045': 24,'alpha017': 23,'alpha001': 22, 
    #               'alpha007': 19,'alpha022': 19,'alpha025': 19, 'alpha030': 19,  'alpha047': 19
    #               'alpha076': 18, 'alpha092': 18, 
    #               'alpha073': 17,         
    #               'alpha008': 15, 'alpha071': 15, 
    #               } # with lots of NULL values
    # new_alpha_lst = [48, 56, 58, 59, 63, 76, 79, 82, 87, 89, 93, 97]

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


def get_alpha101_table_from_db(to_csv = False):

    db = database_utils()
    db.connect()
    # Fetch the input stock_data table.
    query_input = f'''
        SELECT *
        FROM datacollection.processed_data
    '''
    db.execute_query(query_input)
    df_stock = pd.DataFrame(db.fetch_results(), 
                      columns=["Date", "Open", "High", "Low", "Close", "Volume", "Market_Cap", "Ticker", 
                               "IndClass_Sector", "IndClass_Industry", "Typical_Price", "Return", "VWAP_5"])

    df_stock["Date"] = (pd.to_datetime(df_stock["Date"]) .dt.strftime("%Y-%m-%d"))
    
    # Fetch the alpha101 table.
    query_alpha = f'''
        SELECT * 
        FROM datacollection.alpha101
    '''
    db.execute_query(query_alpha)

    df_101 = pd.DataFrame(db.fetch_results())
    df_101.rename(columns={df_101.columns[0]: "Date", df_101.columns[1]: "Ticker"}, inplace=True)
    df_101["Date"] = pd.to_datetime(df_101["Date"]).dt.strftime("%Y-%m-%d")

    # Fetch the market index table.
    query_index = f'''
        SELECT * 
        FROM datacollection.index_data
    '''
    db.execute_query(query_index)
    df_idx = pd.DataFrame(db.fetch_results(), 
                      columns=["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume", "Ticker", 
                               "IndexName"])

    df_idx["Date"] = (pd.to_datetime(df_idx["Date"]) .dt.strftime("%Y-%m-%d"))

    db.close_connection()
    
    if to_csv:
        df_stock.to_csv("stock_data.csv", index=False)
        df_101.to_csv("alpha101.csv", index=False)
        df_idx.to_csv("index_data.csv", index=False)
    
    return df_stock, df_101, df_idx


def get_updated_sentiment_table_from_db():
    db = database_utils()
    db.connect()
    # Fetch the input stock_data table.
    query_input = f'''SELECT * FROM nlp.sentiment_aggregated_newdate'''
    db.execute_query(query_input)
    df_sentiment = pd.DataFrame(db.fetch_results(), 
                      columns=["Ticker", "Date", "Positive", "Negative", "Neutral", "Surprise", "Joy", "Anger", 
                               "Fear", "Sadness", "Disgust", "Intent Sentiment"])
    df_sentiment["Date"] = (pd.to_datetime(df_sentiment["Date"]) .dt.strftime("%Y-%m-%d"))

    db.close_connection()

    return df_sentiment


def get_sentiment_table_from_db():
    db = database_utils()
    db.connect()
    # Fetch the input stock_data table.
    query_input = f'''SELECT * FROM nlp.sentiment_aggregated_data'''
    db.execute_query(query_input)
    df_sentiment = pd.DataFrame(db.fetch_results(), 
                      columns=["Ticker", "Date", "Positive", "Negative", "Neutral", "Surprise", "Joy", "Anger", 
                               "Fear", "Sadness", "Disgust", "Intent Sentiment"])
    df_sentiment["Date"] = (pd.to_datetime(df_sentiment["Date"]) .dt.strftime("%Y-%m-%d"))

    query_input = f'''SELECT * FROM nlp.sentiment_aggregated_data_filter'''
    db.execute_query(query_input)
    df_sentiment_filter = pd.DataFrame(db.fetch_results(), 
                      columns=["Ticker", "Date", "Positive", "Negative", "Neutral", "Surprise", "Joy", "Anger", 
                               "Fear", "Sadness", "Disgust", "Intent Sentiment"])
    df_sentiment_filter["Date"] = (pd.to_datetime(df_sentiment["Date"]) .dt.strftime("%Y-%m-%d"))
    db.close_connection()

    return df_sentiment, df_sentiment_filter

 


if __name__ == '__main__':
    get_alpha101_table_from_db(to_csv=True)

    


