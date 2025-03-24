#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# libraries
import yfinance as yf
import pandas as pd
import sys


# In[46]:


# Set date
today = pd.Timestamp.today().strftime("%Y-%m-%d")
previous_day = (pd.to_datetime(today) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
print(today)


# # Get Stock Data

# In[37]:


# company list
nasdaq_companies = [ "ADBE", "AMD", "ABNB", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", "ADI", "ANSS",
                     "AAPL", "AMAT", "APP", "ARM", "ASML", "AZN", "TEAM", "ADSK", "ADP", "AXON", 
                     "BKR", "BIIB", "BKNG", "AVGO", "CDNS", "CDW", "CHTR", "CTAS", "CSCO", "CCEP", 
                     "CTSH", "CMCSA", "CEG", "CPRT", "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", 
                     "FANG", "DASH", "EA", "EXC", "FAST", "FTNT", "GEHC", "GILD", "GFS", "HON", 
                     "IDXX", "INTC", "INTU", "ISRG", "KDP", "KLAC", "KHC", "LRCX", "LIN", "LULU", 
                     "MAR", "MRVL", "MELI", "META", "MCHP", "MU", "MSFT", "MSTR", "MDLZ", "MDB", 
                     "MNST", "NFLX", "NVDA", "NXPI", "ORLY", "ODFL", "ON", "PCAR", "PLTR", "PANW", 
                     "PAYX", "PYPL", "PDD", "PEP", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", 
                     "TTWO", "TMUS", "TSLA", "TXN", "TTD", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS" ]

# Initialize a list to store data
all_stock_data = []

for ticker in nasdaq_companies:
    stock = yf.Ticker(ticker)

    # Get historical market data (including Adjusted Close)
    hist_data = stock.history(start=previous_day, end=today, auto_adjust=False)[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    hist_data = hist_data.rename(columns={"Adj Close": "Adj_Close"})  # Rename column

    # Get financial ratios and company info
    financials = stock.info
    factors = {
        "PE": financials.get("trailingPE", None),
        "PB": financials.get("priceToBook", None),
        "PS": financials.get("priceToSalesTrailing12Months", None),
        "ROE": financials.get("returnOnEquity", None),
        "PM": financials.get("profitMargins", None),
        "IN": financials.get("ebitdaMargins", None),
        "Market_Cap": financials.get("marketCap", None),  # Get directly from Yahoo Finance
        "IndClass_Sector": financials.get("sector", "Unknown"),
        "IndClass_Industry": financials.get("industry", "Unknown")
    }

    # Convert financial factors to DataFrame and duplicate for all dates
    factors_df = pd.DataFrame([factors] * len(hist_data), index=hist_data.index)

    # Combine market data with financial factors
    combined_data = hist_data.join(factors_df)
    combined_data["Ticker"] = ticker  # Add ticker column
    all_stock_data.append(combined_data)

# Merge all data into a single DataFrame
final_stock_df = pd.concat(all_stock_data)

# Check if final_stock_df is empty, and terminate if no data
if final_stock_df.empty:
    print(f"No data available for the specified date range ({previous_day} to {today}), terminating the process.")
    sys.exit()  # Terminate the program if final data is empty

# Save to CSV file
final_stock_df.to_csv("/root/automation/stock_data.csv", index=True)

print("File saved successfully.")

# Display sample data
print(final_stock_df.head())


# # Get Index Data

# In[45]:


# Define index list and their Yahoo Finance tickers
indices = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "Russell 1000": "^RUI",
    "Russell 3000": "^RUA",
    "Wilshire 5000": "^W5000"
}

# List to store all data
all_index_data = []

# Loop through each index to fetch data
for index_name, ticker in indices.items():
    print(f"Fetching data for {index_name} ({ticker})...")
    
    # Download historical data with auto_adjust=False to get accurate 'Adj Close'
    stock = yf.Ticker(ticker)
    data = stock.history(start=previous_day, end=today, auto_adjust=False)[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    data = data.rename(columns={"Adj Close": "Adj_Close"})  # Rename column
    
    # Reset index to move Date from index to a column
    data.reset_index(inplace=True)
    
    # Add a Ticker and Index Name column
    data["Ticker"] = ticker
    data["Index"] = index_name  # Add a new column for index name
    
    # Rename columns to match required format
    data.rename(columns={
        "Adj Close": "Adj_close"
    }, inplace=True)
    
    # Select only the necessary columns
    data = data[["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume", "Ticker", "Index"]]
    
    # Append to the list
    all_index_data.append(data)

# Merge all index data into a single DataFrame
final_index_df = pd.concat(all_index_data, ignore_index=True)

# Check if final_stock_df is empty, and terminate if no data
if final_index_df.empty:
    print(f"No data available for the specified date range ({previous_day} to {today}), terminating the process.")
    sys.exit()  # Terminate the program if final data is empty

# Save the data to a CSV file
final_index_df.to_csv("/root/automation/index_data.csv", index=False)

print("File saved successfully.")

# Display the first few rows of the data
print(final_index_df.head())

