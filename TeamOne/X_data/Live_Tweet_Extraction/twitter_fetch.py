import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import psycopg2

print("🚀 All comapnies 50 tweets Final")

# Twitter API config
API_KEY = "93af433a8ff843819d702acbfadcc895"
base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
headers = {"X-API-Key": API_KEY}

# List of tickers
tickers = [
    "ADBE", "AMD", "ABNB", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", 
    "ADI", "ANSS", "AAPL", "AMAT", "APP", "ARM", "ASML", "AZN", 
    "TEAM", "ADSK", "ADP", "AXON", "BKR", "BIIB", "BKNG", "AVGO", 
    "CDNS", "CDW", "CHTR", "CTAS", "CSCO", "CCEP", "CTSH", "CMCSA", 
    "CEG", "CPRT", "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", 
    "VRTX", "WBD", "WDAY", "XEL", "ZS", "QCOM", "REGN", "ROP", "ROST"
]

# Number of tweets per ticker
max_tweets = 50
all_data = []

# Timestamp for tracking the run
start_time = datetime.now()
print("🕒 Data extraction started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

# Fetch tweets for each ticker
for ticker in tickers:
    print(f"\n🔍 Fetching tweets for ${ticker}")
    
    query = f"${ticker} lang:en"
    params = {
        "query": query,
        "queryType": "Latest",
        "cursor": ""
    }