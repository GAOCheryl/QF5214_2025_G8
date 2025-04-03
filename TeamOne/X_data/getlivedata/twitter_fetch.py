import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import psycopg2

print("69 Companies and 40 tweets per extraction")

# Twitter API config
API_KEY = "4e94cbd5c04e44fb8712203476911270"
base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
headers = {"X-API-Key": API_KEY}

# List of tickers
tickers = [
    "ADBE", "AMD", "ABNB", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", 
    "ADI", "ANSS", "AAPL", "AMAT", "APP", "ARM", "ASML", "AZN", 
    "TEAM", "ADSK", "ADP", "AXON", "BKR", "BIIB", "BKNG", "AVGO", 
    "CDNS", "CDW", "CHTR", "CTAS", "CSCO", "CCEP", "CTSH", "CMCSA", 
    "CEG", "CPRT", "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", 
    "VRTX", "WBD", "WDAY", "XEL", "ZS", "QCOM", "REGN", "ROP", "ROST",
    "FAST", "FTNT", "GEHC", "GILD" ,"ON", "PCAR", "PLTR", "PANW",
    "PAYX", "PYPL", "PDD", "PEP","SBUX", "SNPS", "TTWO", 
    "TMUS","TSLA", "TXN", "TTD", "VRSK"
]

# Number of tweets per ticker
max_tweets = 40
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

    ticker_tweets = []
    page_number = 1
    pbar = tqdm(total=max_tweets, desc=f"Collecting ${ticker}", unit="tweet", dynamic_ncols=True)

    while len(ticker_tweets) < max_tweets:
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"❌ Error {response.status_code} for {ticker}: {response.text}")
            break

        data = response.json()
        tweets = data.get("tweets", [])

        ticker_tweets.extend(tweets)
        pbar.update(len(tweets))

        print(f"\n🔹 Page {page_number} (${ticker}): Retrieved {len(tweets)} tweets")

        if not data.get("has_next_page") or len(ticker_tweets) >= max_tweets:
            break

        params["cursor"] = data.get("next_cursor", "")
        page_number += 1

    pbar.close()
    ticker_tweets = ticker_tweets[:max_tweets]

    print(f"✅ Finished fetching ${ticker}: {len(ticker_tweets)} tweets\n")

    for t in ticker_tweets:
        all_data.append({
            "company": ticker,
            "tweet_count": 1,
            "text": t.get("text"),
            "created_at": t.get("createdAt"),
            "retweets": t.get("retweetCount"),
            "likes": t.get("likeCount"),
            "url": t.get("url"),
            "id": t.get("id")
        })

# Convert to DataFrame
df = pd.DataFrame(all_data)
print(f"📦 Total tweets collected: {len(df)}")

# PostgreSQL connection config
db_config = {
    "host": "pgm-t4n365kyk1sye1l7eo.pgsql.singapore.rds.aliyuncs.com",
    "port": "5555",
    "dbname": "QF5214",
    "user": "postgres",
    "password": "qf5214G8"
}


# Insert into PostgreSQL
try:
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    print("✅ Connected to database")

    insert_query = """
        INSERT INTO datacollection.tweets_live
        (company, tweet_count, text, created_at, retweets, likes, url, id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            company = EXCLUDED.company,
            tweet_count = EXCLUDED.tweet_count,
            text = EXCLUDED.text,
            created_at = EXCLUDED.created_at,
            retweets = EXCLUDED.retweets,
            likes = EXCLUDED.likes,
            url = EXCLUDED.url
    """

    for _, row in df.iterrows():
        cur.execute(insert_query, (
            row["company"],
            row["tweet_count"],
            row["text"],
            row["created_at"],
            row["retweets"],
            row["likes"],
            row["url"],
            row["id"] 
        ))

    conn.commit()
    print(f"✅ Inserted {len(df)} rows into datacollection.tweets_live")

except Exception as e:
    print("❌ Failed to insert into database:", e)

finally:
    if conn:
        cur.close()
        conn.close()
        print("🔌 Database connection closed")

