'''
nasdaq_companies = [ "ADBE", "AMD", "ABNB", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", 
"ADI", "ANSS", "AAPL", "AMAT", "APP", "ARM", "ASML", "AZN", 
"TEAM", "ADSK", "ADP", "AXON", "BKR", "BIIB", "BKNG", "AVGO", 
"CDNS", "CDW", "CHTR", "CTAS", "CSCO", "CCEP", "CTSH", "CMCSA", 
"CEG", "CPRT", "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", 
"FANG", "DASH", "EA", "EXC", "FAST", "FTNT", "GEHC", "GILD", 
"GFS", "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP", "KLAC", 
"KHC", "LRCX", "LIN", "LULU", "MAR", "MRVL", "MELI", "META", 
"MCHP", "MU", "MSFT", "MSTR", "MDLZ", "MDB", "MNST", "NFLX", 
"NVDA", "NXPI", "ORLY", "ODFL", "ON", "PCAR", "PLTR", "PANW", 
"PAYX", "PYPL", "PDD", "PEP", "QCOM", "REGN", "ROP", "ROST", 
"SBUX", "SNPS", "TTWO", "TMUS", "TSLA", "TXN", "TTD", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS" ]
'''

#############
# -*- coding: utf-8 -*-
import asyncio
import csv
import os
import json
from datetime import datetime, timedelta
from random import randint
from configparser import ConfigParser
from twikit import Client, TooManyRequests
import httpx
import nest_asyncio
nest_asyncio.apply()

# -------------------------------------
# Configuration Parameters
# -------------------------------------
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 3, 1)
MINIMUM_TWEETS = 6
PROGRESS_FILE = 'progress_log.json'
LOG_FILE = 'log.txt'

# Prompt user
csv_filename = input("Enter the CSV filename strictly in the form: tweets_nasdaq100_4.csv: ")
company_input = input('Enter tickers in the form: "APP", "ARM", "ASML", "AZN": ')
nasdaq_companies = [
    x.strip().strip('"').strip("'")
    for x in company_input.split(",")
]
print("Tickers:", nasdaq_companies)

# -------------------------------------
# Load progress log if available
# -------------------------------------
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        progress_log = json.load(f)
else:
    progress_log = {}

# -------------------------------------
# Logging helper
# -------------------------------------
def log_message(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {message}\n"
    print(full_msg.strip())
    with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
        log_file.write(full_msg)

# -------------------------------------
# Async Tweet Fetch Function
# -------------------------------------
async def get_tweets(client, query, tweets=None):
    if tweets is None:
        log_message(f"Searching tweets with query: {query}")
        tweets = await client.search_tweet(query, product='Top')
    else:
        wait_time = randint(5, 10)
        log_message(f"Getting next tweets after {wait_time} seconds ...")
        await asyncio.sleep(wait_time)
        tweets = await tweets.next()
    return tweets

# -------------------------------------
# Main Async Function
# -------------------------------------
async def main():
    start_time = datetime.now()
    log_message("🔄 Script started.")

    # Load login config
    config = ConfigParser()
    config.read('config.ini')
    username = config['X']['username']
    email = config['X']['email']
    password = config['X']['password']

    # Login
    client = Client(language='en-US',
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.2.4231.78 Safari/537.36")
    await client.login(auth_info_1=username, auth_info_2=email, password=password)
    client.save_cookies('cookies.json')
    client.load_cookies('cookies.json')
    log_message("✅ Login successful.")

    # Create file if not exists; otherwise append
    file_exists = os.path.exists(csv_filename)
    with open(csv_filename, 'a', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

        if not file_exists:
            writer.writerow([
                'Company',
                'Tweet_Count',
                'Text',
                'Created_At',
                'Retweets',
                'Likes',
                'URL'
            ])
            csv_file.flush()

        for company in nasdaq_companies:
            log_message(f"🚀 Start crawling tweets for {company}")

            # Resume point
            if company in progress_log:
                current_date = datetime.strptime(progress_log[company], '%Y-%m-%d') + timedelta(days=1)
                log_message(f"📌 Resuming {company} from {current_date.date()}")
            else:
                current_date = START_DATE

            global_count = 0

            while current_date < END_DATE:
                next_date = current_date + timedelta(days=1)
                query_str = (
                    f"({company}) lang:en "
                    f"since:{current_date.strftime('%Y-%m-%d')} "
                    f"until:{next_date.strftime('%Y-%m-%d')}"
                )

                tweets = None
                day_count = 0

                while True:
                    try:
                        tweets = await get_tweets(client, query_str, tweets)

                    except TooManyRequests as e:
                        rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
                        log_message(f"⚠️ Rate limit hit. Sleeping until {rate_limit_reset}")
                        wait_time = (rate_limit_reset - datetime.now()).total_seconds()
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)
                        continue

                    except httpx.ReadTimeout:
                        log_message("⏱️ ReadTimeout occurred. Retrying in 10 seconds...")
                        await asyncio.sleep(10)
                        continue

                    except Exception as e:
                        log_message(f"❌ Unexpected error: {e}")
                        await asyncio.sleep(5)
                        continue

                    if not tweets:
                        log_message(f"No more tweets for {company} on {current_date.date()}")
                        break

                    for t in tweets:
                        clean_text = t.text.replace('\n', ' ').replace('\r', ' ')
                        if company.upper() not in clean_text.upper():
                            continue

                        global_count += 1
                        day_count += 1

                        user_screen_name = getattr(t.user, 'screen_name', None) or getattr(t.user, 'username', '')
                        tweet_id = getattr(t, 'id', None)
                        url = f"https://twitter.com/{user_screen_name}/status/{tweet_id}" if user_screen_name and tweet_id else ""

                        writer.writerow([
                            company,
                            global_count,
                            clean_text,
                            t.created_at,
                            t.retweet_count,
                            t.favorite_count,
                            url
                        ])
                        csv_file.flush()

                    log_message(f"{company} {current_date.date()} collected {day_count} tweets.")

                    if day_count >= MINIMUM_TWEETS:
                        break

                # Save progress after each day
                progress_log[company] = current_date.strftime('%Y-%m-%d')
                with open(PROGRESS_FILE, 'w') as f:
                    json.dump(progress_log, f)

                current_date = next_date

            log_message(f"✅ Done crawling for {company}. Total tweets collected: {global_count}")

    elapsed_time = datetime.now() - start_time
    log_message(f"🎉 All Done! Total time taken: {elapsed_time}")

# -------------------------------------
# Run
# -------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log_message(f"💥 Unhandled exception: {e}")
