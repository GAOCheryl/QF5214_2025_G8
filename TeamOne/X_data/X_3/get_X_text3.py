# -*- coding: utf-8 -*-
'''
@Time  :  2025/03/02 15:15:52
@Author  :  River
'''
 
import asyncio
import csv
from datetime import datetime, timedelta
from random import randint
from configparser import ConfigParser
from twikit import Client, TooManyRequests
import nest_asyncio
nest_asyncio.apply()

# -------------------------------------
# Configuration Parameters
# -------------------------------------
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 3, 1)    

nasdaq_companies = ["ADI", "ANSS", "AAPL", "AMAT"]    #Need change
MINIMUM_TWEETS = 6  

async def get_tweets(client, query, tweets=None):
    """
    Asynchronously fetch tweets using the twikit.Client.
    
    If 'tweets' is None, start a new search with the given query.
    Otherwise, fetch the next page of tweets by calling tweets.next().
    """
    if tweets is None:
        print(f"{datetime.now()} - Searching tweets with query: {query}")
        tweets = await client.search_tweet(query, product='Top')
    else:
        wait_time = randint(5, 10)
        print(f"{datetime.now()} - Getting next tweets after {wait_time} seconds ...")
        await asyncio.sleep(wait_time)
        tweets = await tweets.next()
    return tweets

async def main():
    start_time = datetime.now()
    
    # -------------------------------------
    # Load Configuration
    # -------------------------------------
    config = ConfigParser()
    config.read('config.ini')
    username = config['X']['username']
    email = config['X']['email']
    password = config['X']['password']

    # -------------------------------------
    # Initialize and Login to twikit Client
    # -------------------------------------
    client = Client(language='en-US',
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36")
    await client.login(auth_info_1=username, auth_info_2=email, password=password)
    client.save_cookies('cookies.json')
    client.load_cookies('cookies.json')

    # -------------------------------------
    # Open CSV file for writing tweet data.
    # Use utf-8-sig encoding (to include BOM) and newline='' to avoid extra blank lines.
    # -------------------------------------File name need change
    with open('tweets_nasdaq100_3.csv', 'w', newline='', encoding='utf-8-sig') as csv_file:       
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        writer.writerow([
            'Company',
            'Tweet_Count',
            'Text',
            'Created_At',
            'Retweets',
            'Likes',
            'URL'
        ])
        csv_file.flush()  # Flush header to disk immediately

        # -------------------------------------
        # Loop through each company ticker.
        # -------------------------------------
        for company in nasdaq_companies:
            print(f"\n===== Start crawling tweets for {company} =====\n")
            current_date = START_DATE
            global_count = 0

            # Iterate day-by-day within the specified date range.
            while current_date < END_DATE:
                next_date = current_date + timedelta(days=1)
                # Append financial keywords to restrict tweets to finance/stock-related content.
                query_str = (
                    f"({company}) lang:en "
                    f"since:{current_date.strftime('%Y-%m-%d')} "
                    f"until:{next_date.strftime('%Y-%m-%d')}"
                )

                tweets = None  # Reset tweets for the new day.
                day_count = 0  # Counter for tweets collected on the current day.

                # Paginate through tweet results until no more tweets are available.
                while True:
                    try:
                        tweets = await get_tweets(client, query_str, tweets)
                    except TooManyRequests as e:
                        rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
                        print(f"{datetime.now()} - Rate limit reached. Waiting until {rate_limit_reset}")
                        wait_time = (rate_limit_reset - datetime.now()).total_seconds()
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)
                        continue

                    if not tweets:
                        print(f"{datetime.now()} - No more tweets found for {company} on {current_date.date()}")
                        break

                    # Iterate over each tweet in the current batch.
                    for t in tweets:
                        # Remove the like filter; process all tweets.
                        # Clean tweet text by removing newline characters.
                        clean_text = t.text.replace('\n', ' ').replace('\r', ' ')
                        
                        # Only process tweet if the tweet text contains the company keyword.
                        if company.upper() not in clean_text.upper():
                            continue

                        global_count += 1
                        day_count += 1

                        # Construct tweet URL.
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
                        csv_file.flush()  # Flush after each row

                    print(f"{datetime.now()} - {company} {current_date.date()} collected {day_count} tweets so far...")

                    if day_count >= MINIMUM_TWEETS:
                        break

                current_date = next_date

            print(f"===== Done crawling for {company}. Total tweets collected: {global_count} =====\n")

    elapsed_time = datetime.now() - start_time
    print(f"{datetime.now()} - All Done! Total time taken: {elapsed_time}")

# Execute the main coroutine using asyncio.run()
asyncio.run(main())
