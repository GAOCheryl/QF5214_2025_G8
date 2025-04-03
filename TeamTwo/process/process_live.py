from sqlalchemy import create_engine, text
from datetime import datetime
import os
import json
import pandas as pd
from pathlib import Path
from filter import clean_text, filter_text, filter_irrelevant_comments, remove_stock_symbols_flexible
from nlp_v1 import SentimentEmotionAnalyzer

analyzer = SentimentEmotionAnalyzer()

user = "postgres"
pw = "qf5214"
host = "134.122.167.14"
port = 5555
dbname = "QF5214"
db = create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{dbname}")

progfile = "progress_sentiment_live.json"
if os.path.exists(progfile):
    with open(progfile, "r") as f:
        prog = json.load(f)
    lastprocessed = prog.get("last_created_at")
else:
    lastprocessed = "2025-03-01 00:00:00"

print(f"Last processed time: {lastprocessed}")

q = """
    SELECT company, tweet_count, text, created_at, retweets, likes, url, id
    FROM datacollection.tweets_live
    WHERE created_at > '""" + lastprocessed + """'
    ORDER BY created_at ASC
"""
with db.connect() as conn:
    resultdf = pd.read_sql(text(q), conn)

if len(resultdf) == 0:
    print("No new data to process.")
    exit()

print(f"Got {len(resultdf)} new tweets")

resultdf = resultdf.rename(columns={
    "company": "Company",
    "text": "Text",
    "created_at": "Created_At"
})

resultdf['Cleaned_Text'] = resultdf['Text'].apply(clean_text)
filtered = resultdf[resultdf.apply(lambda row: not filter_irrelevant_comments(row['Cleaned_Text'], row['Company']), axis=1)].copy()
filtered['Cleaned_Text'] = filtered['Cleaned_Text'].apply(remove_stock_symbols_flexible)
data = filtered[filtered['Cleaned_Text'].apply(filter_text)].copy()

data.to_sql(
    name="filtered_tweets_live",
    con=db,
    if_exists="append",
    index=False,
    schema="nlp"
)
print(f"Saved filtered data")

data["Created_At"] = pd.to_datetime(data["Created_At"])
data["Date"] = data["Created_At"].dt.strftime("%Y/%m/%d")

sentdata = []
okcount = 0
for i, row in enumerate(data.itertuples(index=False)):
    try:
        res = analyzer.nlp(row.Cleaned_Text)
        sentdata.append({
            "company": row.Company,
            "text": row.Cleaned_Text,
            "created_at": row.Created_At.strftime("%Y-%m-%d %H:%M:%S"),
            "Date": row.Created_At.strftime("%Y/%m/%d"),
            "Positive": res[0],
            "Negative": res[1],
            "Neutral": res[2],
            "Surprise": res[3],
            "Joy": res[4],
            "Anger": res[5],
            "Fear": res[6],
            "Sadness": res[7],
            "Disgust": res[8],
            "Emotion Confidence": res[9],
            "Intent Sentiment": res[10],
            "Confidence": res[11]
        })
    except:
        print(f"Error on row {i}")
        continue

    if len(sentdata) >= 100:
        tmp = pd.DataFrame(sentdata)
        try:
            tmp.to_sql("sentiment_live", db, if_exists="append", index=False, schema="nlp")
            okcount = okcount + len(sentdata)
            sentdata = []
        except Exception as e:
            print(f"DB error! Trying one by one...")
            for thing in sentdata:
                try:
                    pd.DataFrame([thing]).to_sql("sentiment_live", db, if_exists="append", index=False, schema="nlp")
                    okcount = okcount + 1
                except:
                    continue
            sentdata = []

if len(sentdata) > 0:
    tmp = pd.DataFrame(sentdata)
    try:
        tmp.to_sql("sentiment_live", db, if_exists="append", index=False, schema="nlp")
        okcount = okcount + len(sentdata)
    except:
        print(f"Final save error! Going one by one.")
        for thing in sentdata:
            try:
                pd.DataFrame([thing]).to_sql("sentiment_live", db, if_exists="append", index=False, schema="nlp")
                okcount = okcount + 1
            except:
                continue

print(f"Total saved: {okcount}")

newest = data["Created_At"].max().strftime("%Y-%m-%d %H:%M:%S")
with open(progfile, "w") as f:
    json.dump({"last_created_at": newest}, f)
print(f"Checkpoint updated: {newest}")

sql = """
    SELECT company, "Date", 
           CAST("Positive" AS FLOAT), CAST("Negative" AS FLOAT), CAST("Neutral" AS FLOAT), 
           CAST("Surprise" AS FLOAT), CAST("Joy" AS FLOAT), CAST("Anger" AS FLOAT), 
           CAST("Fear" AS FLOAT), CAST("Sadness" AS FLOAT), CAST("Disgust" AS FLOAT), 
           "Intent Sentiment"
    FROM nlp.sentiment_live
"""
allsents = pd.read_sql(text(sql), db)
allsents = allsents[allsents["Emotion Confidence"] >= 0.4]

dailystats = allsents.groupby(["company", "Date"]).agg({
    "Positive": "mean",
    "Negative": "mean",
    "Neutral": "mean",
    "Surprise": "mean",
    "Joy": "mean",
    "Anger": "mean",
    "Fear": "mean",
    "Sadness": "mean",
    "Disgust": "mean",
    "Intent Sentiment": lambda x: x.value_counts().idxmax()
}).reset_index()

existing = pd.read_sql(
    text('SELECT company, "Date" FROM nlp.sentiment_aggregated_live'),
    db
)
dailystats = dailystats.merge(existing, on=["company", "Date"], how="left", indicator=True)
dailystats = dailystats[dailystats["_merge"] == "left_only"].drop(columns=["_merge"])

if not dailystats.empty:
    dailystats.to_sql(
        name="sentiment_aggregated_live",
        con=db,
        if_exists="append",
        index=False,
        schema="nlp"
    )
    print(f"Saved {len(dailystats)} new aggregated rows.")
else:
    print("No new data to aggregate.")

print("All done! Stats saved.")
