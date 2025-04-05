import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import os
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from nlp.SentimentEmotionAnalyzer import SentimentEmotionAnalyzer

analyzer = SentimentEmotionAnalyzer()

user = "postgres"
pw = "qf5214G8"
host = "pgm-t4n365kyk1sye1l7eo.pgsql.singapore.rds.aliyuncs.com"
port = 5555
dbname = "QF5214"

db = create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{dbname}")

files = [
    "filtered_tweets_nasdaq100_one.csv",
    "filtered_tweets_nasdaq100_two.csv",
]
path = "./data/"

def getbpfile(filename):
    stem = Path(filename).stem
    return f"progress_{stem}.json"

for fname in files:
    fpath = os.path.join(path, fname)
    print(f"Starting file: {fname}")

    bpfile = getbpfile(fname)

    if os.path.exists(bpfile):
        with open(bpfile, "r") as f:
            prog = json.load(f)
        startidx = prog.get("last_index", 0)
    else:
        startidx = 0

    resultdf = pd.read_csv(fpath)
    resultdf.rename(columns={"Company": "company", "Cleaned_Text": "text", "Created_At": "created_at"}, inplace=True)
    resultdf["created_at"] = pd.to_datetime(resultdf["created_at"], format="%a %b %d %H:%M:%S %z %Y")
    resultdf["Date"] = resultdf["created_at"].dt.strftime("%Y/%m/%d")

    total = len(resultdf)
    okcount = 0
    sentdata = []

    for i, row in enumerate(resultdf.itertuples(index=False)):
        if i < startidx:
            continue

        company = row.company
        text = str(row.text)
        dt = row.created_at
        created = dt.strftime("%Y-%m-%d %H:%M:%S")
        date = dt.strftime("%Y/%m/%d")

        try:
            res = analyzer.nlp(text)
            sentdata.append({
                "company": company,
                "text": text,
                "created_at": created,
                "Date": date,
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

        if len(sentdata) >= 100:
            tmp = pd.DataFrame(sentdata)
            tblname = f"sentiment_raw_data_{Path(fname).stem.split('_')[-1]}"
            try:
                tmp.to_sql(
                    name=tblname,
                    con=db,
                    if_exists="append",
                    index=False,
                    schema="nlp"
                )
            except:
                print(f"DB error at row {i}, trying one by one.")
                for thing in sentdata:
                    try:
                        pd.DataFrame([thing]).to_sql(
                            name=tblname,
                            con=db,
                            if_exists="append",
                            index=False,
                            schema="nlp"
                        )
                    except:
                        realrow = i - len(sentdata) + sentdata.index(thing) + 1
                        print(f"Failed at row {realrow}")
            okcount = okcount + len(sentdata)
            sentdata = []

            if okcount % 1000 == 0:
                print(f"Saved {okcount} rows from {fname}")

            with open(bpfile, "w") as f:
                json.dump({"last_index": i + 1}, f)

    if len(sentdata) > 0:
        tmp = pd.DataFrame(sentdata)
        tblname = f"sentiment_raw_data_{Path(fname).stem.split('_')[-1]}"
        try:
            tmp.to_sql(
                name=tblname,
                con=db,
                if_exists="append",
                index=False,
                schema="nlp"
            )
        except:
            print(f"Final save error! Going one by one.")
            for thing in sentdata:
                try:
                    pd.DataFrame([thing]).to_sql(
                        name=tblname,
                        con=db,
                        if_exists="append",
                        index=False,
                        schema="nlp"
                    )
                except:
                    print(f"Final single insert failed.")
        okcount = okcount + len(sentdata)
        with open(bpfile, "w") as f:
            json.dump({"last_index": total}, f)

    print(f"Done with {fname}, saved {okcount} rows.")
