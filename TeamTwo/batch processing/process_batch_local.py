import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import os
import json
from pathlib import Path

analyzer = SentimentEmotionAnalyzer()

# PostgreSQL connection settings
db_user = "postgres"
db_password = "qf5214G8"
db_host = "pgm-t4n365kyk1sye1l7eo.pgsql.singapore.rds.aliyuncs.com"
db_port = 5555
db_name = "QF5214"

# Create engine
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

input_files = [
    "filtered_tweets_nasdaq100_twenty.csv"
]
base_path = "C:/Users/20757/"

def get_breakpoint_file(filename):
    stem = Path(filename).stem
    return f"progress_{stem}.json"

for file_name in input_files:
    file_path = os.path.join(base_path, file_name)
    print(f"Starting file: {file_name}")

    bp_file = get_breakpoint_file(file_name)

    if os.path.exists(bp_file):
        with open(bp_file, "r") as f:
            progress = json.load(f)
        start_index = progress.get("last_index", 0)
    else:
        start_index = 0

    df = pd.read_csv(file_path)
    df.rename(columns={"Company": "company", "Cleaned_Text": "text", "Created_At": "created_at"}, inplace=True)
    df["created_at"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y")
    df["Date"] = df["created_at"].dt.strftime("%Y/%m/%d")

    total_rows = len(df)
    success_count = 0
    to_insert = []

#BATCH NLP
for i, row in enumerate(df.itertuples(index=False), start=0):
    if i < start_index:
        continue

    if i % batch_size == 0:  
        batch_texts = []
        batch_indices = []
    
    company = row.company
    text = str(row.text)
    created_at_dt = row.created_at
    created_at = created_at_dt.strftime("%Y-%m-%d %H:%M:%S")
    date_str = created_at_dt.strftime("%Y/%m/%d")
    
    batch_texts.append(text)
    batch_indices.append(i)
    
    if len(batch_texts) == batch_size or i == len(df) - 1:
        try:
            batch_results = analyzer.batch_nlp(batch_texts)
            
            for idx, text_idx in enumerate(batch_indices):
                result = batch_results[idx]
                to_insert.append({
                    "company": df.iloc[text_idx].company,
                    "text": batch_texts[idx],
                    "created_at": df.iloc[text_idx].created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "Date": df.iloc[text_idx].created_at.strftime("%Y/%m/%d"),
                    "Positive": result[0],  # finbert_scores.get("positive", 0.0)
                    "Negative": result[1],  # finbert_scores.get("negative", 0.0)
                    "Neutral": result[2],   # finbert_scores.get("neutral", 0.0)
                    "Surprise": result[3],  # agg_emotions["surprise"]
                    "Joy": result[4],      # agg_emotions["joy"]
                    "Anger": result[5],     # agg_emotions["anger"]
                    "Fear": result[6],      # agg_emotions["fear"]
                    "Sadness": result[7],   # agg_emotions["sadness"]
                    "Disgust": result[8],   # agg_emotions["disgust"]
                    "Emotion Confidence": result[9],  # emotion_conf
                    "Intent Sentiment": result[10],    # intent_label
                    "Confidence": result[11]           # intent_conf
                })
                
            with open(bp_file, "w") as f:
                json.dump({"last_index": i + 1}, f)
                
        except Exception as e:
            print(f"Error processing}")

        if len(to_insert) >= 100:
            df_batch = pd.DataFrame(to_insert)
            table_name = f"sentiment_raw_data_{Path(file_name).stem.split('_')[-1]}"
            try:
                df_batch.to_sql(
                    name=table_name,
                    con=engine,
                    if_exists="append",
                    index=False,
                    schema="nlp"
                )
            except Exception as e:
                print(f"Batch insert failed at row {i}, retrying single inserts.")
                for j, item in enumerate(to_insert):
                    try:
                        pd.DataFrame([item]).to_sql(
                            name=table_name,
                            con=engine,
                            if_exists="append",
                            index=False,
                            schema="nlp"
                        )
                    except Exception as e2:
                        actual_row = i - len(to_insert) + j + 1
                        print(f"Single insert failed at row {actual_row}")
            success_count += len(to_insert)
            to_insert = []

            if success_count % 1000 == 0:
                print(f"Inserted {success_count} rows from {file_name}")

            with open(bp_file, "w") as f:
                json.dump({"last_index": i + 1}, f)

    if to_insert:
        df_batch = pd.DataFrame(to_insert)
        table_name = f"sentiment_raw_data_{Path(file_name).stem.split('_')[-1]}"
        try:
            df_batch.to_sql(
                name=table_name,
                con=engine,
                if_exists="append",
                index=False,
                schema="nlp"
            )
        except Exception as e:
            print(f"Final batch insert failed, retrying single inserts.")
            for item in to_insert:
                try:
                    pd.DataFrame([item]).to_sql(
                        name=table_name,
                        con=engine,
                        if_exists="append",
                        index=False,
                        schema="nlp"
                    )
                except Exception as e2:
                    print(f"Final single insert failed.")
        success_count += len(to_insert)
        with open(bp_file, "w") as f:
            json.dump({"last_index": total_rows}, f)

    print(f"Finished processing {file_name}, total inserted: {success_count} rows.")
