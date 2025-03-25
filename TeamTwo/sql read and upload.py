import pandas as pd
from sqlalchemy import create_engine, text as sql_text
from datetime import datetime

analyzer = SentimentEmotionAnalyzer()

# PostgreSQL connection settings
db_user = "postgres"
db_password = "qf5214"
db_host = "134.122.167.14"
db_port = 5555
db_name = "QF5214"

# Create engine
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

target_companies = ["ADP"]

for target_company in target_companies:
    to_insert = []
    inserted_count = 0

    existing_df = pd.read_sql(
        f"""
        SELECT company, text, created_at
        FROM nlp.sentiment_raw_data
        WHERE company = '{target_company}'
        """,
        engine
    )
    existing_set = set(
        (row["company"], row["text"], row["created_at"].strftime("%Y-%m-%d %H:%M:%S"))
        for _, row in existing_df.iterrows()
    )

    data_query = f"""
        SELECT company, text, created_at
        FROM datacollection.x_batch
        WHERE company = '{target_company}'
        ORDER BY created_at ASC
    """
    df = pd.read_sql(data_query, engine)

    if df.empty:
        print(f"No data found for company {target_company}.")
        continue

    for row in df.itertuples(index=False):
        company = row.company
        text = str(row.text)
        created_at_dt = pd.to_datetime(row.created_at)
        created_at = created_at_dt.strftime("%Y-%m-%d %H:%M:%S")
        date_str = created_at_dt.strftime("%Y/%m/%d")

        if (company, text, created_at) in existing_set:
            continue 

        result = analyzer.nlp(text)

        to_insert.append({
            "company": company,
            "text": text,
            "created_at": created_at,
            "Date": date_str,
            "Positive": result[0],
            "Negative": result[1],
            "Neutral": result[2],
            "Surprise": result[3],
            "Joy": result[4],
            "Anger": result[5],
            "Fear": result[6],
            "Sadness": result[7],
            "Disgust": result[8],
            "Emotion Confidence": result[9],
            "Intent Sentiment": result[10],
            "Confidence": result[11]
        })

        if len(to_insert) >= 100:
            pd.DataFrame(to_insert).to_sql(
                name="sentiment_raw_data",
                con=engine,
                if_exists="append",
                index=False,
                schema="nlp"
            )
            inserted_count += len(to_insert)
            to_insert = []

    if to_insert:
        pd.DataFrame(to_insert).to_sql(
            name="sentiment_raw_data",
            con=engine,
            if_exists="append",
            index=False,
            schema="nlp"
        )
        inserted_count += len(to_insert)

    print(f"Inserted {inserted_count} new records for {target_company}.")