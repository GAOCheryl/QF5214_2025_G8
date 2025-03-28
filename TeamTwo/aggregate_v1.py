import pandas as pd
from sqlalchemy import create_engine

db_user = "postgres"
db_password = "qf5214"
db_host = "134.122.167.14"
db_port = 5555
db_name = "QF5214"

engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

query = """
    SELECT company, "Date", 
           CAST("Positive" AS FLOAT), CAST("Negative" AS FLOAT), CAST("Neutral" AS FLOAT), 
           CAST("Surprise" AS FLOAT), CAST("Joy" AS FLOAT), CAST("Anger" AS FLOAT), 
           CAST("Fear" AS FLOAT), CAST("Sadness" AS FLOAT), CAST("Disgust" AS FLOAT), 
           "Intent Sentiment"
    FROM nlp.sentiment_raw_data
"""
df = pd.read_sql(query, engine)

aggregated_df = df.groupby(["company", "Date"]).agg({
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

aggregated_df.to_sql(
    name="sentiment_aggregated_data",
    con=engine,
    if_exists="append",
    index=False,
    schema="nlp"
)

print("Aggregated sentiment data written to sentiment_aggregated_data table.")
