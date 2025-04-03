import pandas as pd
from sqlalchemy import create_engine

user = "postgres"
pw = "qf5214"
host = "134.122.167.14"
port = 5555
dbname = "QF5214"

db = create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{dbname}")

sql = """
    SELECT company, "Date", 
           CAST("Positive" AS FLOAT), CAST("Negative" AS FLOAT), CAST("Neutral" AS FLOAT), 
           CAST("Surprise" AS FLOAT), CAST("Joy" AS FLOAT), CAST("Anger" AS FLOAT), 
           CAST("Fear" AS FLOAT), CAST("Sadness" AS FLOAT), CAST("Disgust" AS FLOAT), 
           "Intent Sentiment",
           CAST("Emotion Confidence" AS FLOAT) AS "Emotion Confidence"
    FROM nlp.sentiment_raw_data
"""

df = pd.read_sql(sql, db)
df = df[df["Emotion Confidence"] >= 0.4]

dailystats = df.groupby(["company", "Date"]).agg({
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

dailystats.to_sql(
    name="sentiment_aggregated_data_filter",
    con=db,
    if_exists="append",
    index=False,
    schema="nlp"
)

print("Aggregated sentiment data written to sentiment_aggregated_data table.")
