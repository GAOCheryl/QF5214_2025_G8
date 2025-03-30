import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from sqlalchemy import create_engine

# --- Page Config ---
st.set_page_config(page_title="Market Sentiment Trends", layout="wide")
st_autorefresh(interval=60000, key="refresh_time")

# --- Time Zones ---
sgt = pytz.timezone("Asia/Singapore")
ny = pytz.timezone("America/New_York")
now_sgt = datetime.now(sgt)
now_ny = datetime.now(ny)
date_today = now_sgt.strftime("%A, %d %B %Y")
time_sgt = now_sgt.strftime("%H:%M")
time_ny = now_ny.strftime("%H:%M")

# --- Date & Time Display ---
st.markdown(
    f"""
    <div style="text-align: center; padding: 10px 0; font-size: 18px; color: #444;">
        <b>{date_today}</b><br>
        Singapore: {time_sgt} &nbsp;&nbsp;|&nbsp;&nbsp; New York: {time_ny}
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Market Sentiment Trends")

# Spacer
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# --- DB Connection ---
host = "134.122.167.14"
port = "5555"
database = "QF5214"
user = "postgres"
password = "qf5214"

db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
engine = create_engine(db_url)

# --- Get latest available date from trading table ---
try:
    date_query = "SELECT MAX(\"Date\") AS latest_date FROM tradingstrategy.dailytrading"
    latest_date_result = pd.read_sql(date_query, engine)
    latest_trading_date = latest_date_result["latest_date"].iloc[0]
    latest_trading_date = pd.to_datetime(latest_trading_date).date()

    if latest_trading_date:
        sentiment_date_obj = latest_trading_date - timedelta(days=1)
        sentiment_date_str = sentiment_date_obj.strftime('%Y/%m/%d')

        st.markdown(
            f"""
            <div style="text-align: center; color: #999; font-size: 16px; margin-top: -10px;">
                ⚠️ <i>Testing Phase – trading data from <b>{latest_trading_date}</b>, sentiment from <b>{sentiment_date_str}</b></i>
            </div>
            """,
            unsafe_allow_html=True
        )

        ticker_query = f"""
            SELECT DISTINCT \"Ticker\"
            FROM tradingstrategy.dailytrading
            WHERE \"Date\" = '{latest_trading_date}'
            LIMIT 5
        """
        tickers_df = pd.read_sql(ticker_query, engine)
        available_tickers = tickers_df["Ticker"].tolist()

        if available_tickers:
            selected_company = st.selectbox("Select a Company", available_tickers)
        else:
            st.warning("No tickers found for the latest available trading date.")
            selected_company = None
    else:
        st.warning("No available trading data found in the database.")
        selected_company = None

except Exception as e:
    st.error(f"Database error: {e}")
    selected_company = None

# --- Combine sentiment data from multiple sources ---
def load_combined_sentiment_data(company: str, start_date: str, end_date: str):
    tables = ["nlp.sentiment_aggregated_data", "nlp.sentiment_aggregated_live", "nlp.sentiment_aggregated_newdate"]
    combined_df = pd.DataFrame()
    
    for table in tables:
        query = f"""
            SELECT \"Date\", \"company\", \"Surprise\", \"Joy\", \"Anger\", \"Fear\", \"Sadness\", \"Disgust\"
            FROM {table}
            WHERE (\"company\" = '{company}' OR \"company\" = '${company}')
            AND \"Date\" >= '{start_date}' AND \"Date\" <= '{end_date}'
        """
        try:
            df = pd.read_sql(query, engine)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            st.warning(f"Failed to fetch from {table}: {e}")

    if not combined_df.empty:
        combined_df.drop_duplicates(subset=["Date", "company"], keep="last", inplace=True)
        combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors='coerce')
        combined_df = combined_df.dropna(subset=["Date"])
        return combined_df.sort_values("Date")
    return pd.DataFrame()

# --- Timeframe Toggle & Dynamic Sentiment Trend Chart ---
try:
    st.markdown(
        f"<h4 style='margin-top: 40px; margin-bottom: 5px;'>Sentiment Trend Visualization</h4>",
        unsafe_allow_html=True
    )

    with st.container():
        timeframe = st.radio(
            "",
            ["1W", "1M"],
            index=1,
            horizontal=True,
            label_visibility="collapsed"
        )

    days_back = 5 if timeframe == "1W" else 30
    start_date = (sentiment_date_obj - timedelta(days=days_back)).strftime('%Y/%m/%d')
    end_date = sentiment_date_str

    history_df = load_combined_sentiment_data(selected_company, start_date, end_date)

    if not history_df.empty:
        sentiment_cols = ["Surprise", "Joy", "Anger", "Fear", "Sadness", "Disgust"]
        for col in sentiment_cols:
            history_df[col] = pd.to_numeric(history_df[col], errors="coerce").round(3)

        history_df = history_df.dropna(subset=sentiment_cols, how="all")

        clean_df = history_df.melt(
            id_vars="Date",
            value_vars=sentiment_cols,
            var_name="Sentiment",
            value_name="Score"
        ).dropna(subset=["Score"])

        max_score = clean_df["Score"].max()

        fig = px.line(
            clean_df,
            x="Date", y="Score", color="Sentiment",
            markers=True,
            template="simple_white",
            color_discrete_map={
                "Surprise": "#FFDAB9",
                "Joy": "#AEC6CF",
                "Anger": "#F4C2C2",
                "Fear": "#D8BFD8",
                "Sadness": "#FFE4E1",
                "Disgust": "#BFD8B8"
            }
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            yaxis=dict(tickformat=".2f", range=[0, max_score + 0.05]),
            legend_title="",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment data available for this timeframe.")

except Exception as e:
    st.error(f"Error loading sentiment chart: {e}")