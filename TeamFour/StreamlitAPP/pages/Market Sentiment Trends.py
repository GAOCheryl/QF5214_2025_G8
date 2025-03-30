import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, date
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
        SGT: {time_sgt} &nbsp;&nbsp;|&nbsp;&nbsp; New York: {time_ny}
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Market Sentiment Trends")

# --- DB Connection ---
host = "134.122.167.14"
port = "5555"
database = "QF5214"
user = "postgres"
password = "qf5214"

db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
engine = create_engine(db_url)

# --- Get latest available date ---
try:
    date_query = "SELECT MAX(\"Date\") AS latest_date FROM tradingstrategy.dailytrading"
    latest_date_result = pd.read_sql(date_query, engine)
    latest_date = latest_date_result["latest_date"].iloc[0]

    if latest_date:
        # Label: Testing Phase
        st.markdown(
            f"""
            <div style="text-align: center; color: #999; font-size: 16px; margin-top: -10px;">
                ⚠️ <i>Testing Phase – using latest available data from <b>{latest_date}</b></i>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Fetch tickers from that latest date
        ticker_query = f"""
            SELECT DISTINCT "Ticker"
            FROM tradingstrategy.dailytrading
            WHERE "Date" = '{latest_date}'
            LIMIT 5
        """
        tickers_df = pd.read_sql(ticker_query, engine)
        available_tickers = tickers_df["Ticker"].tolist()

        if available_tickers:
            selected_company = st.selectbox("Select a Company", available_tickers)
        else:
            st.warning("No tickers found for the latest available date.")
            selected_company = None
    else:
        st.warning("No available trading data found in the database.")
        selected_company = None

except Exception as e:
    st.error(f"Database error: {e}")
    selected_company = None

# --- Sentiment Chart (Mocked) ---
if selected_company:
    company_seed = abs(hash(selected_company)) % (2**32)
    np.random.seed(company_seed)
    sentiment_data = {
        "Time": ["T1", "T2", "T3", "T4", "T5"],
        "Joy": np.random.uniform(0.3, 0.9, 5),
        "Optimism": np.random.uniform(0.4, 0.95, 5),
        "Anger": np.random.uniform(0.1, 0.4, 5),
        "Sadness": np.random.uniform(0.1, 0.3, 5),
        "Fear": np.random.uniform(0.05, 0.25, 5)
    }
    df_sentiment = pd.DataFrame(sentiment_data)
    df_melted = df_sentiment.melt(id_vars=["Time"], var_name="Sentiment", value_name="Score")

    st.subheader(f"Sentiment Trends Over Time: {selected_company}")
    fig = px.line(
        df_melted, 
        x="Time", y="Score", color="Sentiment",
        markers=True,
        template="simple_white",
        color_discrete_map={
            "Joy": "#6495ED",
            "Optimism": "#ADD8E6",
            "Anger": "#F4C2C2",
            "Sadness": "#D8BFD8",
            "Fear": "#D2691E"
        }
    )
    fig.update_layout(yaxis_title="Sentiment Score", xaxis_title="Time")
    st.plotly_chart(fig, use_container_width=True)
