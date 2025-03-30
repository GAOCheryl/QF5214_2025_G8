import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from streamlit_autorefresh import st_autorefresh

# --- Page Config ---
st.set_page_config(page_title="Sentiment & Stock Performance", layout="wide")
st_autorefresh(interval=60000, key="refresh_time")

# --- Time Display (SGT & NY) ---
sgt = pytz.timezone("Asia/Singapore")
ny = pytz.timezone("America/New_York")
now_sgt = datetime.now(sgt)
now_ny = datetime.now(ny)

date_today = now_sgt.strftime("%A, %d %B %Y")
time_sgt = now_sgt.strftime("%H:%M")
time_ny = now_ny.strftime("%H:%M")

st.markdown(
    f"""
    <div style="text-align: center; padding: 5px 0; font-size: 16px; color: #444;">
        <b>{date_today}</b><br>
        Singapore: {time_sgt} &nbsp;&nbsp;|&nbsp;&nbsp; New York: {time_ny}
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Sentiment & Stock Performance Relation")

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
        # Calculate T - 1 for sentiment
        sentiment_date_obj = latest_trading_date - timedelta(days=1)
        sentiment_date_str = sentiment_date_obj.strftime('%Y/%m/%d')  # <-- Use slashes for sentiment table

        # Label: Testing Phase
        st.markdown(
            f"""
            <div style="text-align: center; color: #999; font-size: 16px; margin-top: -10px;">
                ⚠️ <i>Testing Phase – trading data from <b>{latest_trading_date}</b>, sentiment from <b>{sentiment_date_str} and backwards</b></i>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Fetch tickers from latest trading date
        ticker_query = f"""
            SELECT DISTINCT "Ticker"
            FROM tradingstrategy.dailytrading
            WHERE "Date" = '{latest_trading_date}'
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

# --- Sentiment vs Return Scatter Plot (1 Year) ---
try:
    st.markdown(
        f"<h4 style='margin-top: 40px;'>Sentiment Score vs. Daily Return (1 Year) – {selected_company}</h4>",
        unsafe_allow_html=True
    )

    sentiment_return_query = f"""
        SELECT 
            s."Date", 
            s."company", 
            s."Positive", 
            s."Negative", 
            s."Neutral", 
            d."Close",
            d."Ticker"
        FROM nlp.sentiment_aggregated_data s
        JOIN datacollection.stock_data d 
            ON s."Date"::date = d."Date"
        WHERE 
            (s."company" = '{selected_company}' OR s."company" = '${selected_company}')
            AND d."Ticker" = '{selected_company}'
            AND s."Date"::date >= CURRENT_DATE - INTERVAL '1 year'
        ORDER BY s."Date"
    """
    joined_df = pd.read_sql(sentiment_return_query, engine)

    if not joined_df.empty:
        joined_df.sort_values("Date", inplace=True)
        joined_df["Close"] = pd.to_numeric(joined_df["Close"], errors="coerce")
        joined_df["Return"] = joined_df["Close"].pct_change().round(3)

        # Melt for scatter plot
        melted_df = joined_df.melt(
            id_vars=["Date", "Return"],
            value_vars=["Positive", "Negative", "Neutral"],
            var_name="Sentiment",
            value_name="Score"
        ).dropna()

        melted_df["Score"] = pd.to_numeric(melted_df["Score"], errors="coerce").round(3)
        melted_df = melted_df.dropna(subset=["Score", "Return"])

        fig = px.scatter(
            melted_df,
            x="Score",
            y="Return",
            color="Sentiment",
            opacity=0.6,
            template="simple_white",
            color_discrete_map={
                "Positive": "#96C38D",  # medium aquamarine (darker pastel green)
                "Negative": "#e57373",  # soft crimson/pastel red
                "Neutral": "#ffdd57"    # darker sunflower pastel yellow
            }

        )
        fig.update_layout(
            xaxis_title="Sentiment Score",
            yaxis_title="Daily Return",
            height=500,
            legend_title=""
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No data available for this company in the past year.")

except Exception as e:
    st.error(f"Error loading sentiment-return relationship: {e}")
