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
                ⚠️ <i>Testing Phase – trading data from <b>{latest_trading_date}</b>, sentiment from <b>{sentiment_date_str}</b></i>
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

# --- Sentiment Chart (from sentiment_aggregated_data) ---
if selected_company:
    try:
        sentiment_query = f"""
            SELECT "company", "Date", "Positive", "Negative", "Neutral",
                   "Surprise", "Joy", "Anger", "Fear", "Sadness", "Disgust",
                   "Intent Sentiment"
            FROM nlp.sentiment_aggregated_data
            WHERE ("company" = '{selected_company}' OR "company" = '${selected_company}')
            AND "Date" = '{sentiment_date_str}'
            LIMIT 1
        """
        sentiment_df = pd.read_sql(sentiment_query, engine)

        if not sentiment_df.empty:
            # Extract the overall intent sentiment
            overall_sentiment = sentiment_df["Intent Sentiment"].iloc[0]

           # Capitalize first letter
            overall_sentiment = overall_sentiment.capitalize()

            # Background color mapping for the sentiment word only
            bg_color_map = {
                "Buy": "#b8f2c2",      # pastel green
                "Sell": "#f6b6b6",     # pastel red
                "Neutral": "#fff2b2"   # pastel yellow
            }
            bg_color = bg_color_map.get(overall_sentiment, "#eeeeee")

            # Styled sentiment badge centered
            st.markdown(
                f"""
                <div style="text-align: left; margin-top: 10px; margin-bottom: 20px;">
                    <span style="font-size: 18px; font-weight: 600;">
                        Overall Sentiment:
                        <span style="
                            background-color: {bg_color};
                            padding: 6px 14px;
                            border-radius: 12px;
                            color: black;
                        ">
                            {overall_sentiment}
                        </span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )



            # Select and format sentiment score columns
            display_df = sentiment_df[[
                "Positive", "Negative", "Neutral",
                "Surprise", "Joy", "Anger", "Fear", "Sadness", "Disgust"
            ]].T.reset_index()
            display_df.columns = ["Sentiment", "Score"]
            display_df["Score"] = pd.to_numeric(display_df["Score"], errors="coerce").round(3)


            # Show as a simple clean table
            # Show as a smaller centered table
            st.markdown(
                """
                <div style="display: flex; justify-content: center; margin-top: 10px;">
                    <div style="width: 400px;">
                """,
                unsafe_allow_html=True
            )

            st.dataframe(display_df.set_index("Sentiment"), use_container_width=False)

            st.markdown("</div></div>", unsafe_allow_html=True)


        else:
            st.warning(f"No sentiment data found for {selected_company} on {sentiment_date_str}.")
    except Exception as e:
        st.error(f"Error loading sentiment data: {e}")

# --- Historical Sentiment Trend Chart (1 month) ---
try:
    # Calculate historical date (30 days before T-1)
    historical_start_date = (sentiment_date_obj - timedelta(days=30)).strftime('%Y/%m/%d')

    history_query = f"""
        SELECT "Date", "Surprise", "Joy", "Anger", "Fear", "Sadness", "Disgust"
        FROM nlp.sentiment_aggregated_data
        WHERE ("company" = '{selected_company}' OR "company" = '${selected_company}')
        AND "Date" BETWEEN '{historical_start_date}' AND '{sentiment_date_str}'
        ORDER BY "Date" ASC
    """

    history_df = pd.read_sql(history_query, engine)

    if not history_df.empty:
        # Convert dates
        history_df["Date"] = pd.to_datetime(history_df["Date"], format="%Y/%m/%d")

        # Melt for Plotly
        history_melted = history_df.melt(id_vars="Date", var_name="Sentiment", value_name="Score")

        st.markdown(
            "<h4 style='margin-top: 40px;'>Sentiment Trend (Past 30 Days)</h4>",
            unsafe_allow_html=True
        )

        fig = px.line(
            history_melted,
            x="Date", y="Score", color="Sentiment",
            template="simple_white",
            markers=True,
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
            legend_title="",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical sentiment data available for this company.")
except Exception as e:
    st.error(f"Error loading sentiment trend data: {e}")
