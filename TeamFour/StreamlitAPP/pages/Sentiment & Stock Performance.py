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

st.title("Sentiment & Stock Performance")

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

# --- Get latest available date and tickers (long + short) ---
try:
    date_query = "SELECT MAX(\"Date\") AS latest_date FROM tradingstrategy.dailytrading"
    latest_date_result = pd.read_sql(date_query, engine)
    latest_trading_date = latest_date_result["latest_date"].iloc[0]
    latest_trading_date = pd.to_datetime(latest_trading_date).date()
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
        SELECT DISTINCT "Ticker", "Position_Type"
        FROM tradingstrategy.dailytrading
        WHERE "Date" = '{latest_trading_date}'
        LIMIT 10
    """
    tickers_df = pd.read_sql(ticker_query, engine)
    available_tickers = tickers_df["Ticker"].tolist()
    ticker_position_map = dict(zip(tickers_df["Ticker"], tickers_df["Position_Type"]))

    if available_tickers:
        selected_company = st.selectbox("Select a Company", available_tickers)
        position_type = ticker_position_map.get(selected_company, "Unknown")
    else:
        st.warning("No tickers found for the latest available trading date.")
        selected_company = None
        position_type = None

except Exception as e:
    st.error(f"Database error: {e}")
    selected_company = None
    position_type = None

# --- Combine sentiment data from multiple sources ---
def load_combined_sentiment_data(company: str, start_date: str, end_date: str):
    tables = ["nlp.sentiment_aggregated_data", "nlp.sentiment_aggregated_live", "nlp.sentiment_aggregated_newdate"]
    combined_df = pd.DataFrame()

    for table in tables:
        query = f"""
            SELECT "Date", "company", "Surprise", "Joy", "Anger", "Fear", "Sadness", "Disgust", "Positive", "Negative", "Neutral", "Intent Sentiment"
            FROM {table}
            WHERE ("company" = '{company}' OR "company" = '${company}')
            AND "Date" >= '{start_date}' AND "Date" <= '{end_date}'
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


st.markdown(f"<h5 style='margin-top: 20px;'> Selected Company: {selected_company} ({position_type})</h5>", unsafe_allow_html=True)


# --- Sentiment vs Return Scatter Plot (1 Year) ---
try:
    st.markdown(
        f"<h4 style='margin-top: 40px;'>Sentiment Score vs. Daily Return</h4>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-size: 18px; color: #666; margin-top: -12px;'>Data Range: 1 Year</p>",
        unsafe_allow_html=True
    )


    sentiment_sources = [
        "nlp.sentiment_aggregated_data",
        "nlp.sentiment_aggregated_live",
        "nlp.sentiment_aggregated_newdate"
    ]

    sentiment_data = pd.DataFrame()
    for source in sentiment_sources:
        query = f"""
            SELECT s."Date", s."company", s."Positive", s."Negative", s."Neutral"
            FROM {source} s
            WHERE (s."company" = '{selected_company}' OR s."company" = '${selected_company}')
            AND s."Date"::date >= CURRENT_DATE - INTERVAL '1 year'
        """
        try:
            df = pd.read_sql(query, engine)
            sentiment_data = pd.concat([sentiment_data, df], ignore_index=True)
        except:
            pass

    sentiment_data.drop_duplicates(subset=["Date", "company"], keep="last", inplace=True)
    sentiment_data["Date"] = pd.to_datetime(sentiment_data["Date"], errors="coerce")
    sentiment_data = sentiment_data.dropna(subset=["Date"])

    stock_query = f"""
        SELECT "Date", "Ticker", "Close"
        FROM datacollection.stock_data
        WHERE "Ticker" = '{selected_company}'
        AND "Date" >= CURRENT_DATE - INTERVAL '1 year'
    """
    stock_df = pd.read_sql(stock_query, engine)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")

    # Merge on Date
    merged_df = pd.merge(sentiment_data, stock_df, left_on="Date", right_on="Date")
    merged_df.sort_values("Date", inplace=True)
    merged_df["Close"] = pd.to_numeric(merged_df["Close"], errors="coerce")
    merged_df["Return"] = merged_df["Close"].pct_change().round(3)

    # Melt for plotting
    melted_df = merged_df.melt(
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
            "Positive": "#96C38D",
            "Negative": "#e57373",
            "Neutral": "#ffdd57"
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

except Exception as e:
    st.error(f"Error loading sentiment-return relationship: {e}")


# import streamlit as st
# import plotly.express as px
# import pandas as pd
# import numpy as np
# import pytz
# from datetime import datetime, timedelta
# from sqlalchemy import create_engine
# from streamlit_autorefresh import st_autorefresh

# # --- Page Config ---
# st.set_page_config(page_title="Sentiment & Stock Performance", layout="wide")
# st_autorefresh(interval=60000, key="refresh_time")

# # --- Time Display (SGT & NY) ---
# sgt = pytz.timezone("Asia/Singapore")
# ny = pytz.timezone("America/New_York")
# now_sgt = datetime.now(sgt)
# now_ny = datetime.now(ny)

# date_today = now_sgt.strftime("%A, %d %B %Y")
# time_sgt = now_sgt.strftime("%H:%M")
# time_ny = now_ny.strftime("%H:%M")

# st.markdown(
#     f"""
#     <div style="text-align: center; padding: 5px 0; font-size: 16px; color: #444;">
#         <b>{date_today}</b><br>
#         Singapore: {time_sgt} &nbsp;&nbsp;|&nbsp;&nbsp; New York: {time_ny}
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# st.title("Sentiment & Stock Performance Relation")

# # --- DB Connection ---
# host = "134.122.167.14"
# port = "5555"
# database = "QF5214"
# user = "postgres"
# password = "qf5214"
# db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
# engine = create_engine(db_url)

# # --- Latest trading date & ticker selection ---
# try:
#     date_query = "SELECT MAX(\"Date\") AS latest_date FROM tradingstrategy.dailytrading"
#     latest_date_result = pd.read_sql(date_query, engine)
#     latest_trading_date = latest_date_result["latest_date"].iloc[0]
#     latest_trading_date = pd.to_datetime(latest_trading_date).date()

#     if latest_trading_date:
#         sentiment_date_obj = latest_trading_date - timedelta(days=1)
#         sentiment_date_str = sentiment_date_obj.strftime('%Y/%m/%d')

#         st.markdown(
#             f"""
#             <div style="text-align: center; color: #999; font-size: 16px; margin-top: -10px;">
#                 ⚠️ <i>Testing Phase – trading data from <b>{latest_trading_date}</b>, sentiment from <b>{sentiment_date_str} and backwards</b></i>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#         ticker_query = f"""
#             SELECT DISTINCT "Ticker"
#             FROM tradingstrategy.dailytrading
#             WHERE "Date" = '{latest_trading_date}'
#             LIMIT 5
#         """
#         tickers_df = pd.read_sql(ticker_query, engine)
#         available_tickers = tickers_df["Ticker"].tolist()

#         if available_tickers:
#             selected_company = st.selectbox("Select a Company", available_tickers)
#         else:
#             st.warning("No tickers found for the latest available trading date.")
#             selected_company = None
#     else:
#         st.warning("No available trading data found in the database.")
#         selected_company = None

# except Exception as e:
#     st.error(f"Database error: {e}")
#     selected_company = None

# # --- Sentiment vs Return Scatter Plot (1 Year) from multiple sources ---
# def load_combined_sentiment_scores(company: str):
#     tables = [
#         "nlp.sentiment_aggregated_data",
#         "nlp.sentiment_aggregated_live",
#         "nlp.sentiment_aggregated_newdate"
#     ]
#     combined_df = pd.DataFrame()

#     for table in tables:
#         query = f"""
#             SELECT "Date", "company", "Positive", "Negative", "Neutral"
#             FROM {table}
#             WHERE ("company" = '{company}' OR "company" = '${company}')
#               AND "Date" >= CURRENT_DATE - INTERVAL '1 year'
#         """
#         try:
#             df = pd.read_sql(query, engine)
#             combined_df = pd.concat([combined_df, df], ignore_index=True)
#         except Exception as e:
#             st.warning(f"Failed to fetch from {table}: {e}")

#     if not combined_df.empty:
#         combined_df.drop_duplicates(subset=["Date", "company"], keep="last", inplace=True)
#         combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors='coerce')
#         return combined_df.dropna(subset=["Date"])
#     return pd.DataFrame()

# if selected_company:
#     try:
#         st.markdown(
#             f"<h4 style='margin-top: 40px;'>Sentiment Score vs. Daily Return (1 Year) – {selected_company}</h4>",
#             unsafe_allow_html=True
#         )

#         # Load sentiment data from multiple tables
#         sentiment_df = load_combined_sentiment_scores(selected_company)

#         # Load stock close prices
#         price_query = f"""
#             SELECT "Date", "Close", "Ticker"
#             FROM datacollection.stock_data
#             WHERE "Ticker" = '{selected_company}'
#               AND "Date" >= CURRENT_DATE - INTERVAL '1 year'
#         """
#         price_df = pd.read_sql(price_query, engine)
#         price_df["Date"] = pd.to_datetime(price_df["Date"], errors='coerce')
#         price_df = price_df.dropna(subset=["Date"])

#         if not sentiment_df.empty and not price_df.empty:
#             merged = pd.merge(sentiment_df, price_df, on="Date")
#             merged.sort_values("Date", inplace=True)
#             merged["Close"] = pd.to_numeric(merged["Close"], errors="coerce")
#             merged["Return"] = merged["Close"].pct_change().round(3)

#             melted_df = merged.melt(
#                 id_vars=["Date", "Return"],
#                 value_vars=["Positive", "Negative", "Neutral"],
#                 var_name="Sentiment",
#                 value_name="Score"
#             ).dropna()

#             melted_df["Score"] = pd.to_numeric(melted_df["Score"], errors="coerce").round(3)
#             melted_df = melted_df.dropna(subset=["Score", "Return"])

#             fig = px.scatter(
#                 melted_df,
#                 x="Score",
#                 y="Return",
#                 color="Sentiment",
#                 opacity=0.6,
#                 template="simple_white",
#                 color_discrete_map={
#                     "Positive": "#96C38D",
#                     "Negative": "#e57373",
#                     "Neutral": "#ffdd57"
#                 }
#             )
#             fig.update_layout(
#                 xaxis_title="Sentiment Score",
#                 yaxis_title="Daily Return",
#                 height=500,
#                 legend_title=""
#             )
#             fig.add_hline(y=0, line_dash="dot", line_color="gray")
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("Insufficient data to display sentiment-return relationship.")

#     except Exception as e:
#         st.error(f"Error loading sentiment-return relationship: {e}")
