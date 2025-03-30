import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from streamlit_autorefresh import st_autorefresh


# Page Config
st.set_page_config(page_title="Market Sentiment Trends", layout="wide")
 
# Auto-refresh every 60 seconds (60000 ms)
st_autorefresh(interval=60000, key="refresh_time")


# Get current times
sgt = pytz.timezone("Asia/Singapore")
ny = pytz.timezone("America/New_York")
now_sgt = datetime.now(sgt)
now_ny = datetime.now(ny)

# Format date and time
date_today = now_sgt.strftime("%A, %d %B %Y")
time_sgt = now_sgt.strftime("%H:%M")
time_ny = now_ny.strftime("%H:%M")

# Centered, minimal date/time display
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

# Sample Companies
nasdaq_companies = [
    "Apple", "Microsoft", "Amazon", "NVIDIA", "Meta", 
    "Tesla", "Alphabet", "Intel", "PepsiCo", "Cisco"
]

# Dropdown to select company
selected_company = st.selectbox("Select a Company", nasdaq_companies)

# Generate company-specific sentiment data (mocked with unique seeds)
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

# Interactive Line Chart (Pastel Color Theme from Page 3)
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
