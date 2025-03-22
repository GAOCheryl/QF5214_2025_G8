import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Market Sentiment Trends", layout="wide")
st.title("Market Sentiment Trends")

# Sample Companies
nasdaq_companies = [
    "Apple", "Microsoft", "Amazon", "NVIDIA", "Meta", 
    "Tesla", "Alphabet", "Intel", "PepsiCo", "Cisco"
]

# Dropdown to select company
selected_company = st.selectbox("Select a Company", nasdaq_companies)

# Simulate sentiment data for each company (in a real app, you'd load actual data)
sentiment_data = {
    "Time": ["T1", "T2", "T3", "T4", "T5"],
    "Joy": [0.4, 0.5, 0.6, 0.7, 0.8],
    "Optimism": [0.6, 0.7, 0.75, 0.8, 0.85],
    "Anger": [0.2, 0.3, 0.2, 0.25, 0.3],
    "Sadness": [0.1, 0.15, 0.2, 0.18, 0.22],
    "Fear": [0.05, 0.08, 0.1, 0.12, 0.15]
}
df_sentiment = pd.DataFrame(sentiment_data)

df_melted = df_sentiment.melt(id_vars=["Time"], var_name="Sentiment", value_name="Score")

# Interactive Line Chart (Minimalist Theme)
st.subheader(f"Sentiment Trends Over Time: {selected_company}")
fig = px.line(df_melted, x="Time", y="Score", color="Sentiment",
              markers=True,
              template="simple_white",
              color_discrete_map={"Joy": "#4682B4", "Optimism": "#1E90FF", "Anger": "#DC143C", "Sadness": "#87CEFA", "Fear": "#FF4500"})
fig.update_layout(yaxis_title="Sentiment Score", xaxis_title="Time")
st.plotly_chart(fig, use_container_width=True)

# Word Cloud for Financial News Topics (mocked per company)
st.subheader(f"Financial News Topic Frequency: {selected_company}")
word_freq = {
    "Tech Stocks": 50, "Inflation": 40, "Federal Reserve": 35, "Interest Rates": 30,
    "Oil Prices": 25, "Recession": 22, "Earnings": 20, "GDP": 18, "Crypto": 15, "Unemployment": 12
}
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

# Display Word Cloud
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
