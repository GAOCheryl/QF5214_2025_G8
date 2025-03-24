import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

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

# Generate word frequencies uniquely for each company
topics = ["Tech Stocks", "Inflation", "Federal Reserve", "Interest Rates", "Oil Prices", "Recession", "Earnings", "GDP", "Crypto", "Unemployment"]
np.random.seed(company_seed + 1)
frequencies = np.random.randint(10, 60, size=len(topics))
word_freq = dict(zip(topics, frequencies))

# Word Cloud in pastel color theme
def pastel_color_func(*args, **kwargs):
    pastel_colors = ["#AEC6CF", "#FFDAB9", "#CBAACB", "#FFB347", "#BFD8B8", "#F4C2C2", "#F0E68C"]
    return np.random.choice(pastel_colors)

wordcloud = WordCloud(
    width=800, height=400,
    background_color="white",
    color_func=pastel_color_func
).generate_from_frequencies(word_freq)

# Display Word Cloud
st.subheader(f"Financial News Topic Frequency: {selected_company}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
