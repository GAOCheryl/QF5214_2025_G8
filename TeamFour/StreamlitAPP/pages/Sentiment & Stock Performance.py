import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="Sentiment & Stock Performance", layout="wide")
st.title("Sentiment & Stock Performance Relation")

# Sample Companies
nasdaq_companies = [
    "Apple", "Microsoft", "Amazon", "NVIDIA", "Meta", 
    "Tesla", "Alphabet", "Intel", "PepsiCo", "Cisco"
]

# Dropdown to select company
selected_company = st.selectbox("Select a Company", nasdaq_companies)

# Generate Mock Data: Different for each company (for demo purposes)
company_seed = abs(hash(selected_company)) % (2**32)  # consistent but unique seed
np.random.seed(company_seed)

# Generate company-specific sentiment-return relationship
sentiment_scores = np.random.uniform(-1, 1, 300)
company_bias = np.random.uniform(-0.005, 0.015)  # simulate bias per company
daily_returns = sentiment_scores * company_bias + np.random.normal(0, 0.01, 300)

df_scatter = pd.DataFrame({
    "Sentiment Score": sentiment_scores,
    "Daily Stock Return (%)": daily_returns
})


# Scatter Plot: Sentiment vs. Return
st.subheader(f"Sentiment Score vs. Daily Return: {selected_company}")
fig1 = px.scatter(df_scatter, x="Sentiment Score", y="Daily Stock Return (%)",
                 opacity=0.6, color_discrete_sequence=["#F4C542"], template="simple_white")
fig1.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
fig1.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
st.plotly_chart(fig1, use_container_width=True)

# Bar Plot: Emotion-Specific Average Return (mock data)
emotion_bias = np.random.uniform(-0.01, 0.015, 4)  # unique return profile per company
df_emotions = pd.DataFrame({
    "Emotion": ["Joy", "Optimism", "Pessimism", "Fear"],
    "Average Daily Stock Return (%)": emotion_bias
})


st.subheader(f"Emotion-Specific Sentiment vs. Stock Return: {selected_company}")
fig2 = px.bar(df_emotions, x="Emotion", y="Average Daily Stock Return (%)",
              color="Emotion",
              color_discrete_map={
                  "Joy": "#6495ED",
                  "Optimism": "#ADD8E6",
                  "Pessimism": "#F4C2C2",
                  "Fear": "#D2691E"
              },
              template="simple_white")
fig2.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
st.plotly_chart(fig2, use_container_width=True)
