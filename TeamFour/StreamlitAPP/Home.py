import streamlit as st

st.set_page_config(page_title="Stock Analysis App", page_icon="📈", layout="wide")

# Display Group Name in small font at top left
st.markdown("<h4 style='text-align: left; font-size: 14px;'>QF5214 Group 8</h4>", unsafe_allow_html=True)

# Display Main Header
st.markdown("<h1 style='text-align: center;'>From Sentiment to Strategy: Integrating NLP-Derived Factors into Quantitative Investment Models</h1>", unsafe_allow_html=True)

# Display Group Members in smaller italic font
st.markdown("""
*Group Members: Gao XuanRong, Hur Sinhaeng, Li LingYan, Liu Yang, Ren ZhiNan, Zhang YiChen, Zhou Zheng, Zhang Leyan, Lee Jiazhe, Mei Su*
""", unsafe_allow_html=True)

# Display Objective
st.markdown("""
### Objective
To extract market sentiment from textual data, quantify it into sentiment factors, and integrate these factors into a multi-factor investment model. We will build a real-time, interactive dashboard that visualizes all relevant data dynamically.
""")