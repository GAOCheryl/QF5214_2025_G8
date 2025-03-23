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

# Report Link Section (replacing Objective)
st.markdown("""
<hr style='border: 1px solid #ccc; margin-top: 30px; margin-bottom: 10px;'>

<h4 style='text-align: center; color: #333;'>📄 Project Report</h4>
<p style='text-align: center; font-size: 16px;'>
    <a href='https://your-report-link.com' target='_blank' style='text-decoration: none; color: #1a73e8;'>
        Click here to view our full project report
    </a>
</p>

<hr style='border: 1px solid #ccc; margin-top: 10px;'>
""", unsafe_allow_html=True)
