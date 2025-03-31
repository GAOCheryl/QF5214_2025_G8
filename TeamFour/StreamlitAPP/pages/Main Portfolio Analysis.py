from datetime import datetime, timedelta
import sys
import subprocess
import streamlit as st
import os
from streamlit_autorefresh import st_autorefresh
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

st.set_page_config(page_title="Portfolio Analysis", layout="wide")
st_autorefresh(interval=60000, key="refresh_time")

import pytz

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
    <div style="text-align: center; padding: 5px 0; font-size: 16px; color: #444;">
        <b>{date_today}</b><br>
        Singapore: {time_sgt} &nbsp;&nbsp;|&nbsp;&nbsp; New York: {time_ny}
    </div>
    """,
    unsafe_allow_html=True
)
 

st.title("Portfolio Analysis")

# Add backtest button
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Run Latest Backtest", help="Click to run backtest analysis with latest data"):
        st.info("Running backtest analysis, please wait...")
        
        # Get backtest script path
        backtest_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest/vector_backtest.py")
        
        # Use subprocess to run backtest script
        try:
            # Use python interpreter to run script, ensuring the same environment
            python_executable = sys.executable
            subprocess.run([python_executable, backtest_script_path], check=True)
            st.success("Backtest completed!")
            st.rerun()  # Reload page to display new results
        except subprocess.CalledProcessError as e:
            st.error(f"Backtest failed: {str(e)}")
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

# Use correct relative path
long_only_backtest_chart_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),r"backtest/backtest_results/long-only_backtest_chart.html")
long_short_backtest_chart_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), r"backtest/backtest_results/long-short_backtest_chart.html")

# 展示单多回测结果
st.subheader("Long-only Strategy Backtest Results")
if os.path.exists(long_only_backtest_chart_path):
    with open(long_only_backtest_chart_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=1400, width=1600, scrolling=True)
else:
    st.error(f"Long-only backtest result file doesn't exist. Please run backtest first. Path: {long_only_backtest_chart_path}")

# 展示多空回测结果 
st.subheader("Long-short Strategy Backtest Results")
if os.path.exists(long_short_backtest_chart_path):
    with open(long_short_backtest_chart_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=1100, width=1600, scrolling=True)
else:
    st.error(f"Long-short backtest result file doesn't exist. Please run backtest first. Path: {long_short_backtest_chart_path}")


# Add IC Comparison Chart
st.markdown("---")
st.subheader("Cumulative IC and Rank IC Comparison Analysis")

# Get IC comparison chart path
ic_comparison_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest/interactive_comparison_ic_and_rank_ic.html")

# Check if file exists
if os.path.exists(ic_comparison_path):
    # Read HTML file content
    with open(ic_comparison_path, "r", encoding="utf-8") as f:
        ic_html_content = f.read()
    
    # Display IC comparison chart
    st.components.v1.html(ic_html_content, height=1200, width=1050, scrolling=True)
else:
    st.warning(f"IC comparison chart file does not exist. Path: {ic_comparison_path}")