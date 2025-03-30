import streamlit as st
import os
from datetime import datetime, timedelta
import sys
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

st.set_page_config(page_title="Portfolio Analysis", layout="wide")
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
backtest_chart_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest/backtest_results/backtest_chart.html")

# Check if file exists
if os.path.exists(backtest_chart_path):
    # Read HTML file content
    with open(backtest_chart_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Increase height and width to fully display the chart
    st.components.v1.html(html_content, height=1400, width=1500, scrolling=True)
else:
    st.error(f"Backtest result file doesn't exist. Please run backtest first. Path: {backtest_chart_path}")


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
    st.components.v1.html(ic_html_content, height=1200, width=1200, scrolling=True)
else:
    st.warning(f"IC comparison chart file does not exist. Path: {ic_comparison_path}")