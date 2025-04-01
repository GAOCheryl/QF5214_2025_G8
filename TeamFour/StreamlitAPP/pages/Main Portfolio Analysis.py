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

# 获取HTML文件路径
def get_html_paths():
    # 使用正确的相对路径
    long_only_backtest_chart_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),r"backtest/backtest_results/long-only_backtest_chart.html")
    long_short_backtest_chart_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), r"backtest/backtest_results/long-short_backtest_chart.html")
    ic_comparison_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest/interactive_comparison_ic_and_rank_ic.html")
    return long_only_backtest_chart_path, long_short_backtest_chart_path, ic_comparison_path

# 显示HTML内容函数
def display_html_content(file_path, height, width, error_message):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=height, width=width, scrolling=True)
    else:
        st.error(error_message)

# 添加回测按钮
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Run Latest Backtest", help="Click to run backtest analysis with latest data"):
        st.info("运行回测分析中，请稍候...")
        
        # 获取回测脚本路径
        backtest_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest/vector_backtest.py")
        cumulative_ic_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest/cumulative_IC.py")
        
        # 使用subprocess运行回测脚本
        try:
            # 使用相同的Python解释器运行脚本，确保环境一致
            python_executable = sys.executable
            # 运行主回测脚本
            subprocess.run([python_executable, backtest_script_path], check=True)
            # 运行累积IC分析脚本
            subprocess.run([python_executable, cumulative_ic_script_path], check=True)
            st.success("回测完成！")
            st.rerun()  # 重新加载页面以显示新结果
        except subprocess.CalledProcessError as e:
            st.error(f"回测失败: {str(e)}")
        except Exception as e:
            st.error(f"发生错误: {str(e)}")

# 获取HTML文件路径
long_only_path, long_short_path, ic_path = get_html_paths()

# 展示单多回测结果
st.subheader("Long-only Strategy Backtest Results")
display_html_content(
    long_only_path, 
    height=1400, 
    width=1600, 
    error_message=f"Long-only回测结果文件不存在。请先运行回测。路径: {long_only_path}"
)

# 展示多空回测结果 
st.subheader("Long-short Strategy Backtest Results")
display_html_content(
    long_short_path, 
    height=1100, 
    width=1600, 
    error_message=f"Long-short回测结果文件不存在。请先运行回测。路径: {long_short_path}"
)

# 添加IC比较图表
st.markdown("---")
st.subheader("Cumulative IC and Rank IC Comparison Analysis")

# 展示IC比较图表
display_html_content(
    ic_path, 
    height=1200, 
    width=1050, 
    error_message=f"IC比较图表文件不存在。路径: {ic_path}"
)