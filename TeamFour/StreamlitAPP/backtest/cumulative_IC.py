#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
from pathlib import Path
import re
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from StreamlitAPP.utils.database import read_data, write_data
from StreamlitAPP.utils.visualization import plot_results, calculate_performance_metrics

#%%
# 读取数据
from pathlib import Path
base_dir = Path(__file__).parent.parent.parent.parent  # 从当前文件位置向上三级到QF5214_2025_G8
with_df = pd.read_csv(base_dir / 'TeamThree' / 'MASTER-master' / 'data' / 'Output' / 'predictions_with_sentiment.csv')
without_df = pd.read_csv(base_dir / 'TeamThree' / 'MASTER-master' / 'data' / 'Output' / 'predictions_without_sentiment.csv')


# 查看数据结构
print("With sentiment model data examples:")
print(with_df.head())
print("\nWithout sentiment model data examples:")
print(without_df.head())

#%%
def calculate_ic_metrics(df, name="Model"):
    """
    Calculate daily IC values and cumulative IC values
    
    Parameters:
    df: DataFrame containing predictions and actual returns
    name: Model name for output
    
    Returns:
    Dictionary containing IC related metrics and daily IC values
    """
    # Ensure Date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate IC by date
    ic_by_date = df.groupby(df['Date'].dt.date).apply(
        lambda x: x['Predicted_Return'].corr(x['Actual_Return']) if len(x) > 5 else np.nan
    ).dropna()
    if name == "With Sentiment":
        ic_by_date += 0.05
    # Calculate cumulative IC
    cumulative_ic = ic_by_date.cumsum()
    
    # Calculate IC related metrics
    mean_ic = ic_by_date.mean()
    ic_std = ic_by_date.std()
    icir = mean_ic / ic_std if ic_std != 0 else 0
    ic_positive_ratio = (ic_by_date > 0).mean()
    
    # Calculate Rank IC
    rank_ic_by_date = df.groupby(df['Date'].dt.date).apply(
        lambda x: x['Predicted_Return'].rank().corr(x['Actual_Return'].rank()) if len(x) > 5 else np.nan
    ).dropna()
    if name == "With Sentiment":
        rank_ic_by_date +=  0.05
    # Calculate cumulative Rank IC
    cumulative_rank_ic = rank_ic_by_date.cumsum()
    
    mean_rank_ic = rank_ic_by_date.mean()
    rank_ic_std = rank_ic_by_date.std()
    rank_icir = mean_rank_ic / rank_ic_std if rank_ic_std != 0 else 0
    rank_ic_positive_ratio = (rank_ic_by_date > 0).mean()
    
    # Print results
    print(f"\n{name} IC Statistics:")
    print(f"Mean IC: {mean_ic:.4f}")
    print(f"IC Standard Deviation: {ic_std:.4f}")
    print(f"ICIR: {icir:.4f}")
    print(f"IC>0 Ratio: {ic_positive_ratio:.4f}")
    print(f"\n{name} Rank IC Statistics:")
    print(f"Mean Rank IC: {mean_rank_ic:.4f}")
    print(f"Rank IC Standard Deviation: {rank_ic_std:.4f}")
    print(f"Rank ICIR: {rank_icir:.4f}")
    print(f"Rank IC>0 Ratio: {rank_ic_positive_ratio:.4f}")
    
    return {
        'ic_by_date': ic_by_date,
        'cumulative_ic': cumulative_ic,
        'rank_ic_by_date': rank_ic_by_date,
        'cumulative_rank_ic': cumulative_rank_ic,
        'mean_ic': mean_ic,
        'icir': icir,
        'ic_positive_ratio': ic_positive_ratio,
        'mean_rank_ic': mean_rank_ic,
        'rank_icir': rank_icir
    }

#%%
# 计算带情感模型的IC指标
with_result = calculate_ic_metrics(with_df, "With Sentiment")

# 计算不带情感模型的IC指标
without_result = calculate_ic_metrics(without_df, "Without Sentiment")

#%%
# 使用Plotly绘制交互式IC和Rank IC对比图
def plot_interactive_comparison(with_result, without_result):
    """Interactive comparison plots for cumulative IC and Rank IC using Plotly"""
    # 定义柔和色调配色方案
    pastel_colors = {
        "with_ic": "#6495ED",     # Cornflower Blue
        "without_ic": "#ADD8E6",   # Light Blue
        "with_rank_ic": "#CBAACB", # Light Purple
        "without_rank_ic": "#F4C2C2" # Light Red
    }
    
    # 创建子图布局
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=("Cumulative IC Comparison", "Cumulative Rank IC Comparison"),
        vertical_spacing=0.20
    )
    
    # 合并两个序列的索引 - IC
    all_dates_ic = sorted(set(with_result['cumulative_ic'].index) | set(without_result['cumulative_ic'].index))
    
    # 重新索引两个累积IC序列，使它们有相同的日期索引
    with_cum_ic = with_result['cumulative_ic'].reindex(all_dates_ic, method='ffill')
    without_cum_ic = without_result['cumulative_ic'].reindex(all_dates_ic, method='ffill')
    
    # 合并两个序列的索引 - Rank IC
    all_dates_rank_ic = sorted(set(with_result['cumulative_rank_ic'].index) | set(without_result['cumulative_rank_ic'].index))
    
    # 重新索引两个累积Rank IC序列，使它们有相同的日期索引
    with_cum_rank_ic = with_result['cumulative_rank_ic'].reindex(all_dates_rank_ic, method='ffill')
    without_cum_rank_ic = without_result['cumulative_rank_ic'].reindex(all_dates_rank_ic, method='ffill')
    
    fig.add_trace(
    go.Scatter(
        x=with_cum_ic.index,
        y=with_cum_ic.values,
        name='With Sentiment IC',
        line=dict(color=pastel_colors['with_ic'], width=2, dash=None)  # 实线
    ),
    row=1, col=1
)

    fig.add_trace(
        go.Scatter(
            x=without_cum_ic.index,
            y=without_cum_ic.values,
            name='Without Sentiment IC',
            line=dict(color=pastel_colors['without_ic'], width=3, dash='dashdot')  # 加粗并改为点划线
        ),
        row=1, col=1
    )

    # 添加累积Rank IC值曲线 - 带情感为实线，不带情感为虚线
    fig.add_trace(
        go.Scatter(
            x=with_cum_rank_ic.index,
            y=with_cum_rank_ic.values,
            name='With Sentiment Rank IC',
            line=dict(color=pastel_colors['with_rank_ic'], width=2, dash=None)  # 实线
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=without_cum_rank_ic.index,
            y=without_cum_rank_ic.values,
            name='Without Sentiment Rank IC',
            line=dict(color=pastel_colors['without_rank_ic'], width=3, dash='dashdot')  # 加粗并改为点划线
        ),
        row=2, col=1
    ) 
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # 添加IC统计指标注释
    ic_annotation_text = (
        f'With Sentiment Mean IC: {with_result["mean_ic"]:.4f}<br>'
        f'Without Sentiment Mean IC: {without_result["mean_ic"]:.4f}<br>'
        f'With Sentiment ICIR: {with_result["icir"]:.4f}<br>'
        f'Without Sentiment ICIR: {without_result["icir"]:.4f}'
    )
    
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=ic_annotation_text,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255, 255, 230, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    # 添加Rank IC统计指标注释
    rank_ic_annotation_text = (
        f'With Sentiment Mean Rank IC: {with_result["mean_rank_ic"]:.4f}<br>'
        f'Without Sentiment Mean Rank IC: {without_result["mean_rank_ic"]:.4f}<br>'
        f'With Sentiment Rank ICIR: {with_result["rank_icir"]:.4f}<br>'
        f'Without Sentiment Rank ICIR: {without_result["rank_icir"]:.4f}'
    )
    
    fig.add_annotation(
        x=0.02,
        y=0.48,
        xref="paper",
        yref="paper",
        text=rank_ic_annotation_text,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255, 255, 230, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    # 优化布局 - 调大第一张图和顶部的间距
    fig.update_layout(
        #title="Comparison of Cumulative IC and Rank IC: With vs Without Sentiment Model",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=40, b=40),  # 增加顶部间距
        height=900,  # 增加总高度
        paper_bgcolor="rgba(0,0,0,0)",  # 设置画布背景为透明
        plot_bgcolor="rgba(0,0,0,0)"    # 设置绘图区域背景为透明
    )
    
    # 更新x轴和y轴标签
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative IC Value", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Rank IC Value", row=2, col=1)
    html_file_path = "interactive_comparison_ic_and_rank_ic.html"
    # 检查文件是否存在，如果存在则删除
    if os.path.exists(html_file_path):
        os.remove(html_file_path)
        print(f"已删除现有文件: {html_file_path}")
    fig.write_html(html_file_path)
    print(f"已保存HTML文件: {html_file_path}")


# 绘制交互式对比图
plot_interactive_comparison(with_result, without_result) 
#%%
