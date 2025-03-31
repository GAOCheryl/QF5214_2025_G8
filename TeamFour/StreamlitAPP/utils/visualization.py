import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from StreamlitAPP.utils.database import read_data, write_data

def plot_results(equity_curve=None, evaluate=None, title="Backtest Results", data_source=None):
    """
    使用Plotly绘制回测结果
    
    Args:
        equity_curve: DataFrame, 包含策略净值的时间序列，如果为None则从数据库读取
        evaluate: DataFrame, 包含性能评估指标，如果为None则从数据库读取
        title: str, 图表标题
        data_source: str, 数据库表路径，例如'QF5214.visualization.equity_curve'
        
    Returns:
        plotly.graph_objects.Figure: Plotly图表对象
    """
    # 如果没有提供equity_curve和evaluate，从数据库读取
    if equity_curve is None or evaluate is None:
        if data_source is None:
            data_source = 'QF5214.visualization.backtest_results'
        try:
            equity_curve = read_data(f'SELECT * FROM {data_source}_equity')
            equity_curve.set_index('Date', inplace=True)
            evaluate = read_data(f'SELECT * FROM {data_source}_metrics')
            evaluate.set_index('Type', inplace=True)
        except Exception as e:
            print(f"从数据库读取数据失败: {str(e)}")
            return None
    
    # 定义简约柔和色调配色方案
    pastel_colors = {
        "Strategy": "#6495ED",     # 矢车菊蓝
        "S&P500": "#ADD8E6",       # 淡蓝色
        "Excess_Return": "#CBAACB",  # 淡紫色
        "Drawdown": "#F4C2C2",     # 淡红色
        "Beta": "#AEC6CF",         # 淡青色
        "Sharpe": "#BFD8B8"        # 淡绿色
    }
    
    # 创建子图 - 调整为左右各四个子图
    fig = make_subplots(
        rows=4, 
        cols=2, 
        specs=[
            [{"colspan": 1}, {"rowspan": 1, "type": "table"}], 
            [{"colspan": 1}, {"rowspan": 1, "type": "table"}],
            [{"colspan": 1}, {"rowspan": 1, "type": "table"}],
            [{"colspan": 1}, {"rowspan": 1, "type": "table"}]
        ], 
        row_heights=[0.37, 0.23, 0.2, 0.2],  # 调整行高比例，为表格提供更多空间
        column_widths=[0.7, 0.3],  # 增加右侧列宽比例
        subplot_titles=("Net Value", "Performance Metrics", "Drawdown", "Drawdown Stats", "Rolling Beta (3 months)", "Rolling Beta Stats (3 months)", "Rolling Sharpe (3 months)", "Rolling Sharpe Stats (3 months)"),
        vertical_spacing=0.08,  # 减小垂直间距以提供更多空间
        horizontal_spacing=0.05
    )
    
    # 绘制净值曲线
    for strategy in equity_curve.columns:
        if 'rolling' not in strategy.lower() and strategy != 'Excess_Return':  # 不绘制滚动指标和超额收益
            color = pastel_colors.get(strategy, "#F0E68C")  # 默认使用淡黄色
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index, 
                    y=equity_curve[strategy], 
                    name=strategy, 
                    line=dict(width=2, color=color)
                ),
                row=1, col=1
            )
    
    # 绘制累积超额收益
    if 'Excess_Return' in equity_curve.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index, 
                y=equity_curve['Excess_Return'], 
                name='Cumulative Excess Return', 
                line=dict(width=2, dash='dash', color=pastel_colors['Excess_Return'])
            ),
            row=1, col=1
        )
    
    # 绘制策略回撤
    strategy = 'Strategy'
    equity_series = equity_curve[strategy]
    cummax = equity_series.cummax()
    drawdown = (equity_series / cummax - 1)
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, 
            y=drawdown.values, 
            name=f"{strategy} Drawdown", 
            fill='tozeroy', 
            line=dict(color=pastel_colors['Drawdown'])
        ),
        row=2, col=1
    )
    
    # 绘制超额收益回撤
    if 'Excess_Return' in equity_curve.columns:
        excess_series = equity_curve['Excess_Return']
        excess_cummax = excess_series.cummax()
        excess_drawdown = (excess_series / excess_cummax - 1)
        
        fig.add_trace(
            go.Scatter(
                x=excess_drawdown.index, 
                y=excess_drawdown.values, 
                name="Excess Return Drawdown", 
                fill='tozeroy', 
                line=dict(color=pastel_colors['Excess_Return'], dash='dash')
            ),
            row=2, col=1
        )
    
    # 绘制滚动Beta
    if 'Rolling_3M_Beta' in equity_curve.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index, 
                y=equity_curve['Rolling_3M_Beta'], 
                name='Rolling Beta (3 months)', 
                line=dict(color=pastel_colors['Beta'])
            ),
            row=3, col=1
        )
        # 添加Beta=1的参考线
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index, 
                y=[1] * len(equity_curve), 
                name='Beta=1', 
                line=dict(color='lightgrey', dash='dash')
            ),
            row=3, col=1
        )
    
    # 绘制滚动Sharpe
    if 'Rolling_3M_Sharpe' in equity_curve.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index, 
                y=equity_curve['Rolling_3M_Sharpe'], 
                name='Rolling Sharpe (3 months)', 
                line=dict(color=pastel_colors['Sharpe'])
            ),
            row=4, col=1
        )
        # 添加Sharpe=0的参考线
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index, 
                y=[0] * len(equity_curve), 
                name='Sharpe=0', 
                line=dict(color='lightgrey', dash='dash')
            ),
            row=4, col=1
        )
    
    # 创建主要指标表格数据
    table_data = [
        ["Metrics"] + [col for col in equity_curve.columns if 'rolling' not in col.lower() and col != 'Excess_Return'],
    ]
    
    # 如果有超额收益数据，添加到表格
    if 'Excess_Return' in equity_curve.columns and 'Excess_Return' in evaluate.index:
        table_data[0].append('Excess_Return')
    
    # 添加各项指标
    metrics = ['Total Return', 'Annual Return', 'Annual Volatility', 'Sharpe', 'Max Drawdown', 'Calmar']
    original_metrics = ['总收益', '年化收益', '年化波动率', 'sharpe', '最大回撤', '卡尔玛']
    
    for i, metric in enumerate(metrics):
        row_data = [metric]
        for strategy in table_data[0][1:]:
            if strategy in evaluate.index:
                if metric in ['Total Return', 'Annual Return', 'Annual Volatility', 'Max Drawdown']:
                    row_data.append(f"{evaluate.loc[strategy, original_metrics[i]]:.2%}")
                else:
                    row_data.append(f"{evaluate.loc[strategy, original_metrics[i]]:.2f}")
            else:
                row_data.append('N/A')
        table_data.append(row_data)
    
    # 添加主表格（使用简约的表格样式）
    fig.add_trace(
        go.Table(
            header=dict(
                values=table_data[0],
                fill_color='#E5ECF6',  # 浅蓝灰色背景
                align=['center'] * len(table_data[0]),
                font=dict(size=12, family="Arial", color="#2a3f5f"),
                height=28,
                line_width=0.5,
                line_color='white'
            ),
            cells=dict(
                values=[[row[0] for row in table_data[1:]]] + 
                      [[row[i+1] for row in table_data[1:]] for i in range(len(table_data[0])-1)],
                fill_color='white',
                align=['center'] + ['center'] * (len(table_data[0])-1),
                font=dict(size=11, color="#2a3f5f"),
                height=22,
                line_width=0.5,
                line_color='#E5ECF6'
            )
        ),
        row=1, col=2
    )
    
    # 创建回撤统计表格
    drawdown_table_data = [["Drawdown Stats", "Strategy", "S&P500"]]
    if 'Excess_Return' in equity_curve.columns:
        drawdown_table_data = [["Drawdown Stats", "Strategy", "S&P500", "Excess Return"]]
    
    # 回撤指标
    drawdown_metrics = [
        ('平均回撤', 'Average Drawdown'),
        ('最大回撤', 'Max Drawdown'),
        ('最大回撤修复天数', 'Recovery Days')
    ]
    
    for db_metric, display_name in drawdown_metrics:
        row_data = [display_name]
        # 添加策略数据
        if 'Strategy' in evaluate.index and db_metric in evaluate.loc['Strategy']:
            if db_metric == '最大回撤':
                row_data.append(f"{evaluate.loc['Strategy', db_metric]:.2%}")
            elif db_metric == '平均回撤':
                row_data.append(f"{evaluate.loc['Strategy', db_metric]:.2%}")
            else:
                # 修复天数可能是数字或字符串
                if isinstance(evaluate.loc['Strategy', db_metric], str):
                    row_data.append(evaluate.loc['Strategy', db_metric])
                else:
                    row_data.append(f"{evaluate.loc['Strategy', db_metric]:.0f}")
        else:
            row_data.append('N/A')
            
        # 添加基准数据
        if 'S&P500' in evaluate.index and db_metric in evaluate.loc['S&P500']:
            if db_metric == '最大回撤':
                row_data.append(f"{evaluate.loc['S&P500', db_metric]:.2%}")
            elif db_metric == '平均回撤':
                row_data.append(f"{evaluate.loc['S&P500', db_metric]:.2%}")
            else:
                # 修复天数可能是数字或字符串
                if isinstance(evaluate.loc['S&P500', db_metric], str):
                    row_data.append(evaluate.loc['S&P500', db_metric])
                else:
                    row_data.append(f"{evaluate.loc['S&P500', db_metric]:.0f}")
        else:
            row_data.append('N/A')
            
        # 添加超额收益数据
        if 'Excess_Return' in equity_curve.columns:
            if 'Excess_Return' in evaluate.index and db_metric in evaluate.loc['Excess_Return']:
                if db_metric == '最大回撤':
                    row_data.append(f"{evaluate.loc['Excess_Return', db_metric]:.2%}")
                elif db_metric == '平均回撤':
                    row_data.append(f"{evaluate.loc['Excess_Return', db_metric]:.2%}")
                else:
                    # 修复天数可能是数字或字符串
                    if isinstance(evaluate.loc['Excess_Return', db_metric], str):
                        row_data.append(evaluate.loc['Excess_Return', db_metric])
                    else:
                        row_data.append(f"{evaluate.loc['Excess_Return', db_metric]:.0f}")
            else:
                row_data.append('N/A')
        
        drawdown_table_data.append(row_data)
    
    # 添加回撤统计表格（简约柔和风格）
    fig.add_trace(
        go.Table(
            header=dict(
                values=drawdown_table_data[0],
                fill_color='#F4C2C2',  # 淡红色背景
                align=['center'] * len(drawdown_table_data[0]),
                font=dict(size=12, family="Arial", color="#2a3f5f"),
                height=28,
                line_width=0.5,
                line_color='white'
            ),
            cells=dict(
                values=[[row[0] for row in drawdown_table_data[1:]]] + 
                       [[row[i+1] for row in drawdown_table_data[1:]] for i in range(len(drawdown_table_data[0])-1)],
                fill_color='white',
                align=['center'] * len(drawdown_table_data[0]),
                font=dict(size=11, color="#2a3f5f"),
                height=22,
                line_width=0.5,
                line_color='#F4C2C2'
            )
        ),
        row=2, col=2
    )
    
    # 创建滚动Beta统计指标表格
    beta_table_data = [["Beta Stats (3 months)", "Value"]]
    
    rolling_beta_metrics = [
        ('滚动Beta平均值', 'Mean'),
        ('滚动Beta标准差', 'Std Dev'),
        ('滚动Beta中位数', 'Median'), 
        ('滚动Beta 25%分位', '25% Quantile'),
        ('滚动Beta 75%分位', '75% Quantile')
    ]
    
    for metric, display_name in rolling_beta_metrics:
        if 'Strategy' in evaluate.index and metric in evaluate.loc['Strategy']:
            beta_table_data.append([display_name, f"{evaluate.loc['Strategy', metric]:.3f}"])
    
    # 添加Beta统计表格（简约风格）
    fig.add_trace(
        go.Table(
            header=dict(
                values=beta_table_data[0],
                fill_color=pastel_colors['Beta'],
                align=['center', 'center'],
                font=dict(size=12, family="Arial", color="#2a3f5f"),
                height=28,
                line_width=0.5,
                line_color='white'
            ),
            cells=dict(
                values=[[row[0] for row in beta_table_data[1:]], [row[1] for row in beta_table_data[1:]]],
                fill_color='white',
                align=['center', 'center'],
                font=dict(size=11, color="#2a3f5f"),
                height=22,
                line_width=0.5,
                line_color=pastel_colors['Beta']
            )
        ),
        row=3, col=2
    )
    
    # 创建滚动Sharpe统计指标表格
    sharpe_table_data = [["Sharpe Stats (3 months)", "Value"]]
    
    rolling_sharpe_metrics = [
        ('滚动Sharpe平均值', 'Mean'),
        ('滚动Sharpe标准差', 'Std Dev'),
        ('滚动Sharpe中位数', 'Median'), 
        ('滚动Sharpe 25%分位', '25% Quantile'),
        ('滚动Sharpe 75%分位', '75% Quantile')
    ]
    
    for metric, display_name in rolling_sharpe_metrics:
        if 'Strategy' in evaluate.index and metric in evaluate.loc['Strategy']:
            sharpe_table_data.append([display_name, f"{evaluate.loc['Strategy', metric]:.3f}"])
    
    # 添加Sharpe统计表格（简约风格）
    fig.add_trace(
        go.Table(
            header=dict(
                values=sharpe_table_data[0],
                fill_color=pastel_colors['Sharpe'],
                align=['center', 'center'],
                font=dict(size=12, family="Arial", color="#2a3f5f"),
                height=28,
                line_width=0.5,
                line_color='white'
            ),
            cells=dict(
                values=[[row[0] for row in sharpe_table_data[1:]], [row[1] for row in sharpe_table_data[1:]]],
                fill_color='white',
                align=['center', 'center'],
                font=dict(size=11, color="#2a3f5f"),
                height=22,
                line_width=0.5,
                line_color=pastel_colors['Sharpe']
            )
        ),
        row=4, col=2
    )
    
    # 更新布局（简约风格）
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.98,
            font=dict(size=20, color="#2a3f5f")
        ),
        height=1300,
        width=1500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(color="#2a3f5f")
        ),
        margin=dict(t=100, b=120, l=50, r=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # 更新x轴和y轴标签
    fig.update_yaxes(
        title_text="Net Value", 
        row=1, col=1, 
        gridcolor='#E5ECF6', 
        title_font=dict(color="#2a3f5f")
    )
    fig.update_yaxes(
        title_text="Drawdown", 
        row=2, col=1, 
        gridcolor='#E5ECF6', 
        title_font=dict(color="#2a3f5f")
    )
    fig.update_yaxes(
        title_text="Beta (3 months)", 
        row=3, col=1, 
        gridcolor='#E5ECF6', 
        title_font=dict(color="#2a3f5f")
    )
    fig.update_yaxes(
        title_text="Sharpe (3 months)", 
        row=4, col=1, 
        gridcolor='#E5ECF6', 
        title_font=dict(color="#2a3f5f")
    )
    
    # 更新所有x轴的网格线颜色为淡色
    fig.update_xaxes(gridcolor='#E5ECF6')
    
    # 调整子图标题位置
    for i, annotation in enumerate(fig['layout']['annotations']):
        # 调整子图标题的垂直位置，防止与坐标轴重叠
        if i < 8:  # 所有子图标题
            annotation['y'] = annotation['y'] + 0.02
            annotation['font'] = dict(color="#2a3f5f")
    
    return fig

# 计算性能评估指标的函数
def calculate_performance_metrics(equity_curve):
    """
    计算回测的性能评估指标
    
    Args:
        equity_curve: DataFrame, 包含策略净值的时间序列
        
    Returns:
        DataFrame: 包含性能评估指标的DataFrame
    """
    evaluate = pd.DataFrame()
    
    # 计算每个策略的指标
    for strategy in equity_curve.columns:
        # 跳过滚动指标列
        if 'rolling' in strategy.lower():
            continue
            
        equity_series = equity_curve[strategy]
        
        # 计算日收益率
        daily_returns = equity_series.pct_change().dropna()
        
        # 计算总收益
        total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
        
        # 计算年化收益
        days = len(daily_returns)
        annual_return = (1 + total_return) ** (250 / days) - 1
        
        # 计算年化波动率
        annual_volatility = daily_returns.std() * np.sqrt(250)
        
        # 计算夏普比率
        sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # 计算最大回撤
        cummax = equity_series.cummax()
        drawdown = (equity_series / cummax - 1)
        max_drawdown = drawdown.min()
        
        # 计算平均回撤
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # 计算最大回撤修复天数
        max_dd_end_idx = drawdown.idxmin()
        max_dd_start_idx = equity_series.loc[:max_dd_end_idx].idxmax()
        
        # 找到修复点（回到最大回撤前的净值）
        if max_dd_end_idx != drawdown.index[-1]:  # 确保不是最后一个点
            # 找到修复日期
            recovery_date = None
            for date in drawdown.loc[max_dd_end_idx:].index:
                if equity_series[date] >= equity_series[max_dd_start_idx]:
                    recovery_date = date
                    break
            
            # 计算修复天数
            if recovery_date is not None:
                recovery_days = len(drawdown.loc[max_dd_end_idx:recovery_date])
            else:
                recovery_days = "Not Recovered"
        else:
            # 如果最大回撤发生在最后一个点，那么肯定没有恢复
            recovery_days = "Not Recovered"
        
        # 计算卡尔玛比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
        
        # 添加到评估指标DataFrame
        metrics_dict = {
            'Type': strategy,
            '总收益': total_return,
            '年化收益': annual_return,
            '年化波动率': annual_volatility,
            'sharpe': round(sharpe, 3),
            '最大回撤': max_drawdown,
            '平均回撤': avg_drawdown,
            '最大回撤修复天数': recovery_days,
            '卡尔玛': round(calmar, 3),
            '最大回撤起点': max_dd_start_idx,
            '最大回撤终点': max_dd_end_idx
        }
        
        # 添加滚动Beta统计数据
        if 'Rolling_3M_Beta' in equity_curve.columns and strategy == 'Strategy':
            rolling_beta = equity_curve['Rolling_3M_Beta'].dropna()
            if not rolling_beta.empty:
                metrics_dict.update({
                    '滚动Beta平均值': round(rolling_beta.mean(), 3),
                    '滚动Beta标准差': round(rolling_beta.std(), 3),
                    '滚动Beta中位数': round(rolling_beta.median(), 3),
                    '滚动Beta 25%分位': round(rolling_beta.quantile(0.25), 3),
                    '滚动Beta 75%分位': round(rolling_beta.quantile(0.75), 3)
                })
        
        # 添加滚动Sharpe统计数据
        if 'Rolling_3M_Sharpe' in equity_curve.columns and strategy == 'Strategy':
            rolling_sharpe = equity_curve['Rolling_3M_Sharpe'].dropna()
            if not rolling_sharpe.empty:
                metrics_dict.update({
                    '滚动Sharpe平均值': round(rolling_sharpe.mean(), 3),
                    '滚动Sharpe标准差': round(rolling_sharpe.std(), 3),
                    '滚动Sharpe中位数': round(rolling_sharpe.median(), 3),
                    '滚动Sharpe 25%分位': round(rolling_sharpe.quantile(0.25), 3),
                    '滚动Sharpe 75%分位': round(rolling_sharpe.quantile(0.75), 3)
                })
        
        evaluate = evaluate._append(metrics_dict, ignore_index=True)
    
    # 设置索引
    evaluate.index = evaluate['Type'].tolist()
    evaluate.drop('Type', axis=1, inplace=True)
    
    return evaluate
