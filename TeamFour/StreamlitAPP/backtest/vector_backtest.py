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
from StreamlitAPP.utils.database import read_data, write_data, get_market_data
from StreamlitAPP.utils.visualization import plot_results, calculate_performance_metrics


#%%
try:
    position_df = read_data('QF5214.tradingstrategy.dailytrading')
    #print(position_df.head())
except Exception as e:
    print(f"读取持仓数据出错: {str(e)}")
    position_df = None
position_start_date=position_df['Date'].min()
position_end_date=position_df['Date'].max()
#print(position_start_date)
#%%

try:
    price_df = read_data(f'SELECT * FROM datacollection.stock_data WHERE "Date" >= \'{position_start_date}\' AND "Date" <= \'{position_end_date}\'')
    #print(price_df.head())
except Exception as e:
    print(f"读取股票价格数据出错: {str(e)}")
    price_df = None

#%%
# 读取指数数据
try:
    index_df = read_data(f'SELECT * FROM datacollection.index_data WHERE "Date" >= \'{position_start_date}\' AND "Date" <= \'{position_end_date}\'')
    #print(index_df.head())
    index_df = index_df[['Date', 'Adj_Close']]
except Exception as e:
    print(f"读取指数数据出错: {str(e)}")
    index_df = None



#%%
#暂时将position_df的Weight列设置为0.2
position_df['Weight'] = 0.2
price_df=price_df[['Ticker','Date','Adj_Close']]

# 转换持仓数据的日期
if position_df['Date'].dtype != 'datetime64[ns]':
    position_df['Date'] = pd.to_datetime(position_df['Date'])

# 转换价格数据的日期
if price_df['Date'].dtype != 'datetime64[ns]':
    price_df['Date'] = pd.to_datetime(price_df['Date'])

# 转换指数数据的日期
if index_df is not None and index_df['Date'].dtype != 'datetime64[ns]':
    index_df['Date'] = pd.to_datetime(index_df['Date'])

# 获取三个DataFrame的起始和终止时间
dfs = [position_df, price_df, index_df]
df_names = ['position_df', 'price_df', 'index_df']

# 初始化最晚的起始时间和最早的终止时间
latest_start_date = None
earliest_end_date = None

# 扫描所有DataFrame的时间范围
for df, name in zip(dfs, df_names):
    if df is not None and not df.empty:
        start_date = df['Date'].min()
        end_date = df['Date'].max()
        
        print(f"{name} 时间范围: {start_date} 到 {end_date}")
        
        # 更新最晚的起始时间
        if latest_start_date is None or start_date > latest_start_date:
            latest_start_date = start_date
            
        # 更新最早的终止时间
        if earliest_end_date is None or end_date < earliest_end_date:
            earliest_end_date = end_date

print(f"统一时间范围: {latest_start_date} 到 {earliest_end_date}")

# 将三个DataFrame的时间范围统一
for i, (df, name) in enumerate(zip(dfs, df_names)):
    if df is not None and not df.empty:
        filtered_df = df[(df['Date'] >= latest_start_date) & (df['Date'] <= earliest_end_date)]
        dfs[i] = filtered_df
        print(f"{name} 过滤后行数: {len(filtered_df)}")

# 更新原始DataFrame
position_df, price_df, index_df = dfs




merged_df = pd.merge(position_df, price_df, on=['Ticker', 'Date'], how='left')
merged_df = pd.merge(merged_df, index_df, on=['Date'], how='left', suffixes=('', '_S&P500'))
print(merged_df.head())


#%%
# 向量化回测核心函数
def run_backtest(position_df, price_df=None, title="Backtest", save_results=True, data_source='QF5214.visualization.backtest_results'):
    """
    执行向量化回测
    
    Args:
        position_df: DataFrame, 持仓数据
        price_df: DataFrame, 价格数据，如果为None则从数据库获取
        title: str, 回测标题
        save_results: bool, 是否保存结果
        data_source: str, 保存结果的数据库表路径，默认为'QF5214.visualization.backtest_results'
        
    Returns:
        tuple: (evaluate, equity_curve, fig)
            - evaluate: 评估指标DataFrame
            - equity_curve: 净值曲线DataFrame
            - fig: Plotly图表对象
    """
    print("开始向量化回测...")
    
    # 确保持仓数据不为空
    if position_df is None or position_df.empty:
        print("错误: 持仓数据为空")
        return None, None, None
    
    # 确保日期列是datetime类型
    if 'Date' in position_df.columns and position_df['Date'].dtype != 'datetime64[ns]':
        position_df['Date'] = pd.to_datetime(position_df['Date'])
    
    # 获取持仓数据的日期范围
    position_start_date = position_df['Date'].min()
    position_end_date = position_df['Date'].max()
    
    # 获取持仓中的股票列表
    if 'Ticker' in position_df.columns:
        tickers = position_df['Ticker'].unique().tolist()
    else:
        print("错误: 持仓数据中缺少'Ticker'列")
        return None, None, None
    
    # 如果未提供价格数据，从数据库获取
    if price_df is None:
        try:
            # 使用新的get_market_data函数获取价格数据
            price_df = get_market_data(
                start_date=position_start_date.strftime('%Y-%m-%d'), 
                end_date=position_end_date.strftime('%Y-%m-%d'),
                tickers=tickers
            )
            
            if price_df is None or price_df.empty:
                print("错误: 无法从数据库获取价格数据")
                return None, None, None
                
            # 只保留需要的列
            price_df = price_df[['Ticker', 'Date', 'Adj_Close']]
        except Exception as e:
            print(f"获取价格数据出错: {str(e)}")
            return None, None, None
    
    # 确保价格数据的日期列是datetime类型
    if 'Date' in price_df.columns and price_df['Date'].dtype != 'datetime64[ns]':
        price_df['Date'] = pd.to_datetime(price_df['Date'])
    
    # 获取指数数据作为基准
    try:
        index_df = read_data(f'SELECT * FROM datacollection.index_data WHERE "Date" >= \'{position_start_date}\' AND "Date" <= \'{position_end_date}\'')
        if index_df is not None and not index_df.empty:
            index_df = index_df[['Date', 'Adj_Close']]
            
            # 确保指数数据的日期列是datetime类型
            if index_df['Date'].dtype != 'datetime64[ns]':
                index_df['Date'] = pd.to_datetime(index_df['Date'])
        else:
            print("警告: 无法获取指数数据，将创建一个假数据基准")
            index_df = None
    except Exception as e:
        print(f"读取指数数据出错: {str(e)}")
        index_df = None
    
    # 统一日期范围
    dfs = [position_df, price_df, index_df]
    df_names = ['position_df', 'price_df', 'index_df']
    
    # 初始化最晚的起始时间和最早的终止时间
    latest_start_date = None
    earliest_end_date = None
    
    # 扫描所有DataFrame的时间范围
    for df, name in zip(dfs, df_names):
        if df is not None and not df.empty:
            start_date = df['Date'].min()
            end_date = df['Date'].max()
            
            print(f"{name} 时间范围: {start_date} 到 {end_date}")
            
            # 更新最晚的起始时间
            if latest_start_date is None or start_date > latest_start_date:
                latest_start_date = start_date
                
            # 更新最早的终止时间
            if earliest_end_date is None or end_date < earliest_end_date:
                earliest_end_date = end_date
    
    print(f"统一时间范围: {latest_start_date} 到 {earliest_end_date}")
    
    # 将三个DataFrame的时间范围统一
    for i, (df, name) in enumerate(zip(dfs, df_names)):
        if df is not None and not df.empty:
            filtered_df = df[(df['Date'] >= latest_start_date) & (df['Date'] <= earliest_end_date)]
            dfs[i] = filtered_df
            print(f"{name} 过滤后行数: {len(filtered_df)}")
    
    # 更新原始DataFrame
    position_df, price_df, index_df = dfs
    
    # 合并持仓数据和价格数据用于计算指数比较
    merged_df = pd.merge(position_df, price_df, on=['Ticker', 'Date'], how='left')
    if index_df is not None:
        merged_df = pd.merge(merged_df, index_df, on=['Date'], how='left', suffixes=('', '_S&P500'))
    
    # 获取所有交易日期
    all_dates = sorted(position_df['Date'].unique())
    
    # 初始化净值曲线，添加基准指数
    equity_curve = pd.DataFrame(index=all_dates, columns=['Strategy', 'S&P500', 'Excess_Return'])
    equity_curve.fillna(1.0, inplace=True)
    
    # 按日期遍历计算收益
    for i in range(1, len(all_dates)):
        current_date = all_dates[i]
        prev_date = all_dates[i-1]
        
        # 获取前一天的持仓
        prev_positions = position_df[position_df['Date'] == prev_date]
        
        # 获取当天和前一天的价格数据
        current_prices = price_df[price_df['Date'] == current_date]
        prev_prices = price_df[price_df['Date'] == prev_date]
        
        # 获取当天和前一天的指数数据
        if index_df is not None:
            current_index = merged_df[merged_df['Date'] == current_date]['Adj_Close_S&P500'].iloc[0] if not merged_df[merged_df['Date'] == current_date].empty else None
            prev_index = merged_df[merged_df['Date'] == prev_date]['Adj_Close_S&P500'].iloc[0] if not merged_df[merged_df['Date'] == prev_date].empty else None
            
            # 计算基准指数收益率
            if current_index is not None and prev_index is not None and prev_index > 0:
                index_return = current_index / prev_index - 1
                equity_curve.loc[current_date, 'S&P500'] = equity_curve.loc[prev_date, 'S&P500'] * (1 + index_return)
            else:
                # 如果缺少数据，净值保持不变
                equity_curve.loc[current_date, 'S&P500'] = equity_curve.loc[prev_date, 'S&P500']
        else:
            # 如果没有指数数据，给S&P500设置一个微小的随机收益率
            random_return = np.random.normal(0.0003, 0.005)  # 均值为0.03%，标准差为0.5%
            equity_curve.loc[current_date, 'S&P500'] = equity_curve.loc[prev_date, 'S&P500'] * (1 + random_return)
        
        # 检查是否有足够的数据
        if prev_positions.empty or current_prices.empty or prev_prices.empty:
            print(f"警告: {prev_date}或{current_date}缺少必要数据，跳过该日计算")
            # 如果缺少数据，净值保持不变
            equity_curve.loc[current_date, 'Strategy'] = equity_curve.loc[prev_date, 'Strategy']
            # 超额收益也保持不变
            equity_curve.loc[current_date, 'Excess_Return'] = equity_curve.loc[prev_date, 'Excess_Return']
            continue
        
        # 合并持仓和价格数据
        merged_data = pd.merge(
            prev_positions[['Ticker', 'Weight']],  # 取前一天的持仓
            current_prices[['Ticker', 'Adj_Close']],  # 当天的收盘价
            on='Ticker',
            how='inner'  # 内连接，只保留同时存在于两个表中的股票
        )
        
        # 再合并前一天的价格
        merged_data = pd.merge(
            merged_data,
            prev_prices[['Ticker', 'Adj_Close']],  # 前一天的收盘价
            on='Ticker',
            how='inner',
            suffixes=('', '_prev')  # 区分当天和前一天的价格
        )
        
        # 检查合并后是否有数据
        if merged_data.empty:
            print(f"警告: {current_date}没有可用的股票数据，跳过该日计算")
            equity_curve.loc[current_date, 'Strategy'] = equity_curve.loc[prev_date, 'Strategy']
            # 超额收益也保持不变
            equity_curve.loc[current_date, 'Excess_Return'] = equity_curve.loc[prev_date, 'Excess_Return']
            continue
        
        # 计算每支股票的收益率
        merged_data['Return'] = merged_data['Adj_Close'] / merged_data['Adj_Close_prev'] - 1
        
        # 计算加权收益率
        merged_data['Weighted_Return'] = merged_data['Weight'] * merged_data['Return']
        
        # 计算当天的总收益率
        day_return = merged_data['Weighted_Return'].sum()
        
        # 更新净值曲线
        equity_curve.loc[current_date, 'Strategy'] = equity_curve.loc[prev_date, 'Strategy'] * (1 + day_return)
        
        # 计算当天的超额收益率
        strategy_return = equity_curve.loc[current_date, 'Strategy'] / equity_curve.loc[prev_date, 'Strategy'] - 1
        benchmark_return = equity_curve.loc[current_date, 'S&P500'] / equity_curve.loc[prev_date, 'S&P500'] - 1
        excess_return = strategy_return - benchmark_return
        
        # 更新累积超额收益
        equity_curve.loc[current_date, 'Excess_Return'] = equity_curve.loc[prev_date, 'Excess_Return'] * (1 + excess_return)
    
    # 计算滚动指标
    # 首先计算日收益率
    daily_returns = equity_curve.pct_change().dropna()
    
    # 计算滚动3个月(63个交易日)夏普比率
    rolling_window = 63  # 约3个月的交易日
    
    if len(daily_returns) >= rolling_window:
        # 对策略计算滚动夏普
        rolling_returns = daily_returns['Strategy'].rolling(window=rolling_window)
        rolling_mean = rolling_returns.mean() * 252  # 年化
        rolling_std = rolling_returns.std() * np.sqrt(252)  # 年化
        equity_curve['Rolling_3M_Sharpe'] = rolling_mean / rolling_std
        
        # 计算滚动3个月Beta
        if 'S&P500' in daily_returns.columns:
            # 使用协方差计算beta
            rolling_cov = daily_returns['Strategy'].rolling(window=rolling_window).cov(daily_returns['S&P500'])
            rolling_var = daily_returns['S&P500'].rolling(window=rolling_window).var()
            equity_curve['Rolling_3M_Beta'] = rolling_cov / rolling_var
    
    # 计算评估指标
    evaluate = calculate_performance_metrics(equity_curve)
    
    # 保存结果
    if save_results:
        # 重置索引以便将日期作为列保存到数据库
        equity_curve_db = equity_curve.reset_index()
        
        # 保存到数据库
        try:
            # 保存净值曲线
            write_data(equity_curve_db, f"{data_source}_equity")
            print(f"净值曲线已保存到 {data_source}_equity 表中")
            
            # 重置索引以便将类型作为列保存到数据库
            evaluate_db = evaluate.reset_index().rename(columns={'index': 'Type'})
            
            # 保存评估指标
            write_data(evaluate_db, f"{data_source}_metrics")
            print(f"评估指标已保存到 {data_source}_metrics 表中")
        except Exception as e:
            print(f"保存数据到数据库失败: {str(e)}")
            
            # 创建备份文件目录
            result_dir = 'backtest_results'
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # 备份保存到文件
            equity_curve.to_csv(os.path.join(result_dir, "backtest_equity.csv"))
            evaluate.to_csv(os.path.join(result_dir, "backtest_metrics.csv"))
            print(f"数据已备份保存到 {result_dir} 目录")
    
    # 绘制结果
    fig = plot_results(equity_curve, evaluate, title)
    
    # 如果成功创建了图形，可以保存图形文件
    if fig is not None and save_results:
        result_dir = 'backtest_results'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        fig.write_html(os.path.join(result_dir, "backtest_chart.html"))
    
    return evaluate, equity_curve, fig



#%%

# 执行回测，添加S&P500指数作为基准
# evaluate, equity_curve, fig = run_backtest(position_df, price_df=None, title="Backtest")

# # 注释掉显示图表的代码，避免在运行脚本时弹出图表
# # if fig is not None:
# #     fig.show()

# # 打印评估指标
# print("\n======== 回测结果 ========")
# print("评估指标:")
# print(evaluate)
    
#%%

# 添加一个main函数，方便脚本单独运行
def main():
    try:
        # 读取持仓数据
        position_df = read_data('QF5214.tradingstrategy.dailytrading')
        if position_df is None or position_df.empty:
            print("错误: 无法读取持仓数据或持仓数据为空")
            return
            
        position_start_date = position_df['Date'].min()
        position_end_date = position_df['Date'].max()
        
        # 读取股票价格数据
        price_df = read_data(f'SELECT * FROM datacollection.stock_data WHERE "Date" >= \'{position_start_date}\' AND "Date" <= \'{position_end_date}\'')
        if price_df is None or price_df.empty:
            print("错误: 无法读取股票价格数据或价格数据为空")
            return
            
        price_df = price_df[['Ticker', 'Date', 'Adj_Close']]
        
        # 设置权重（示例中固定为0.2）
        position_df['Weight'] = 0.2
        
        # 执行回测
        print("开始执行回测...")
        evaluate, equity_curve, fig = run_backtest(position_df, price_df=price_df, title="Backtest")
        
        # 打印结果
        if evaluate is not None:
            print("\n======== 回测结果 ========")
            print("评估指标:")
            print(evaluate)
            print("回测完成，结果已保存。")
        else:
            print("回测失败，未能生成评估指标。")
            
    except Exception as e:
        print(f"回测过程中发生错误: {str(e)}")

# 当脚本直接运行时执行main函数
if __name__ == "__main__":
    main()


