#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
from pathlib import Path
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from StreamlitAPP.utils.database import read_data, write_data
from StreamlitAPP.utils.visualization import plot_results, calculate_performance_metrics



#%%


#%% 持仓数据
try:
    position_df = read_data('QF5214.tradingstrategy.dailytrading')
    #print(position_df.head())
except Exception as e:
    print(f"读取持仓数据出错: {str(e)}")
    position_df = None
position_start_date=position_df['Date'].min()
position_end_date=position_df['Date'].max()
print(position_end_date)
#%% 行情数据
try:
    price_df = read_data(f'SELECT * FROM datacollection.stock_data WHERE "Date" >= \'{position_start_date}\' AND "Date" <= \'{position_end_date}\'')
    #print(price_df.head())
except Exception as e:
    print(f"读取股票价格数据出错: {str(e)}")
    price_df = None

#%% 指数数据
try:
    index_df = read_data(f'SELECT * FROM datacollection.index_data WHERE "Date" >= \'{position_start_date}\' AND "Date" <= \'{position_end_date}\' AND "Index" = \'NASDAQ 100\'')
    #print(index_df.head())
    index_df = index_df[['Date', 'Adj_Close']]
except Exception as e:
    print(f"读取指数数据出错: {str(e)}")
    index_df = None


#%% 完成数据准备
long_position_df=position_df[position_df['Position_Type']=='Long']
# 将Short仓位的权重取反
short_position_df = position_df[position_df['Position_Type']=='Short']
short_position_df['Weight'] = short_position_df['Weight'] * -1
position_df = pd.concat([long_position_df, short_position_df])
position_df.sort_values(by='Date', inplace=True)
print(position_df.head(15))
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
merged_df = pd.merge(merged_df, index_df, on=['Date'], how='left', suffixes=('', '_NASDAQ100'))
merged_df.dropna(subset=['Weight','Adj_Close','Adj_Close_NASDAQ100'], inplace=True)
merged_df.sort_values(by='Date', inplace=True)
merged_df.reset_index(drop=True, inplace=True)
#print(merged_df.tail(15))


#%% 向量化回测
def run_backtest(merged_df, title="Backtest", save_results=True, data_source='QF5214.visualization.backtest_results', is_long_short=True):
    """
    执行向量化回测
    
    Args:
        merged_df: DataFrame, 已合并的持仓和价格数据
        title: str, 回测标题
        save_results: bool, 是否保存结果
        data_source: str, 保存结果的数据库表路径，默认为'QF5214.visualization.backtest_results'
        is_long_short: bool, 是否为多空回测，默认为True。如果为False，则只使用多头仓位
        
    Returns:
        tuple: (evaluate, equity_curve, fig)
            - evaluate: 评估指标DataFrame
            - equity_curve: 净值曲线DataFrame
            - fig: Plotly图表对象
    """
    print("开始向量化回测...")
    
    # 确保数据不为空
    if merged_df is None or merged_df.empty:
        print("错误: 合并数据为空")
        return None, None, None
    
    # 确保日期列是datetime类型
    if 'Date' in merged_df.columns and merged_df['Date'].dtype != 'datetime64[ns]':
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    # 根据回测类型处理持仓数据
    position_df = merged_df[['Date', 'Ticker', 'Weight', 'Position_Type']].copy()
    
    if is_long_short:
        # 多空回测：所有权重除以2
        position_df['Weight'] = position_df['Weight'] / 2
        print("执行多空回测，所有权重已除以2")
        prefix = "long-short"
    else:
        # 单多回测：只使用多头仓位
        position_df = position_df[position_df['Position_Type'] == 'Long']
        print("执行单多回测，只使用多头仓位")
        prefix = "long-only"
    
    # 获取所有交易日期
    all_dates = sorted(position_df['Date'].unique())
    
    # 初始化净值曲线，根据回测类型决定是否添加基准指数
    if is_long_short:
        equity_curve = pd.DataFrame(index=all_dates, columns=['Strategy'])
    else:
        equity_curve = pd.DataFrame(index=all_dates, columns=['Strategy', 'NASDAQ100', 'Excess_Return'])
    equity_curve.fillna(1.0, inplace=True)
    
    # 按日期遍历计算收益
    for i in tqdm(range(1, len(all_dates))):
        current_date = all_dates[i]
        prev_date = all_dates[i-1]
        
        # 获取前一天的持仓
        prev_positions = position_df[position_df['Date'] == prev_date]
        
        # 获取当天和前一天的合并数据
        current_data = merged_df[merged_df['Date'] == current_date]
        prev_data = merged_df[merged_df['Date'] == prev_date]
        
        # 如果不是多空回测，计算基准指数收益率
        if not is_long_short:
            current_index = current_data['Adj_Close_NASDAQ100'].iloc[0] if not current_data.empty else None
            prev_index = prev_data['Adj_Close_NASDAQ100'].iloc[0] if not prev_data.empty else None
            
            if current_index is not None and prev_index is not None and prev_index > 0:
                index_return = current_index / prev_index - 1
                equity_curve.loc[current_date, 'NASDAQ100'] = equity_curve.loc[prev_date, 'NASDAQ100'] * (1 + index_return)
            else:
                equity_curve.loc[current_date, 'NASDAQ100'] = equity_curve.loc[prev_date, 'NASDAQ100']
        
        # 检查是否有足够的数据
        if prev_positions.empty or current_data.empty or prev_data.empty:
            print(f"警告: {prev_date}或{current_date}缺少必要数据，跳过该日计算")
            equity_curve.loc[current_date, 'Strategy'] = equity_curve.loc[prev_date, 'Strategy']
            if not is_long_short:
                equity_curve.loc[current_date, 'Excess_Return'] = equity_curve.loc[prev_date, 'Excess_Return']
            continue
        
        # 合并持仓和价格数据
        merged_positions = pd.merge(
            prev_positions[['Ticker', 'Weight']],  # 取前一天的持仓
            current_data[['Ticker', 'Adj_Close']],  # 当天的收盘价
            on='Ticker',
            how='inner'  # 内连接，只保留同时存在于两个表中的股票
        )
        
        # 再合并前一天的价格
        merged_positions = pd.merge(
            merged_positions,
            prev_data[['Ticker', 'Adj_Close']],  # 前一天的收盘价
            on='Ticker',
            how='inner',
            suffixes=('', '_prev')  # 区分当天和前一天的价格
        )
        
        # 检查合并后是否有数据
        if merged_positions.empty:
            print(f"警告: {current_date}没有可用的股票数据，跳过该日计算")
            equity_curve.loc[current_date, 'Strategy'] = equity_curve.loc[prev_date, 'Strategy']
            if not is_long_short:
                equity_curve.loc[current_date, 'Excess_Return'] = equity_curve.loc[prev_date, 'Excess_Return']
            continue
        
        # 计算每支股票的收益率
        merged_positions['Return'] = merged_positions['Adj_Close'] / merged_positions['Adj_Close_prev'] - 1
        
        # 计算加权收益率
        merged_positions['Weighted_Return'] = merged_positions['Weight'] * merged_positions['Return']
        
        # 计算当天的总收益率
        day_return = merged_positions['Weighted_Return'].sum()
        
        # 更新净值曲线
        equity_curve.loc[current_date, 'Strategy'] = equity_curve.loc[prev_date, 'Strategy'] * (1 + day_return)
        
        # 如果不是多空回测，计算超额收益
        if not is_long_short:
            strategy_return = equity_curve.loc[current_date, 'Strategy'] / equity_curve.loc[prev_date, 'Strategy'] - 1
            benchmark_return = equity_curve.loc[current_date, 'NASDAQ100'] / equity_curve.loc[prev_date, 'NASDAQ100'] - 1
            excess_return = strategy_return - benchmark_return
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
        
        # 只在单多回测时计算Beta
        if not is_long_short:
            # 计算滚动3个月Beta
            if 'NASDAQ100' in daily_returns.columns:
                # 使用协方差计算beta
                rolling_cov = daily_returns['Strategy'].rolling(window=rolling_window).cov(daily_returns['NASDAQ100'])
                rolling_var = daily_returns['NASDAQ100'].rolling(window=rolling_window).var()
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
            write_data(equity_curve_db, f"{data_source}_{prefix}_equity")
            print(f"净值曲线已保存到 {data_source}_{prefix}_equity 表中")
            
            # 重置索引以便将类型作为列保存到数据库
            evaluate_db = evaluate.reset_index().rename(columns={'index': 'Type'})
            
            # 保存评估指标
            write_data(evaluate_db, f"{data_source}_{prefix}_metrics")
            print(f"评估指标已保存到 {data_source}_{prefix}_metrics 表中")
        except Exception as e:
            print(f"保存数据到数据库失败: {str(e)}")
            
            # 创建备份文件目录
            result_dir = 'backtest_results'
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # 备份保存到文件
            equity_curve.to_csv(os.path.join(result_dir, f"{prefix}_backtest_equity.csv"))
            evaluate.to_csv(os.path.join(result_dir, f"{prefix}_backtest_metrics.csv"))
            print(f"数据已备份保存到 {result_dir} 目录")
    
    # 绘制结果
    fig = plot_results(equity_curve, evaluate, title, is_long_short=is_long_short)
    
    # 如果成功创建了图形，可以保存图形文件
    if fig is not None and save_results:
        result_dir = 'backtest_results'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # 保存HTML前先检查文件是否存在，存在则删除
        html_file_path = os.path.join(result_dir, f"{prefix}_backtest_chart.html")
        if os.path.exists(html_file_path):
            try:
                os.remove(html_file_path)
                print(f"已删除现有文件: {html_file_path}")
            except Exception as e:
                print(f"删除文件失败: {str(e)}")
        
        fig.write_html(html_file_path) 
    
    return evaluate, equity_curve, fig


#%%
# 添加一个main函数，方便脚本单独运行
def main():
    try:
        # 使用已经在文件开头准备好的merged_df直接进行回测
        global merged_df
        
        # 执行多空回测
        print("开始执行多空回测...")
        evaluate_long_short, equity_curve_long_short, fig_long_short = run_backtest(
            merged_df, 
            title="Long-Short Strategy", 
            is_long_short=True,
        )
        
        # 执行单多回测
        print("\n开始执行单多回测...")
        evaluate_long_only, equity_curve_long_only, fig_long_only = run_backtest(
            merged_df, 
            title="Long-Only Strategy", 
            is_long_short=False,
        )
        
        # 打印多空回测结果
        if evaluate_long_short is not None:
            print("\n======== 多空回测结果 ========")
            print("评估指标:")
            print(evaluate_long_short)
        else:
            print("多空回测失败，未能生成评估指标。")
            
        # 打印单多回测结果
        if evaluate_long_only is not None:
            print("\n======== 单多回测结果 ========")
            print("评估指标:")
            print(evaluate_long_only)
        else:
            print("单多回测失败，未能生成评估指标。")
            
        print("回测完成，结果已保存。")
            
    except Exception as e:
        print(f"回测过程中发生错误: {str(e)}")

#%%

# 当脚本直接运行时执行main函数
if __name__ == "__main__":
    main() 

#%%
