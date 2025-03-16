import backtrader as bt
import pandas as pd
import numpy as np
import talib


# load historical prices, predicted returns, and sentiment scores
data = # parse_dates=["Date"], index_col="Date"
predicted_returns = # parse_dates=["Date"], index_col="Date"
sentiments = # parse_dates=["Date"], index_col="Date"

# if the input type is dataframe, we can use the following code
data = data.merge(predicted_returns, on=["Date", "Stock"], how="left")
data = data.merge(sentiments, on=["Date", "Stock"], how="left")

# Backtrader Strategy
class PredictedReturnStrategy(bt.Strategy):
    params = dict(rsi_period=14, atr_period=14, stop_loss_mult=2, take_profit_mult=3)
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.params.rsi_period)
        self.atr = bt.indicators.ATR(self.datas[0], period=self.params.atr_period)
    
    def next(self):
        predicted_return = self.datas[0].predicted_return[0]
        sentiment = self.datas[0].sentiment_score[0]
        atr_value = self.atr[0]
        
        if self.position:
            # Check stop-loss and take-profit conditions
            if self.dataclose[0] <= self.stop_loss or self.dataclose[0] >= self.take_profit:
                self.close()
        
        else:
            if self.rsi[0] < 30 and predicted_return > 0:
                self.stop_loss = self.dataclose[0] - (self.params.stop_loss_mult * abs(predicted_return) * atr_value)
                self.take_profit = self.dataclose[0] + (self.params.take_profit_mult * abs(predicted_return) * atr_value)
                self.buy()
            
            elif self.rsi[0] > 70 and predicted_return < 0:
                self.stop_loss = self.dataclose[0] + (self.params.stop_loss_mult * abs(predicted_return) * atr_value)
                self.take_profit = self.dataclose[0] - (self.params.take_profit_mult * abs(predicted_return) * atr_value)
                self.sell()

# Load data into Backtrader
class PandasDataExtended(bt.feeds.PandasData):
    lines = ('predicted_return', 'sentiment_score',)
    params = (('predicted_return', -1), ('sentiment_score', -1))

cerebro = bt.Cerebro()
cerebro.addstrategy(PredictedReturnStrategy)

data_feed = PandasDataExtended(dataname=data)
cerebro.adddata(data_feed)

# Set cash and commission
cerebro.broker.set_cash(100000)
cerebro.broker.setcommission(commission=0.001)

# benchmark (S&P 500)
benchmark = # from database parse_dates=["Date"], index_col="Date"
cerebro.addobserver(bt.observers.Benchmark, data=benchmark)

# Run backtest
results = cerebro.run()

# Performance Metrics
portfolio_value = cerebro.broker.getvalue()
sharpe_ratio = (np.mean(results[0].analyzers.returns.get_analysis()) / np.std(results[0].analyzers.returns.get_analysis())) * np.sqrt(252)
drawdown = results[0].analyzers.drawdown.get_analysis()['max']['drawdown']
win_rate = results[0].analyzers.trades.get_analysis()['won']['total'] / (results[0].analyzers.trades.get_analysis()['won']['total'] + results[0].analyzers.trades.get_analysis()['lost']['total'])

print(f"Final Portfolio Value: ${portfolio_value:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {drawdown:.2f}%")
print(f"Win Rate: {win_rate:.2%}")
cerebro.plot()
