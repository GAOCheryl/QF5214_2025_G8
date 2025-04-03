# QF5214_Group 8 Team_3 README

# Project README

This project implements the MASTER (Multi-head Attention-based Sentiment and Technical factors with External Regulation) model—a deep learning strategy that integrates technical indicators with sentiment analysis—to forecast stock returns and drive a quantitative trading strategy. This repository contains the model code, data processing pipelines, and scripts for backtesting and execution.

## 1. Docker Environment Setting
This section explains how to build and run the Docker container that hosts the trading strategy.

### 1.1 Dockerfile
Below is an annotated Dockerfile that sets up the necessary environment, installs dependencies, schedules a daily run at 1:00 PM, and runs cron in the foreground:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including gcc and set timezone
RUN apt-get update && apt-get -y install cron gcc python3-dev tzdata git

# Extra dependencies that might be needed for some Python packages
RUN apt-get install -y --no-install-recommends build-essential libpq-dev

# Set timezone to Beijing time
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Copy requirements.txt and remove qlib entry
COPY requirements.txt /tmp/original_requirements.txt
RUN cat /tmp/original_requirements.txt | grep -v qlib > /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install qlib directly from GitHub
RUN pip install --no-cache-dir git+https://github.com/microsoft/qlib.git@main

# Copy all required Python scripts and ensure execution permissions
COPY final_sentiment_strategy.py .
COPY master_strategy.py .
COPY base_model_strategy.py .
COPY database_utils.py .
# Copy the alpha_101 folder
COPY alpha_101 ./alpha_101/
COPY model ./model/
RUN chmod +x final_sentiment_strategy.py

# Create data directories that might be needed
RUN mkdir -p data/Input data/Output model/Output

# Create a bash script to run your Python script with proper environment
RUN echo '#!/bin/bash' > /app/run_script.sh && \
    echo 'cd /app' >> /app/run_script.sh && \
    echo 'export PYTHONPATH=$PYTHONPATH:/data:/app' >> /app/run_script.sh && \
    echo '/usr/local/bin/python /app/final_sentiment_strategy.py' >> /app/run_script.sh
RUN chmod +x /app/run_script.sh

# Set up cron job with proper environment
RUN (crontab -l 2>/dev/null; echo "0 13 * * * /app/run_script.sh >> /var/log/cron.log 2>&1") | crontab -

# Create log file
RUN touch /var/log/cron.log && chmod 0666 /var/log/cron.log

# Start cron in foreground to keep container running
CMD service cron start && tail -f /var/log/cron.log
```

### 1.2 Running the Docker Container
Follow these steps to build and run your Docker container:
```sh
# Build the Docker image
docker build -t sentiment-strategy .

# Run the container in detached mode
docker run -d --name sentiment-daily sentiment-strategy

# Check container logs to verify it's running correctly
docker logs sentiment-daily

# To stop and remove the container if needed
# docker stop sentiment-daily
# docker rm sentiment-daily
```

### 1.3 Requirement List
#### 1.3.1 Python Dependencies
Include the following in your <code>requirements.txt</code> file:
```
pandas
numpy
pyqlib
pickle5
optuna
pandas_market_calendars
sqlalchemy
psycopg2-binary
datetime
tensorflow
torch
matplotlib
scikit-learn
```

#### 1.3.2 System Requirements
- OS: Linux/macOS/Windows with Docker installed
- Python: 3.9+
- Disk Space: At least 2GB for Docker image and dependencies
- Memory: At least 4GB RAM recommended for model operations

## 2. Data Processing
The repository includes Python modules for data processing. Key scripts include:
-   [database_utils.py](#database_utilspy)
-   [alpha_generator.py](#alpha_generatorpy)

###  database_utils
<code>database_utils.py</code> provides a simple and reusable class, DatabaseUtils, for handling PostgreSQL database connections and queries in Python. It includes methods for connecting to the database, executing queries, fetching results, and closing connections.
- `connect(self, if_return)`: Establish a PostgreSQL database connection

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| if_return | whether return cursor, conn and engine | bool | False |
- `execute_query(self, query, params)`: Execute SQL queries with optional parameters

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| query | query | str | / |
| params | Variables are specified either with positional (%s) or named (%(name)s) placeholders. [Ref](https://www.psycopg.org/docs/usage.html#query-parameters) | tuple, dictionary, list | None |
- `df_to_sql_table(self,df, type_list, schema, table_name, drop_table)`: save dataframe as a new table under a given schema

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| df | The DataFrame to be inserted into the database | pandas.dataframe | / |
| type_list | A list specifying the data types for each column in the DataFrame. It should match the order of the columns in df | list | / |
| schema | The DataFrame to be inserted into the database | str | / |
| table_name | The name of the table in which the DataFrame will be stored | str | / |
| drop_table | Drop table if table exists | bool | True| 
- `set_schema(schema)`: Set Schema for the Session as default

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| scheme | scheme in database | str | / |
- fetch_results(): Fetch query results easily
- close_connection(): Ensure proper connection closure
- Handle errors and rollbacks gracefully

###  alpha_generator
<code>alpha_generator.py</code> provides functionality to generate WorldQuant Alpha101 factors, process stock data, index data, and sentiment data for quantitative trading models. It includes methods for data preprocessing, running alpha calculations by ticker, and retrieving data from PostgreSQL databases.

- `data_preprocessing(df)`: Prepares stock data for alpha calculation

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| df | DataFrame containing stock price and volume data | pandas.DataFrame | False |

- `run_by_ticker(df, tickers, alpha_indices)`: Calculates alpha factors for each ticker

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| df | Preprocessed DataFrame | pandas.DataFrame | False |
| tickers | List of stock tickers to process | list | False |
| alpha_indices | List of alpha factor indices to calculate | list | False |

- `generate_alphas(input_schema, input_table_name, save, output_schema, output_table_name, if_return)`: generate_alphas(input_schema, input_table_name, save, output_schema, output_table_name, if_return)

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| input_schema | Database schema for input data | str | 'datacollection' |
| input_table_name | Whether to save results to database | str | 'stock_data' |
| save | Whether to save results to database | bool | False |
| output_schema | Schema for output data | str | 'datacollection' |
| output_table_name | Table for storing alpha factors | str | 'alpha101' |
| if_return | Whether to return DataFrames | bool | False |

- `get_alpha101_table_from_db(to_csv)`: Retrieves alpha factors, stock data, and index data

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| to_csv | Whether to save data to CSV files | bool | False |

- `get_updated_sentiment_table_from_db()`: Retrieves latest sentiment data
Returns two DataFrames: live sentiment data and new sentiment data

- `get_sentiment_table_from_db()`: Retrieves historical sentiment data
Returns two DataFrames: complete sentiment history and filtered sentiment data


## 3. Model Description
The MASTER model is designed to forecast stock returns using a multi-headed attention mechanism that fuses technical factors with sentiment indicators.

### 3.1 Model Parameters
List and describe the key parameters used in the model, including their meanings, possible values, and how they impact the results.

| Parameter Name | Description | Possible Values | Default Value |
|---------------|-------------|-----------------|---------------|
| d_feat | Number of price-based features | Integer > 0 | 106 |
| d_model | Hidden dimension size for model | Integer > 0 | 512 |
| t_nhead | Number of attention heads for temporal features | Integer > 0 | 4 |
| s_nhead | Number of attention heads for sentiment features | Integer > 0 | 2 |
| T_dropout_rate | Dropout rate for temporal feature attention | 0.0-1.0 | 0.7 |
| S_dropout_rate | Dropout rate for sentiment feature attention | 0.0-1.0 | 0.7 |
| beta | Weight factor for sentiment regulation | Float > 0 | 5 |
| gate_input_start_index | Starting index for sentiment gate input | Integer ≥ 0 | 106 |
| gate_input_end_index | Ending index for sentiment gate input | Integer > gate_input_start_index | 130 |
| n_epochs | Number of training epochs | Integer > 0 | 100 |
| lr | Learning rate | Float > 0 | 1e-4 |
| GPU | GPU device ID (-1 for CPU) | Integer ≥ -1 | 0 |
| train_stop_loss_thred | Threshold for early stopping | Float > 0 | 0.0007 |

### 3.2 Input Data List
The model requires the following inputs:

| Feature Name | Description | Data Type | Example |
|-------------|-------------|-----------|---------|
| Price Data | Stock OHLC and volume information | Float | Close: 150.23 |
| Alpha Factors | WorldQuant Alpha101 technical indicators | Float | alpha001: 0.245 |
| Sentiment Metrics | Emotion/sentiment scores from text analysis | Float | Positive: 0.78 |
| Index Data | Market benchmark data | Float | S&P500_Close: 4200.5 |
| Market Cap | Company market capitalization | Float | 2.1e9 |
| Return | Daily price return (prediction target) | Float | 0.023 |

The input data is processed into a qlib-compatible format using TSDataSampler with a step length of 8, allowing the model to analyze patterns across multiple trading days.

### 3.3 Output Data List
Describe the output data generated by the model, specifying structure and format in a table.

| Output Name | Description | Data Type | Example |
|------------|-------------|-----------|---------|
| Predicted_Return | Forecasted stock returns | pandas.Series | 0.0145 |
| real_returns | Series of actual returns indexed by datetime and instrument | pandas.Series | 0.023 |
| real_prices | Series of stock prices indexed by datetime and instrument | pandas.Series | 150.23 |
| market_cap | Series of market capitalizations indexed by datetime and instrument | pandas.Series | 2.1e9 |
| metrics | Dictionary containing model performance metrics (IC, ICIR, RIC, RICIR) | dict | {'IC': 0.215, 'ICIR': 1.72} |

The model's outputs are used to construct a trading strategy that selects the top and bottom N stocks based on predicted returns. These stocks are assigned equal weights within their respective long/short position types, and trading costs are factored into position sizing.

### 3.4 Baseline Model List

| Model Name | Description | Assumptions |
|------------|-------------|------------|
| MASTER without Sentiment | A version of the MASTER model that excludes sentiment data from the prediction process | - Technical factors alone can predict stock movement<br>- Market behavior is primarily driven by price patterns<br>- Historical price information is sufficient for prediction |

The baseline MASTER model follows the same architecture as the full model but eliminates sentiment factors from consideration. This provides a direct comparison to evaluate the incremental value of sentiment analysis in stock prediction.
Key differences in the baseline model:

Uses only price-based features (d_feat = 97 vs. 106 in the full model)
Maintains the same multi-head attention mechanism but focuses exclusively on technical factors
Loads parameters from 'model/model_training_without_sentiment_0.pkl' rather than the sentiment-enabled version
Outputs predictions to 'data/Output/predictions_without_sentiment.csv'

The baseline model serves as a control to quantify the predictive power added by incorporating sentiment analysis. By comparing performance metrics such as Information Coefficient (IC) and Rank Information Coefficient (RIC) between the two models, we can measure the value of sentiment data in improving prediction accuracy across different market conditions.

Model parameters:
- d_feat = 97 (number of technical features)
- d_model = 128 (hidden dimension size)
- t_nhead = 4 (temporal attention heads)
- s_nhead = 2 (spatial attention heads)
- dropout = 0.7 (dropout rate for regularization)
- Learning rate = 1e-5 (smaller than full model to prevent overfitting with fewer features)

This baseline offers a more conservative approach to stock prediction, less subject to noise from sentiment measures but potentially missing important market reaction signals contained in text-based data.

### 3.5 Performance Table
Compare the model's performance against baseline models using key performance metrics.

| Model Name | IC | RIC | RIC | RICIR |
|------------|--------|--------|--------|--------|
| MASTER without Sentiment | 0.1016 | 0.0941 | 1.68 | 1.54 |
| MASTER with Sentiment | 0.1054 | 0.0965 | 1.75 | 1.62 |

The performance table demonstrates that incorporating sentiment features into the MASTER model produces measurable improvements across all evaluation metrics. The Information Coefficient (IC) shows a 3.7% improvement, while the Rank Information Coefficient (RIC) increases by 2.6% when sentiment analysis is included.
These improvements, while modest in absolute terms, are significant in quantitative finance where even small predictive edges can translate to substantial returns when applied across a diversified portfolio. The higher IC and RIC values indicate that sentiment data captures additional signals not present in technical indicators alone, validating the multi-modal approach to stock prediction.
The ICIR and RICIR metrics, which measure the consistency of prediction accuracy over time, also show improvement with the sentiment-enhanced model, suggesting that sentiment features contribute to more stable performance across varying market conditions.

## 4. Trading Strategy
Our trading strategy leverages the predictive power of the MASTER model to generate daily trading signals for a portfolio of stocks, with a focus on capitalizing on both upward and downward price movements.

### 4.1 Execution Rules
The strategy operates on the following core execution principles:

1. Signal Generation: 
- Each day, stocks are ranked based on their predicted returns from the MASTER model
- Top 5 stocks with highest predicted returns are selected for long positions
- Bottom 5 stocks with lowest predicted returns are selected for short positions (in long-short strategies)

2. Position Management:
- If a stock was selected yesterday but not today, the position is closed at today's price
- If a stock continues to be selected, it remains in the portfolio without rebalancing
- New selections are added to the portfolio with appropriate position sizes

3. Rebalancing Frequency:
- The portfolio is adjusted daily based on the selection signals
- This high-frequency approach aims to capture short-term market inefficiencies identified by the model

4. Trade Execution:
- Trades are assumed to be executed at market close prices
- A 0.1% transaction cost is applied to both entry and exit trades

### 4.2 Risk Management
The strategy incorporates several risk management techniques:

1. Position Sizing:
- Initial capital of $1,000,000 is allocated across selected positions
- Two weighting methods are employed:
    - Equal Weight: Each position receives 1/N of the allocated capital
    - Market Cap Weighted: Positions are sized proportionally to their market capitalization

2. Diversification:
- Portfolio maintains exposure to 5 stocks on each side (long/short)
- Selections span multiple sectors to reduce sector-specific risk

3. Dollar Neutrality (for long-short strategies):
- Equal capital allocation between long and short sides
- Provides partial hedge against overall market movements

4. Transaction Cost Management:
- 0.1% trading cost per transaction is factored into position sizing
- Positions that remain in the portfolio are not churned unnecessarily

### 4.3 Backtesting Results
The strategy was backtested across four variants:

| Strategy | Approach | Weighting | Sharpe Ratio | Max Drawdown |
|------------|--------|--------|--------|--------|
| Strategy 1 | Long-Only | Equal Weight | 0.162 | -13.64% |
| Strategy 2 | Long-Only | Market Cap | 0.044 | -16.93% |
| Strategy 3 | Long-Short | Equal Weight | 0.381 | -3.50% |
| Strategy 4 | Long-Short | Market Cap | 0.155 | -7.89% |

Key Insights:
- Strategy 3 (Equal Weight and Long-Short) significantly outperforms all other approaches with the highest Sharpe Ratio and lowest drawdown
- Market Cap weighting consistently underperforms equal weighting
- Long-short approaches demonstrate better risk-adjusted returns than long-only strategies
- Equal weighting appears to better capture the alpha generated by the MASTER model predictions

The superior performance of Strategy 3 suggests that:
- The model effectively identifies both outperforming and underperforming stocks
- Neutralizing market exposure improves risk-adjusted returns
- Equal weighting avoids concentration risk and better utilizes model signals across the portfolio

## 5. Running the Model
The MASTER model integrates sentiment and technical factors to forecast returns. Refer to the following files:
- <code> master_strategy.py</code> : Contains the core model architecture and prediction routines.
- <code>final_sentiment_strategy.py</code> : Implements the complete trading strategy using model predictions.
- <code>base_model_strategy.py</code> : Provides baseline model implementations for comparison.

After predictions, trading signals are generated, positions are sized (accounting for trading costs), and orders are executed based on daily rebalancing.

## 6. Deployment and Data Storage
### 6.1 CSV & Database Integration
- Updated predictions and allocation files are stored as CSV files.
- The final trading allocations are also inserted into a PostgreSQL database table (tradingstrategy.dailytrading) using SQLAlchemy.
- More details of the database allocation can be seen in <code>MASTER-master/README.md</code>

### 6.2 PostgreSQL Connection

Ensure that your PostgreSQL credentials (user, password, host, port, and database name) are configured properly. The code uses SQLAlchemy’s engine to connect and insert data.

## 7. References
```latex
@inproceedings{li2024master,
  title={Master: Market-guided stock transformer for stock price forecasting},
  author={Li, Tong and Liu, Zhaoyang and Shen, Yanyan and Wang, Xue and Chen, Haokun and Huang, Sen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={162--170},
  year={2024}
}
```