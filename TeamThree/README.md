# QF5214_Group 8 Team_3 README

# Project README

## 1. Docker Environment Setting
Provide details on how to set up the Docker environment for running the model.

### 1.1 Dockerfile
```dockerfile
# Use official Python image as a base
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including gcc
RUN apt-get update && apt-get -y install cron gcc python3-dev

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script
COPY final_sentiment_strategy.py .

# Create crontab file
RUN echo "0 13 * * * cd /app && python /app/final_sentiment_strategy.py >> /var/log/cron.log 2>&1" > /etc/cron.d/script-cron
RUN chmod 0644 /etc/cron.d/script-cron

# Apply cron job
RUN crontab /etc/cron.d/script-cron

# Create log file
RUN touch /var/log/cron.log

# Run cron in foreground
CMD ["cron", "-f"]
```

### 1.2 Running the Docker Container
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
List all necessary dependencies for running the model.
```
pandas
numpy
qlib==0.0.2.dev20
tensorflow
torch
matplotlib
scikit-learn
optuna
pandas_market_calendars
sqlalchemy
psycopg2-binary
datetime
```

#### 1.3.2 System Requirements
- OS: Linux/macOS/Windows with Docker installed
- Python: 3.9+
- Disk Space: At least 2GB for Docker image and dependencies
- Memory: At least 4GB RAM recommended for model operations

## 2. Data Processing
Below are the base python files for Data Processing.
-   [database_utils.py](#database_utilspy)
-   [alpha_generator.py](#alpha_generatorpy)

###  database_utils
database_utils.py provides a simple and reusable class, DatabaseUtils, for handling PostgreSQL database connections and queries in Python. It includes methods for connecting to the database, executing queries, fetching results, and closing connections.
- `connect(self, if_return)`: Establish a PostgreSQL database connection

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| if_return | whether return cursor, conn and engine | bool | Flase |
- `execute_query(self, query, params)`: Execute SQL queries with optional parameters

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| query | query | str | / |
| params | Variables are specified either with positional (%s) or named (%(name)s) placeholders. [Ref](https://www.psycopg.org/docs/usage.html#query-parameters) | tuple, dictionary, list | None |
- `df_to_sql_table(self,df, type_list, schema, table_name, drop_table)`: save dataframe as a new table under a given schema

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| df | The DataFrame to be inserted into the database | pandas.dataframe | \ |
| type_list | A list specifying the data types for each column in the DataFrame. It should match the order of the columns in df | list | \ |
| schema | The DataFrame to be inserted into the database | str | \ |
| table_name | The name of the table in which the DataFrame will be stored | str | \ |
| drop_table | Drop table if table exists | bool | True| 
- `set_schema(schema)`: Set Schema for the Session as default

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| scheme | scheme in database | str | / |
- fetch_results(): Fetch query results easily
- close_connection(): Ensure proper connection closure
- Handle errors and rollbacks gracefully

###  alpha_generator
alpha_generator.py provides functionality to generate WorldQuant Alpha101 factors, process stock data, index data, and sentiment data for quantitative trading models. It includes methods for data preprocessing, running alpha calculations by ticker, and retrieving data from PostgreSQL databases.

- `data_preprocessing(df)`: Prepares stock data for alpha calculation

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| df | DataFrame containing stock price and volume data | pandas.DataFrame | Flase |

- `run_by_ticker(df, tickers, alpha_indices)`: Calculates alpha factors for each ticker

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| df | Preprocessed DataFrame | pandas.DataFrame | Flase |
| tickers | List of stock tickers to process | list | Flase |
| alpha_indices | List of alpha factor indices to calculate | list | Flase |

- `generate_alphas(input_schema, input_table_name, save, output_schema, output_table_name, if_return)`: generate_alphas(input_schema, input_table_name, save, output_schema, output_table_name, if_return)

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| input_schema | Database schema for input data | str | 'datacollection' |
| input_table_name | Whether to save results to database | str | 'stock_data' |
| save | Whether to save results to database | bool | Flase |
| output_schema | Schema for output data | str | 'datacollection' |
| output_table_name | Table for storing alpha factors | str | 'alpha101' |
| if_return | Whether to return DataFrames | bool | Flase |

- `get_alpha101_table_from_db(to_csv)`: Retrieves alpha factors, stock data, and index data

| Parameter Name | Description | Type | Default Value |
|---------------|-------------|-----------------|---------------|
| to_csv | Whether to save data to CSV files | bool | Flase |

- `get_updated_sentiment_table_from_db()`: Retrieves latest sentiment data
Returns two DataFrames: live sentiment data and new sentiment data

- `get_sentiment_table_from_db()`: Retrieves historical sentiment data
Returns two DataFrames: complete sentiment history and filtered sentiment data


## 3. Model Description
Provide a brief introduction to the model, including its purpose, core methodology, and any relevant theoretical background.

### 3.1 Model Parameters
List and describe the key parameters used in the model, including their meanings, possible values, and how they impact the results.

| Parameter Name | Description | Possible Values | Default Value |
|---------------|-------------|-----------------|---------------|
| Param1 | Description of Param1 | Range/Options | Default |
| Param2 | Description of Param2 | Range/Options | Default |

### 3.2 Input Data List
Describe the input data required for the model, specifying the structure and format in a table.

| Feature Name | Description | Data Type | Example |
|-------------|-------------|-----------|---------|
| Feature1 | Description of Feature1 | Type | Example Value |
| Feature2 | Description of Feature2 | Type | Example Value |

### 3.3 Output Data List
Describe the output data generated by the model, specifying structure and format in a table.

| Output Name | Description | Data Type | Example |
|------------|-------------|-----------|---------|
| Output1 | Description of Output1 | Type | Example Value |
| Output2 | Description of Output2 | Type | Example Value |

### 3.4 Baseline Model List
List and describe baseline models used for comparison, including their methodologies and assumptions.

| Model Name | Description | Assumptions |
|------------|-------------|------------|
| Model1 | Description of Model1 | Assumption1, Assumption2 |
| Model2 | Description of Model2 | Assumption1, Assumption2 |

### 3.5 Performance Table
Compare the model's performance against baseline models using key performance metrics.

| Model Name | Metric1 | Metric2 | Metric3 |
|------------|--------|--------|--------|
| Model1 | Value1 | Value2 | Value3 |
| Model2 | Value1 | Value2 | Value3 |

## 4. Trading Strategy
Describe the trading strategy implemented using the model's outputs, including execution rules, risk management, and backtesting results.

### 4.1 Execution Rules
- Describe how the strategy decides to enter and exit trades.

### 4.2 Risk Management
- Detail risk controls such as stop-loss, take-profit, and position sizing.

### 4.3 Backtesting Results
- Provide insights from historical performance testing, including key statistics and visualizations.

## 5. Conclusion
Summarize findings, limitations, and potential future improvements.

## 6. References
List any relevant papers, articles, or resources that informed the model and strategy design.

