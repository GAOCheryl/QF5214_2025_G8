# QF5214_Group 8 Team_3 README

## 1. Overview
This folder provides a framework for computing WorldQuant Alpha 101 factors and storing the results in a PostgreSQL database. The core functionality is implemented in Python and supports data preprocessing, factor computation, and database interaction.

### Base Files
Below are the essential Python files that form the backbone of the repository:
-   [alpha_generator.py](#alpha_generatorpy) – Computes Alpha 101 factors for each stock and stores them in a PostgreSQL database.
-   [utils.py](#utilspy) – Contains the `Alphas` class, which implements WorldQuant Alpha 101 factors.

## 2. Function Description
### 2.1 alpha_generator.py
The script `alpha_generator.py` extracts stock data from a PostgreSQL database, preprocesses it, computes WorldQuant Alpha 101 factors, and stores the results back into the database.

#### **Main Functions:**
- `data_preprocessing(df)`: Prepares the stock data for alpha factor calculations.

| Parameter Name | Description | Type |
|---------------|-------------|------|
| `df` | Input DataFrame containing stock data | `pandas.DataFrame` |
- `run_by_ticker(df, tickers, alpha_indices)`: Computes specified Alpha 101 factors for each stock.

| Parameter Name | Description | Type |
|---------------|-------------|------|
| `df` | Preprocessed stock data | `pandas.DataFrame` |
| `tickers` | List of unique stock tickers | `list` |
| `alpha_indices` | List of Alpha 101 indices to compute | `list[int]` |

- `df_to_sql_table(df, type_list, schema, table_name)`: Stores the computed alpha factors into the PostgreSQL database.

| Parameter Name | Description | Type |
|---------------|-------------|------|
| `df` | DataFrame to be inserted into the database | `pandas.DataFrame` |
| `type_list` | List of column data types | `list` |
| `schema` | Database schema name | `str` |
| `table_name` | Name of the table where data will be stored | `str` |

### 2.2 utils.py
The utils.py module provides auxiliary functions for common operations used in Alpha 101 computations, including rolling statistics, ranking, scaling, and time-series transformations.
#### Main Functions:
- `ts_sum(df, window)`: Computes rolling sum over a specified window.
- `sma(df, window)`: Computes simple moving average (SMA).
- `stddev(df, window)`: Computes rolling standard deviation.
- `correlation(x, y, window)`: Computes rolling correlation between two series.
- `covariance(x, y, window)`: Computes rolling covariance between two series.
- `ts_min(df, window)`: Computes rolling minimum value.
- `ts_max(df, window)`: Computes rolling maximum value.
- `delta(df, period)`: Computes the difference between the current value and the value period days ago.
- `delay(df, period)`: Computes lagged values.
- `rank(df)`: Computes cross-sectional rank.
- `scale(df, k)`: Scales the DataFrame such that the sum of absolute values equals k.
- `decay_linear(df, period)`: Computes liner weighted moving average.
- `IndNeutralize(df, level)`: Cross-sectioally neutralizes values within a given industry sector.
These functions are essential for constructing and transforming factors in the Alphas class.


## 3. License
The [utils.py](#utilspy) is based on open-source implementations of WorldQuant Alpha 101 formulas. If you modify or use this code, please cite the original repository: [yli188/WorldQuant_alpha101_code](https://github.com/yli188/WorldQuant_alpha101_code).

