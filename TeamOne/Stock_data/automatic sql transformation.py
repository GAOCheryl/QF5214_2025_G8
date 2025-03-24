#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# libraries
import pandas as pd
import sys

import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from io import StringIO


# In[ ]:


# Database connection parameters
database = "QF5214"
username = "postgres"
password = "qf5214"
host = "134.122.167.14"
port = 5555


# In[20]:


# Get the previous day's date
today = pd.Timestamp.today().strftime("%Y-%m-%d")
previous_day = (pd.to_datetime(today) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
print(previous_day)


# # Transform Stock Data

# In[ ]:


# Fill NaN with 0
csv_file_path = '/root/automation/stock_data.csv'
df = pd.read_csv(csv_file_path)

# Check if the DataFrame is empty
if df.empty:
    print("The DataFrame is empty. Terminating the program.")
    sys.exit()  # exit the program

# Get the first date in the DataFrame
sample_date = pd.to_datetime(df['Date'].iloc[0]).strftime("%Y-%m-%d")

# Compare if the first date equals the previous day's date
if sample_date != previous_day:
    print(f"The first date in the DataFrame ({sample_date}) does not match {previous_day}. Terminating the program.")
    sys.exit()  # Exit if the first date does not match previous_day

df.fillna(0, inplace=True)

df.to_csv(csv_file_path, index=False)
print("The NaN values have been replaced with 0 in the stock_data file.")


# In[10]:


# CSV file path
csv_file_path = '/root/automation/stock_data.csv'

# Extract schema and table name from CSV file name
schema_name = "datacollection"
table_name = os.path.splitext(os.path.basename(csv_file_path))[0]

# Read CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Define columns that should be stored as numerical types
numeric_columns = {
    "Open": "DOUBLE PRECISION",
    "High": "DOUBLE PRECISION",
    "Low": "DOUBLE PRECISION",
    "Close": "DOUBLE PRECISION",
    "Adj_Close": "DOUBLE PRECISION",
    "Volume": "DOUBLE PRECISION",
    "PE": "DOUBLE PRECISION",
    "PB": "DOUBLE PRECISION",
    "PS": "DOUBLE PRECISION",
    "ROE": "DOUBLE PRECISION",
    "PM": "DOUBLE PRECISION",
    "IN": "DOUBLE PRECISION",
    "Market_Cap": "DOUBLE PRECISION"
}

# Define columns that should be stored as datetime
datetime_columns = {
    "Date": "DATE" 
}

# Convert necessary columns to appropriate types in pandas
for col in numeric_columns.keys():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, setting invalid values as NaN

for col in datetime_columns.keys():
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime, setting invalid values as NaT

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname=database, 
    user=username, 
    password=password, 
    host=host, 
    port=port
)
cursor = conn.cursor()

# Ensure the schema exists
cursor.execute(sql.SQL("SET search_path TO {};").format(sql.Identifier(schema_name)))

# Check if the table exists
cursor.execute(sql.SQL("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = %s AND table_name = %s
    );
"""), (schema_name, table_name))

table_exists = cursor.fetchone()[0]

if not table_exists:
    print(f"Table {schema_name}.{table_name} does not exist. Please create the table first.")
    sys.exit()  # Terminate the program if the table does not exist

# Use COPY to insert data in bulk
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False, header=False, date_format='%Y-%m-%d %H:%M:%S')  # 确保日期格式正确
csv_buffer.seek(0)

# Use COPY to insert the data from the StringIO buffer
cursor.copy_expert(
    sql.SQL("COPY {}.{} FROM STDIN WITH CSV NULL 'NaN'").format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name)
    ),
    csv_buffer
)

# Commit changes and close the connection
conn.commit()
cursor.close()
conn.close()

print(f"Stock data successfully written into {schema_name}.{table_name}")


# # Transform Index Data

# In[ ]:


# Fill NaN with 0
index_file_path = '/root/automation/index_data.csv'
index_df = pd.read_csv(index_file_path)

# Check if the DataFrame is empty
if index_df.empty:
    print("The DataFrame is empty. Terminating the program.")
    sys.exit()  # exit the program

# Get the first date in the DataFrame
index_sample_date = pd.to_datetime(index_df['Date'].iloc[0]).strftime("%Y-%m-%d")

# Compare if the first date equals the previous day's date
if index_sample_date != previous_day:
    print(f"The first date in the DataFrame ({index_sample_date}) does not match {previous_day}. Terminating the program.")
    sys.exit()  # Exit if the first date does not match previous_day

index_df.fillna(0, inplace=True)

index_df.to_csv(index_file_path, index=False)
print("The NaN values have been replaced with 0 in the index_data file.")


# In[16]:


# CSV file path
index_file_path = '/root/automation/index_data.csv'

# Extract schema and table name from CSV file name
schema_name = "datacollection"
table_name = os.path.splitext(os.path.basename(index_file_path))[0]

# Read CSV file into a DataFrame
index_df = pd.read_csv(index_file_path)

# Define columns that should be stored as numerical types
numeric_columns = {
    "Open": "DOUBLE PRECISION",
    "High": "DOUBLE PRECISION",
    "Low": "DOUBLE PRECISION",
    "Close": "DOUBLE PRECISION",
    "Adj_Close": "DOUBLE PRECISION",
    "Volume": "DOUBLE PRECISION",
}

# Define columns that should be stored as datetime
datetime_columns = {
    "Date": "DATE" 
}

# Convert necessary columns to appropriate types in pandas
for col in numeric_columns.keys():
    if col in index_df.columns:
        index_df[col] = pd.to_numeric(index_df[col], errors='coerce')  # Convert to numeric, setting invalid values as NaN

for col in datetime_columns.keys():
    if col in index_df.columns:
        index_df[col] = pd.to_datetime(index_df[col], errors='coerce')  # Convert to datetime, setting invalid values as NaT

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname=database, 
    user=username, 
    password=password, 
    host=host, 
    port=port
)
cursor = conn.cursor()

# Ensure the schema exists
cursor.execute(sql.SQL("SET search_path TO {};").format(sql.Identifier(schema_name)))

# Check if the table exists
cursor.execute(sql.SQL(""" 
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = %s AND table_name = %s
    );
"""), (schema_name, table_name))

table_exists = cursor.fetchone()[0]

if not table_exists:
    print(f"The table {table_name} does not exist in schema {schema_name}.")
    cursor.close()
    conn.close()
    sys.exit()  # Exit if table does not exist

# Use COPY to insert data in bulk
csv_buffer = StringIO()
index_df.to_csv(csv_buffer, index=False, header=False, date_format='%Y-%m-%d %H:%M:%S')  # Ensure correct date format
csv_buffer.seek(0)

# Use COPY to insert the data from the StringIO buffer
cursor.copy_expert(
    sql.SQL("COPY {}.{} FROM STDIN WITH CSV NULL 'NaN'").format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name)
    ),
    csv_buffer
)

# Commit changes and close the connection
conn.commit()
cursor.close()
conn.close()

print(f"Index data successfully written into {schema_name}.{table_name}")

