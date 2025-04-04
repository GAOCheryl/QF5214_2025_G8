#%% transform
import os
import psycopg2
import csv
from io import StringIO
from datetime import datetime

# Database connection parameters
# database = "QF5214"
username = "postgres"
password = "qf5214G8"
host = "pgm-t4n365kyk1sye1l7eo.pgsql.singapore.rds.aliyuncs.com"
port = 5555

# Folder path containing your CSV files
folder_path = r"C:\Users\Mr.river\Desktop\transfer"

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    user=username,
    password=password,
    host=host,
    port=port
)
cur = conn.cursor()

# 1. Create the datacollection schema if it doesn't exist
cur.execute("CREATE SCHEMA IF NOT EXISTS datacollection;")
conn.commit()

# 2. Create the X_batch table (if it doesn't already exist).
#    Adjust column names and data types to match your CSV structure.
create_table_query = """
CREATE TABLE IF NOT EXISTS datacollection.X_batch (
    Company TEXT,
    Tweet_Count INT,
    Text TEXT,
    Created_At TIMESTAMP,
    Retweets INT,
    Likes INT,
    URL TEXT
);
"""
cur.execute(create_table_query)
conn.commit()

# 3. Collect all CSV files in the specified folder matching the naming pattern
csv_files = [
    f for f in os.listdir(folder_path)
    if f.startswith("tweets_nasdaq100_") and f.endswith(".csv")
]

# 4. Process each CSV file and copy its data into the X_batch table
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    print(f"Processing {file_path} ...")
    
    # Prepare an in-memory buffer to store transformed CSV data
    transformed_data = StringIO()
    writer = csv.writer(transformed_data)

    # Read the original CSV file
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the header row
        writer.writerow(header)  # Write the header row into our buffer

        # Find the indexes of 'Company' and 'Created_At' columns
        try:
            company_idx = header.index('Company')
        except ValueError:
            raise Exception(f"'Company' column not found in {csv_file}")

        try:
            created_at_idx = header.index('Created_At')
        except ValueError:
            raise Exception(f"'Created_At' column not found in {csv_file}")

        # Transform each row's data
        for row in reader:
            # 4.1 Remove the leading "$" from the Company column if present
            if row[company_idx].startswith('$'):
                row[company_idx] = row[company_idx][1:].strip()

            # 4.2 Convert the Created_At column to "YYYY-MM-DD HH:MM:SS" format
            original_date_str = row[created_at_idx].strip()
            if original_date_str:
                # Example: "Sat Jan 01 09:39:49 +0000 2022"
                dt = datetime.strptime(original_date_str, "%a %b %d %H:%M:%S %z %Y")
                new_date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                row[created_at_idx] = new_date_str

            writer.writerow(row)

    # Reset the StringIO pointer so we can read from the beginning
    transformed_data.seek(0)

    # 5. Use the COPY command for bulk insertion
    #    Make sure the columns listed match your table definition and CSV headers
    copy_sql = """
        COPY datacollection.X_batch (Company, Tweet_Count, Text, Created_At, Retweets, Likes, URL)
        FROM STDIN WITH CSV HEADER
    """
    cur.copy_expert(copy_sql, transformed_data)
    conn.commit()
    print(f"Finished importing {csv_file}.")

# 6. Close the cursor and connection
cur.close()
conn.close()