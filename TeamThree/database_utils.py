import os
import psycopg2
from psycopg2 import OperationalError
from sqlalchemy import create_engine

class database_utils:
    """Handles database connections and provides utility functions."""
    
    def __init__(self):
        self.database = os.getenv("DB_NAME", "QF5214")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "qf5214G8")
        self.host = os.getenv("DB_HOST", "pgm-t4n365kyk1sye1l7eo.pgsql.singapore.rds.aliyuncs.com")
        self.port = os.getenv("DB_PORT", "5555")
        self.conn = None
        self.cursor = None

    def connect(self, if_return = False):
        """Establishes a database connection."""
        try:
            self.conn = psycopg2.connect(
                dbname=self.database,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            self.cursor = self.conn.cursor()
            self.engine = create_engine(f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}')
            if if_return:
                return self.cursor, self.conn, self.engine
        except OperationalError as e:
            print(f"Database connection error: {e}")

    def execute_query(self, query, params=None):
        """Executes a query and commits changes if applicable."""
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error executing query: {e}")
            
    def df_to_sql_table(self, df, type_list, schema, table_name, drop_table = True):
        """Inserts a DataFrame into a table in the database."""
        try:
            # set the data type
            for i in range(len(df.columns)):
                df[df.columns[i]] = df[df.columns[i]].astype(type_list[i])
                
            # insert the DataFrame into the table under certain schema
            if drop_table:
                self.execute_query(f"DROP TABLE IF EXISTS {schema}.{table_name}")
            df.to_sql(table_name, self.engine, schema=schema, if_exists='replace', index=False)
            
        except Exception as e:
            print(f"Error inserting DataFrame into table: {e}")
    
    def set_schema(self, schema):
        """Connects to a specific schema."""
        self.execute_query(f"SET search_path TO {schema}")

    def fetch_results(self):
        """Fetches all rows from the last executed query."""
        return self.cursor.fetchall()

    def close_connection(self):
        """Closes the database connection properly."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


# if __name__ == "__main__":
#     db = database_utils()
#     db.connect()
    
#     # Fetch all table names from datacollection schema
#     db.execute_query("""
#         SELECT table_name 
#         FROM information_schema.tables 
#         WHERE table_schema = 'datacollection'
#     """)
#     tables = db.fetch_results()

#     # Iterate through tables and get column names for each
#     for table in tables:
#         table_name = table[0]
#         print(f"\n🔹 Table: {table_name}")

#         # Fetch column names for the current table
#         db.execute_query(f"""
#             SELECT column_name 
#             FROM information_schema.columns 
#             WHERE table_name = %s
#         """, (table_name,))
#         columns = db.fetch_results()

#         column_names = [col[0] for col in columns]
#         print(f"   Columns: {', '.join(column_names)}")
        

#     db.close_connection()
    


# if __name__ == "__main__":
#     import pandas as pd
    # db = database_utils()
    # db.connect()
    # query_input = f'''
    #     SELECT company, created_at
    #     FROM datacollection.x_batch
    # '''
    # # nlp.
    # db.execute_query(query_input)
    # df_stock = pd.DataFrame(db.fetch_results())
    # df_stock.columns = ['company', 'created_at']
    # df_stock['created_at'] = pd.to_datetime(df_stock['created_at']).dt.date
    # # Step 2: Group by 'company' and count unique dates in 'created_at'
    # result = df_stock.groupby('company')[['created_at']].nunique()
    # print(type(result))
    # result.to_csv('series_log.csv')
    # print(result)
    # db.close_connection()