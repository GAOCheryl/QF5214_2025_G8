import os
import psycopg2
from psycopg2 import OperationalError

class DatabaseManager:
    """Handles database connections and provides utility functions."""
    
    def __init__(self):
        self.database = os.getenv("DB_NAME", "QF5214")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "qf5214")
        self.host = os.getenv("DB_HOST", "134.122.167.14")
        self.port = os.getenv("DB_PORT", "5555")
        self.conn = None
        self.cursor = None

    def connect(self):
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

    def fetch_results(self):
        """Fetches all rows from the last executed query."""
        return self.cursor.fetchall()

    def close_connection(self):
        """Closes the database connection properly."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    db = DatabaseManager()
    db.connect()
    
    # Fetch all table names from datacollection schema
    db.execute_query("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'datacollection'
    """)
    tables = db.fetch_results()

    # Iterate through tables and get column names for each
    for table in tables:
        table_name = table[0]
        print(f"\n🔹 Table: {table_name}")

        # Fetch column names for the current table
        db.execute_query(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s
        """, (table_name,))
        columns = db.fetch_results()

        column_names = [col[0] for col in columns]
        print(f"   Columns: {', '.join(column_names)}")
        

    db.close_connection()