import os
import sqlite3
from sqlalchemy import create_engine, inspect, MetaData, Table, Column, String, Integer, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from typing import Optional, Dict, List, Union

Base = declarative_base()

class DatabaseManager:
    def __init__(self, connection_string: str = None):
        """
        Initialize the database manager.
        
        Args:
            connection_string (str, optional): SQLAlchemy connection string. 
                If None, defaults to SQLite in data/customized/database.
        """
        if connection_string is None:
            db_path = os.path.join('data', 'customized', 'database')
            os.makedirs(db_path, exist_ok=True)
            connection_string = f'sqlite:///{os.path.join(db_path, "saged.db")}'
        
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
        
    def initialize_database(self):
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)
        
    def get_table_names(self) -> List[str]:
        """Get list of all tables in the database."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        return table_name in self.get_table_names()
    
    def create_table(self, table_name: str, columns: Dict[str, type]):
        """
        Create a new table with specified columns.
        
        Args:
            table_name (str): Name of the table to create
            columns (Dict[str, type]): Dictionary of column names and their types
        """
        column_definitions = []
        for col_name, col_type in columns.items():
            if col_type == str:
                column_definitions.append(Column(col_name, String))
            elif col_type == int:
                column_definitions.append(Column(col_name, Integer))
            elif col_type == float:
                column_definitions.append(Column(col_name, Float))
            elif col_type == dict:
                column_definitions.append(Column(col_name, JSON))
        
        table = Table(table_name, self.metadata, *column_definitions)
        table.create(self.engine)
        
    def drop_table(self, table_name: str):
        """Drop a table from the database."""
        if self.table_exists(table_name):
            table = Table(table_name, self.metadata)
            table.drop(self.engine)
            
    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """
        Save a DataFrame to a database table.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            table_name (str): Name of the table to save to
            if_exists (str): How to behave if the table exists ('fail', 'replace', 'append')
        """
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
        
    def load_dataframe(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Load a table from the database into a DataFrame.
        
        Args:
            table_name (str): Name of the table to load
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing the table data, or None if table doesn't exist
        """
        if not self.table_exists(table_name):
            return None
        return pd.read_sql_table(table_name, self.engine)
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            pd.DataFrame: Query results
        """
        return pd.read_sql_query(query, self.engine)
    
    def get_table_schema(self, table_name: str) -> Dict:
        """
        Get the schema of a table.
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            Dict: Dictionary containing column names and their types
        """
        if not self.table_exists(table_name):
            return {}
        
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        return {col['name']: str(col['type']) for col in columns}

def initialize_database(connection_string: str = None) -> DatabaseManager:
    """
    Initialize the database and return a DatabaseManager instance.
    
    Args:
        connection_string (str, optional): SQLAlchemy connection string
        
    Returns:
        DatabaseManager: Initialized database manager
    """
    db_manager = DatabaseManager(connection_string)
    db_manager.initialize_database()
    return db_manager

# Example usage:
if __name__ == "__main__":
    # Initialize database with default SQLite connection
    db = initialize_database()
    
    # Example: Create a table for keywords
    if not db.table_exists('keywords'):
        db.create_table('keywords', {
            'domain': str,
            'concept': str,
            'keyword': str,
            'metadata': dict
        })
    
    # Example: Save some data
    df = pd.DataFrame({
        'domain': ['test'],
        'concept': ['test_concept'],
        'keyword': ['test_keyword'],
        'metadata': [{'type': 'test'}]
    })
    db.save_dataframe(df, 'keywords')
    
    # Example: Load data
    loaded_df = db.load_dataframe('keywords')
    print(loaded_df) 