# -*- coding: utf-8 -*-
# @Time       : 2025/2/15 18:02
# @Author     : Marverlises
# @File       : paper_sql.py
# @Description: Database handling for paper data
import logging
import sqlite3
import os
import json
from sqlite3 import Error
from typing import List, Dict, Any, Optional, Union, Tuple

from config import settings
from modules.utils import Utils

logger = logging.getLogger(__name__)

class PaperSQL:
    """
    Database interface for storing and retrieving paper data.
    Handles connections, table creation, and data operations.
    """
    
    def __init__(self, db_file=settings.DB_SAVE_PATH):
        """
        Initialize the database connection.
        
        Args:
            db_file (str): Path to the SQLite database file
        """
        self.db_file = db_file
        self.connection = None
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)
        
        # Create connection
        self.create_connection()
        
    def create_connection(self) -> None:
        """
        Create a database connection to the SQLite database.
        
        Returns:
            None
        """
        try:
            self.connection = sqlite3.connect(self.db_file)
            self.connection.row_factory = sqlite3.Row  # Return rows as dictionaries
            logger.info(f"Connection established to {self.db_file}")
        except Error as e:
            logger.error(f"Error creating connection: {e}")
            raise
            
    def close_connection(self) -> None:
        """
        Close the database connection.
        
        Returns:
            None
        """
        if self.connection:
            self.connection.close()
            logger.info("Connection closed")
            
    def __enter__(self):
        """
        Enter context manager for with statement.
        
        Returns:
            PaperSQL: Self instance
        """
        if not self.connection:
            self.create_connection()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager for with statement.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
            
        Returns:
            None
        """
        self.close_connection()

    def create_table(self, table_name: str, columns: List[str]) -> None:
        """
        Create a new table with the given name and columns.
        
        Args:
            table_name (str): Name of the table to create
            columns (List[str]): List of column names
            
        Returns:
            None
        """
        try:
            if not self.connection:
                self.create_connection()
                
            cursor = self.connection.cursor()
            
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if cursor.fetchone():
                logger.info(f"Table '{table_name}' already exists")
                return
                
            # Create column definitions with all columns as TEXT except id
            column_definitions = ', '.join([f"{col} TEXT" for col in columns])
            
            # Create the table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {column_definitions}
                )
            ''')
            
            # Create indexes on commonly used columns
            for index_col in ['title', 'year', 'authors']:
                if index_col in columns:
                    index_name = f"idx_{table_name}_{index_col}"
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({index_col})")
                    
            self.connection.commit()
            logger.info(f"Table '{table_name}' created successfully with {len(columns)} columns")
        except Error as e:
            logger.error(f"Error creating table {table_name}: {e}")
            raise

    def delete_table(self, table_name: str) -> bool:
        """
        Delete a table from the database.
        
        Args:
            table_name (str): Name of the table to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.connection:
                self.create_connection()
                
            cursor = self.connection.cursor()
            cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
            self.connection.commit()
            logger.info(f"Table '{table_name}' deleted successfully")
            return True
        except Error as e:
            logger.error(f"Error deleting table {table_name}: {e}")
            return False

    def insert_data(self, table_name: str, data: Dict[str, Any]) -> int:
        """
        Insert data into a table.
        
        Args:
            table_name (str): Name of the table
            data (Dict[str, Any]): Data to insert as column-value pairs
            
        Returns:
            int: Row ID of the inserted data or -1 if failed
        """
        try:
            if not self.connection:
                self.create_connection()
                
            # Process each field value to ensure compatibility
            processed_data = {}
            for key, value in data.items():
                processed_data[key] = Utils.convert_data(value)

            cursor = self.connection.cursor()
            
            # Generate SQL for insertion
            columns = ', '.join(processed_data.keys())
            placeholders = ', '.join(['?'] * len(processed_data))
            
            cursor.execute(f'''
                INSERT INTO {table_name} ({columns}) VALUES ({placeholders})
            ''', tuple(processed_data.values()))
            
            self.connection.commit()
            row_id = cursor.lastrowid
            logger.debug(f"Data inserted into '{table_name}' with ID {row_id}")
            return row_id
        except Error as e:
            logger.error(f"Error inserting data into {table_name}: {e}")
            self.connection.rollback()
            return -1

    def fetch_data(self, 
                  table_name: str, 
                  conditions: Optional[Dict[str, Any]] = None, 
                  limit: Optional[int] = None,
                  order_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch data from a table with optional filtering.
        
        Args:
            table_name (str): Name of the table
            conditions (Dict[str, Any], optional): Filter conditions as column-value pairs
            limit (int, optional): Maximum number of rows to return
            order_by (str, optional): Column name to order by
            
        Returns:
            List[Dict[str, Any]]: List of matching rows as dictionaries
        """
        try:
            if not self.connection:
                self.create_connection()
                
            cursor = self.connection.cursor()
            
            # Build the query
            query = f'SELECT * FROM {table_name}'
            params = []
            
            # Add WHERE clause if conditions are provided
            if conditions:
                where_clauses = []
                for column, value in conditions.items():
                    where_clauses.append(f"{column} = ?")
                    params.append(value)
                    
                if where_clauses:
                    query += f" WHERE {' AND '.join(where_clauses)}"
            
            # Add ORDER BY clause if specified
            if order_by:
                query += f" ORDER BY {order_by}"
                
            # Add LIMIT clause if specified
            if limit:
                query += f" LIMIT {limit}"
                
            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            result = []
            for row in rows:
                row_dict = {}
                for key in row.keys():
                    row_dict[key] = row[key]
                    # Try to parse JSON fields
                    if isinstance(row[key], str) and (row[key].startswith('{') or row[key].startswith('[')):
                        try:
                            row_dict[key] = json.loads(row[key])
                        except json.JSONDecodeError:
                            pass
                result.append(row_dict)
                
            logger.info(f"Fetched {len(result)} rows from '{table_name}'")
            return result
        except Error as e:
            logger.error(f"Error fetching data from {table_name}: {e}")
            return []

    def update_data(self, 
                   table_name: str, 
                   row_id: int, 
                   data: Dict[str, Any]) -> bool:
        """
        Update data in a table by row ID.
        
        Args:
            table_name (str): Name of the table
            row_id (int): ID of the row to update
            data (Dict[str, Any]): New data as column-value pairs
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.connection:
                self.create_connection()
                
            # Process each field value to ensure compatibility
            processed_data = {}
            for key, value in data.items():
                processed_data[key] = Utils.convert_data(value)
                
            cursor = self.connection.cursor()
            
            # Build the SET clause
            set_clauses = [f"{column} = ?" for column in processed_data.keys()]
            params = list(processed_data.values())
            params.append(row_id)
            
            query = f'''
                UPDATE {table_name}
                SET {', '.join(set_clauses)}
                WHERE id = ?
            '''
            
            cursor.execute(query, params)
            self.connection.commit()
            
            rows_affected = cursor.rowcount
            if rows_affected > 0:
                logger.info(f"Updated row {row_id} in '{table_name}'")
                return True
            else:
                logger.warning(f"No rows updated in '{table_name}' for ID {row_id}")
                return False
        except Error as e:
            logger.error(f"Error updating data in {table_name}: {e}")
            self.connection.rollback()
            return False
    
    def search_papers(self, 
                     query: str, 
                     tables: Optional[List[str]] = None,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers matching the query across specified tables.
        
        Args:
            query (str): Search query
            tables (List[str], optional): List of tables to search. If None, searches all tables.
            limit (int, optional): Maximum number of results per table. Defaults to 10.
            
        Returns:
            List[Dict[str, Any]]: List of matching papers with table name added
        """
        try:
            if not self.connection:
                self.create_connection()
                
            cursor = self.connection.cursor()
            results = []
            
            # Get all tables if not specified
            if not tables:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
            search_terms = [f'%{term}%' for term in query.split()]
            
            for table in tables:
                try:
                    # Check if columns exist in the table
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [row[1] for row in cursor.fetchall()]
                    
                    search_columns = []
                    for col in ['title', 'abstract', 'keywords']:
                        if col in columns:
                            search_columns.append(col)
                            
                    if not search_columns:
                        continue
                        
                    # Build the WHERE clause for each search term
                    where_clauses = []
                    params = []
                    
                    for term in search_terms:
                        term_clause = []
                        for col in search_columns:
                            term_clause.append(f"{col} LIKE ?")
                            params.append(term)
                        where_clauses.append(f"({' OR '.join(term_clause)})")
                    
                    # Combine all terms with AND
                    where_combined = ' AND '.join(where_clauses)
                    
                    query = f'''
                        SELECT * FROM {table}
                        WHERE {where_combined}
                        LIMIT {limit}
                    '''
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Process and add to results
                    for row in rows:
                        row_dict = {}
                        for key in row.keys():
                            row_dict[key] = row[key]
                            # Try to parse JSON fields
                            if isinstance(row[key], str) and (row[key].startswith('{') or row[key].startswith('[')):
                                try:
                                    row_dict[key] = json.loads(row[key])
                                except json.JSONDecodeError:
                                    pass
                        
                        # Add table name to results
                        row_dict['table_name'] = table
                        results.append(row_dict)
                except Error as e:
                    logger.error(f"Error searching in table {table}: {e}")
            
            logger.info(f"Found {len(results)} papers matching query: {query}")
            return results
        except Error as e:
            logger.error(f"Error in paper search: {e}")
            return []

    def load_column_from_db(self, table_name: str, column_name: str) -> List[str]:
        """
        Load values of a specific column from the database.
        
        Args:
            table_name (str): Name of the table
            column_name (str): Name of the column to retrieve
            
        Returns:
            List[str]: List of column values
        """
        try:
            if not self.connection:
                self.create_connection()
                
            cursor = self.connection.cursor()
            
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not cursor.fetchone():
                logger.error(f"Table '{table_name}' does not exist")
                return []
                
            # Check if column exists
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            if column_name not in columns:
                logger.error(f"Column '{column_name}' does not exist in table '{table_name}'")
                return []
            
            # Fetch the column values
            cursor.execute(f"SELECT {column_name} FROM {table_name}")
            rows = cursor.fetchall()
            
            values = []
            for row in rows:
                value = row[0]
                if value is not None:
                    # Try to parse JSON fields
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                    values.append(value)
                    
            logger.info(f"Loaded {len(values)} {column_name} values from '{table_name}'")
            return values
        except Error as e:
            logger.error(f"Error loading column '{column_name}' from '{table_name}': {e}")
            raise
            
    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics about a table.
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            Dict[str, Any]: Dictionary of table statistics
        """
        try:
            if not self.connection:
                self.create_connection()
                
            cursor = self.connection.cursor()
            
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not cursor.fetchone():
                logger.error(f"Table '{table_name}' does not exist")
                return {"exists": False}
                
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            # Get unique values for key columns
            stats = {
                "exists": True,
                "row_count": row_count,
                "columns": columns,
                "unique_values": {}
            }
            
            for col in ['year', 'conference']:
                if col in columns:
                    cursor.execute(f"SELECT DISTINCT {col} FROM {table_name}")
                    stats["unique_values"][col] = [row[0] for row in cursor.fetchall()]
                    
            return stats
        except Error as e:
            logger.error(f"Error getting statistics for table '{table_name}': {e}")
            return {"exists": False, "error": str(e)}

    def persist_paper_info(self):
        """
        Persist paper information to the database.
        """
        raise NotImplementedError("This method should be implemented by subclasses")
