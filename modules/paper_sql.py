# -*- coding: utf-8 -*-
# @Time       : 2025/2/15 18:02
# @Author     : Marverlises
# @File       : paper_sql.py
# @Description: PyCharm
import logging
import sqlite3
from config import settings
from sqlite3 import Error
from modules.utils import Utils
from typing import List

logger = logging.getLogger(__name__)


class PaperSQL:
    def __init__(self, db_file=settings.DB_SAVE_PATH):
        """初始化数据库连接"""
        self.db_file = db_file
        self.connection = None
        self.create_connection()

    def create_connection(self):
        """创建数据库连接"""
        try:
            self.connection = sqlite3.connect(self.db_file)
            logger.info(f"Connection established to {self.db_file}")
        except Error as e:
            logger.error(f"Error creating connection: {e}")

    def create_table(self, table_name, columns):
        """根据表名和列定义动态创建表"""
        try:
            cursor = self.connection.cursor()
            column_definitions = ', '.join([f"{col} TEXT" for col in columns])
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {column_definitions}
                )
            ''')
            self.connection.commit()
            logger.info(f"Table '{table_name}' created successfully")
        except Error as e:
            logger.error(f"Error creating table {table_name}: {e}")

    def delete_table(self, table_name):
        """删除指定的表"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
            self.connection.commit()
            logger.info(f"Table '{table_name}' deleted successfully")
        except Error as e:
            logger.error(f"Error deleting table {table_name}: {e}")

    def insert_data(self, table_name, data):
        """插入数据到指定的表"""
        try:
            # 转换每个字段的值
            for key, value in data.items():
                data[key] = Utils.convert_data(value)

            cursor = self.connection.cursor()
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?'] * len(data))
            cursor.execute(f'''
                INSERT INTO {table_name} ({columns}) VALUES ({placeholders})
            ''', tuple(data.values()))
            self.connection.commit()
            logger.info(f"Data prepared to be inserted into '{table_name}'")
        except Error as e:
            logger.error(f"Error inserting data into {table_name}: {e}")
            self.connection.rollback()
            raise

    def fetch_data(self, table_name, year=None):
        """根据年份查询特定表中的数据"""
        try:
            cursor = self.connection.cursor()
            if year:
                cursor.execute(f'''
                    SELECT * FROM {table_name} WHERE year = ?
                ''', (year,))
            else:
                cursor.execute(f'SELECT * FROM {table_name}')
            return cursor.fetchall()
        except Error as e:
            logger.error(f"Error fetching data from {table_name}: {e}")
            return []

    def close_connection(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("Connection closed")

    def persist_paper_info(self):
        """
        Persist paper information to the database.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def load_column_from_db(self, table_name: str, column_name: str) -> List[str]:
        """
        Get all the values of a column from a table in the database.
        """
        try:
            if not self.connection:
                self.create_connection()
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT {column_name} FROM {table_name}")
            rows = cursor.fetchall()
            logger.info(f"Loaded {len(rows)} {column_name} from the table '{table_name}'.")
            return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"Error loading {column_name} from database: {e}")
            raise e


if __name__ == '__main__':
    # # 测试删除table
    # paper_sql = PaperSQL(db_file='../data/papers.db')
    # # 查询
    # data = paper_sql.fetch_data('IJCAI_2024')
    # print(data)

    import sqlite3

    # 连接到 SQLite 数据库
    conn = sqlite3.connect('../data/papers.db')

    # 创建一个游标对象
    cursor = conn.cursor()

    # 执行 SQL 查询，选择 abstract 列
    cursor.execute("SELECT abstracts FROM IJCAI_2024")

    # 获取查询结果
    abstracts = cursor.fetchall()

    # 打印查询结果
    for abstract in abstracts:
        print(abstract[0])

    # 关闭连接
    cursor.close()
    conn.close()
