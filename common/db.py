import os
import abc

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

class AbstractDB(abc.ABC):
    """Abstract base class for DB abstractions"""

    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.engine: Engine | None = None

    @abc.abstractmethod
    def connect(self):
        """Connect to the database"""
        pass

    def disconnect(self):
        """Disconnect from DB"""
        if self.engine:
            self.engine.dispose()
            self.engine = None

    def query(self, query: str, params: dict | None = None) -> pd.DataFrame:
        """Execute a read (SELECT) query and return rows as a dataframe."""
        if not self.engine:
            raise ConnectionError("Database connection not established")
        try:
            with self.engine.connect() as connection:
                stmt = text(query) if isinstance(query, str) else query
                return pd.read_sql(stmt, connection, params=params)
        except SQLAlchemyError as e:
            raise RuntimeError("Database connection failed") from e

    def execute(self, query: str, params: dict | None = None) -> int:
        """Execute a write (INSERT, UPDATE, DELETE) query and return affected row count."""
        if not self.engine:
            raise ConnectionError("Database engine not initialized. Call connect() first.")
        try:
            with self.engine.begin() as connection:
                result = connection.execute(text(query), params or {})
                return result.rowcount
        except SQLAlchemyError as e:
            raise RuntimeError(f"Non-query execution failed: {e}")


class PgDB(AbstractDB):
    """Postgres DB"""
    def __init__(self):
        username = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        database = os.getenv("DB_DATABASE")
        conn_str =  f'postgresql://{username}:{password}@{host}/{database}?sslmode=require&channel_binding=require'
        super().__init__(conn_str)

    def connect(self):
        """Connect to the database"""
        try:
            self.engine = create_engine(self.connection_string)
        except SQLAlchemyError as e:
            raise RuntimeError("Database connection failed") from e

