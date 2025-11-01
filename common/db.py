import abc
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    String,
    Float,
    Table,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

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

    def session(self) -> Session:
        """Create a new session"""
        if not self.engine:
            raise ConnectionError("Database connection not established")
        return Session(self.engine)

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
            raise ConnectionError(
                "Database engine not initialized. Call connect() first."
            )
        try:
            with self.engine.begin() as connection:
                stmt = text(query) if isinstance(query, str) else query
                result = connection.execute(stmt, params or {})
                return result
        except SQLAlchemyError as e:
            raise RuntimeError(f"Non-query execution failed: {e}")


class PgDB(AbstractDB):
    """Postgres DB"""

    def __init__(self):
        username = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        database = os.getenv("DB_DATABASE")
        conn_str = f"postgresql://{username}:{password}@{host}/{database}?sslmode=require&channel_binding=require"
        super().__init__(conn_str)

        self.metadata = MetaData()
        self.models: dict[str, Table] = {}

    def connect(self, init_models: bool = False):
        """Connect to the database"""
        try:
            self.engine = create_engine(self.connection_string)
            if init_models:
                self.metadata = MetaData()
                self.metadata.create_all(self.engine)
                self._init_models()
        except SQLAlchemyError as e:
            raise RuntimeError("Database connection failed") from e

    def _init_models(self):
        topic_table = Table(
            "topics",
            self.metadata,
            Column("id", String, primary_key=True),
            Column("cluster_id", String),
            Column("title", String),
            Column("description", String),
            Column("type", String),
            Column("user_id", String),
            Column("score", Float),
            Column("created_at", DateTime),
            Column("updated_at", DateTime),
        )
        topic_tags_table = Table(
            "topics_tags",
            self.metadata,
            Column("topic_id", String, primary_key=True),
            Column("tag_id", String, primary_key=True),
            Column("user_id", String),
        )
        topic_bookmarks_table = Table(
            "topics_bookmarks",
            self.metadata,
            Column("topic_id", String, primary_key=True),
            Column("bookmark_id", String, primary_key=True),
            Column("user_id", String),
        )
        self.models = {
            "topic": topic_table,
            "topic_tags": topic_tags_table,
            "topic_bookmarks": topic_bookmarks_table,
        }
