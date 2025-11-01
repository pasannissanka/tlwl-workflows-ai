import logging
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy.sql import insert

from common.db import AbstractDB, PgDB

logger = logging.getLogger(__name__)


class Data:
    """Inject data from the database into the topic generation workflow"""

    def __init__(self):
        self.db: AbstractDB = PgDB()

    def connect(self, init_models: bool = False) -> None:
        """Connect to the database"""
        logger.debug(f"Connecting to the database with init_models: {init_models}")
        self.db.connect(init_models=init_models)

    def query_bookmarks(self, user_id: str) -> pd.DataFrame:
        """Query all the bookmarks with their tags from the database"""
        bookmarks_df = None
        query = """
                select b.id as bookmark_id,
                       b.url,
                       b.title,
                       b.description,
                       b."language",
                       b.created_at,
                       b.updated_at,
                       t.id as tag_id,
                       t."key",
                       t."name"
                from bookmark_tag bt
                         inner join tag t on bt.tag_id = t.id
                         inner join bookmark b on b.id = bt.bookmark_id
                where b.user_id = :user_id
                """
        try:
            logger.debug(f"Querying bookmarks for user {user_id}")
            self.db.connect()
            bookmarks_df = self.db.query(query, {"user_id": user_id})
        except Exception as e:
            logger.error(f"Failed to query bookmarks for user {user_id}: {e}")
            raise RuntimeError("Failed to query bookmarks") from e
        finally:
            logger.debug("DB connection closed")
            self.db.disconnect()
        if bookmarks_df is None or bookmarks_df.empty:
            logger.warning(f"No bookmarks found for user {user_id}")
            return []
        logger.debug(f"Bookmarks queried successfully for user {user_id}")
        return bookmarks_df

    def _query_user(self, user_id: str) -> pd.DataFrame:
        query = "select * from user u where u.id = :user_id"
        user_df = self.db.query(query, {"user_id": user_id})
        if user_df.empty:
            raise RuntimeError("No user found")
        return user_df.to_dict("records")[0]

    def _query_tags(self, user_id: str) -> pd.DataFrame:
        query = "select * from tag t where t.user_id = :user_id"
        tag_df = self.db.query(query, {"user_id": user_id})
        if tag_df.empty:
            raise RuntimeError("No tag found")
        print(tag_df)
        return tag_df

    def insert_topics(self, data: dict[str, Any], user_id: str) -> None:
        """Insert topics into the database

        Args:
            data (dict[str, Any]): data {
                "cluster_id": str,
                "title": str,
                "description": str,
                "type": str,
                "tags": list[str],
                "bookmarks": list[str],
                "score": float,
            }
            user_id (str): The user id to insert the topics for.

        Raises:
            RuntimeError: If the topics cannot be inserted.
        """
        logger.debug(f"Inserting topics for user {user_id}")
        session = self.db.session()

        try:
            topic_table = self.db.models["topic"]
            topic_tags_table = self.db.models["topic_tags"]
            topic_bookmarks_table = self.db.models["topic_bookmarks"]
            with session.begin():
                logger.debug("Session started")
                topic_id = uuid.uuid4()
                topic_data = [
                    {
                        "id": str(topic_id),
                        "cluster_id": data["cluster_id"],
                        "title": data["title"],
                        "description": data["description"],
                        "type": data["type"],
                        "user_id": user_id,
                        "score": data["score"],
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    }
                ]
                session.execute(insert(topic_table), topic_data)

                tags = [
                    {"topic_id": str(topic_id), "tag_id": str(tag), "user_id": user_id}
                    for tag in data["tags"]
                ]
                session.execute(insert(topic_tags_table), tags)

                bookmarks = [
                    {
                        "topic_id": str(topic_id),
                        "bookmark_id": str(bookmark),
                        "user_id": user_id,
                    }
                    for bookmark in data["bookmarks"]
                ]
                session.execute(insert(topic_bookmarks_table), bookmarks)

                session.commit()
                logger.debug("Session committed")
        except Exception as e:
            logger.error(f"Failed to insert topic: {e}")
            session.rollback()
            logger.debug("Session rolled back")
            raise RuntimeError(f"Failed to insert topic: {e}")
        finally:
            logger.debug("Session closed")
            session.close()
        logger.debug(f"Topics inserted successfully for user {user_id}")
