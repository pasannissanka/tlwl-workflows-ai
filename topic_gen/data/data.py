import json
import logging
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy.sql import insert, update

from common.db import AbstractDB, PgDB

logger = logging.getLogger(__name__)


class Data:
    """Inject data from the database into the topic generation workflow"""

    def __init__(self):
        self.db: PgDB = PgDB()

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
                       t."name",
                       bt.relevance_score
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
            topic_table = self.db.topic_table
            topic_tags_table = self.db.topic_tags_table
            topic_bookmarks_table = self.db.topic_bookmarks_table
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
                        "job_id": data["job_id"],
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

    def create_background_job(self, user_id: str, type: str) -> None:
        """Insert a background job into the database and return the job id

        Args:
            user_id (str): The user id to insert the background job for.
            type (str): The type of background job to insert.

        Returns:
            str: The id of the inserted background job.
        """
        background_job_data = {
            "user_id": user_id,
            "type": type,
            "metadata": json.dumps({}),
            "status": "running",
            "started_at": datetime.now().isoformat(),
        }
        try:
            logger.debug(f"Inserting background job for user {user_id}")
            self.db.connect(init_models=True)
            background_job_table = self.db.background_job_table
            session = self.db.session()
            with session.begin():
                logger.debug("Session started")
                background_job_id = str(uuid.uuid4())
                background_job_data["id"] = background_job_id
                session.execute(insert(background_job_table), [background_job_data])
                session.commit()
                logger.debug(
                    f"Background job inserted successfully, job id: {background_job_id}"
                )
                return background_job_id
        except Exception as e:
            logger.error(f"Failed to insert background job: {e}")
            session.rollback()
            logger.debug("Session rolled back")
            raise RuntimeError(f"Failed to insert background job: {e}")
        finally:
            session.close()
            logger.debug("Session closed")

    def update_background_job(
        self, job_id: str, status: str, metadata: dict = None, error: str = None
    ) -> None:
        """Update a background job in the database and return the updated job id

        Args:
            job_id (str): The id of the background job to update.
            status (str): The status of the background job to update.
            metadata (dict, optional): The metadata of the background job to update. Defaults to None.
            error (str, optional): The error of the background job to update. Defaults to None.

        Raises:
            RuntimeError: If the background job cannot be updated.
        """
        try:
            logger.debug(f"Updating background job {job_id}")
            self.db.connect(init_models=True)
            session = self.db.session()
            with session.begin():
                logger.debug("Session started")
                session.execute(
                    update(self.db.background_job_table)
                    .where(self.db.background_job_table.c.id == job_id)
                    .values(
                        status=status,
                        metadata=json.dumps(metadata),
                        error=error,
                        completed_at=(
                            datetime.now().isoformat()
                            if status == "completed"
                            else None
                        ),
                        updated_at=datetime.now().isoformat(),
                    )
                    .returning(self.db.background_job_table.c.id)
                )
                session.commit()
                logger.debug("Background job updated successfully")
        except Exception as e:
            logger.error(f"Failed to update background job: {e}")
            session.rollback()
            logger.debug("Session rolled back")
            raise RuntimeError(f"Failed to update background job: {e}")
        finally:
            session.close()
            logger.debug("Session closed")

    def get_latest_background_job(self, user_id: str, type: str) -> dict | None:
        query = """
        select * from background_jobs bj where 
            bj.user_id = :user_id and 
            bj.type = :type and 
            bj.status = :status
        order by bj.created_at desc limit 1
        """
        try:
            logger.debug(f"Getting latest background job for user {user_id}")
            self.db.connect()
            background_job_df = self.db.query(
                query, {"user_id": user_id, "type": type, "status": "completed"}
            )
            if background_job_df.empty:
                logger.warning(
                    f"No previous successful background jobs found for user {user_id} and type {type}"
                )
                return None
            return background_job_df.to_dict("records")[0]
        except Exception as e:
            logger.error(f"Failed to get latest background job for user {user_id}: {e}")
            raise RuntimeError(
                f"Failed to get latest background job for user {user_id}: {e}"
            )
        finally:
            logger.debug("Connection closed")
            self.db.disconnect()
