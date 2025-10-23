import pandas as pd

from common.db import AbstractDB, PgDB

class InjectData:
    """Inject data from the database into the topic generation workflow"""
    
    def __init__(self):
        self.db: AbstractDB = PgDB()

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
            self.db.connect()
            bookmarks_df = self.db.query(query, {"user_id": user_id})
        except Exception as e:
            raise RuntimeError("Failed to query bookmarks") from e
        finally:
            self.db.disconnect()
        if bookmarks_df is None or bookmarks_df.empty:
            return []
        return bookmarks_df

    def _query_user(self, user_id: str) -> pd.DataFrame:
        query = "select * from user u where u.id = :user_id"
        user_df = self.db.query(query, {"user_id": user_id})
        if user_df.empty:
            raise RuntimeError("No user found")
        return user_df.to_dict('records')[0]

    def _query_tags(self, user_id: str) -> pd.DataFrame:
        query = "select * from tag t where t.user_id = :user_id"
        tag_df = self.db.query(query, {"user_id": user_id})
        if tag_df.empty:
            raise RuntimeError("No tag found")
        print(tag_df)
        return tag_df



