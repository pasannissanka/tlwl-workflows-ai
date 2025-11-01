from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np
import pandas as pd

from common.llm import OpenAILLM
from topic_gen.agent.agent import TopicGenAgent
from topic_gen.clustering.cluster import AgglomerativeCluster
from topic_gen.data.cleanup import CleanUpInput
from topic_gen.data.data import Data
from topic_gen.prompt import TOPIC_GEN_SYSTEM_PROMPT, TOPIC_GEN_USER_PROMPT

logger = logging.getLogger(__name__)


class TopicGen:
    """TopicGen class for topic generation"""

    def __init__(self):
        self.data = Data()
        self.clustering = AgglomerativeCluster(distance_threshold=1.2)
        self.agent = TopicGenAgent()

        # Data attributes
        self.user_id: str = None
        self.raw_df: pd.DataFrame = None
        self.df: pd.DataFrame = None
        self.cluster_df: pd.DataFrame = None

    def generate(self, user_id: str) -> pd.DataFrame:
        """Generate topics from a dataframe

        Args:
            user_id (str): The user id to generate topics for.

        Returns:
            pd.DataFrame: The dataframe with the "topic" column (str) added to the dataframe
        """
        logger.info(f"Generating topics for user {user_id}")
        self.user_id = user_id
        self.df = self.ingest(user_id)
        self.df = self.filter_data(self.df)
        self.df = self.cluster_data(self.df)
        self.df = self.generate_topics(self.df)
        self.save_topics(self.df)

        logger.debug(f"Generated topics for user {user_id}")
        return self.df

    def ingest(self, user_id: str) -> None:
        """Ingest data from the database and return a cleaned dataframe"""
        bookmarks_df = self.data.query_bookmarks(user_id)
        logger.debug(
            f"Bookmarks dataframe size: {len(bookmarks_df)}, dimensions: {bookmarks_df.shape}"
        )
        df = CleanUpInput.cleanup(bookmarks_df)
        logger.debug(f"Cleaned dataframe size: {len(df)}, dimensions: {df.shape}")
        return df

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the dataframe based on the number of bookmarks per tag"""
        bookmark_counts = (
            df.groupby(["tag_id", "key", "name"])
            .size()
            .reset_index(name="bookmark_count")
        )
        # Filter tags with more than 5 bookmarks
        filtered_tags = bookmark_counts[
            bookmark_counts["bookmark_count"] > 5
        ]  # TODO: Make this a configurable parameter
        # Filter the original dataframe to only include bookmarks with the filtered tags
        df = df[df["tag_id"].isin(filtered_tags["tag_id"])]
        logger.debug(f"Filtered dataframe size: {len(df)}, dimensions: {df.shape}")
        return df

    def cluster_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cluster the dataframe based on the tags"""

        tag_keys_df = df[["tag_id", "key"]].drop_duplicates()
        cluster_df = self.clustering.fit(tag_keys_df)

        df = pd.merge(cluster_df, df, on="tag_id", how="left")
        # Drop, Rename unnecessary columns
        df = df.drop(columns=["key_x"])
        df = df.rename(columns={"key_y": "key"})
        logger.debug(f"Clustered dataframe size: {len(df)}, dimensions: {df.shape}")
        return df

    def generate_topics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate topics for all clusters

        Args:
            df (pd.DataFrame): The dataframe to generate topics for.

        Returns:
            pd.DataFrame: A dataframe with the "topic" column (str) added to the dataframe
        """
        cluster_ids = df["cluster"].unique()
        logger.debug(f"Generating topics for {len(cluster_ids)} clusters")

        max_workers = min(len(cluster_ids), 4)
        logger.debug(f"Using {max_workers} workers for topic generation")
        # Use ThreadPoolExecutor for parallel topic generation
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_cluster = {
                executor.submit(
                    self._submit_topic_generation_job,
                    df[df["cluster"] == cluster_id],
                ): cluster_id
                for cluster_id in cluster_ids
            }

            # Process completed tasks
            for future in as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    result = future.result()
                    # Add the topic and score to the dataframe
                    df.loc[df["cluster"] == cluster_id, "topic"] = result["topic"]
                    df.loc[df["cluster"] == cluster_id, "description"] = result[
                        "description"
                    ]
                    df.loc[df["cluster"] == cluster_id, "score"] = result["score"]
                except Exception as exc:
                    logger.error(f"Cluster {cluster_id} generated an exception: {exc}")
                    # Set a default topic and score for failed clusters
                    df.loc[df["cluster"] == cluster_id, "topic"] = np.nan
                    df.loc[df["cluster"] == cluster_id, "description"] = np.nan
                    df.loc[df["cluster"] == cluster_id, "score"] = np.nan
        df = df.dropna(subset=["topic", "description", "score"]).reset_index(drop=True)
        logger.debug("Generated topics successfully")
        return df

    def _submit_topic_generation_job(self, cluster_df: pd.DataFrame) -> str:
        """Generate a topic for a specific cluster ID

        Args:
            cluster_id (int): The cluster ID to generate a topic for

        Returns:
            str: The generated topic
        """
        print(f"Generating topic for cluster {cluster_df.head(10)}")

        tags_df = cluster_df[["tag_id", "key"]].drop_duplicates()
        tags = tags_df["key"].unique().tolist()

        bookmarks_df = cluster_df[
            [
                "bookmark_id",
                "title",
                "description",
                "language",
                "created_at",
                "updated_at",
            ]
        ].drop_duplicates()
        titles = bookmarks_df["title"].unique().tolist()

        response = self.agent.invoke(tags=tags, titles=titles)
        print(response)
        topic = response.topic
        description = response.description
        score = response.score

        logger.debug(
            f"Topic generation successful: topic: {topic}, description: {description}, score: {score}"
        )
        return {
            "topic": topic,
            "description": description,
            "score": score,
        }

    def save_topics(self, df: pd.DataFrame) -> None:
        """Save topics to the database

        Args:
            df (pd.DataFrame): The dataframe to persist topics for.

        Returns:
            None
        """
        cluster_ids = df["cluster"].unique()

        self.data.connect(init_models=True)
        with ThreadPoolExecutor(max_workers=min(len(cluster_ids), 4)) as executor:
            future_to_cluster = {
                executor.submit(
                    self._submit_data_insertion_job,
                    df[df["cluster"] == cluster_id],
                ): cluster_id
                for cluster_id in cluster_ids
            }
            for future in as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Cluster {cluster_id} failed to persist: {exc}")
        return None

    def _submit_data_insertion_job(self, cluster_df: pd.DataFrame) -> None:
        data = {
            "cluster_id": int(cluster_df["cluster"].unique()[0]),
            "title": str(cluster_df["topic"].unique()[0]),
            "description": str(cluster_df["description"].unique()[0]),
            "type": "topic",
            "tags": cluster_df["tag_id"].unique().tolist(),
            "bookmarks": cluster_df["bookmark_id"].unique().tolist(),
            "score": float(cluster_df["score"].unique()[0]),
        }
        self.data.insert_topics(data, self.user_id)
        return None

    def _calculate_score(self, tags: list[str], bookmarks: list[str]) -> float:
        """Calculate the score for a cluster

        Args:
            tags (list[str]): The tags for the cluster
            bookmarks (list[str]): The bookmarks for the cluster

        Returns:
            float: The score for the cluster
        """
        # TODO: Implement a proper score calculation
        return len(tags) + len(bookmarks)
