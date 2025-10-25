import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from common.llm import OpenAILLM
from topic_gen.clustering.cluster import AgglomerativeCluster
from topic_gen.input.cleanup import CleanUpInput
from topic_gen.input.inject_data import InjectData
from topic_gen.prompt import TOPIC_GEN_SYSTEM_PROMPT, TOPIC_GEN_USER_PROMPT


class TopicGen:
    """TopicGen class for topic generation"""

    def __init__(self):
        self.inject = InjectData()
        self.clustering = AgglomerativeCluster()
        self.llm = OpenAILLM(model="gpt-4o-mini")

    def generate(self, user_id: str) -> pd.DataFrame:
        """Generate topics from a dataframe

        Args:
            user_id (str): The user id to generate topics for.

        Returns:
            pd.DataFrame: The dataframe with the "topic" column (str) added to the dataframe
        """
        ####################################################################################################################
        # Input stage                                                                                                      #
        ####################################################################################################################
        self.raw_df = self.inject.query_bookmarks(user_id)
        self.df = CleanUpInput.cleanup(self.raw_df)

        print(self.df.head(10))

        ####################################################################################################################
        # Filtering stage                                                                                                  #
        ####################################################################################################################
        # Group by tag_id, key, and name, then count bookmarks for each tag_id
        bookmark_counts = (
            self.df.groupby(["tag_id", "key", "name"])
            .size()
            .reset_index(name="bookmark_count")
        )
        # Filter tags with more than 5 bookmarks
        filtered_tags = bookmark_counts[
            bookmark_counts["bookmark_count"] > 5
        ]  # TODO: Make this a configurable parameter
        # Filter the original dataframe to only include bookmarks with the filtered tags
        self.df = self.df[self.df["tag_id"].isin(filtered_tags["tag_id"])]

        ####################################################################################################################
        # Clustering stage                                                                                                 #
        ####################################################################################################################
        # Get the unique tag keys dataframe
        tag_keys_df = self.df[["tag_id", "key"]].drop_duplicates()
        self.cluster_df = self.clustering.fit(tag_keys_df)
        self.df = pd.merge(self.cluster_df, self.df, on="tag_id", how="left")

        # Drop, Rename unnecessary columns
        self.df = self.df.drop(columns=["key_x"])
        self.df = self.df.rename(columns={"key_y": "key"})

        ####################################################################################################################
        # Topic generation stage                                                                                           #
        ####################################################################################################################
        # Get unique cluster IDs
        cluster_ids = self.df["cluster"].unique()
        
        # Use ThreadPoolExecutor for parallel topic generation
        with ThreadPoolExecutor(max_workers=min(len(cluster_ids), 4)) as executor:
            # Submit all tasks
            future_to_cluster = {
                executor.submit(self._generate_topic_for_cluster, cluster_id): cluster_id 
                for cluster_id in cluster_ids
            }
            
            # Process completed tasks
            for future in as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    topic = future.result()
                    print(f"Cluster {cluster_id}: {topic}")
                    # Add the topic to the dataframe
                    self.df.loc[self.df["cluster"] == cluster_id, "topic"] = topic
                except Exception as exc:
                    print(f"Cluster {cluster_id} generated an exception: {exc}")
                    # Set a default topic for failed clusters
                    self.df.loc[self.df["cluster"] == cluster_id, "topic"] = "Unknown Topic"

        return self.df

    def _generate_topic_for_cluster(self, cluster_id: int) -> str:
        """Generate a topic for a specific cluster ID
        
        Args:
            cluster_id (int): The cluster ID to generate a topic for
            
        Returns:
            str: The generated topic
        """
        cluster_df = self.df[self.df["cluster"] == cluster_id]
        return self._generate_topic(cluster_df)

    def _generate_topic(self, df: pd.DataFrame) -> str:
        """Generate a topic from a dataframe

        Args:
            df (pd.DataFrame): The dataframe to generate a topic from.

        Returns:
            str: The generated topic.
        """
        print(f"Generating topic for cluster {df.head(10)}")

        tags_df = df[["tag_id", "key"]].drop_duplicates()
        tags = tags_df["key"].unique().tolist()

        bookmarks_df = df[
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

        system_prompt = TOPIC_GEN_SYSTEM_PROMPT
        user_prompt = TOPIC_GEN_USER_PROMPT(tags, titles)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.llm.generate(messages=messages)
        print(response)
        return response.choices[0].message.content.strip()
