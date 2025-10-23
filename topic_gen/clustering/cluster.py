import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
import numpy as np


class AbstractCluster(ABC):
    """AbstractCluster class"""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self._model_name = model
        self.embedding_model = SentenceTransformer(self._model_name)

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the cluster model"""
        pass


class KMeansCluster(AbstractCluster):
    """KMeansCluster class"""

    def __init__(self, model: str = "all-MiniLM-L6-v2", n_clusters: int = 10):
        super().__init__(model)
        self._n_clusters = n_clusters

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the kmeans cluster model

        Args:
            df (pd.DataFrame): The dataframe to fit the cluster model to.
            Must have a "key" column.
            Must have no missing values in the "key" column.

        Returns:
            pd.DataFrame: The dataframe with the "cluster" column (int) added to the dataframe
        """
        assert "key" in df.columns, "Dataframe must have a 'key' column"
        assert (
            df["key"].notna().all()
        ), "Dataframe must have no missing values in the 'key' column"

        self.df = df
        self.embeddings = self.embedding_model.encode(df["key"])
        # Perform kmean clustering
        self.kmeans = KMeans(n_clusters=self._n_clusters)
        self.kmeans.fit(self.embeddings)
        self.df["cluster"] = self.kmeans.labels_
        return self.df


class AgglomerativeCluster(AbstractCluster):
    """AgglomerativeCluster class for agglomerative clustering"""

    def __init__(
        self, model: str = "all-MiniLM-L6-v2", distance_threshold: float = 1.5
    ):
        """AgglomerativeCluster class initializer

        Args:
            model (str, optional): The model to use for the embedding. Defaults to "all-MiniLM-L6-v2".
            distance_threshold (float, optional): The distance threshold to use for the agglomerative clustering. Defaults to 1.5.
        """
        super().__init__(model)
        self._distance_threshold = distance_threshold

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the agglomerative cluster model

        Args:
            df (pd.DataFrame): The dataframe to fit the cluster model to.
            Must have a "key" column.
            Must have no missing values in the "key" column.

        Returns:
            pd.DataFrame: The dataframe with the "cluster" column (int) added to the dataframe
        """
        assert "key" in df.columns, "Dataframe must have a 'key' column"
        assert (
            df["key"].notna().all()
        ), "Dataframe must have no missing values in the 'key' column"

        self.df = df
        self.embeddings = self.embedding_model.encode(df["key"].tolist())
        # Normalize the embeddings
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        # Perform agglomerative clustering
        self.agglomerative = AgglomerativeClustering(
            distance_threshold=self._distance_threshold, n_clusters=None
        )
        self.agglomerative.fit(self.embeddings)
        self.df["cluster"] = self.agglomerative.labels_
        return self.df
