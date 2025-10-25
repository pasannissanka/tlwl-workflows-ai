import pandas as pd


class CleanUpInput:
    @staticmethod
    def cleanup(df: pd.DataFrame) -> pd.DataFrame:
        """Cleanup the input dataframe

        Args:
            df (pd.DataFrame): The input dataframe to cleanup.

        Returns:
            pd.DataFrame: The cleaned input dataframe.
        """
        # Remove rows with empty title
        df = df[df["title"].notna()]
        # Remove rows with empty url
        df = df[df["url"].notna()]
        # Remove rows with empty tag key
        df = df[df["key"].notna()]
        return df
