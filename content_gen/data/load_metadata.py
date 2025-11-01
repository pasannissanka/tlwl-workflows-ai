import logging
from common.db import AbstractDB, PgDB
import pandas as pd

logger = logging.getLogger(__name__)


class LoadMetadata:
    def __init__(self):
        self.db: AbstractDB = PgDB()

    def connect(self, init_models: bool = False) -> None:
        """Connect to the database"""
        logger.debug(f"Connecting to the database with init_models: {init_models}")
        self.db.connect(init_models=init_models)
        
    def load_metadata(self, user_id: str, topic_id: str) -> pd.DataFrame:
        topic_df =  None
        # query = """
