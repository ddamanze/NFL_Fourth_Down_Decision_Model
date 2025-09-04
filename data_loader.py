import pyarrow.parquet as pq
import pandas as pd
import logging

from pandas.io.parquet import to_parquet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""Github website to pull play by play data"""
base_url = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet'

class DataLoader:
    def __init__(self, years=None, local_data_dir = './data'):
        """If no years are given, use 2020 and later as the years to load"""
        if years is None:
            years = [2020, 2021, 2022, 2023, 2024]
        self.years = years
        self.local_data_dir = local_data_dir
        self.df = pd.DataFrame()
        self.is_data_loaded = False

    def load_data(self):
        """Pull from Github"""
        all_years = []
        for year in self.years:
            try:
                url = f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet'
                df_read = pd.read_parquet(url, engine='pyarrow')
                all_years.append(df_read)
                logger.info(f"Successfully loaded data for {year}")
            except Exception as e:
                logger.warning(f"Failed to load data for {year}: {e}")

        if all_years:
            self.df = pd.concat(all_years, ignore_index=True)
        else:
            logger.warning("No data was loaded.")
            self.df = pd.DataFrame()

        return self.df
