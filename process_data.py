import pyarrow.parquet as pq
import pandas as pd
import logging
from data_loader import DataLoader
from added_features import (
week, added_wp, decision_classes, distance_bin, field_position,
score_difference, fourth_and_one, get_season_year, compute_distance_success_rates,
average_epa_calculations
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""Process data which includes feature engineering, cleaning and filtering the data.
Three dataframes are given as a result, dataframe for modeling, dataframe with only punts & FGs,
and full dataframe."""
class ProcessData:
    def __init__(self, include_epa: bool=True):
        self.df_model = pd.DataFrame()
        self.df_punt_fg = pd.DataFrame()
        self.include_epa = include_epa
        self.df = pd.DataFrame()

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Apply feature engineering function
        df_cleaned = self._feature_engineering(df)
        # Apply data cleaning function
        df_cleaned = self._clean_data(df_cleaned)

        # Dataframe for model training will only be run and pass plays
        self.df_model = df_cleaned[(df_cleaned['play_type'].isin(['run', 'pass']))]
        self.df_punt_fg =  df_cleaned[(df_cleaned['play_type'].isin(['punt', 'field_goal']))]
        self.df = df_cleaned
        return self.df, self.df_model, self.df_punt_fg

    def _clean_data(self, df):
        # Filter out 2 point conversion plays
        df = df[df['two_point_attempt'] == 0]

        # Convert game date to a datetime instead of a string
        df['game_date'] = df['game_date'].astype('datetime64[ns]')

        # Features filtered down to what is useful for model training and analysis
        features = ['play_id', 'game_id', 'week','game_date', 'down', 'ydstogo', 'yardline_100', 'posteam',
                        'half_seconds_remaining', 'game_seconds_remaining', 'game_half', 'qtr', 'play_type','posteam_type',
                    'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'posteam_score', 'defteam_score',
                    'score_differential', 'fg_prob', 'td_prob', 'wp', 'def_wp','drive_first_downs', 'drive_inside20',
                    'posteam_coach','distance_bin', 'posteam_wp_post', 'defteam_wp_post', 'decision_class', 'field_position',
                    'year', 'score_diff_bins', 'is_fourth_and_one', 'defteam', 'home_team', 'away_team', 'home_wp_post',
                    'away_wp_post', 'home_coach', 'away_coach', 'distance_success_rate', 'fourth_down_converted', 'two_point_attempt']
        logger.info(f"Cleaning data, features: {features}")

        # Include Expected Points (EP) and Expected Points Added (EPA) for postgame model simulation
        if self.include_epa:
            features += ['ep', 'epa']

        # All features filtered down to the ones listed above.
        df = df[features]

        # Convert the following columns to booleans
        convert_to_bool = ['fourth_down_converted', 'fourth_down_failed', 'field_goal_attempt', 'punt_attempt',
                           'drive_inside20']

        for col in convert_to_bool:
            if col in df.columns:
                df[col] = df[col].astype('boolean')
            else:
                logger.warning(f"Boolean column {col} not found, skipping")

        # Convert the following columns to integers
        convert_to_int = ['play_id', 'down', 'ydstogo', 'yardline_100', 'half_seconds_remaining',
                           'game_seconds_remaining', 'qtr', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
                          'posteam_score', 'defteam_score', 'score_differential', 'drive_first_downs']

        for col in convert_to_int:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
                except Exception as e:
                    logger.error(f"Failed to convert column {col} to int: {e}")
            else:
                logger.warning(f"Integer column {col} not found, skipping")

        # If there are any missing values, give a warning
        if df.isnull().any().any():
            logger.warning("There are missing values in the dataframe.")

        logger.info(f"Shape after cleaning: {df.shape}")

        return df

    """Apply feature engineering from added features script"""
    def _feature_engineering(self, df):
        logger.info("Adding created features...")
        df = week(df)
        df = added_wp(df)
        df = distance_bin(df)
        df = field_position(df)
        df = score_difference(df)
        df = fourth_and_one(df)
        df = decision_classes(df)
        df = get_season_year(df)
        df = compute_distance_success_rates(df)
        df = average_epa_calculations(df)
        return df
