import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from joblib import load
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from streamlit import cache_resource
from streamlit import cache_data

from utils import param_dist
from config import CONFIG
from process_data import ProcessData
import logging
import xgboost as xgb
import requests
from io import BytesIO
from sklearn.metrics import mean_absolute_error, accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set mode to postgame
# MODE = 'postgame'
# cfg = CONFIG[mode]
#
# columns_to_exclude = cfg['exclude_cols']
model_path_url = "https://github.com/ddamanze/NFL_Fourth_Down_Decision_Model/releases/download/v1.0-datasets/model_{mode}.joblib"
model_path_conversion_url = "https://github.com/ddamanze/NFL_Fourth_Down_Decision_Model/releases/download/v1.0-datasets/model_path_conversion_probability.joblib"


# Cached loader function outside the class
@cache_resource
def load_model_from_url(_url: str):
    """Load joblib model from GitHub release URL, cached by Streamlit."""
    try:
        response = requests.get(_url)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {_url}: {e}")


class ModelTrainer:
    def __init__(self, df: pd.DataFrame, mode='postgame'):
        process_data = ProcessData()
        df, df_model, df_punt_fg = process_data.run(df)
        self.df_model = df_model.copy()
        self.df_punt_fg = df_punt_fg.copy()
        self.df = df.copy()
        self.mode = mode

        # Load models from GitHub Releases
        self.pipeline_model = load_model_from_url(model_path_url.format(mode=self.mode))
        self.pipeline_conversion_probability = load_model_from_url(model_path_conversion_url)

        cfg = CONFIG[self.mode]
        self.columns_to_exclude = cfg['exclude_cols']

        # Column filtering and transformer
        self.column_filter()
        self.build_transformer()

    def build_transformer(self):
        if isinstance(self.data_cat_cols, str):
            self.data_cat_cols = [self.data_cat_cols]
        transformer = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.data_cat_cols)
        ], remainder='passthrough')
        return transformer

    def column_filter(self):
        data_num = self.df_model.select_dtypes(include=['int64', 'float64']).copy()
        data_cat = self.df_model.select_dtypes(include=['object', 'category', 'boolean']).copy()
        self.data_cat_cols = [col for col in data_cat.columns if col not in self.columns_to_exclude]
        self.data_num_cols = [col for col in data_num.columns if col not in self.columns_to_exclude]
        logger.info(f"Columns excluded from model: {self.columns_to_exclude}")
        logger.info(f"Numeric columns used: {self.data_num_cols}")
        logger.info(f"Categorical columns used: {self.data_cat_cols}")
        return self.data_cat_cols

    def predict(self, df: pd.DataFrame):
        df_ready = df.drop(columns=[col for col in self.columns_to_exclude if col in df.columns])
        return self.pipeline_model.predict(df_ready)

    def predict_fourth_probability(self, df: pd.DataFrame):
        df_ready = df.drop(columns=[col for col in self.columns_to_exclude if col in df.columns])
        probabilities = self.pipeline_conversion_probability.predict_proba(df_ready)
        df_ready['fourth_down_probability'] = probabilities[:, 1]
        return df_ready


# Streamlit-safe cached prediction functions
# @cache_data
# def cached_predict(_df: pd.DataFrame, mode: str = "postgame"):
#     trainer = ModelTrainer(_df, mode=mode)
#     return trainer.predict(_df)


# @cache_data
# def cached_predict_fourth_probability(_df: pd.DataFrame, mode: str = "postgame"):
#     trainer = ModelTrainer(_df, mode=mode)
#     return trainer.predict_fourth_probability(_df)

    # def model(self):
    #     x = self.df.drop(columns=COLUMNS_TO_DROP_MODEL, axis=1)
    #     y = self.df['posteam_wp_post']
    #     # Train Test Split before One Hot to make it easier for future extractions and predictions
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #
    #     regressor = xgb.XGBRegressor()
    #     transformer = self.feature_pipeline.build_transformer(self.data_cat_cols)
    #     pipeline = Pipeline(steps=[('preprocessor', transformer),
    #                                ('model', regressor)])
    #     pipeline.fit(x_train, y_train)
    #     pipeline = joblib.load('model_postgame.joblib')
    #     self.pipeline_model = pipeline

    # def column_filter(self):
    #     data_num = self.df.select_dtypes(include=['int64', 'float64']).copy()
    #     data_cat = self.df.select_dtypes(include=['object', 'category', 'boolean']).copy()
    #     data_cat_cols = [col for col in data_cat.columns if col not in columns_to_exclude]
    #     data_num_cols = [col for col in data_num.columns if col not in columns_to_exclude]
    #
    #     logger.info(f"Using mode {MODE}")
    #     logger.info(f"Columns excluded from model: {columns_to_exclude}")
    #     logger.info(f"Numeric columns used: {data_num_cols}")
    #     logger.info(f"Categorical columns used: {data_cat_cols}")
    #
    #     return data_cat_cols
