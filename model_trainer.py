import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from utils import param_dist
from config import CONFIG
from process_data import ProcessData
import logging
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set mode to postgame
# MODE = 'postgame'
# cfg = CONFIG[mode]
#
# columns_to_exclude = cfg['exclude_cols']

class ModelTrainer:
    def __init__(self, df: pd.DataFrame, model_path=None, model_path_conversion_probability=None, mode='postgame'):
        process_data = ProcessData()
        df, df_model, df_punt_fg = process_data.run(df)
        self.df_model = pd.DataFrame(df_model) #df_model
        self.df_punt_fg = pd.DataFrame(df_punt_fg) #df_punt_fg
        self.df = pd.DataFrame(df)
        self.mode = mode
        if model_path is None:
            model_path = f'/Users/ddamanze/PycharmProjects/PythonProject/model_{self.mode}.joblib'
        self.model_path = model_path
        if model_path_conversion_probability is None:
            model_path_conversion_probability = f'/Users/ddamanze/PycharmProjects/PythonProject/model_path_conversion_probability.joblib'
        self.model_path_conversion_probability = model_path_conversion_probability

        cfg = CONFIG[self.mode]
        columns_to_exclude = cfg['exclude_cols']
        self.columns_to_exclude = columns_to_exclude
        self.column_filter()
        self.build_transformer()

        self.pipeline_model = None
        self.pipeline_conversion_probability = None
        self.pipeline_field_goal_probability = None


        # Option to load the model pipeline if it is already in your local environment
        try:
            self.pipeline_model = joblib.load(model_path)
            if not isinstance(self.pipeline_model, Pipeline):
                raise TypeError("Model is not a valid sklearn Pipeline. Check the model file or retrain.")
            logger.info(f"Model successfully loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}, retraining...")
            self.model()

        try:
            self.pipeline_conversion_probability = joblib.load(model_path_conversion_probability)
            if not isinstance(self.pipeline_conversion_probability, Pipeline):
                raise TypeError("Logistic model is not a valid sklearn Pipeline. Check the model file or retrain.")
            logger.info(f"Model successfully loaded from {self.model_path_conversion_probability}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path_conversion_probability}: {e}, retraining...")
            self.logistic_regression_model()

    """Tranforms and trains the model to make predictions for post-play win probability."""
    def model(self):
        if self.pipeline_model is not None:
            logger.info("Model already loaded. Skipping training.")
            return self.pipeline_model

        if os.path.exists(self.model_path):
            logger.info(f"Model already loaded. Skipping training. ({self.model_path})")
            self.pipeline_model = joblib.load(self.model_path)
            return self.pipeline_model

        # Filter columns and preprocess the data
        x = self.df_model.drop(columns=self.columns_to_exclude, axis=1, errors='ignore')
        y = self.df_model['posteam_wp_post']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Build transformer and pipeline
        transformer = self.build_transformer()
        #XGBoost Regressor used for modeling
        pipeline = Pipeline(steps=[('preprocessor', transformer),
                                   ('model', xgb.XGBRegressor())])

        # pipeline.fit(x_train, y_train)
        """Hyperparameter Tuning to improve model performance."""
        random_search_cv = RandomizedSearchCV(pipeline, param_distributions=param_dist, scoring='neg_mean_absolute_error')
        random_search_cv.fit(x_train, y_train)
        best_model = random_search_cv.best_estimator_
        best_params = random_search_cv.best_params_
        logger.info(f"Best model parameters: {best_params}")

        try:
            # Dumps model training into a joblib in local environment
            joblib.dump(best_model, self.model_path)
            logger.info(f"Model training complete and saved as model_{self.mode}.joblib")# Save the trained model
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
        self.pipeline_model = best_model
        return self.pipeline_model

    """Logistic Regression model to predict the probability of converting a 4th down"""
    def logistic_regression_model(self):
        fourth_conversion_mode = 'realtime'
        cfg_fourth_conv = CONFIG[fourth_conversion_mode]
        columns_to_exclude_fourth_conv = cfg_fourth_conv['exclude_cols']

        if self.pipeline_conversion_probability is not None:
            logger.info("Model already loaded. Skipping training.")
            return self.pipeline_conversion_probability

        if os.path.exists(self.model_path_conversion_probability):
            logger.info(f"Model already loaded. Skipping training. ({self.model_path_conversion_probability})")
            self.pipeline_conversion_probability = joblib.load(self.model_path_conversion_probability)
            return self.pipeline_conversion_probability

        # Filter columns and preprocess the data
        x = self.df_model.drop(columns=columns_to_exclude_fourth_conv, axis=1, errors='ignore')
        y = self.df_model['fourth_down_converted']
        x_train, self.x_test, y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Build transformer and pipeline
        transformer = self.build_transformer()

        pipeline_conversion_probability = Pipeline(steps=[('preprocessor', transformer),
                                   ('model', LogisticRegression())])

        pipeline_conversion_probability.fit(x_train, y_train)

        try:
            joblib.dump(pipeline_conversion_probability, self.model_path_conversion_probability)
            logger.info(f"Model training complete and saved as model_path_conversion_probability.joblib")  # Save the trained model
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
        self.pipeline_conversion_probability = pipeline_conversion_probability
        return self.pipeline_conversion_probability


    """Builds the transformer as One Hot Encoder for any categorical columns included in the model"""
    def build_transformer(self):
        # Ensure self.data_cat_cols is a list
        if isinstance(self.data_cat_cols, str):
            self.data_cat_cols = [self.data_cat_cols]
        transformer = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.data_cat_cols)
        ], remainder='passthrough')
        return transformer

    """Filters columns using in model into categorical and numerical. Categorical will be transformed with One
    Hot Encoder and numerical columns will pass through."""
    def column_filter(self):
        # Separate numeric and categorical columns
        data_num = self.df_model.select_dtypes(include=['int64', 'float64']).copy()
        data_cat = self.df_model.select_dtypes(include=['object', 'category', 'boolean']).copy()
        self.data_cat_cols = [col for col in data_cat.columns if col not in self.columns_to_exclude]
        self.data_num_cols = [col for col in data_num.columns if col not in self.columns_to_exclude]

        if isinstance(self.data_cat_cols, str):
            self.data_cat_cols = [self.data_cat_cols]

        logger.info(f"Columns excluded from model: {self.columns_to_exclude}")
        logger.info(f"Numeric columns used: {self.data_num_cols}")
        logger.info(f"Categorical columns used: {self.data_cat_cols}")

        return self.data_cat_cols

    """Prediction function. Drops any columns that were not used in pipeline training"""
    def predict(self, df):
        if not isinstance(self.pipeline_model, Pipeline):
            raise TypeError("Model is not a valid sklearn Pipeline. Check the model file or retrain.")

        drop_cols = [col for col in self.columns_to_exclude if col in df.columns]
        df_model_ready = df.drop(columns=drop_cols)
        try:
            # Make predictions
            predictions = self.pipeline_model.predict(df_model_ready)
            logger.info(f"Predictions made successfully")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def predict_fourth_probability(self, df):
        if not isinstance(self.pipeline_conversion_probability, Pipeline):
            raise TypeError("Model is not a valid sklearn Pipeline. Check the model file or retrain.")

        drop_cols = [col for col in self.columns_to_exclude if col in df.columns]
        df_model_ready = df.drop(columns=drop_cols)
        try:
            # Make predictions
            probabilities = self.pipeline_conversion_probability.predict_proba(df_model_ready)
            logger.info(f"Predictions made successfully")
            df_model_ready['fourth_down_probability'] = probabilities[:, 1]
            return df_model_ready
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

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
