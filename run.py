import pandas as pd
from process_data import ProcessData
#from utils import compute_distance_success_rates
from config import CONFIG
from streamlit import cache_data
from model_trainer import ModelTrainer, cached_predict

from model_simulation import (transform_inputs_success, transform_inputs_failure,
                              transform_inputs_fg_success, transform_inputs_fg_failure,
                              transform_inputs_punt, decision_time)


class RunPipeline:
    def __init__(self, df: pd.DataFrame, mode='postgame'):
        self.original_df = df
        self.mode = mode
        self.model_trainer = ModelTrainer(self.original_df, mode=mode)
        self.df_model = self.model_trainer.df_model
        self.df_punt_fg = self.model_trainer.df_punt_fg
        self.df = self.model_trainer.df.loc[:, ~self.model_trainer.df.columns.duplicated()]
        self.base_df = self.create_base_df()
        self.post_pred_df = pd.DataFrame()

    def create_base_df(self):
        base_df = self.df_model[self.df_model['down'] == 4]
        base_df = pd.concat([base_df, self.df_punt_fg])
        base_df = base_df.loc[:, ~base_df.columns.duplicated()]
        return base_df

    def run_pipeline(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.base_df.copy()

        # Use cached prediction functions to avoid Streamlit hashing errors
        input_df = self.model_trainer.predict_fourth_probability(df)
        input_df['fourth_success'] = cached_predict(input_df.apply(transform_inputs_success, axis=1, result_type='expand'), mode=self.mode)
        input_df['fourth_failure'] = cached_predict(input_df.apply(transform_inputs_failure, axis=1, result_type='expand'), mode=self.mode)

        input_df.loc[input_df['yardline_100'] == input_df['ydstogo'], 'fourth_success'] = (
            1 - input_df.loc[input_df['yardline_100'] == input_df['ydstogo'], 'fourth_success']
        )
        input_df['fourth_failure'] = 1 - input_df['fourth_failure']
        input_df['go_for_it_decision'] = (
            input_df['fourth_success'] * input_df['fourth_down_probability'] +
            input_df['fourth_failure'] * (1 - input_df['fourth_down_probability'])
        )

        input_df['fg_success'] = cached_predict(input_df.apply(transform_inputs_fg_success, axis=1, result_type='expand'), mode=self.mode)
        input_df['fg_failure'] = cached_predict(input_df.apply(transform_inputs_fg_failure, axis=1, result_type='expand'), mode=self.mode)
        input_df['fg_success'] = 1 - input_df['fg_success']
        input_df['fg_failure'] = 1 - input_df['fg_failure']
        input_df['kick_fg_decision'] = (
            input_df['fg_success'] * input_df['fg_prob'] +
            input_df['fg_failure'] * (1 - input_df['fg_prob'])
        )

        input_df['punt'] = cached_predict(input_df.apply(transform_inputs_punt, axis=1, result_type='expand'), mode=self.mode)
        input_df['punt'] = 1 - input_df['punt']

        input_df['model_recommendation'] = input_df.apply(decision_time, axis=1)

        columns_to_add_back = [
            'game_id','posteam','decision_class','play_id','year','week','wp','def_wp',
            'posteam_wp_post','defteam_wp_post','posteam_coach','game_date','defteam',
            'home_team','away_team','home_wp_post','away_wp_post','home_coach','away_coach',
            'two_point_attempt','distance_success_rate','fourth_down_converted','play_type'
        ]
        input_df[columns_to_add_back] = df[columns_to_add_back].values
        self.post_pred_df = input_df
        return input_df
