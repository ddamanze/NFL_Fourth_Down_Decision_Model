import pandas as pd
from .process_data import ProcessData
#from utils import compute_distance_success_rates
from .config import CONFIG
from .model_trainer import ModelTrainer
from .model_simulation import (transform_inputs_success, transform_inputs_failure,
                              transform_inputs_fg_success, transform_inputs_fg_failure,
                              transform_inputs_punt, decision_time)


class RunPipeline:
    def __init__(self, df: pd.DataFrame, mode='postgame'):
        self.original_df = df
        self.mode = mode
        model_path = 'model_postgame.joblib' if self.mode == 'postgame' else 'model_realtime.joblib'
        self.model_trainer = ModelTrainer(self.original_df, model_path=model_path, mode=mode)
        self.mode = self.model_trainer.mode
        self.df_model = self.model_trainer.df_model
        self.df_punt_fg = self.model_trainer.df_punt_fg
        # Re-apply self.df to make sure it is converted correctly from model trainer
        self.df = self.model_trainer.df.loc[:, ~self.model_trainer.df.columns.duplicated()]
        #self.pipeline_model = model_trainer.model()
        self.base_df = self.create_base_df()
        self.base_pred_df = self.run_pipeline()
        self.post_pred_df = pd.DataFrame()


    def create_base_df(self):
        # Dataframes joined together to include all scenarios of 4th downs, going for it, punt, FG
        base_df = self.df_model[self.df_model['down'] == 4]
        base_df = pd.concat([base_df, self.df_punt_fg])
        base_df = base_df.loc[:, ~base_df.columns.duplicated()]
        return base_df

    """Runs pipeline prediction for every simulation and creates a win probability column for each.
    The win probability is flipped in every scenario except a 4th down conversion because the other team
    will be considered as the possessing team. The model predicts the win probability of the possessing team."""
    def run_pipeline(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.base_df.copy()

        # distance_success_lookup = compute_distance_success_rates(filtered_df)
        # filtered_df['distance_success_rate'] = filtered_df['distance_bin'].map(distance_success_lookup).astype('float')
        #input_df = input_df.drop(columns=[col for col in columns_to_exclude if col in input_df.columns])
        # Simulation for 4th down success and failure
        # transform_rows_success = filtered_df.apply(transform_inputs_success, axis=1)
        #df_for_pred = transform_rows_success.drop(columns=columns_to_exclude)
        input_df = self.model_trainer.predict_fourth_probability(df)
        input_df['fourth_success'] = self.model_trainer.predict(input_df.apply(transform_inputs_success, axis=1, result_type='expand'))
        input_df['fourth_failure'] = self.model_trainer.predict(input_df.apply(transform_inputs_failure, axis=1, result_type='expand'))

        # Apply an if function for the instances where a successful 4th down conversion would result in a TD, flipping the win probability
        input_df.loc[input_df['yardline_100'] == input_df['ydstogo'], 'fourth_success'] = (
                1 - input_df.loc[input_df['yardline_100'] == input_df['ydstogo'], 'fourth_success']
        )

        # Win probability flipped because the function for turnover on downs flips the possession arrow to the other team
        input_df['fourth_failure'] = 1 - input_df['fourth_failure']

        # Win probability formula based on probability of converting 4th down vs turnover on downs
        input_df['go_for_it_decision'] = (input_df['fourth_success'] * input_df['fourth_down_probability'])\
        + (input_df['fourth_failure'] * (1 - (input_df['fourth_down_probability'])))

        # Simulation for field goal success and failure
        input_df['fg_success'] = self.model_trainer.predict(input_df.apply(transform_inputs_fg_success, axis=1, result_type='expand'))
        input_df['fg_failure'] = self.model_trainer.predict(input_df.apply(transform_inputs_fg_failure, axis=1, result_type='expand'))

        # Win probability flipped because the other team receives possession regardless of FG outcome
        input_df['fg_success'] = 1 - input_df['fg_success']
        input_df['fg_failure'] = 1 - input_df['fg_failure']

        # Win probability formula based on probability of making FG vs missing
        input_df['kick_fg_decision'] = (input_df['fg_success'] * input_df['fg_prob'])\
        + (input_df['fg_failure'] * (1 - (input_df['fg_prob'])))

        # Simulation for punt
        input_df['punt'] = self.model_trainer.predict(input_df.apply(transform_inputs_punt, axis=1, result_type='expand'))
        # Win probability flipped because of punt
        input_df['punt'] = 1 - input_df['punt']

        input_df['model_recommendation'] = input_df.apply(decision_time, axis=1)
        # filtered_df['model_recommendation'] = self.predict(decision_time(filtered_df))
        columns_to_add_back = ['game_id', 'posteam', 'decision_class', 'play_id', 'year', 'week','wp',
                      'def_wp', 'posteam_wp_post', 'defteam_wp_post', 'posteam_coach', 'game_date',
                      'defteam', 'home_team', 'away_team', 'home_wp_post', 'away_wp_post', 'home_coach',
                      'away_coach', 'two_point_attempt', 'distance_success_rate', 'fourth_down_converted',
                     'play_type']
        input_df[columns_to_add_back] = df[columns_to_add_back].values
        self.post_pred_df = input_df
        return input_df
