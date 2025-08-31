import pandas as pd
import logging
from run import RunPipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Coaches:
    def __init__(self, pipeline_runner: RunPipeline, latest_season=None):
        self.df = pipeline_runner.post_pred_df
        self.coach_stats_df = pd.DataFrame()
        if latest_season is not None:
            self.latest_season = int(latest_season)
        else:
            self.latest_season= int(self.df['year'].max())
        logger.info(f"Latest season: {self.latest_season}")
        self._cached_latest_season = None

    """Creates dataframe for coaching stats. Includes columns that display:
     - when the coach and the model were aligned on the 4th down decision (coach_model_aligned)
     - when the coach was more aggressive in going for it on 4th down when the model recommended to punt or kick (over_aggression)
     - when the coach was more conservative and decided to kick a FG or punt when the model recommended to go for it (missed_opportunity)"""
    def coaching_stats(self):
        if (self._cached_latest_season != self.latest_season) or (self.coach_stats.empty):
            df = self.df.copy()

            # Handle missing data by dropping rows with critical missing values
            df = df.dropna(subset=['model_recommendation', 'decision_class'])

            # 1 when model recommendation and decision class were aligned, 0 when false
            coach_model_aligned = ((df['model_recommendation'] == 'Go For It') &
             (df['decision_class'] == 'Go For It')) | ((df['model_recommendation'] == 'Kick FG') &
             (df['decision_class'] == 'Kick FG')) | ((df['model_recommendation'] == 'Punt') &
             (df['decision_class'] == 'Punt'))
            df['coach_model_aligned'] = coach_model_aligned.astype(int)

            # 1 when decision class was FG or punt and recommendation was go for it, 0 when false
            missed_opportunity = ((df['decision_class'] == 'Kick FG') |
             (df['decision_class'] == 'Punt')) & (df['model_recommendation'] == 'Go For It')
            df['missed_opportunity'] = missed_opportunity.astype(int)

            # 1 when decision class was go for it and model recommendation was FG or punt, 0 when false
            over_aggression = ((df['model_recommendation'] == 'Punt') | (df['model_recommendation'] == 'Kick FG')) & \
             (df['decision_class'] == 'Go For It')
            df['over_aggression'] = over_aggression.astype(int)

            # Filter to active coaches of the latest season
            df['year'] = df['year'].astype(int)
            active_coaches = df[df['year'] == self.latest_season]['posteam_coach'].unique()

            coach_stats = df[df['posteam_coach'].isin(active_coaches)]


            # Group stats by coach
            coach_stats = coach_stats.groupby(['posteam_coach']).agg({'coach_model_aligned': 'sum',
                                                                      'missed_opportunity': 'sum',
                                                          'over_aggression': 'sum'})

            coach_stats['total_plays'] = coach_stats['coach_model_aligned'] + coach_stats['missed_opportunity'] + coach_stats['over_aggression']
            # Net aggressiveness score = when coaches were aggressive subtracted by conservative decisions over the total amount of 4th downs
            coach_stats['net_aggressiveness_score'] = (coach_stats['over_aggression'] - coach_stats['missed_opportunity']) / coach_stats['total_plays']
            coach_stats['alignment_rate'] = coach_stats['coach_model_aligned'] / coach_stats['total_plays']
            self.coach_stats_df = coach_stats.reset_index()
            self._cached_latest_season = self.latest_season
        return self.coach_stats_df