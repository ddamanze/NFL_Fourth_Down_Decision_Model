import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

"""Add week of the NFL season using Game ID"""
def week(df):
    df['week'] = df['game_id'].apply(lambda x: str(x).split("_")[1]).astype(int)
    return df

"""Add win probability based on the possessing team instead of home/away team. This makes it easier when using
the model to predict the post play win probability."""
def added_wp(df):
    required_cols = ['posteam', 'defteam', 'home_team', 'away_team', 'home_wp_post', 'away_wp_post', 'home_coach',
                     'away_coach']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in added_wp: {missing}")

    df['posteam_wp_post'] = np.where(
        df['posteam'] == df['home_team'],
        df['home_wp_post'],
        df['away_wp_post']
    )
    df['defteam_wp_post'] = np.where(
        df['defteam'] == df['home_team'],
        df['home_wp_post'],
        df['away_wp_post']
    )
    df['posteam_coach'] = np.where(
        df['posteam'] == df['home_team'],
        df['home_coach'],
        df['away_coach']
    )
    return df

"""Decision classes created as a feature based on the decision of going for it, punting or kicking the FG.
Will come in handy when filtering down to only 4th down decisions"""
def decision_classes(df):
    df['decision_class'] = np.select(
        [
            df['play_type'].isin(['pass', 'run']),
            df['play_type'] == 'punt',
            df['play_type'] == 'field_goal'
        ],
        ['Go For It', 'Punt', 'Kick FG'],
        default='other'
    )
    return df

"""Created distance bins as a feature to group them based on difficulty of distance. 4th & 4 and 4th & 5 are
typically treated the same by teams. These bins will also be used when calculating the success rate for 4th down
conversion by distance."""
def distance_bin(df):
    df['distance_bin'] = pd.cut(df['ydstogo'], bins=[0, 1, 3, 6, 10, 100],
                                labels=['inches', 'short', 'medium', 'long', 'very long'])
    return df

"""Field positioning group into bins"""
def field_position(df):
    df['field_position'] = pd.cut(df['yardline_100'], bins=[0, 49, 50, 100],
                                  labels=['OPP', '50', 'OWN'])
    return df

"""Score difference grouped into bins based on how many scores a team is up or down by"""
def score_difference(df):
    df['score_diff_bins'] = pd.cut(df['score_differential'], bins = [-100, -16, -8, 8, 16, 100],
                                            labels = ['down_2+_scores', 'down_2_scores',
                                                      'one_score', 'up_2_scores', 'up_2+_scores'])
    return df

"""4th & 1 is the yardage that coaches take the most risk on. Creating a feature to allow the model to see
the difference in 4th down decision based on this criteria. This feature is also used as an indicator the make the
modeling decision a little more aggressive towards going for it."""
def fourth_and_one(df):
    df['is_fourth_and_one'] = ((df['down'] == 4) & (df['ydstogo'] == 1)).astype(int)
    return df

"""Adding the year of the season using game date. Any games between August and February will count towards the same
season."""
def get_season_year(df):
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['year'] = df['game_date'].apply(lambda d: d.year if d.month >= 8 else d.year - 1)
    return df

"""Computes the success rate of 4th down based on the coach and the distance bin created above. 
Coaches that do not have enough experience on 4th down in the database (first year head coaches) will default to a league avg.
This feature is applied during the simulation runs to determine the win probability for converting a 4th down vs turnover on downs"""
def compute_distance_success_rates(df, group_by = 'distance_bin', success_col = 'fourth_down_converted'):
    filtered_df = df[(df['down'] == 4) & (df['play_type'].isin(['run', 'pass']))].sort_values(by=['game_date', 'game_id', 'play_id']).copy()
    # Group and compute total attempts and conversions per distance bin
    grouped_all_teams = filtered_df.groupby('distance_bin')['fourth_down_converted'].agg(['sum', 'count']).reset_index()
    grouped_by_coach = filtered_df.groupby(['posteam_coach', 'distance_bin'])['fourth_down_converted'].agg(
        ['sum', 'count']).reset_index()

    grouped_all_teams['distance_success_rate'] = grouped_all_teams['sum'] / grouped_all_teams['count']
    grouped_by_coach['distance_success_rate_coach'] = grouped_by_coach['sum'] / grouped_by_coach['count']

    # Merge dataframes together
    merged = grouped_by_coach.merge(grouped_all_teams[['distance_bin', 'distance_success_rate']], on='distance_bin',
                                    how='left')

    # First year coaches get at least 2 attempts at every distance_bin
    merged['coach_rate'] = merged.apply(
        lambda row: row['distance_success_rate_coach'] if row['count'] >= 2 else row['distance_success_rate'], axis=1)

    # Convert to dictionary to map
    success_rate_lookup = merged.set_index(['posteam_coach', 'distance_bin'])['coach_rate'].to_dict()

    # Map to df
    df['distance_success_rate'] = df.apply(
        lambda row: success_rate_lookup.get((row['posteam_coach'], row['distance_bin']), None), axis=1).astype('float')

    return df

"""The functions below are for streamlit batch predictions"""
def score_diff_subtraction(df):
    df['score_differential'] = df['posteam_score'] - df['posteam_score']
    return df

def add_yardline_100(df):
    def compute_yardline(row):
        if row['field_position'] == 'OPP':
            return row['yardline_100']
        elif row['field_position'] == 'OWN':
            return 100 - row['yardline_100']
        elif row['field_position'] == 'Midfield':
            return 50
        else:
            return 0

    df['yardline_100'] = df.apply(compute_yardline, axis=1)
    return df

def drive_inside20(df):
    df['drive_inside20'] = np.where(
        (df['field_position'] == 'Opp') & (df['yardline_100'] < 20),
        True,
        False
    )
    return df


def game_half(df):
    df['game_half'] = np.select(
        [
            (df['qtr'] == 1) | (df['qtr'] == 2),
            (df['qtr'] == 4) & (df['overtime'] == 'Yes'),
            (df['qtr'] == 3) | (df['qtr'] == 4) & (df['overtime'] == 'No')
        ],
        ['Half1', 'Overtime', 'Half2'],
        default='Half2'
    )
    return df

def half_seconds_remaining(df):
    df['half_seconds_remaining'] = np.where(
        (df['qtr'] == 1) & (df['qtr'] == 3),
        (df['minutes'] * 60 + df['seconds']) + 900,
        (df['minutes'] * 60 + df['seconds'])
    )
    return df

def game_seconds_remaining(df):
    df['game_seconds_remaining'] = np.where(
        df['game_half'] == 'Half1',
        df['half_seconds_remaining'] * 1800,
        df['half_seconds_remaining']
    )
    return df

def get_knn_features(base_df, knn_df, n_neighhbors=20):
    features = ['down', 'ydstogo', 'yardline_100', 'half_seconds_remaining', 'game_seconds_remaining', 'score_differential']
    scaler = StandardScaler()
    X_model = scaler.fit_transform(base_df[features])

    knn = NearestNeighbors(n_neighbors=n_neighhbors)
    knn.fit(X_model)

    # Transform new df
    X_target = scaler.transform(knn_df[features])

    # Store results
    knn_features = []

    for row in X_target:
        indices = knn.kneighbors([row], return_distance=False)[0]
        neighbors = base_df.iloc[indices]

        row_result = {
            'posteam_timeouts_remaining': neighbors['posteam_timeouts_remaining'].mean(),
            'defteam_timeouts_remaining': neighbors['defteam_timeouts_remaining'].mean(),
            'fg_prob': neighbors['fg_prob'].median(),
            'td_prob': neighbors['td_prob'].median(),
            'drive_first_downs': neighbors['drive_first_downs'].mean()
        }

        knn_features.append(row_result)

    # Convert to DataFrame and return
    return pd.DataFrame(knn_features, index=knn_df.index)

def average_epa_calculations(df):
    epa_calc_df = df[((df['down'] == 3) | (df['down'] == 4)) & (df['play_type'].isin(['pass', 'run']))]
    offensive_epa = epa_calc_df.groupby(['posteam']).agg({'epa': 'mean'}).reset_index()
    defensive_epa = epa_calc_df.groupby(['defteam']).agg({'epa': 'mean'}).reset_index()

    offensive_epa_dict = offensive_epa.set_index(['posteam'])['epa'].to_dict()
    defensive_epa_dict = defensive_epa.set_index(['defteam'])['epa'].to_dict()

    df['avg_offensive_epa'] = df.apply(lambda row: offensive_epa_dict.get(row['posteam']), axis=1).astype(
    'float')
    df['avg_defensive_epa'] = df.apply(lambda row: defensive_epa_dict.get(row['defteam']), axis=1).astype(
        'float')
    return df