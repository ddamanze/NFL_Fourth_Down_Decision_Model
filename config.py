"""Columns to exclude when training the model. These are nice to have once training and predictions
are complete, but add additional features that do not add value to the model."""
COLUMNS_TO_EXCLUDE = ['game_id', 'posteam', 'decision_class', 'play_id', 'year', 'week','wp',
                      'def_wp', 'posteam_wp_post', 'defteam_wp_post', 'posteam_coach', 'game_date',
                      'defteam', 'home_team', 'away_team', 'home_wp_post', 'away_wp_post', 'home_coach',
                      'away_coach', 'two_point_attempt', 'distance_success_rate', 'fourth_down_converted',
                        'fourth_down_failed', 'field_goal_attempt', 'punt_attempt', 'play_type', 'field_goal_result',
                      'two_point_attempt']

MODEL_SIMULATION_COLUMNS = [
    'down',
    'ydstogo',
    'yardline_100',
    'half_seconds_remaining',
    'game_seconds_remaining',
    'game_half',
    'qtr',
    'posteam_type',
    'posteam_timeouts_remaining',
    'defteam_timeouts_remaining',
    'posteam_score',
    'defteam_score',
    'score_differential',
    'fg_prob',
    'td_prob',
    'drive_first_downs',
    'drive_inside20',
    'distance_bin',
    'field_position',
    'score_diff_bins',
    'is_fourth_and_one',
    'distance_success_rate'
]

"""Expected points (EP) and Expected Points Added (EPA) only available via Github after games have ended.
This is used as a toggle in case the model simulation is ran for in-game situations where EP and EPA
are not accessible."""
POSTGAME_ONLY_COLS = ['ep', 'epa']

CONFIG = {'realtime': {
    'exclude_cols': COLUMNS_TO_EXCLUDE + POSTGAME_ONLY_COLS,
    'include_epa': False,
},
'postgame': {
    'exclude_cols': COLUMNS_TO_EXCLUDE,
    'include_epa': True,}
}

"""Average punt added to use in punt simulation. Slightly adjusted within model simulation depending on the field
position of the punting team."""
AVG_PUNT = 45
