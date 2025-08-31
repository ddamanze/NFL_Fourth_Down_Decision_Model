
param_dist = {
    'n_estimators': [100, 300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.01, 0.1, 1, 5],
}

def run_streamlit_preloads(df, pre_loaded_scenario, pre_loaded_score_diff):
    # Take out all instances where coaching decisions will lean towards going for it because they are trailing late in game
    desperation_to_exclude = df[((df['game_half']=='Half2') & (df['half_seconds_remaining'] <= 120) & (df['score_differential'] < -3))\
    | ((df['game_half']=='Half2') & (df['half_seconds_remaining'] <= 300) & (df['score_differential'] <= -13))\
        | ((df['game_half']=='Half2') & (df['half_seconds_remaining'] <= 480) & (df['score_differential'] < -14))]

    df = df.loc[~df.index.isin(desperation_to_exclude.index)]

    if (pre_loaded_score_diff == 'Trailing') & (pre_loaded_scenario == '4th & 1 OWN 40YD Line'):
        pre_loaded_df = df[(df['score_differential'] < 0) & (df['ydstogo'] == 1) &
                           (df['yardline_100'] == 60)].sample(n=1)

    elif (pre_loaded_score_diff == 'Tie Game') & (pre_loaded_scenario == '4th & 1 OWN 40YD Line'):
        pre_loaded_df = df[(df['score_differential'] == 0) & (df['ydstogo'] == 1) &
                           (df['yardline_100'] == 60)].sample(n=1)

    elif (pre_loaded_score_diff == 'Leading') & (pre_loaded_scenario == '4th & 1 OWN 40YD Line'):
        pre_loaded_df = df[(df['score_differential'] > 0) & (df['ydstogo'] == 1) &
                           (df['yardline_100'] == 60)].sample(n=1)

    elif (pre_loaded_score_diff == 'Trailing') & (pre_loaded_scenario == '4th & short 50YD Line'):
        pre_loaded_df = df[(df['score_differential'] < 0) & (df['distance_bin'] == 'short') &
                           (df['yardline_100'] == 50)].sample(n=1)

    elif (pre_loaded_score_diff == 'Tie Game') & (pre_loaded_scenario == '4th & short 50YD Line'):
        pre_loaded_df = df[(df['score_differential'] == 0) & (df['distance_bin'] == 'short') &
                           (df['yardline_100'] == 50)].sample(n=1)

    elif (pre_loaded_score_diff == 'Leading') & (pre_loaded_scenario == '4th & short 50YD Line'):
        pre_loaded_df = df[(df['score_differential'] > 0) & (df['distance_bin'] == 'short') &
                           (df['yardline_100'] == 50)].sample(n=1)

    elif (pre_loaded_score_diff == 'Trailing') & (pre_loaded_scenario == '4th & 1 OPP 40YD Line'):
        pre_loaded_df = df[(df['score_differential'] < 0) & (df['ydstogo'] == 1) &
                           (df['yardline_100'] == 40)].sample(n=1)

    elif (pre_loaded_score_diff == 'Tie Game') & (pre_loaded_scenario == '4th & 1 OPP 40YD Line'):
        pre_loaded_df = df[(df['score_differential'] == 0) & (df['ydstogo'] == 1) &
                           (df['yardline_100'] == 40)].sample(n=1)

    elif (pre_loaded_score_diff == 'Leading') & (pre_loaded_scenario == '4th & 1 OPP 40YD Line'):
        pre_loaded_df = df[(df['score_differential'] > 0) & (df['ydstogo'] == 1) &
                           (df['yardline_100'] == 40)].sample(n=1)

    elif (pre_loaded_score_diff == 'Trailing') & (pre_loaded_scenario == '4th & short in FG Range'):
        pre_loaded_df = df[(df['score_differential'] < 0) & (df['distance_bin'] == 'short') &
                           (df['yardline_100'] <= 35)].sample(n=1)

    elif (pre_loaded_score_diff == 'Tie Game') & (pre_loaded_scenario == '4th & short in FG Range'):
        pre_loaded_df = df[(df['score_differential'] == 0) & (df['distance_bin'] == 'short') &
                           (df['yardline_100'] <= 35)].sample(n=1)

    elif (pre_loaded_score_diff == 'Leading') & (pre_loaded_scenario == '4th & short in FG Range'):
        pre_loaded_df = df[(df['score_differential'] > 0) & (df['distance_bin'] == 'short') &
                           (df['yardline_100'] <= 35)].sample(n=1)

    elif (pre_loaded_score_diff == 'Trailing') & (pre_loaded_scenario == '4th & 1 in FG Range'):
        pre_loaded_df = df[(df['score_differential'] < 0) & (df['ydstogo'] == 1) &
                           (df['yardline_100'] <= 35)].sample(n=1)

    elif (pre_loaded_score_diff == 'Tie Game') & (pre_loaded_scenario == '4th & 1 in FG Range'):
        pre_loaded_df = df[(df['score_differential'] == 0) & (df['ydstogo'] == 1) &
                           (df['yardline_100'] <= 35)].sample(n=1)

    elif (pre_loaded_score_diff == 'Leading') & (pre_loaded_scenario == '4th & 1 in FG Range'):
        pre_loaded_df = df[(df['score_differential'] > 0) & (df['ydstogo'] == 1) &
                           (df['yardline_100'] <= 35)].sample(n=1)

    elif (pre_loaded_score_diff == 'Trailing') & (pre_loaded_scenario == '4th & Goal 3YD Line'):
        pre_loaded_df = df[(df['score_differential'] < 0) & (df['ydstogo'] == 3) &
                           (df['yardline_100'] == 3)].sample(n=1)

    elif (pre_loaded_score_diff == 'Tie Game') & (pre_loaded_scenario == '4th & Goal 3YD Line'):
        pre_loaded_df = df[(df['score_differential'] == 0) & (df['ydstogo'] == 3) &
                           (df['yardline_100'] == 3)].sample(n=1)

    elif (pre_loaded_score_diff == 'Leading') & (pre_loaded_scenario == '4th & Goal 3YD Line'):
        pre_loaded_df = df[(df['score_differential'] > 0) & (df['ydstogo'] == 3) &
                           (df['yardline_100'] == 3)].sample(n=1)
    else:
        raise ValueError('Unrecognized score difference or scenario')

    return pre_loaded_df

def model_misses(row):
    if (row['decision_class'] == 'Go For It') & (row['model_recommendation'] == 'Kick FG'):
        return row['kick_fg_decision'] - row['go_for_it_decision'] # this SHOULD be a negative number. The more negative, the more aggressive the call was
    elif (row['decision_class'] == 'Go For It') & (row['model_recommendation'] == 'Punt'):
        return row['punt'] - row['go_for_it_decision'] # this SHOULD be a negative number. The more negative, the more aggressive the call was
    elif (row['decision_class'] == 'Kick FG') & (row['model_recommendation'] == 'Punt'):
        return row['punt'] - row['kick_fg_decision'] # this SHOULD be a negative number. The more negative, the more aggressive the call was
    elif (row['decision_class'] == 'Kick FG') & (row['model_recommendation'] == 'Go For It'):
        return row['kick_fg_decision'] - row['go_for_it_decision'] # this SHOULD be a negative number. The more negative, the more conservative the call was
    elif (row['decision_class'] == 'Punt') & (row['model_recommendation'] == 'Go For It'):
        return row['punt'] - row['go_for_it_decision']
    elif (row['decision_class'] == 'Punt') & (row['model_recommendation'] == 'Kick FG'):
        return row['punt'] - row['kick_fg_decision']
    else:
        return 0

def scenario_sentence(qtr, half_seconds_remaining, score_differential,coach, team, defteam, week, ydstogo, yardline_100, decision, model_decision, probability):
    yard_desc = f"the opponent's {yardline_100}-yard line" if yardline_100 <= 50 else f"their own {yardline_100}-yard line"
    if decision == 'Go For It':
        return (f"{coach} ({team}) chose to {decision} on 4th and {ydstogo} from {yard_desc} in the {qtr} of week {week} with {half_seconds_remaining} remaining against {defteam} up/down by {score_differential}.\n"
            f"The model recommended to {model_decision} with a {probability}.2% chance of converting.")
    else:
        return (f"{coach} ({team}) chose to {decision} on 4th and {ydstogo} from {yard_desc} in the {qtr} of week {week} with {half_seconds_remaining} remaining against {defteam} up/down by {score_differential}.\n"
            f"The model recommended to {model_decision} with a {probability:.2%} chance of converting.")
