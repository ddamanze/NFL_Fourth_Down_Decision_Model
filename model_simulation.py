from config import AVG_PUNT
import numpy as np

"""Function to simulate a successful 4th down conversion. Assumptions are:
- If it is 4th and goal, team scores 7 points for a TD
- Yardage gained on the play is only the yardage needed to convert the 4th down.
- 40 seconds are run off the clock by the time the next play begins."""
def transform_inputs_success(row):
  row = row.copy()
  # Could potentially include scenario here where TD is scored when yardline_100 is the same as yds_to_go
  # Taking these out for now since they do not impact the wp predictions
  # row['fourth_down_converted'] = True
  # row['fourth_down_failed'] = False
  if row['yardline_100'] == row['ydstogo']:
    # If its 4th & goal, simulate a change of possession with 7 points added
    row['distance_bin'] = 'long'
    row['yardline_100'] = 30
    row['down'] = 1
    row['ydstogo'] = 10
    row['half_seconds_remaining'] = row['half_seconds_remaining'] - 40
    row['game_seconds_remaining'] = row['game_seconds_remaining'] - 40
    # Score differential flipped with the possession arrow
    row['score_differential'] = -row['score_differential'] + 7
    row['drive_first_downs'] = 0
    if row['posteam_type'] == 'away':
      row['posteam_type'] = 'home'
    else:
      row['posteam_type'] = 'away'
    row['field_position'] = 'OWN'
  else:
    # Team keeps possession, gets first down
    row['distance_bin'] = 'long'
    row['yardline_100'] = row['yardline_100'] - row['ydstogo']
    row['down'] = 1
    row['ydstogo'] = 10
    row['half_seconds_remaining'] = row['half_seconds_remaining'] - 40
    row['game_seconds_remaining'] = row['game_seconds_remaining'] - 40
    row['score_differential'] = row['score_differential']
    row['drive_first_downs'] = row['drive_first_downs'] + 1
    row['posteam_type'] = row['posteam_type']
    if row['yardline_100'] < 50:
      row['field_position'] = 'OPP'
    elif row['yardline_100'] == 50:
      row['field_position'] = 'Midfield'
    else:
      row['field_position'] = 'OWN'
  return row

"""Function to simulate an unsuccessful 4th down conversion. Assumptions are:
- Defending team now takes over possession.
- Field position and yard line 100 are flipped since possession arrow changes.
- 40 seconds are run off the clock by the time the next play begins."""
def transform_inputs_failure(row):
  row = row.copy()
  # row['fourth_down_converted'] = False
  # row['fourth_down_failed'] = True
  row['distance_bin'] = 'long'
  row['yardline_100'] = 100 - (10 - row['ydstogo'])
  row['down'] = 1
  row['ydstogo'] = 10
  row['half_seconds_remaining'] = row['half_seconds_remaining'] - 40
  row['game_seconds_remaining'] = row['game_seconds_remaining'] - 40
  # Score differential flipped with the possession arrow
  row['score_differential'] = -row['score_differential']
  row['drive_first_downs'] = 0
  if row['posteam_type'] == 'away':
    row['posteam_type'] = 'home'
  else:
    row['posteam_type'] = 'away'
  if row['yardline_100'] < 50:
    row['field_position'] = 'OPP'
  elif row['yardline_100'] == 50:
    row['field_position'] = 'Midfield'
  else:
    row['field_position'] = 'OWN'
  return row

"""Function to simulate a successful FG kick. Assumptions are:
- Defending team now takes over possession at the line of a touchback. 
- Kickoff play is skipped in this scenario due to its exclusion from the model.
- 40 seconds are run off the clock by the time the next play begins."""
def transform_inputs_fg_success(row):
  row = row.copy()
  # row['fourth_down_converted'] = False
  # row['fourth_down_failed'] = False
  row['distance_bin'] = 'long'
  row['yardline_100'] = 30 #or 35
  row['down'] = 1
  row['ydstogo'] = 10
  row['half_seconds_remaining'] = row['half_seconds_remaining'] - 40
  row['game_seconds_remaining'] = row['game_seconds_remaining'] - 40
  row['drive_first_downs'] = 0
  # Points added to the team that scored
  row['score_differential'] = -row['score_differential'] + 3
  if row['posteam_type'] == 'away':
    row['posteam_type'] = 'home'
  else:
    row['posteam_type'] = 'away'
  if row['yardline_100'] < 50:
    row['field_position'] = 'OPP'
  elif row['yardline_100'] == 50:
    row['field_position'] = 'Midfield'
  else:
    row['field_position'] = 'OWN'
  return row

"""Function to simulate an unsuccessful FG kick. Assumptions are:
- Defending team now takes over possession at the line of the missed kicked (usually 7 yards before line of scrimmage). 
- Kickoff play is skipped in this scenario due to its exclusion from the model.
- 40 seconds are run off the clock by the time the next play begins."""
def transform_inputs_fg_failure(row):
  row = row.copy()
  # row['fourth_down_converted'] = False
  # row['fourth_down_failed'] = False
  row['distance_bin'] = 'long'
  # kicks are 7-8yds from LOS
  row['yardline_100'] = 100 - (row['yardline_100'] + 7)
  row['down'] = 1
  row['ydstogo'] = 10
  row['half_seconds_remaining'] = row['half_seconds_remaining'] - 40
  row['game_seconds_remaining'] = row['game_seconds_remaining'] - 40
  # Score differential flipped with the possession arrow
  row['score_differential'] = -row['score_differential']
  row['drive_first_downs'] = 0
  if row['posteam_type'] == 'away':
    row['posteam_type'] = 'home'
  else:
    row['posteam_type'] = 'away'
  if row['yardline_100'] < 50:
    row['field_position'] = 'OPP'
  elif row['yardline_100'] == 50:
    row['field_position'] = 'Midfield'
  else:
    row['field_position'] = 'OWN'
  return row

"""Function to simulate a punt. Assumptions are:
- Defending team now takes over where the average punt lands.
- Touchback will occur when a punt is near or at the opposing team's 40 yard line.
- Punters will add more power and less accuracy when they are very far from 
scoring position (80 yard line and further back)
- 40 seconds are run off the clock by the time the next play begins."""
def transform_inputs_punt(row):
  row = row.copy()
  # Could add in average punt yards
  # row['fourth_down_converted'] = False
  # row['fourth_down_failed'] = False
  row['distance_bin'] = 'long'
  if row['yardline_100'] <= 40:
    row['yardline_100'] = 20
  elif row['yardline_100'] >= 80:
    row['yardline_100'] = 100 - (row['yardline_100'] - (AVG_PUNT + 3))
  else:
    row['yardline_100'] = 100 - (row['yardline_100'] - AVG_PUNT)
  #add prob of touchback by distance later?
  row['down'] = 1
  row['ydstogo'] = 10
  row['half_seconds_remaining'] = row['half_seconds_remaining'] - 40
  row['game_seconds_remaining'] = row['game_seconds_remaining'] - 40
  row['drive_first_downs'] = 0
  # Score differential flipped with the possession arrow
  row['score_differential'] = -row['score_differential']
  if row['posteam_type'] == 'away':
    row['posteam_type'] = 'home'
  else:
    row['posteam_type'] = 'away'
  if row['yardline_100'] < 50:
    row['field_position'] = 'OPP'
  elif row['yardline_100'] == 50:
    row['field_position'] = 'Midfield'
  else:
    row['field_position'] = 'OWN'
  return row

"""Function to simulate the decisions. A aggression toggle has been added for situations on 4th & 1. 3% added to
the post-play win probability prediction to make 4th & 1 more likely to have a "Go For It" recommendation.
Assumptions for the the model simulation decision are:
- Desperation mode when a team is down by more than a FG with 2 minutes left in the 4th quarter,
or when a team is down almost 2 TDs with 5 minutes left in the game. Simulation will always default to going for it.
- If you are not in scoring position (past the 50 yard line), kicking a FG is not an option.
- If you are within 40 yards of the endzone, punting is not an option.
- If it is 4th & long and the team is not in desperation mode, going for it is not an option."""
def decision_time(row):
  row = row.copy()
  desperation = False
  go_wp = row['go_for_it_decision']
  fg_wp = row['kick_fg_decision']
  punt_wp = row['punt']

  # If possessing team is not past the 50, kicking fg not an option
  if row['field_position'] != 'OPP':
    fg_wp = -np.inf

  # Negate points for field goal farther than 55 yards out?

  # If it is the 4th quarter with less than 8min left, down by multiple scores
  if (row['qtr'] == 4) and (row['half_seconds_remaining'] <= 480) and (row['score_differential'] < -14):
    desperation = True

  # If it is the 4th quarter with less than 5min left, down by multiple scores
  if (row['qtr'] == 4) and (row['half_seconds_remaining'] <= 300) and (row['score_differential'] <= -14):
    desperation = True

  # If it is the 4th quarter with less than 2min left, down by more than a field goal
  if (row['qtr'] == 4) and (row['half_seconds_remaining'] <= 120) and (row['score_differential'] < -3)\
          and (row['posteam_timeouts_remaining'] < 3):
    desperation = True

  # If trailing in overtime, desperation is true
  if (row['game_half'] == 'Overtime') and (row['score_differential'] < 0):
    desperation = True

  if desperation:
    go_wp = 1
    fg_wp = -np.inf
    punt_wp = -np.inf

  #Compare aggressiveness, add a slider to the dashboard/model deployment
  if (row['is_fourth_and_one'] == 1) & (row['yardline_100'] == 1):
    go_wp += 0.03

  # If win probability super close for punt vs go for it deep on their own side of the field, lean punt
  if (go_wp - punt_wp < .01) & (row['yardline_100'] > 60):
    punt_wp += 0.01

  if (row['down'] == 4) and (row['ydstogo'] >= 6) and not desperation:
    go_wp = -np.inf

  if ((row['game_half'] == 'Overtime') and (row['yardline_100'] <= 35) and (row['is_fourth_and_one'] == 0) and
          (row['score_differential'] >= -3)):
    fg_wp = 1

  # If overtime and the team is on their own side of the field, more conservative
  if ((row['game_half'] == 'Overtime') and (row['score_differential'] == 0) and (row['field_position'] == 'OWN') and
          (row['is_fourth_and_one'] == 0)):
    go_wp = -np.inf

  if ((row['score_differential'] >= -3) and (row['yardline_100'] <= 40) and (row['half_seconds_remaining'] <= 60) and
    (row['qtr'] == 4)):
    fg_wp = 1

  if (row['qtr'] == 2) and (row['half_seconds_remaining'] <= 10) and (row['yardline_100'] <= 35) and (row['yardline_100'] >= 10):
    fg_wp = 1

  # If possessing team is within 40 yards of endzone, no punt
  if row['yardline_100'] <= 40:
    punt_wp = -np.inf

  wp_values = [go_wp, fg_wp, punt_wp]
  decision_options = ['Go For It', 'Kick FG', 'Punt']
  max_idx = np.argmax(wp_values)
  best_decision = decision_options[max_idx]
  return best_decision