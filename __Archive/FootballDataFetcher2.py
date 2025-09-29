#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 08:00:02 2025

@author: alexander
"""

#%%minutes_played
# 📋 Configuration Variables
LEAGUE_NAME   = "Premier League"
COUNTRY_NAME  = "England"
# LEAGUE_NAME   = "La Liga"
# COUNTRY_NAME  = "Spain"
# LEAGUE_NAME   = "Bundesliga"
# COUNTRY_NAME  = "Germany"
SEASON        = 2023

# File paths (adjust if needed)         
SAVE_DIR         = f"{COUNTRY_NAME}_{LEAGUE_NAME}/{SEASON}"
FIXTURES_FILE = "fixtures.json"
EVENTS_FILE   = "match_events.json"

#%%
# 🛠 Imports & Setup
import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load JSON helper
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# Compute absolute minute including extra
def get_minute(ev):
    extra = ev['time'].get('extra') or 0
    return ev['time']['elapsed'] + extra

# Assign names
def get_name(pid):
    if pid.startswith('p'):
        return player_id_to_name[int(pid[1:])]
    else:
        return pid  # group id remains as is

# Paths
fixtures_path = os.path.join(SAVE_DIR, FIXTURES_FILE)
events_path   = os.path.join(SAVE_DIR, EVENTS_FILE)

fixtures_data = load_json(fixtures_path)['response']
match_events  = load_json(events_path)

#%%
# 🏗 Identify all used players by team and create player ID to name mapping
team_players = {}
player_id_to_name = {}  # New dictionary to map player ID to player name

for fix in fixtures_data:
    fid = str(fix['fixture']['id'])
    league_round = fix['league'].get('round', '')
    if not league_round.startswith('Regular Season'):
        continue
    events = match_events[fid]
    lineups = events.get('lineups', [])
    for lineup in lineups:
        team_id = lineup['team']['id']
        if team_id not in team_players:
            team_players[team_id] = set()
        for player in lineup['startXI']:
            pid = player['player']['id']
            pname = player['player']['name']
            team_players[team_id].add(pid)
            player_id_to_name[pid] = pname  # Map ID to name

    for ev in events['events']:
        if ev['type'] == 'subst':
            team_id = ev['team']['id']

            pid_in = ev['player']['id']
            pname_in = ev['player']['name']
            team_players[team_id].add(pid_in)
            player_id_to_name[pid_in] = pname_in

            pid_out = ev.get('assist', {}).get('id')
            pname_out = ev.get('assist', {}).get('name')
            if pid_out:
                team_players[team_id].add(pid_out)
                player_id_to_name[pid_out] = pname_out

# Create pandas DataFrame with player columns
all_player_ids = sorted({pid for players in team_players.values() for pid in players})
segment_df = pd.DataFrame(columns=[f'p{pid}' for pid in all_player_ids])
meta_df = pd.DataFrame(columns=['fixture', 'start', 'end', 'duration', 'goal_diff'])

#%%
# 🏟 Build segments for each match
for fix in fixtures_data:
    fid = str(fix['fixture']['id'])
    league_round = fix['league'].get('round', '')

    if not league_round.startswith('Regular Season'):
        continue

    events = match_events[fid]
    lineups = events.get('lineups', [])

    home_team = fix['teams']['home']['name']
    away_team = fix['teams']['away']['name']
    home_id = fix['teams']['home']['id']
    away_id = fix['teams']['away']['id']

    # Total match duration
    match_end_minute = max([get_minute(ev) for ev in events['events']] + [90])

    # Initial lineup
    current_players = {}
    for lineup in lineups:
        sign = 1 if lineup['team']['id'] == home_id else -1
        for px in lineup['startXI']:
            current_players[px['player']['id']] = sign

    subs_events = sorted([ev for ev in events['events'] if ev['type']=='subst'], key=get_minute)

    # Build segments
    segment_start = 0
    for sub_minute in sorted(set([get_minute(ev) for ev in subs_events] + [match_end_minute])):
        segment_end = sub_minute
        duration = segment_end - segment_start

        # Goals in segment
        home_goals = sum(1 for ev in events['events'] if ev['type']=='Goal' and home_id==ev['team']['id'] and segment_start < get_minute(ev) <= segment_end)
        away_goals = sum(1 for ev in events['events'] if ev['type']=='Goal' and away_id==ev['team']['id'] and segment_start < get_minute(ev) <= segment_end)
        goal_diff = home_goals - away_goals

        # Segment players row
        row = {f'p{pid}': 0 for pid in all_player_ids}
        for pid, sign in current_players.items():
            row[f'p{pid}'] = sign

        segment_df = pd.concat([segment_df, pd.DataFrame([row])], ignore_index=True)
        meta_row = {'fixture': fid, 'home_team': home_team, 'away_team': away_team, 'home_id': home_id, 'away_id': away_id, 'start': segment_start, 'end': segment_end, 'duration': duration, 'goal_diff': goal_diff}
        meta_df = pd.concat([meta_df, pd.DataFrame([meta_row])], ignore_index=True)

        # Update lineup for next segment
        for ev in subs_events:
            if get_minute(ev) == segment_end:
                pid_in = ev.get('assist', {}).get('id')
                pid_out = ev['player']['id']
                team_sign = 1 if ev['team']['id'] == home_id else -1
        
                del current_players[pid_out]  # Properly remove player substituted out
                current_players[pid_in] = team_sign  # Add player substituted in

        segment_start = segment_end

# Combine metadata with segments
segment_df = pd.concat([meta_df, segment_df], axis=1)

# Display the first few segments
print(segment_df.head())

# sanity checks
segment_op_df = segment_df.drop(columns=['fixture', 'home_team','away_team','home_id','away_id','start', 'end', 'duration', 'goal_diff'])

print((segment_op_df == 1).sum(axis=1).value_counts().sort_index())
print((segment_op_df == -1).sum(axis=1).value_counts().sort_index())



#%%     match summaries
# Collect match-level details
match_summary = []

for fix in fixtures_data:
    fid = str(fix['fixture']['id'])
    league_round = fix['league'].get('round', '')

    if not league_round.startswith('Regular Season'):
        continue

    home_team = fix['teams']['home']['name']
    away_team = fix['teams']['away']['name']
    home_team_id = fix['teams']['home']['id']
    away_team_id = fix['teams']['away']['id']

    events = match_events[fid]
    lineups = events.get('lineups', [])

    home_id = fix['teams']['home']['id']
    away_id = fix['teams']['away']['id']

    # Campaign matchday
    matchday = int(league_round.replace('Regular Season - ', ''))


    # Extract starting lineup player IDs for each team
    players_start = set(px['player']['id'] for lineup in lineups for px in lineup['startXI'])

    #players_subbed_out = set(ev['assist']['id'] for ev in events['events'] if ev['type'] == 'subst' and ev.get('assist', {}).get('id'))
    players_subbed_out = set(ev['player']['id'] for ev in events['events'] if ev['type'] == 'subst' and ev.get('player', {}).get('id'))

    unchanged_players = len(players_start - players_subbed_out)

    subs_by_minute = {}
    for ev in events['events']:
        if ev['type'] == 'subst':
            minute = get_minute(ev)
            team = ev['team']['id']
            subs_by_minute.setdefault((team, minute), []).append(ev)

    total_sub_events = len(subs_by_minute)

    home_goals = sum(ev['team']['id'] == home_id for ev in events['events'] if ev['type'] == 'Goal')
    away_goals = sum(ev['team']['id'] == away_id for ev in events['events'] if ev['type'] == 'Goal')

    match_summary.append({
        'matchday': matchday,
        'fixture_id': fid,
        'home_team': home_team,
        'away_team': away_team,
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'unchanged_players': unchanged_players,
        'sub_events': total_sub_events,
        'home_goals': home_goals,
        'away_goals': away_goals
    })

match_summary_df = pd.DataFrame(match_summary)


#%%     team summaries
# Team-level summary
team_summary = []

for team_id, players in team_players.items():
    team_matches = match_summary_df[(match_summary_df['home_team'] == team_id) | (match_summary_df['away_team'] == team_id)]

    wins = ((team_matches['home_team'] == team_id) & (team_matches['home_goals'] > team_matches['away_goals'])).sum() + \
           ((team_matches['away_team'] == team_id) & (team_matches['away_goals'] > team_matches['home_goals'])).sum()

    draws = (team_matches['home_goals'] == team_matches['away_goals']).sum()

    losses = len(team_matches) - wins - draws

    goals_scored = ((team_matches['home_team'] == team_id) * team_matches['home_goals'] + \
                    (team_matches['away_team'] == team_id) * team_matches['away_goals']).sum()

    goals_conceded = ((team_matches['home_team'] == team_id) * team_matches['away_goals'] + \
                      (team_matches['away_team'] == team_id) * team_matches['home_goals']).sum()

    # always_playing = sum(all(segment_df[f'p{pid}'].abs().sum() == segment_df['duration'].sum() for pid in players))
    always_playing = sum(
    segment_df[f'p{pid}'].abs().sum() == segment_df['duration'].sum()
    for pid in players
        )

    team_summary.append({
        'team_id': team_id,
        'always_playing': always_playing,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'total_players_used': len(players)
    })

team_summary_df = pd.DataFrame(team_summary)


#%%     player grouping
# Player segments grouping
player_cols = [col for col in segment_df if col.startswith('p')]
patterns = segment_df[player_cols].T.apply(lambda row: tuple(row), axis=1)
pattern_groups = patterns.groupby(patterns).groups

group_id_counter = max([int(col[1:]) for col in player_cols]) + 1
player_to_group = {}

for pattern, group in pattern_groups.items():
    if len(group) > 1:
        new_id = f'g{group_id_counter}'
        group_id_counter += 1
        for player_col in group:
            player_to_group[player_col] = new_id

# Update segment_df
segment_df_grouped = segment_df.copy()
for old_id, new_id in player_to_group.items():
    if new_id not in segment_df_grouped:
        segment_df_grouped[new_id] = segment_df_grouped[old_id]
    else:
        segment_df_grouped[new_id] += segment_df_grouped[old_id]
    segment_df_grouped.drop(old_id, axis=1, inplace=True)

# Final outputs
print(match_summary_df.head())
print(team_summary_df.head())
print(segment_df_grouped.head())


segment_op_df_grouped = segment_df_grouped.drop(columns=['fixture', 'home_team','away_team','home_id','away_id','start', 'end', 'duration', 'goal_diff'])





# Convert all X columns to numeric (forcing numeric, errors coerced to NaN)
X_numeric = segment_op_df_grouped.apply(pd.to_numeric, errors='coerce').fillna(0)

# Dependent variable
y_numeric = pd.to_numeric(segment_df_grouped['goal_diff'], errors='coerce').fillna(0)

# Weights (ensure numeric)
weights_numeric = pd.to_numeric(segment_df_grouped['duration'], errors='coerce').fillna(1)

# Add constant
X_numeric = sm.add_constant(X_numeric)

# Run weighted regression
model = sm.WLS(y_numeric, X_numeric, weights=weights_numeric).fit()

# Extract and rank coefficients, excluding intercept
coef = model.params.drop('const').sort_values(ascending=False)

# Prepare DataFrame
coef_df = coef.reset_index()
coef_df.columns = ['player_id', 'impact']


coef_df['player_name'] = coef_df['player_id'].apply(get_name)

print(coef_df.head(20))  # Shows top 20 impactful players/groups


# Calculate minutes played per player/group
minutes_played = segment_op_df_grouped.abs().multiply(segment_df_grouped['duration'], axis=0).sum()

# Assign player names and teams
def get_player_teams(pid):
    if pid.startswith('p'):
        pid_num = int(pid[1:])
        teams_played_for = set()
        player_mask = segment_df_grouped[f'p{pid_num}'] != 0
        teams_played_for.update(segment_df_grouped.loc[player_mask, 'home_team'][segment_df_grouped.loc[player_mask, f'p{pid_num}'] == 1].unique())
        teams_played_for.update(segment_df_grouped.loc[player_mask, 'away_team'][segment_df_grouped.loc[player_mask, f'p{pid_num}'] == -1].unique())
        teams_played_for = sorted(teams_played_for)
        if len(teams_played_for) == 1:
            return teams_played_for[0]
        elif len(teams_played_for) == 2:
            return f"{teams_played_for[0]}, {teams_played_for[1]}"
        else:
            return "Multiple Teams"
    else:
        return "Group of Players"

coef_df['minutes_played'] = coef_df['player_id'].map(minutes_played)
coef_df['team(s)'] = coef_df['player_id'].apply(get_player_teams)

# Final table
coef_df_final = coef_df[['player_id', 'player_name', 'team(s)', 'minutes_played', 'impact']]

print(coef_df_final.head(20))



# Extract standard deviation of coefficients
coef_std = model.bse.drop('const')

# Add standard deviation to coef_df
coef_df['impact_std'] = coef_df['player_id'].map(coef_std)

# Add league positions to the team_summary_df for reference
team_summary_df['points'] = team_summary_df['wins']*3 + team_summary_df['draws']
team_summary_df.sort_values(by=['points', 'goals_scored'], ascending=[False, False], inplace=True)
team_summary_df.reset_index(drop=True, inplace=True)
team_summary_df['league_position'] = team_summary_df.index + 1

# Map team names to league positions
team_name_to_position = {}
for _, row in team_summary_df.iterrows():
    team_name_to_position[row['team_id']] = row['league_position']

# Include league position in coef_df
coef_df['league_position'] = coef_df['team(s)'].map(lambda x: team_name_to_position.get(x, "Multiple Teams" if "," in x else None))

# Add Lasso regression estimates
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_numeric, y_numeric)

# Lasso coefficients
lasso_coef = pd.Series(lasso_model.coef_, index=X_numeric.columns).drop('const')

# Approximate standard deviation of Lasso coefficients (bootstrapping not done here; simple approximation)
lasso_std = np.std(lasso_coef)

# Merge Lasso results
coef_df['lasso_impact'] = coef_df['player_id'].map(lasso_coef)
coef_df['lasso_impact_std'] = lasso_std

# Update final dataframe sorted by impact
coef_df_final = coef_df[['player_id', 'player_name', 'team(s)', 'league_position', 'minutes_played', 'impact', 'impact_std', 'lasso_impact', 'lasso_impact_std']]

coef_df_final.sort_values(by='impact', ascending=False, inplace=True)
coef_df_final.to_excel(SAVE_DIR+"/output.xlsx", sheet_name="OLS", index=False)
with pd.ExcelWriter(SAVE_DIR+"/output.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    coef_df_final.sort_values(by='lasso_impact', ascending=False, inplace=True)
    coef_df_final.to_excel(writer, sheet_name="Lasso", index=False)

print(coef_df_final.head(20))


