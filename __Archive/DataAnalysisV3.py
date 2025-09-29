#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 08:00:02 2025

@author: alexander
"""

#%%
# 📋 Configuration Variables
LEAGUE_NAME   = "Bundesliga"
COUNTRY_NAME  = "Germany"
SEASON        = 2023
VERBOSE_OUTPUT = True  # Set False to suppress segment printing

# File paths (adjust if needed)
SAVE_DIR      = f"{LEAGUE_NAME}_{SEASON}"
SAVE_DIR         = f"{COUNTRY_NAME}_{LEAGUE_NAME}_{SEASON}"
FIXTURES_FILE = "fixtures.json"
EVENTS_FILE   = "match_events.json"

#%%
# 🛠 Imports & Setup
import os
import json
import pandas as pd

# Load JSON helper
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# Compute absolute minute including extra
def get_minute(ev):
    extra = ev['time'].get('extra') or 0
    return ev['time']['elapsed'] + extra

# Paths
fixtures_path = os.path.join(SAVE_DIR, FIXTURES_FILE)
events_path   = os.path.join(SAVE_DIR, EVENTS_FILE)

fixtures_data = load_json(fixtures_path)['response']
match_events  = load_json(events_path)

#%%
# 🏗 Identify all used players by team
team_players = {}
for fix in fixtures_data:
    fid = str(fix['fixture']['id'])
    league_round = fix['league'].get('round', '')

    if not league_round.startswith('Regular Season'):
        continue

    events = match_events[fid]
    lineups = events.get('lineups', [])
    if len(lineups) < 2:
        continue

    for lineup in lineups:
        team_id = lineup['team']['id']
        if team_id not in team_players:
            team_players[team_id] = set()
        for player in lineup['startXI']:
            team_players[team_id].add(player['player']['id'])
    
    for ev in events['events']:
        if ev['type'] == 'subst':
            team_id = ev['team']['id']
            team_players[team_id].add(ev['player']['id'])
            if ev.get('assist', {}).get('id'):
                team_players[team_id].add(ev['assist']['id'])

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
    if len(lineups) < 2:
        continue

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
        meta_row = {'fixture': fid, 'start': segment_start, 'end': segment_end, 'duration': duration, 'goal_diff': goal_diff}
        meta_df = pd.concat([meta_df, pd.DataFrame([meta_row])], ignore_index=True)

        # Verbose output
        if VERBOSE_OUTPUT:
            unchanged_players = sum(1 for pid in current_players)
            subs_this_segment = [ev for ev in subs_events if get_minute(ev)==segment_end]
            num_subs = len(subs_this_segment)
            print(f"Fixture {fid}: {unchanged_players} unchanged players, {num_subs} subs, goal diff {goal_diff}, duration {duration}")

        # Update lineup for next segment
        # Update lineup for next segment
        for ev in subs_events:
            if get_minute(ev) == segment_end:
                pid_out = ev.get('assist', {}).get('id')
                pid_in = ev['player']['id']
                team_sign = 1 if ev['team']['id'] == home_id else -1
        
                if pid_out and pid_out in current_players:
                    del current_players[pid_out]  # Properly remove player substituted out
                current_players[pid_in] = team_sign  # Add player substituted in


        segment_start = segment_end

# Combine metadata with segments
segment_df = pd.concat([meta_df, segment_df], axis=1)

# Display the first few segments
print(segment_df.head())
