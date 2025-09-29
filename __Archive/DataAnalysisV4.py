import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Lasso

class SeasonAnalyzer:
    def __init__(self, league_name="Bundesliga", country_name="Germany", season=2023):
        # Configuration
        self.LEAGUE_NAME = league_name
        self.COUNTRY_NAME = country_name
        self.SEASON = season
        self.SAVE_DIR = f"{country_name}_{league_name}/{season}"
        self.FIXTURES_FILE = "fixtures.json"
        self.EVENTS_FILE = "match_events.json"
        self.PLAYER_FILE = "players.json"
        # Data containers
        self.fixtures_data = None
        self.match_events = None
        self.players = None
        self.team_players = {}        # {team_id: set(player_ids)}
        self.player_id_to_name = {}   # {player_id: player_name}
        self.player_id_to_position = {}
        self.segment_df = None        # DataFrame of segments (with player presence)
        self.segment_df_grouped = None# DataFrame after grouping players
        self.match_summary_df = None
        self.team_summary_df = None
        self.coef_df_final = None
        # Goal and assist counts
        self.goals_by_player = {}     # {player_id: total_goals}
        self.assists_by_player = {}   # {player_id: total_assists}
        self.group_to_players = {}    # {group_id: [member_player_ids]}

    def load_data(self):
        """Load fixtures and match events data from JSON files."""
        fixtures_path = os.path.join(self.SAVE_DIR, self.FIXTURES_FILE)
        events_path   = os.path.join(self.SAVE_DIR, self.EVENTS_FILE)
        player_path   = os.path.join(self.SAVE_DIR, self.PLAYER_FILE)
        with open(fixtures_path, 'r') as f:
            self.fixtures_data = json.load(f)['response']
        with open(events_path, 'r') as f:
            self.match_events = json.load(f)
        with open(player_path, 'r') as f:
            self.players = json.load(f)

    def _get_minute(self, ev):
        """Compute the absolute minute of an event, including extra time."""
        extra = ev['time'].get('extra') or 0
        return ev['time']['elapsed'] + extra

    def identify_players(self):
        """Identify all players who played and map player IDs to names."""
        for fix in self.fixtures_data:
            fid = str(fix['fixture']['id'])
            league_round = fix['league'].get('round', '')
            if not league_round.startswith('Regular Season'):
                continue  # only consider regular season matches
            try:
                events = self.match_events[fid]
            except Exception as e:
                print(f"Error encountered with fid={fid}: {e}")    
            # events = self.match_events[fid]
            # Process starting lineups
            # Modify within identify_players()
            for lineup in events.get('lineups', []):
                team_id = lineup['team']['id']
                if team_id not in self.team_players:
                    self.team_players[team_id] = set()
                for player_entry in lineup['startXI']:
                    pid = player_entry['player']['id']
                    pname = player_entry['player']['name']
                    # Capture position information, checking both 'pos' and 'position'
                    ppos = player_entry['player'].get('pos', player_entry['player'].get('position', 'Unknown'))
                    self.team_players[team_id].add(pid)
                    self.player_id_to_name[pid] = pname
                    self.player_id_to_position[pid] = ppos
                    
            # Process substitutions (to include subbed-in and subbed-out players)
            for ev in events['events']:
                if ev['type'] == 'subst':
                    team_id = ev['team']['id']
                    # Ensure team_id key exists
                    if team_id not in self.team_players:
                        self.team_players[team_id] = set()
                    # Player coming in (stored in 'assist' field for substitutions)
                    pid_in = ev.get('assist', {}).get('id')
                    pname_in = ev.get('assist', {}).get('name')
                    # Player going out
                    pid_out = ev.get('player', {}).get('id')
                    pname_out = ev.get('player', {}).get('name')
                    if pid_in: 
                        self.team_players[team_id].add(pid_in)
                        self.player_id_to_name[pid_in] = pname_in
                    if pid_out:
                        self.team_players[team_id].add(pid_out)
                        self.player_id_to_name[pid_out] = pname_out

        # Initialize the segment DataFrame with a column for each player (prefix 'p')
        all_player_ids = sorted({pid for players in self.team_players.values() for pid in players})
        player_columns = [f'p{pid}' for pid in all_player_ids]
        self.segment_df = pd.DataFrame(columns=player_columns)
        # Metadata DataFrame for segment info
        meta_columns = ['fixture', 'home_team', 'away_team', 'home_id', 'away_id',
                        'start', 'end', 'duration', 'goal_diff']
        meta_df_list = []  # will collect dicts for faster creation

        # Build segments for each match
        for fix in self.fixtures_data:
            fid = str(fix['fixture']['id'])
            league_round = fix['league'].get('round', '')
            if not league_round.startswith('Regular Season'):
                continue
            events = self.match_events[fid]
            home_team = fix['teams']['home']['name']
            away_team = fix['teams']['away']['name']
            home_id = fix['teams']['home']['id']
            away_id = fix['teams']['away']['id']
            # Determine match end minute (use 90 if no events beyond 90)
            match_end_min = max([self._get_minute(ev) for ev in events['events']] + [90])
            # Current players on field (pid -> +1 if home, -1 if away)
            current_players = {}
            for lineup in events.get('lineups', []):
                team_sign = 1 if lineup['team']['id'] == home_id else -1
                for px in lineup['startXI']:
                    current_players[px['player']['id']] = team_sign
            # Get all substitution events sorted by minute
            subs_events = sorted([ev for ev in events['events'] if ev['type'] == 'subst'], key=self._get_minute)
            segment_start = 0
            # Segment end points: each substitution minute and the match end
            segment_end_points = sorted({self._get_minute(ev) for ev in subs_events} | {match_end_min})
            for segment_end in segment_end_points:
                duration = segment_end - segment_start
                # Calculate goals scored by each team in this segment
                home_goals = sum(1 for ev in events['events'] 
                                 if ev['type'] == 'Goal' and ev['team']['id'] == home_id 
                                 and segment_start < self._get_minute(ev) <= segment_end)
                away_goals = sum(1 for ev in events['events'] 
                                 if ev['type'] == 'Goal' and ev['team']['id'] == away_id 
                                 and segment_start < self._get_minute(ev) <= segment_end)
                goal_diff = home_goals - away_goals
                # Build a row for player presence in this segment
                segment_row = {f'p{pid}': 0 for pid in current_players.keys()}
                for pid, sign in current_players.items():
                    segment_row[f'p{pid}'] = sign
                # Append segment row and meta info
                self.segment_df = pd.concat([self.segment_df, pd.DataFrame([segment_row])], ignore_index=True)
                meta_df_list.append({
                    'fixture': fid, 'home_team': home_team, 'away_team': away_team,
                    'home_id': home_id, 'away_id': away_id,
                    'start': segment_start, 'end': segment_end,
                    'duration': duration, 'goal_diff': goal_diff
                })
                # Update current players for next segment (apply all subs at this segment end)
                for ev in (e for e in subs_events if self._get_minute(e) == segment_end):
                    pid_out = ev['player']['id']
                    pid_in  = ev.get('assist', {}).get('id')
                    team_sign = 1 if ev['team']['id'] == home_id else -1
                    if pid_out in current_players:
                        current_players.pop(pid_out, None)
                    if pid_in:
                        current_players[pid_in] = team_sign
                segment_start = segment_end

        # Combine meta info with segment data
        meta_df = pd.DataFrame(meta_df_list, columns=meta_columns)
        self.segment_df = pd.concat([meta_df, self.segment_df], axis=1)
        
        
        
    def calculate_game_metrics(self):
        self.player_game_metrics = {pid: {'games_started':0, 'full_games_played':0, 
                                          'games_subbed_on':0, 'games_subbed_off':0, 
                                          'minutes_played':0} for pid in self.player_id_to_name}
    
        for fix in self.fixtures_data:
            fid = str(fix['fixture']['id'])
            league_round = fix['league'].get('round', '')
            if not league_round.startswith('Regular Season'):
                continue
            events = self.match_events[fid]
            starters = set()
            subs_in = set()
            subs_out = set()
    
            match_end_min = max([self._get_minute(ev) for ev in events['events']] + [90])
    
            for lineup in events.get('lineups', []):
                for px in lineup['startXI']:
                    pid = px['player']['id']
                    starters.add(pid)
                    self.player_game_metrics[pid]['games_started'] += 1
    
            for ev in events['events']:
                if ev['type'] == 'subst':
                    pid_out = ev['player']['id']
                    pid_in = ev.get('assist', {}).get('id')
                    minute = self._get_minute(ev)
    
                    if pid_out:
                        subs_out.add(pid_out)
                        self.player_game_metrics[pid_out]['games_subbed_off'] += 1
                        self.player_game_metrics[pid_out]['minutes_played'] += minute
    
                    if pid_in:
                        subs_in.add(pid_in)
                        self.player_game_metrics[pid_in]['games_subbed_on'] += 1
                        self.player_game_metrics[pid_in]['minutes_played'] += (match_end_min - minute)
    
            # Players who started and weren't subbed out played full games
            for pid in starters - subs_out:
                self.player_game_metrics[pid]['full_games_played'] += 1
                self.player_game_metrics[pid]['minutes_played'] += match_end_min

    def summarize_matches_and_teams(self):
        """Create match summary and team summary DataFrames."""
        match_summary_list = []
        for fix in self.fixtures_data:
            fid = str(fix['fixture']['id'])
            league_round = fix['league'].get('round', '')
            if not league_round.startswith('Regular Season'):
                continue
            home_team = fix['teams']['home']['name']
            away_team = fix['teams']['away']['name']
            home_id   = fix['teams']['home']['id']
            away_id   = fix['teams']['away']['id']
            events = self.match_events[fid]['events']
            # Match day number (if available in round string)
            try:
                matchday = int(league_round.split('-')[-1].strip())
            except:
                matchday = None
            # Count unchanged starters (players who were never subbed out)
            start_players = {px['player']['id'] for lineup in self.match_events[fid].get('lineups', []) for px in lineup['startXI']}
            subbed_out_players = {ev['player']['id'] for ev in events if ev['type'] == 'subst'}
            unchanged_count = len(start_players - subbed_out_players)
            # Count substitution events
            subs_by_minute = {}
            for ev in events:
                if ev['type'] == 'subst':
                    minute = self._get_minute(ev)
                    subs_by_minute.setdefault((ev['team']['id'], minute), []).append(ev)
            total_subs = len(subs_by_minute)
            # Count goals by each team in the match
            home_goals = sum(1 for ev in events if ev['type'] == 'Goal' and ev['team']['id'] == home_id)
            away_goals = sum(1 for ev in events if ev['type'] == 'Goal' and ev['team']['id'] == away_id)
            match_summary_list.append({
                'matchday': matchday,
                'fixture_id': fid,
                'home_team': home_team, 'away_team': away_team,
                'home_team_id': home_id, 'away_team_id': away_id,
                'home_goals': home_goals, 'away_goals': away_goals,
                'sub_events': total_subs,
                'unchanged_players': unchanged_count
            })
        self.match_summary_df = pd.DataFrame(match_summary_list)

        # Build team summary from match summaries
        team_summary_list = []
        for team_id, players in self.team_players.items():
            # Filter matches where this team played home or away
            team_matches = self.match_summary_df[(self.match_summary_df['home_team_id'] == team_id) | 
                                                 (self.match_summary_df['away_team_id'] == team_id)]
            # Calculate results
            wins = ((team_matches['home_team_id'] == team_id) & (team_matches['home_goals'] > team_matches['away_goals'])).sum() \
                 + ((team_matches['away_team_id'] == team_id) & (team_matches['away_goals'] > team_matches['home_goals'])).sum()
            draws = (team_matches['home_goals'] == team_matches['away_goals']).sum()
            losses = len(team_matches) - wins - draws
            goals_scored = ((team_matches['home_team_id'] == team_id) * team_matches['home_goals'] + 
                            (team_matches['away_team_id'] == team_id) * team_matches['away_goals']).sum()
            goals_conceded = ((team_matches['home_team_id'] == team_id) * team_matches['away_goals'] + 
                              (team_matches['away_team_id'] == team_id) * team_matches['home_goals']).sum()
            # Count players who played every minute of the season (always on the pitch)
            total_minutes = self.segment_df['duration'].sum()
            always_playing = sum(
                self.segment_df[f'p{pid}'].abs().sum() == total_minutes 
                for pid in players
            )
            team_summary_list.append({
                'team_id': team_id,
                'wins': wins, 'draws': draws, 'losses': losses,
                'goals_scored': goals_scored, 'goals_conceded': goals_conceded,
                'total_players_used': len(players),
                'always_playing': always_playing
            })
        self.team_summary_df = pd.DataFrame(team_summary_list)
        # Add points and league position
        self.team_summary_df['points'] = self.team_summary_df['wins']*3 + self.team_summary_df['draws']
        self.team_summary_df.sort_values(by=['points','goals_scored'], ascending=[False, False], inplace=True)
        self.team_summary_df.reset_index(drop=True, inplace=True)
        self.team_summary_df['league_position'] = self.team_summary_df.index + 1

    def group_players(self):
        """Group players with identical segment patterns into a combined group."""
        player_cols = [col for col in self.segment_df.columns if col.startswith('p')]
        # Represent each player's presence across all segments as a tuple (pattern)
        patterns = self.segment_df[player_cols].T.apply(lambda row: tuple(row), axis=1)
        pattern_groups = patterns.groupby(patterns).groups  # groups of column labels with identical patterns
        player_to_group = {}
        group_id_counter = max([int(col[1:]) for col in player_cols]) + 1
        # Create group IDs for sets of players with identical patterns
        for pattern, cols in pattern_groups.items():
            if len(cols) > 1:
                new_gid = f'g{group_id_counter}'
                group_id_counter += 1
                for col in cols:
                    player_to_group[col] = new_gid
                # Map group ID to member player IDs (integers)
                member_ids = [int(col[1:]) for col in cols if col.startswith('p')]
                self.group_to_players[new_gid] = member_ids
        # Construct a new DataFrame with grouped columns
        self.segment_df_grouped = self.segment_df.copy()
        for old_col, new_col in player_to_group.items():
            # Sum columns that belong to the same group into the new group column
            if new_col not in self.segment_df_grouped:
                self.segment_df_grouped[new_col] = self.segment_df_grouped[old_col]
            else:
                self.segment_df_grouped[new_col] += self.segment_df_grouped[old_col]
            # Drop the individual player column
            self.segment_df_grouped.drop(columns=old_col, inplace=True)

    def compute_impacts(self):
        """Compute player impact using weighted regression (plus-minus) and prepare final results."""
        # Prepare design matrix X and target y for regression
        drop_cols = ['fixture','home_team','away_team','home_id','away_id','start','end','duration','goal_diff']
        X = self.segment_df_grouped.drop(columns=drop_cols)
        # Convert X to numeric and add intercept
        X_numeric = sm.add_constant(X.apply(pd.to_numeric, errors='coerce').fillna(0))
        y = pd.to_numeric(self.segment_df_grouped['goal_diff'], errors='coerce').fillna(0)
        weights = pd.to_numeric(self.segment_df_grouped['duration'], errors='coerce').fillna(1)
        # Fit weighted OLS regression
        model = sm.WLS(y, X_numeric, weights=weights).fit()
        # Extract and sort coefficients (excluding intercept)
        coef_series = model.params.drop('const').sort_values(ascending=False)
        coef_df = coef_series.reset_index()
        coef_df.columns = ['player_id', 'impact']
        # Map player and group IDs to names
        def get_player_name(pid):
            if pid.startswith('p'):
                return self.player_id_to_name.get(int(pid[1:]), "Unknown")
            else:
                return pid  # group id (e.g., 'g123')
        coef_df['player_name'] = coef_df['player_id'].apply(get_player_name)
        # Calculate minutes played by each player/group (sum of segment durations when present)
        segment_presence = X.abs()  # 1 or 0 (or -1) indicating presence; use abs to count both home/away as presence
        minutes_played = segment_presence.multiply(self.segment_df_grouped['duration'], axis=0).sum()
        coef_df['minutes_played'] = coef_df['player_id'].map(minutes_played.to_dict())
        # Determine team(s) each player/group played for
        def get_teams_for(pid):
            if pid.startswith('p'):
                pid_num = int(pid[1:])
                teams = set()
                # Player appears in home team with +1 or away team with -1 in segments
                if f'p{pid_num}' in self.segment_df_grouped:
                    mask = self.segment_df_grouped[f'p{pid_num}'] != 0
                    home_teams = self.segment_df_grouped.loc[mask & (self.segment_df_grouped[f'p{pid_num}'] == 1), 'home_team'].unique()
                    away_teams = self.segment_df_grouped.loc[mask & (self.segment_df_grouped[f'p{pid_num}'] == -1), 'away_team'].unique()
                    teams.update([t for t in home_teams if pd.notna(t)])
                    teams.update([t for t in away_teams if pd.notna(t)])
                teams = sorted(teams)
                if len(teams) == 1:
                    return teams[0]
                elif len(teams) > 1:
                    return ", ".join(teams)
                else:
                    return None
            else:
                # Group of players may span multiple teams (in case of transfers, etc.)
                # Here we return a generic label for simplicity
                return "Group of Players"
        coef_df['team(s)'] = coef_df['player_id'].apply(get_teams_for)
        # Add league position of team if a single team (otherwise None or "Multiple Teams")
        team_position_map = {row['team_id']: row['league_position'] for _, row in self.team_summary_df.iterrows()}
        coef_df['league_position'] = coef_df['team(s)'].apply(
            lambda t: team_position_map.get(t, "Multiple Teams" if t and "," in str(t) else None)
        )
        # Calculate standard errors for impacts
        impact_std = model.bse.drop('const', errors='ignore')
        coef_df['impact_std'] = coef_df['player_id'].map(impact_std.to_dict())
        # Fit a Lasso regression for comparison (not weighted, as an additional metric)
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_numeric, y)
        lasso_coefs = pd.Series(lasso.coef_, index=X_numeric.columns).drop('const')
        coef_df['lasso_impact'] = coef_df['player_id'].map(lasso_coefs.to_dict())
        coef_df['lasso_impact_std'] = lasso_coefs.std()  # simple std deviation of lasso coefficients as a placeholder
        # Finalize the DataFrame
        self.coef_df_final = coef_df[['player_id', 'player_name', 'team(s)', 'league_position',
                                      'minutes_played', 'impact', 'impact_std', 'lasso_impact', 'lasso_impact_std']]
        
        # within compute_impacts() at the end:
        def player_positions(pid):
            if pid.startswith('p'):
                pid_num = int(pid[1:])
                return self.player_id_to_position.get(pid_num, 'Unknown')
            elif pid.startswith('g'):
                members = self.group_to_players.get(pid, [])
                positions = {self.player_id_to_position.get(m, 'Unknown') for m in members}
                return "_".join(sorted(positions))
            return 'Unknown'
        
        def game_metric(pid, metric):
            if pid.startswith('p'):
                pid_num = int(pid[1:])
                return self.player_game_metrics.get(pid_num, {}).get(metric, 0)
            elif pid.startswith('g'):
                members = self.group_to_players.get(pid, [])
                return sum(self.player_game_metrics.get(m, {}).get(metric, 0) for m in members)
            return 0
        
        self.coef_df_final['position'] = self.coef_df_final['player_id'].apply(player_positions)
        self.coef_df_final['games_started'] = self.coef_df_final['player_id'].apply(lambda pid: game_metric(pid, 'games_started'))
        self.coef_df_final['full_games_played'] = self.coef_df_final['player_id'].apply(lambda pid: game_metric(pid, 'full_games_played'))
        self.coef_df_final['games_subbed_on'] = self.coef_df_final['player_id'].apply(lambda pid: game_metric(pid, 'games_subbed_on'))
        self.coef_df_final['games_subbed_off'] = self.coef_df_final['player_id'].apply(lambda pid: game_metric(pid, 'games_subbed_off'))
        self.coef_df_final['FTE_games_played'] = (self.coef_df_final['minutes_played'] / 90).round(2)


    def add_goals_assists(self):
        """Aggregate total goals and assists for each player across the season and add to results."""
        # Tally goals and assists from all match events
        self.goals_by_player.clear()
        self.assists_by_player.clear()
        for fix in self.fixtures_data:
            league_round = fix['league'].get('round', '')
            if not league_round.startswith('Regular Season'):
                continue
            fid = str(fix['fixture']['id'])
            for ev in self.match_events[fid]['events']:
                if ev.get('type') == 'Goal':
                    detail = ev.get('detail', '')
                    # Exclude own goals from player's goal count:contentReference[oaicite:3]{index=3}
                    if isinstance(detail, str) and 'Own Goal' in detail:
                        continue
                    scorer_id = ev.get('player', {}).get('id')
                    assist_id = ev.get('assist', {}).get('id')
                    if scorer_id:
                        self.goals_by_player[scorer_id] = self.goals_by_player.get(scorer_id, 0) + 1
                    if assist_id:
                        self.assists_by_player[assist_id] = self.assists_by_player.get(assist_id, 0) + 1
        # Add goals and assists columns to the final DataFrame (including groups)
        def total_goals(pid):
            if pid.startswith('p'):
                pid_num = int(pid[1:])
                return self.goals_by_player.get(pid_num, 0)
            elif pid.startswith('g'):
                # Sum goals of all players in this group
                members = self.group_to_players.get(pid, [])
                return sum(self.goals_by_player.get(m, 0) for m in members)
            return 0
        def total_assists(pid):
            if pid.startswith('p'):
                pid_num = int(pid[1:])
                return self.assists_by_player.get(pid_num, 0)
            elif pid.startswith('g'):
                members = self.group_to_players.get(pid, [])
                return sum(self.assists_by_player.get(m, 0) for m in members)
            return 0
        self.coef_df_final['goals'] = self.coef_df_final['player_id'].apply(total_goals)
        self.coef_df_final['assists'] = self.coef_df_final['player_id'].apply(total_assists)

    def run_analysis(self):
        """Execute all analysis steps in order."""
        self.identify_players()
        self.summarize_matches_and_teams()
        self.group_players()
        self.compute_impacts()
        self.add_goals_assists()

    def save_results(self, output_path=None):
        """Save the final results to an Excel file with OLS and Lasso sheets."""
        if output_path is None:
            output_path = os.path.join(self.SAVE_DIR, "output.xlsx")
        # Sort by OLS impact for first sheet, and by Lasso impact for second sheet
        ols_df = self.coef_df_final.sort_values(by='impact', ascending=False)
        lasso_df = self.coef_df_final.sort_values(by='lasso_impact', ascending=False)
        ols_df.to_excel(output_path, sheet_name="OLS", index=False)
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            lasso_df.to_excel(writer, sheet_name="Lasso", index=False)
            
    
    def run(self):
        analyzer.load_data()                  # Loads fixtures and events
        analyzer.identify_players()           # Identifies players and sets up segments
        analyzer.calculate_game_metrics()
        analyzer.summarize_matches_and_teams()# Summarizes matches and teams
        analyzer.group_players()              # Groups players with identical patterns
        analyzer.compute_impacts()            # Computes player impact metrics
        analyzer.add_goals_assists()          # Explicitly calculates goals and assists
        analyzer.save_results()  # Optional, saves output to Excel
            
             



# Example usage
if __name__ == '__main__':
     

    analyzer=SeasonAnalyzer(league_name="Bundesliga", country_name="Germany", season=2022)
    # analyzer=SeasonAnalyzer(league_name="Premier League", country_name="England", season=2023)
    # analyzer=SeasonAnalyzer(league_name="La Liga", country_name="Spain", season=2023)
    
    analyzer.run()               
    # analyzer = SeasonAnalyzer()  # initialize class
    # analyzer = SeasonAnalyzer()
    # analyzer = SeasonAnalyzer()
    # analyzer.load_data()                  # Loads fixtures and events
    # analyzer.identify_players()           # Identifies players and sets up segments
    # analyzer.calculate_game_metrics()
    # analyzer.summarize_matches_and_teams()# Summarizes matches and teams
    # analyzer.group_players()              # Groups players with identical patterns
    # analyzer.compute_impacts()            # Computes player impact metrics
    # analyzer.add_goals_assists()          # Explicitly calculates goals and assists
    # analyzer.save_results()  # Optional, saves output to Excel

    # analyzer.load_data()         # required: loads fixture and match event data
    # analyzer.run_analysis()    
    #results_df = analyzer.display_results()    