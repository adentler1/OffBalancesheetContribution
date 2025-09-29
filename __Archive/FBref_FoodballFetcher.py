#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 16:14:25 2025

@author: alexander
"""

import os, json, time, math, random
import pandas as pd
import soccerdata as sd

class FootballDataFetcher:
    def __init__(self, api_key: str = None, league_name: str = "", country_name: str = "", season_year: int = None,
                 label: str = "fbref", rate_limit_sec: float = 10.0, batch_size: int = 15, debug: bool = False):
        """
        Fetch football data from FBref for a given league and season, and save to analyzer-compatible JSON files.
        
        Parameters:
            api_key (str, optional): API key if using an API-based source (not needed for FBref scraping, kept for interface consistency).
            league_name (str): Name of the league (e.g. "Bundesliga").
            country_name (str): Name of the country (e.g. "Germany").
            season_year (int): Season end year (e.g. 2025 for 2024/25 season).
            label (str): Prefix label for output directory (default "fbref", outputs to "fbref_data/").
            rate_limit_sec (float): Base seconds to sleep between request batches (to avoid overloading the source).
            batch_size (int): Number of matches to fetch per batch request.
            debug (bool): If True, prints debug information during fetching.
        """
        self.league_name = league_name
        self.country_name = country_name
        self.season_year = int(season_year) if season_year is not None else None
        self.api_key = api_key  # Not used for FBref, but kept for compatibility
        self.debug = debug

        # Determine output directory based on label and league
        # e.g., "fbref_data/Germany_Bundesliga/2025"
        self.label = label or ""  # allow empty string to omit label prefix
        if self.label:
            self.out_root = f"{self.label}_data"
        else:
            self.out_root = "."
        self.out_dir = os.path.join(self.out_root, f"{self.country_name}_{self.league_name}", str(self.season_year))
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Prepare file paths for outputs
        self.fixtures_fbref_path = os.path.join(self.out_dir, "fixtures_fbref.json")
        self.events_fbref_path   = os.path.join(self.out_dir, "match_events_fbref.json")
        self.players_fbref_path  = os.path.join(self.out_dir, "players_fbref.json")
        # Analyzer-compatible file paths
        self.fixtures_path = os.path.join(self.out_dir, "fixtures.json")
        self.events_path   = os.path.join(self.out_dir, "match_events.json")
        self.players_path  = os.path.join(self.out_dir, "players.json")
        
        # Internal containers for collected data
        self.match_events = {}  # {match_id_str: { "lineups": [...], "events": [...] }}
        self.players_map = {}   # {team_name: [ {id, name, age, position}, ... ]}

        # Rate limiting settings
        self.batch_size = batch_size
        self.sleep_base = rate_limit_sec
        

    @staticmethod
    def _stable_int_id(s: str) -> int:
        """Generate a stable pseudo-integer ID for a given string (used if no official ID is available)."""
        return abs(hash(str(s))) % (10**9)  # 9-digit max ID (to avoid extremely large numbers)
    
    @staticmethod
    def _coerce_int(x):
        """Try to convert a value to int, return None if it cannot be converted."""
        try:
            return int(x)
        except Exception:
            return None
    
    @staticmethod
    def _map_event_type(event_type: str) -> str:
        """
        Simplify event type to the minimal set of interest:
        - Contains "sub" -> "subst" (substitution event)
        - Contains "goal" -> "Goal" (goal event)
        - Otherwise, return the original string (cards/fouls will be returned as-is, though not used in analysis).
        """
        if not event_type:
            return ""
        t = str(event_type).lower()
        if "sub" in t:
            return "subst"
        if "goal" in t:
            return "Goal"
        return event_type
    
    def _fixtures_to_api_shape(self, schedule_df: pd.DataFrame) -> dict:
        """
        Convert FBref schedule DataFrame to a fixtures payload in API-like format.
        Each match gets a fixture ID and team info. Use official match_id if available, otherwise a stable hash ID.
        """
        fixtures_list = []
        for _, row in schedule_df.iterrows():
            # Determine fixture ID
            mid = self._coerce_int(row.get("match_id"))
            if mid is None:
                # If FBref doesn't provide a numeric match_id, create a hash-based ID from teams and date
                mid = self._stable_int_id(f"{row.get('home_team')} vs {row.get('away_team')} @ {row.get('date')}")
            mid = int(mid)
            # League round info (could be like "Regular Season - 1", etc.)
            round_str = str(row.get("round") or "")
            # Team names
            home_team = str(row.get("home_team") or "")
            away_team = str(row.get("away_team") or "")
            fixture_entry = {
                "fixture": {"id": mid},
                "league": {"round": round_str},
                "teams": {
                    "home": {"id": self._stable_int_id(home_team), "name": home_team},
                    "away": {"id": self._stable_int_id(away_team), "name": away_team}
                }
            }
            fixtures_list.append(fixture_entry)
        return {"response": fixtures_list}
    
    def _lineups_to_api_shape(self, lineups_df: pd.DataFrame) -> dict:
        """
        Build a dictionary of match lineups from the FBref lineup DataFrame.
        Returns a dict: { match_id_str: { "lineups": [ ... ], "events": [] } }.
        Each lineup entry contains a team with its starting XI (players with id, name, position).
        """
        lineup_data = {}
        if not isinstance(lineups_df, pd.DataFrame) or lineups_df.empty:
            return lineup_data
        
        # Normalize column names to lowercase keys for consistency
        cols = {c.lower(): c for c in lineups_df.columns}
        col_match    = cols.get("match_id", "match_id")
        col_team     = cols.get("team", "team")
        col_player   = cols.get("player", "player")
        col_position = cols.get("position", "position")
        col_started  = cols.get("started", "started")
        col_player_id = cols.get("player_id", "player_id")
        
        # Group by match_id to aggregate lineups per match
        for mid, df_match in lineups_df.groupby(col_match):
            # Use the numeric match ID if possible, otherwise a stable hash
            match_key = str(int(self._coerce_int(mid) or self._stable_int_id(mid)))
            match_lineups = []
            # Group by team within the match (two teams per match typically)
            for team_name, df_team in df_match.groupby(col_team):
                team_name = str(team_name or "")
                team_id   = self._stable_int_id(team_name)
                # Filter to starting players (if 'started' column exists and is boolean)
                starters_df = df_team
                if col_started in df_team.columns:
                    starters_df = df_team[df_team[col_started] == True]
                # Build the startXI list for this team
                start_xi = []
                for _, player_row in starters_df.iterrows():
                    raw_player_id = player_row.get(col_player_id)
                    pid = self._coerce_int(raw_player_id)
                    if pid is None:
                        # If no numeric player ID, generate one from name
                        pid = self._stable_int_id(player_row.get(col_player))
                    player_name = player_row.get(col_player)
                    position = player_row.get(col_position) or player_row.get("pos") or None
                    start_xi.append({
                        "player": {
                            "id": int(pid),
                            "name": player_name,
                            "position": position
                        }
                    })
                    # Update players_map (team roster) with this player if not already seen
                    self.players_map.setdefault(team_name, [])
                    # Use a set of (id, name) to ensure uniqueness in roster
                    current_roster = {(p["id"], p["name"]) for p in self.players_map[team_name]}
                    if (int(pid), player_name) not in current_roster:
                        self.players_map[team_name].append({
                            "id": int(pid), "name": player_name, 
                            "age": None, "position": position
                        })
                # Append this team's lineup info
                match_lineups.append({
                    "team": {"id": team_id, "name": team_name},
                    "startXI": start_xi
                })
            # Initialize match entry with lineups; events will be added later
            lineup_data[match_key] = {"lineups": match_lineups, "events": []}
        return lineup_data
    
    def _append_events_to_matches(self, events_df: pd.DataFrame):
        """
        Append event records from the events DataFrame into the self.match_events structure.
        For each match, add a list of event dicts under "events".
        Also update players_map for any substitute players not already in lineups.
        """
        if not isinstance(events_df, pd.DataFrame) or events_df.empty:
            return
        cols = {c.lower(): c for c in events_df.columns}
        col_match  = cols.get("match_id", "match_id")
        col_team   = cols.get("team", "team")
        col_player = cols.get("player", "player")
        col_assist = cols.get("assist", "assist")
        col_min    = cols.get("minute", "minute")
        col_stoppage = cols.get("minute_stoppage_time", "minute_stoppage_time")
        col_type1  = cols.get("event_type", "event_type")
        col_type2  = cols.get("event", "event")  # some versions might use 'event' instead of 'event_type'
        col_desc   = cols.get("event_description", "event_description")
        
        for mid, df_match in events_df.groupby(col_match):
            match_key = str(int(self._coerce_int(mid) or self._stable_int_id(mid)))
            # Ensure the match entry exists in self.match_events (even if no lineup, create placeholder)
            self.match_events.setdefault(match_key, {"lineups": [], "events": []})
            event_list = []
            for _, ev in df_match.iterrows():
                # Determine the unified event type
                etype = ev.get(col_type1)
                if pd.isna(etype) or etype == "" or etype is None:
                    etype = ev.get(col_type2)
                etype = self._map_event_type(etype)
                # Only "Goal" and "subst" types will ultimately be used by analysis, but we will include all events.
                minute = int(self._coerce_int(ev.get(col_min)) or 0)
                extra_min = int(self._coerce_int(ev.get(col_stoppage)) or 0)
                team_name = ev.get(col_team)
                player_name = ev.get(col_player)
                assist_name = ev.get(col_assist)
                # Use stable IDs for team and player (assist is the second player in a substitution or the assister for a goal)
                team_id = self._stable_int_id(team_name)
                player_id = self._stable_int_id(player_name)
                assist_id = self._stable_int_id(assist_name)
                # Build the event record
                event_record = {
                    "time": {"elapsed": minute, "extra": extra_min},
                    "type": etype,
                    "team": {"id": team_id, "name": team_name},
                    "player": {"id": player_id, "name": player_name},
                    "assist": {"id": assist_id, "name": assist_name},
                    "detail": ev.get(col_desc)
                }
                event_list.append(event_record)
                # If this is a substitution event, update players_map with sub-in player (assist) if not already in roster
                if etype == "subst":
                    if assist_name:  # assist field holds the player coming in for substitutions
                        self.players_map.setdefault(team_name, [])
                        roster_set = {(p["id"], p["name"]) for p in self.players_map[team_name]}
                        if (assist_id, assist_name) not in roster_set:
                            self.players_map[team_name].append({
                                "id": assist_id, "name": assist_name, 
                                "age": None, "position": None
                            })
                    # (Optionally, could also ensure the player who went off is in the roster, but if they started they should already be.)
            # Assign the collected events list to the match entry
            self.match_events[match_key]["events"] = event_list
    
    def _sleep_polite(self):
        """Sleep for a random duration between rate_limit_sec and 2*rate_limit_sec (to vary timing)."""
        low = self.sleep_base
        high = self.sleep_base * 2
        time.sleep(random.uniform(low, high))
    
    def run_fetch(self):
        """Execute the data fetch from FBref and save JSON files. Returns a summary report dict."""
        if self.season_year is None:
            raise ValueError("Season year must be specified.")
        
        # Initialize the soccerdata FBref scraper for the specified league and season
        league_key = f"{self.country_name}-{self.league_name}"
        fb = sd.FBref(leagues=[league_key], seasons=self.season_year)
        
        # 1. Retrieve the season schedule (all matches)
        try:
            schedule_df = fb.read_schedule()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch schedule for {league_key} {self.season_year}: {e}")
        
        if not isinstance(schedule_df, pd.DataFrame) or schedule_df.empty:
            # No data retrieved for this league/season
            if self.debug:
                print(f"[{league_key} {self.season_year}] No schedule data found.")
            return {"status": "empty_schedule", "out_dir": self.out_dir}
        
        # Build the fixtures payload from schedule and save it at the end
        fixtures_payload = self._fixtures_to_api_shape(schedule_df)
        
        # Determine the list of match IDs from the schedule
        match_ids = []
        for mid in schedule_df["match_id"]:
            mid_int = self._coerce_int(mid)
            if mid_int is not None:
                match_ids.append(int(mid_int))
            else:
                # Use stable hash for matches lacking an integer ID
                hash_id = self._stable_int_id(mid)
                match_ids.append(int(hash_id))
        
        total_matches = len(match_ids)
        
        # If previous partial data exists, load it to resume
        if os.path.exists(self.events_fbref_path):
            with open(self.events_fbref_path, "r", encoding="utf-8") as f:
                try:
                    self.match_events = json.load(f)
                except json.JSONDecodeError:
                    self.match_events = {}
        if os.path.exists(self.players_fbref_path):
            with open(self.players_fbref_path, "r", encoding="utf-8") as f:
                try:
                    self.players_map = json.load(f)
                except json.JSONDecodeError:
                    self.players_map = {}
        
        # 2. Fetch lineups and events in batches
        processed_count = 0
        for i in range(0, total_matches, self.batch_size):
            batch_ids = match_ids[i: i + self.batch_size]
            # Read lineups for this batch of matches
            try:
                lineups_df = fb.read_lineup(match_id=batch_ids, force_cache=True)
            except Exception as e:
                if self.debug:
                    print(f"[WARN] read_lineup failed for batch starting at index {i}: {e}")
                lineups_df = pd.DataFrame()  # fallback to empty
            
            # Read events for this batch of matches
            try:
                events_df = fb.read_events(match_id=batch_ids, force_cache=True)
            except Exception as e:
                if self.debug:
                    print(f"[WARN] read_events failed for batch starting at index {i}: {e}")
                events_df = pd.DataFrame()
            
            # Merge lineup data into the match_events structure
            lineup_dict = self._lineups_to_api_shape(lineups_df)
            for match_key, match_data in lineup_dict.items():
                # Ensure existing structure, then set lineups (events will be added or overwritten next)
                self.match_events.setdefault(match_key, {"lineups": [], "events": []})
                self.match_events[match_key]["lineups"] = match_data["lineups"]
            # Merge event data into the match_events structure
            self._append_events_to_matches(events_df)
            
            processed_count += len(batch_ids)
            if self.debug:
                print(f"[{league_key} {self.season_year}] Processed {processed_count}/{total_matches} matches")
            # Save partial results to disk (to allow resume on failure mid-way)
            with open(self.events_fbref_path, "w", encoding="utf-8") as f:
                json.dump(self.match_events, f, ensure_ascii=False, indent=2)
            with open(self.players_fbref_path, "w", encoding="utf-8") as f:
                json.dump(self.players_map, f, ensure_ascii=False, indent=2)
            # Polite sleep before the next batch of requests
            self._sleep_polite()
        
        # 3. Save the full fixtures list and analyzer-compatible copies
        with open(self.fixtures_fbref_path, "w", encoding="utf-8") as f:
            json.dump(fixtures_payload, f, ensure_ascii=False, indent=2)
        # Also write the "compatibility" files (without _fbref suffix)
        with open(self.fixtures_path, "w", encoding="utf-8") as f:
            json.dump(fixtures_payload, f, ensure_ascii=False, indent=2)
        with open(self.events_path, "w", encoding="utf-8") as f:
            json.dump(self.match_events, f, ensure_ascii=False, indent=2)
        with open(self.players_path, "w", encoding="utf-8") as f:
            json.dump(self.players_map, f, ensure_ascii=False, indent=2)
        
        # 4. Compute completeness report
        # Build the set of expected match keys (as strings) from the schedule
        expected_match_keys = {str(int(self._coerce_int(m) or self._stable_int_id(m))) for m in schedule_df["match_id"]}
        have_count = 0
        for key in expected_match_keys:
            if key in self.match_events:
                entry = self.match_events[key]
                if entry.get("lineups") and entry.get("events"):
                    # Both lineups and events lists are present (even if empty, they should be lists)
                    if isinstance(entry["lineups"], list) and isinstance(entry["events"], list):
                        if len(entry["lineups"]) > 0 and len(entry["events"]) > 0:
                            have_count += 1
        missing_count = len(expected_match_keys) - have_count
        
        report = {
            "country": self.country_name,
            "league": self.league_name,
            "season_end_year": self.season_year,
            "total_matches": len(expected_match_keys),
            "downloaded": have_count,
            "missing": missing_count,
            "out_dir": self.out_dir,
            "files": {
                "fixtures_fbref": self.fixtures_fbref_path,
                "match_events_fbref": self.events_fbref_path,
                "players_fbref": self.players_fbref_path,
                "fixtures": self.fixtures_path,
                "match_events": self.events_path,
                "players": self.players_path
            }
        }
        # Save a summary report file in the output directory
        report_path = os.path.join(self.out_dir, "backfill_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        # (Optional) also save as CSV for quick view
        pd.DataFrame([report]).to_csv(os.path.join(self.out_dir, "backfill_report.csv"), index=False)
        
        if self.debug:
            print(f"[{league_key} {self.season_year}] Fetch complete. {have_count} matches downloaded, {missing_count} missing.")
        return report
















if __name__ == "__main__":

    fetcher = FootballDataFetcher(
    league_name="Bundesliga",
    country_name="GER",
    season_year=2023,
    label="fbref",
    debug=True
    )
    fetcher.run_fetch()

    
    # jobs = [
    #     ("Netherlands", "Eredivisie"),
    #     ("Portugal", "Primeira Liga"),
    #     ("Mexico", "Liga MX"),
    # ]
    # for year in range(2024, 2023, -1):  # 2024 down to 2014
    #     print(year)
    #     for country, league in jobs:
    #         fetcher = FootballDataFetcher(api_key=None, league_name=league, country_name=country, season_year=year,
    #                                       label="fbref", rate_limit_sec=10.0, debug=True)
    #         fetcher.run_fetch()
    #         # Wait a bit between seasons to avoid any rate-limit issues (if desired)
    #         time.sleep(10)
