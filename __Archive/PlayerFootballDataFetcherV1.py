#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 08:31:37 2025

@author: alexander
"""

import sqlite3
import pandas as pd
import requests
import json
import time
import sys
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Set, Iterable


from GracefulExit import GracefulExit, GracefulExitManager
 

@staticmethod
def _parse_prefixed_pid(val) -> Optional[int]:
    """
    Accepts 'p123', 'P000123', '123', 123 → 123; None if no digits found.
    """
    if val is None:
        return None
    m = re.search(r'\d+', str(val))
    return int(m.group(0)) if m else None



class PlayerFootballDataFetcher:
    """
    Fetches player profiles from the API-FOOTBALL v3 API for given league and season pairs.
    Skips any league-season (or team-season) that appears already in the database.
    Uses a SQLite database to store profiles and avoid duplicate downloads.
    Integrates with GracefulExitManager for clean termination and token usage reporting.
    """
    # API configuration
    API_BASE = "https://v3.football.api-sports.io"
    URL_PLAYERS = f"{API_BASE}/players"
    REQUEST_TIMEOUT = 25
    REQUEST_SLEEP_SECS = 0.30   # Throttle between requests
    BACKOFF_429_SECS = 2.0     # Backoff time on HTTP 429 (rate limit)

    def __init__(self,
                     api_key: str,
                     use_rapidapi: bool = False,
                     profiles_db: str = "player_profiles.db",
                     analysis_db: str = "analysis_results.db",
                     exit_manager: Optional[Any] = None):
        """
        Initialize the data fetcher with API credentials and database path.
        
        :param api_key: API-FOOTBALL API key.
        :param use_rapidapi: Set True if using RapidAPI (adjusts headers accordingly).
        :param profiles_db: Path to the SQLite database for storing player profiles.
        :param exit_manager: Optional GracefulExitManager for graceful exit handling.
        """
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            # Missing API key is a fatal configuration issue
            if exit_manager:
                exit_manager.exit("API key is not provided or invalid", stage="Initialization")
            else:
                raise ValueError("API key is not provided or invalid.")
        
        self.api_key = api_key
        self.use_rapidapi = use_rapidapi
        self.exit_manager = exit_manager
        self.analysis_db = analysis_db   # <--- NEW

        # Prepare HTTP session and headers
        self.session = requests.Session()
        self.headers = (
            {"x-rapidapi-host": "v3.football.api-sports.io", "x-rapidapi-key": self.api_key}
            if self.use_rapidapi else
            {"x-apisports-key": self.api_key}
        )

        # Connect to the SQLite database for player profiles
        try:
            self.conn_profiles = sqlite3.connect(profiles_db)
        except Exception as e:
            if exit_manager:
                exit_manager.exit("Failed to connect to profiles database", detail=str(e), stage="Initialization")
            else:
                raise

        # Optimize SQLite and ensure the profiles table exists
        self._optimize_sqlite(self.conn_profiles)
        self._ensure_profiles_schema()

        # Track last known API request quotas (for token display)
        self.last_requests_remaining_day: Optional[int] = None
        self.last_requests_remaining_minute: Optional[int] = None

    @staticmethod
    def _optimize_sqlite(conn: sqlite3.Connection) -> None:
        """Apply SQLite PRAGMA settings for performance."""
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        conn.commit()

    def _ensure_profiles_schema(self) -> None:
        c = self.conn_profiles.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS player_profiles (
                player_id     INTEGER NOT NULL,
                league_id     INTEGER NOT NULL,
                season        INTEGER NOT NULL,
                team_id       INTEGER,
                team_name     TEXT,
                league_name   TEXT,
                name          TEXT,
                firstname     TEXT,
                lastname      TEXT,
                age           INTEGER,
                birth_date    TEXT,
                birth_place   TEXT,
                birth_country TEXT,
                nationality   TEXT,
                height        TEXT,
                weight        TEXT,
                number        INTEGER,
                position      TEXT,
                photo         TEXT,
                raw_json      TEXT,
                fetched_at    TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                PRIMARY KEY (player_id, league_id, season)
            )
        """)
        self.conn_profiles.commit()

    @staticmethod
    def _coerce_int(val) -> Optional[int]:
        """
        Convert a value to integer if possible.
        e.g., '123' or 123 -> 123; returns None if no digits can be extracted.
        """
        if val is None:
            return None
        match = re.search(r'\d+', str(val))
        return int(match.group(0)) if match else None

    def profile_exists(self, player_id: int) -> bool:
        """Check if a player profile with the given player_id already exists in the database."""
        cur = self.conn_profiles.cursor()
        cur.execute("SELECT 1 FROM player_profiles WHERE player_id = ? LIMIT 1", (player_id,))
        return cur.fetchone() is not None

    def _get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a GET request to the API and return the parsed JSON data.
        Retries on HTTP 429 (rate limit) with a backoff.
        Updates the last known API token counts from response headers.
        """
        while True:
            response = self.session.get(url, headers=self.headers, params=params, timeout=self.REQUEST_TIMEOUT)
            if response.status_code == 429:
                # Hit rate limit, wait and retry
                time.sleep(self.BACKOFF_429_SECS)
                continue
            try:
                response.raise_for_status()
            except requests.RequestException as e:
                # On any HTTP error (other than 429), attempt graceful exit
                if self.exit_manager:
                    reason = "API request failed"
                    detail = f"Request to {url} with {params} failed: {e}"
                    stage = "Fetching player data"
                    # Use last known daily remaining requests (if available) for context
                    remaining = self.last_requests_remaining_day if self.last_requests_remaining_day is not None else None
                    self.exit_manager.exit(reason, detail=detail, stage=stage, requests_remaining=remaining)
                else:
                    raise  # if no exit manager, propagate the exception
            # Parse response JSON
            data = response.json()
            # Update and display remaining token counts from headers (if provided)
            day_left = response.headers.get("x-ratelimit-requests-remaining")
            min_left = response.headers.get("x-ratelimit-remaining")
            if day_left is not None and min_left is not None:
                try:
                    self.last_requests_remaining_day = int(day_left)
                except ValueError:
                    self.last_requests_remaining_day = None
                try:
                    self.last_requests_remaining_minute = int(min_left)
                except ValueError:
                    self.last_requests_remaining_minute = None
                print(f"    quota remaining: day={day_left}, minute={min_left}")
            return data

    @staticmethod
    def _choose_best_stat(stats_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        From a list of statistics entries, choose the one where the player has the most minutes played.
        This is used to pick the primary team/stat entry for the player's profile.
        """
        if not stats_list:
            return None
        best_entry = None
        max_minutes = -1
        for stats in stats_list:
            games = stats.get("games") or {}
            mins = games.get("minutes") or 0
            try:
                mins = int(mins)
            except Exception:
                mins = 0
            if mins > max_minutes:
                max_minutes = mins
                best_entry = stats
        return best_entry or stats_list[0]

    @staticmethod
    def _extract_profile(item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        p = item.get("player") or {}
        stats_list = item.get("statistics") or []
        best = PlayerFootballDataFetcher._choose_best_stat(stats_list) or {}
        games = best.get("games") or {}
        birth = p.get("birth") or {}
        lg = best.get("league") or {}
        tm = best.get("team") or {}
    
        profile_row = {
            "player_id": p.get("id"),
            "league_id": lg.get("id"),
            "season": lg.get("season"),
            "team_id": tm.get("id"),
            "team_name": tm.get("name"),
            "league_name": lg.get("name"),
            "name": p.get("name"),
            "firstname": p.get("firstname"),
            "lastname": p.get("lastname"),
            "age": p.get("age"),
            "birth_date": birth.get("date"),
            "birth_place": birth.get("place"),
            "birth_country": birth.get("country"),
            "nationality": p.get("nationality"),
            "height": p.get("height"),
            "weight": p.get("weight"),
            "number": games.get("number"),
            "position": games.get("position"),
            "photo": p.get("photo"),
            "raw_json": json.dumps(item, ensure_ascii=False),
        }
        return int(p.get("id")), profile_row
    def _insert_profile(self, row: Dict[str, Any]) -> None:
        c = self.conn_profiles.cursor()
        c.execute("""
            INSERT OR IGNORE INTO player_profiles (
                player_id, league_id, season, team_id, team_name, league_name,
                name, firstname, lastname, age,
                birth_date, birth_place, birth_country,
                nationality, height, weight, number, position, photo, raw_json
            ) VALUES (
                :player_id, :league_id, :season, :team_id, :team_name, :league_name,
                :name, :firstname, :lastname, :age,
                :birth_date, :birth_place, :birth_country,
                :nationality, :height, :weight, :number, :position, :photo, :raw_json
            )
        """, row)
        self.conn_profiles.commit()

    def verify_league_season(self, league_id: int, season: int) -> None:
        c = self.conn_profiles.cursor()
        c.execute("""SELECT COUNT(DISTINCT player_id),
                            COUNT(1),
                            COUNT(DISTINCT team_id)
                     FROM player_profiles WHERE league_id=? AND season=?""", (league_id, season))
        dp, rows, teams = c.fetchone() or (0,0,0)
        print(f"[verify] league={league_id}, season={season} -> players={dp}, rows={rows}, teams={teams}")

        # Optional: list teams you’ve covered so far
        c.execute("""SELECT team_id, team_name, COUNT(1) AS n
                     FROM player_profiles WHERE league_id=? AND season=?
                     GROUP BY team_id, team_name ORDER BY n DESC""", (league_id, season))
        rows = c.fetchall()
        if rows:
            print("  Top teams by rows:")
            for team_id, team_name, n in rows[:10]:
                print(f"   - {team_id or 'None'} {team_name or ''}: {n}")
                
                
    def _fetch_teams(self, league_id: int, season: int) -> List[Dict[str, Any]]:
        url = f"{self.API_BASE}/teams"
        data = self._get_json(url, {"league": league_id, "season": season})
        teams = []
        for item in data.get("response") or []:
            t = item.get("team") or {}
            if t.get("id"):
                teams.append({"id": t["id"], "name": t.get("name")})
        return teams
    
    
    def find_missing_players_from_analysis_db(self,
                                              league_id: Optional[int] = None,
                                              season: Optional[int] = None) -> List[int]:
        """
        Scan analysis_results.db for p-prefixed player_ids.
        Return unique numeric ids not present in player_profiles.db.
        Optionally filter by league_id and/or season.
        """
        if not os.path.exists(self.analysis_db):
            print(f"[fallback] analysis DB not found: {self.analysis_db}")
            return []
    
        conn_in = sqlite3.connect(self.analysis_db)
        pids: Set[int] = set()
        try:
            cur = conn_in.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    
            for t in tables:
                try:
                    cols = {r[1].lower(): r[1] for r in cur.execute(f'PRAGMA table_info("{t}")')}
                except sqlite3.Error:
                    continue
                if "player_id" not in cols:
                    continue
    
                query = f'SELECT DISTINCT player_id'
                filters = []
                params = []
                if league_id is not None and "league" in cols:
                    query += f', {cols["league"]}'
                    filters.append(f'{cols["league"]} = ?')
                    params.append(str(league_id))
                if season is not None and "season" in cols:
                    query += f', {cols["season"]}'
                    filters.append(f'{cols["season"]} = ?')
                    params.append(str(season))
    
                query = f'SELECT DISTINCT player_id FROM "{t}"'
                if filters:
                    query += " WHERE " + " AND ".join(filters)
    
                try:
                    for (val,) in cur.execute(query, params):
                        pid = self._parse_prefixed_pid(val)
                        if pid is not None:
                            pids.add(pid)
                except sqlite3.Error:
                    continue
        finally:
            conn_in.close()
    
        # filter out those already present in player_profiles.db
        missing: Set[int] = set()
        cur_out = self.conn_profiles.cursor()
        for pid in pids:
            cur_out.execute("SELECT 1 FROM player_profiles WHERE player_id = ? LIMIT 1", (pid,))
            if cur_out.fetchone() is None:
                missing.add(pid)
        return sorted(missing)
    def count_missing_players(self,
                          league_id: Optional[int] = None,
                              season: Optional[int] = None) -> int:
        missing = self.find_missing_players_from_analysis_db(league_id=league_id, season=season)
        print(f"[fallback] Missing players: {len(missing)}")
        return len(missing)


    
    def fetch_player_by_id(self, player_id: int) -> int:
        """
        Fetch a single player by numeric player_id via /players?id=.
        Returns 1 if a new row was inserted, else 0.
        """
        print(f"\n== Fallback: fetching player_id={player_id} ==")
        data = self._get_json(self.URL_PLAYERS, {"id": player_id})
        items = data.get("response") or []
        if not items:
            print("  No data for this player_id.")
            return 0
    
        # Typically one item, but loop defensively
        inserted = 0
        for item in items:
            try:
                pid, row = self._extract_profile(item)
                # Make sure league/season/team are stamped if present in stats
                # (they already are, via _extract_profile reading the 'best' stats block)
                if not self.profile_exists(pid):
                    self._insert_profile(row)
                    inserted += 1
            except Exception as e:
                print(f"  error processing player {player_id}: {e}", file=sys.stderr)
    
        return inserted


    def fetch_league_season_by_team(self, league_id: int, season: int) -> int:
        print(f"\n== Team-wise fetch for League {league_id}, Season {season} ==")
        new_profiles = 0
        teams = self._fetch_teams(league_id, season)
        if not teams:
            print("  No teams returned for this league/season.")
            return 0
    
        for ti, team in enumerate(teams, 1):
            team_id, team_name = team["id"], team.get("name")
            print(f"  [{ti}/{len(teams)}] Team {team_id} – {team_name}")
            page = 1
            while page <= 3:  # free-plan hard cap safeguard
                data = self._get_json(self.URL_PLAYERS, {"team": team_id, "season": season, "page": page})
                # debug
                params_echo = data.get("parameters")
                pg = data.get("paging") or {}
                print("    params:", params_echo, "| paging:", pg)
    
                items = data.get("response") or []
                if not items:
                    break
                for item in items:
                    try:
                        pid, row = self._extract_profile(item)
                        # stamp league/team/season explicitly (from stats already, but be safe)
                        row["league_id"] = league_id
                        row["season"] = season
                        row["team_id"] = row.get("team_id") or team_id
                        row["team_name"] = row.get("team_name") or team_name
                        if not self.profile_exists(pid):
                            self._insert_profile(row)
                            new_profiles += 1
                    except Exception as e:
                        print(f"    error player: {e}", file=sys.stderr)
                page += 1
                time.sleep(self.REQUEST_SLEEP_SECS)
        return new_profiles
    
    
    def fetch_league_season(self, league_id: int, season: int) -> int:
        print(f"\n== Fetching players for League {league_id}, Season {season} ==")
        new_profiles = 0
        page = 1
        expected_total = None
        empty_retries = 0
        MAX_EMPTY_RETRIES = 3
    
        while True:
            print(f"  Fetching page {page}...")
            data = self._get_json(self.URL_PLAYERS, {"league": league_id, "season": season, "page": page})
            params_echo = data.get("parameters")
            pg = data.get("paging") or {}
            errs = data.get("errors") or {}
            print("  params:", params_echo, "| paging:", pg, "| errors:", errs)
    
            # Free-plan cap detected -> fallback to team-wise
            if "plan" in errs and "maximum value of 3 for the Page parameter" in str(errs.get("plan", "")):
                print("  → Free plan page cap hit. Falling back to team-wise fetch.")
                return self.fetch_league_season_by_team(league_id, season)
    
            if expected_total is None:
                try:
                    expected_total = int(pg.get("total") or 1)
                except Exception:
                    expected_total = 1
    
            items = data.get("response") or []
            if not items:
                empty_retries += 1
                if empty_retries <= MAX_EMPTY_RETRIES and page <= (expected_total or page):
                    wait = 2.0 * empty_retries
                    print(f"  Empty page; retry {empty_retries}/{MAX_EMPTY_RETRIES} after {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                print("  No more results (or retries exhausted).")
                break
    
            empty_retries = 0
            for item in items:
                try:
                    pid, row = self._extract_profile(item)
                    if not self.profile_exists(pid):
                        self._insert_profile(row)
                        new_profiles += 1
                except Exception as e:
                    print(f"  error player: {e}", file=sys.stderr)
            page += 1
            if expected_total and page > expected_total:
                break
            time.sleep(self.REQUEST_SLEEP_SECS)
    
        return new_profiles
    
    def fetch_missing_players(self, analysis_db_path: str, limit: Optional[int] = None) -> int:
        """
        Find & fetch missing players referenced in analysis_results.db but absent from player_profiles.db.
        :param limit: optionally limit how many to fetch (for testing)
        :return: number of inserted player rows
        """
        missing = self.find_missing_players_from_analysis_db(analysis_db_path)
        if not missing:
            print("[fallback] No missing players found.")
            return 0
        if limit is not None:
            missing = missing[:int(limit)]
    
        print(f"[fallback] Missing players to fetch: {len(missing)}")
        inserted = 0
        for i, pid in enumerate(missing, 1):
            try:
                inserted += self.fetch_player_by_id(pid)
            except GracefulExit:
                raise
            except Exception as e:
                print(f"  error fetching player_id={pid}: {e}", file=sys.stderr)
            # gentle throttle
            time.sleep(self.REQUEST_SLEEP_SECS)
        return inserted

    def run_for_pairs(self, pairs: Iterable[Tuple[int, int]]) -> int:
        """
        Process all given (league_id, season) pairs to download player profiles.
        Skips any duplicate or already-downloaded combinations.
        Returns the total number of new profiles inserted across all pairs.
        """
        total_inserted = 0
        seen_pairs: Set[Tuple[int, int]] = set()

        for (lg, season) in pairs:
            lg_id = self._coerce_int(lg)
            season_year = self._coerce_int(season)
            if lg_id is None or season_year is None:
                continue
            if (lg_id, season_year) in seen_pairs:
                continue

            # Skip if this league-season already appears in the database (to avoid re-download)
            cur = self.conn_profiles.cursor()
            pattern = f'%"league":{{"id":{lg_id},%"season":{season_year}%'
            cur.execute("SELECT 1 FROM player_profiles WHERE raw_json LIKE ? LIMIT 1", (pattern,))
            if cur.fetchone():
                print(f"League {lg_id}, Season {season_year} is already in the database. Skipping.")
                seen_pairs.add((lg_id, season_year))
                continue

            seen_pairs.add((lg_id, season_year))
            try:
                new_profiles = self.fetch_league_season(lg_id, season_year)
            except GracefulExit:
                # If a graceful exit was raised, stop processing further pairs
                raise
            except Exception as e:
                # Handle unexpected exception during this pair
                if self.exit_manager:
                    self.exit_manager.exit("Error processing league-season pair",
                                            detail=str(e),
                                            stage=f"League {lg_id} Season {season_year}",
                                            requests_remaining=self.last_requests_remaining_day)
                else:
                    raise

            total_inserted += new_profiles

        return total_inserted

    @staticmethod
    def read_league_season_pairs_from_excel(xlsx_path: str, status_filter: str = "Completed") -> List[Tuple[int, int]]:
        """
        Read (league_id, season) pairs from an Excel file where the 'overall_status' equals the given filter.
        Returns a list of (league_id, season) tuples that match the filter.
        """
        if not os.path.exists(xlsx_path):
            # Excel file not found
            raise FileNotFoundError(f"Excel file not found: {xlsx_path}")
        df = pd.read_excel(xlsx_path)

        # Normalize column names to lowercase for matching
        cols = {c.lower().strip(): c for c in df.columns}
        status_col = None
        for cand in ("overall_status", "status", "overallstatus"):
            if cand in cols:
                status_col = cols[cand]
                break
        if status_col is None:
            raise KeyError("No 'overall_status' column found in the Excel file.")

        # Filter rows based on desired status (e.g., "Completed")
        mask = df[status_col].astype(str).str.strip().str.lower() == status_filter.strip().lower()
        df_filtered = df.loc[mask].copy()
        if df_filtered.empty:
            return []  # No pairs matching the status filter

        # Identify columns for league and season
        league_col = None
        for cand in ("league_id", "league", "leagueid", "league id"):
            if cand in cols:
                league_col = cols[cand]
                break
        if league_col is None:
            raise KeyError("No 'league_id' column found in the Excel file.")
        season_col = None
        for cand in ("season", "year", "season_year", "seasonyear"):
            if cand in cols:
                season_col = cols[cand]
                break
        if season_col is None:
            raise KeyError("No 'season' column found in the Excel file.")

        # Collect league-season pairs from the filtered data
        pairs: List[Tuple[int, int]] = []
        for _, row in df_filtered[[league_col, season_col]].dropna().iterrows():
            lg_val = PlayerFootballDataFetcher._coerce_int(row[league_col])
            ss_val = PlayerFootballDataFetcher._coerce_int(row[season_col])
            if lg_val is not None and ss_val is not None:
                pairs.append((lg_val, ss_val))

        # Deduplicate pairs while preserving original order
        seen: Set[Tuple[int, int]] = set()
        unique_pairs: List[Tuple[int, int]] = []
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)
        return unique_pairs

# ----------------------------- Example CLI --------------------------------
if __name__ == "__main__":
    API_KEY = c
    USE_RAPIDAPI = False  
    ANALYSIS_DB = "analysis_results.db"

    # Assume GracefulExitManager is instantiated as exit_mgr
    # fetcher = PlayerFootballDataFetcher(api_key=API_KEY, use_rapidapi=USE_RAPIDAPI, exit_manager=GracefulExitManager)
    # try:
    #     # Read league-season pairs from Excel (status "Completed")
    #     pairs = PlayerFootballDataFetcher.read_league_season_pairs_from_excel("downloads_progress.xlsx", status_filter="Completed")
    #     if not pairs:
    #         GracefulExitManager.exit("No league-season pairs to process", stage="Input")
    #     # Run the downloader for all pairs
    #     total_new = fetcher.run_for_pairs(pairs)
    #     print(f"Done. New profiles inserted: {total_new}")
    # except GracefulExit:
    #     GracefulExitManager.handle()  # Handle graceful termination (will log details and print reason)


     # instantiate without worrying about graceful exit for now
    fetcher = PlayerFootballDataFetcher(api_key=API_KEY, use_rapidapi=USE_RAPIDAPI)

    test_league_id = 78     # example: Bundesliga
    test_season    = 2022   # example: season year
    n_missing = fetcher.count_missing_players(league_id=test_league_id, season=test_season)
    print(f"Number of missing players: {n_missing}")
    # pick one league/season to test

    try:
        # First try the league-season (will auto-fallback to team-wise on free plan)
        inserted = fetcher.fetch_league_season(test_league_id, test_season)
        print(f"Inserted via league/team: {inserted}")

        # Then fill gaps for any p-prefixed players referenced in analysis_results.db
        backfilled = fetcher.fetch_missing_players(ANALYSIS_DB, limit=None)  # set limit=20 to test
        print(f"Backfilled missing players: {backfilled}")

        
        # new_profiles = fetcher.fetch_league_season(test_league_id, test_season)
        fetcher.verify_league_season(test_league_id, test_season)

        # new_profiles = fetcher.fetch_league_season(test_league_id, test_season)
        print(f"Inserted {inserted} new profiles for league {test_league_id}, season {test_season}.")
        print(f"Inserted {backfilled} new profiles for league {test_league_id}, season {test_season}.")
    finally:
        fetcher.conn_profiles.close()
