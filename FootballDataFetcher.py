#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified APIFootballBulkDownloader (with robust free-plan fallback)

Merges:
  - Season JSON fetch (teams, minimal players per team, fixtures, events)
  - Player profiles -> SQLite (bulk league-season with fallback to team-wise, per-ID backfill)
  - Transfers -> SQLite (first-time fetch; optional refresh; skip heuristics)

Key features:
  - Prominent pipeline: process_league_season(country, league, season, ...)
  - Free plan page-cap fallback in fetch_league_season_profiles() (patched)
  - Skips profiles/transfers already present in DB
  - Exposes per-player and per-season helpers
  - Default sleep = 10s; 429 backoff

External helpers (kept separate as you requested):
  - GracefulExit, GracefulExitManager (soft stub below if not present)

DB:
  - player_profiles.db (table player_profiles)
  - same DB holds transfers tables:
      player_transfers, player_transfers_fetches

JSON output tree:
  {country}_{league}/{season}/
    league.json
    teams.json
    teams_status.json
    players.json
    players_status.json
    fixtures.json
    fixtures_status.json
    match_events.json
    
    
    

player_profiles: ['player_id', 'league_id', 'season', 'team_id', 'team_name', 'league_name', 'name', 'firstname', 'lastname', 'age', 'birth_date', 'birth_place', 'birth_country', 'nationality', 'height', 'weight', 'number', 'position', 'photo', 'raw_json', 'fetched_at']
player_transfers: ['player_id', 'transfer_date', 'type', 'from_team_id', 'from_team_name', 'to_team_id', 'to_team_name', 'fee', 'season_hint', 'raw_json', 'inserted_at']
analysis_results: ['player_id', 'player_name', 'position', 'team(s)', 'league_position', 'country', 'league', 'season', 'minutes_played', 'FTE_games_played', 'games_started', 'full_games_played', 'games_subbed_on', 'games_subbed_off', 'goals', 'assists', 'contribution_ols', 'contribution_ols_std', 'ridge_contribution_alpha_1_0', 'ridge_contribution_alpha_1_0_std', 'ridge_contribution_alpha_10_0', 'ridge_contribution_alpha_10_0_std', 'lasso_contribution_alpha_0_01', 'lasso_contribution_alpha_0_01_std', 'lasso_contribution_alpha_0_001', 'lasso_contribution_alpha_0_001_std', 'partialled_contribution_ols', 'partialled_contribution_ols_std', 'partialled_ridge_contribution_alpha_1_0', 'partialled_ridge_contribution_alpha_1_0_std', 'partialled_ridge_contribution_alpha_10_0', 'partialled_ridge_contribution_alpha_10_0_std', 'partialled_lasso_contribution_alpha_0_01', 'partialled_lasso_contribution_alpha_0_01_std', 'partialled_lasso_contribution_alpha_0_001', 'partialled_lasso_contribution_alpha_0_001_std', 'is_pooled_member', 'pooled_group_key', 'other_players_same_minutes', 'total_regular_games', 'valid_regular_games', 'missing_games_count', 'no_observations']


Author: you + ChatGPT
"""

import os
import sys
import re
import json
import time
import sqlite3
import logging
import traceback
import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import requests


# ----------------------------- Logging setup -----------------------------
LOG_DIR = "./__logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "football_fetcher.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("football")


# ----------------------------- GracefulExit helpers -----------------------------
try:
    from GracefulExit import GracefulExit, GracefulExitManager
except Exception:  # Soft fallback
    class GracefulExit(Exception):
        pass
    class GracefulExitManager:
        def __init__(self): self.ctx = None
        def exit(self, reason: str, **kw):
            self.ctx = type("ExitContext", (), {"reason": reason, **kw})
            raise GracefulExit(reason)
        def handle(self, save_progress_callable=None):
            if save_progress_callable:
                try: save_progress_callable()
                except Exception: pass


# =======================================================================
#                           Unified Downloader
# =======================================================================
class APIFootballBulkDownloader:
    """
    Unified downloader for API-FOOTBALL v3.

    Prominent pipeline:
      process_league_season(country, league, season,
                            fetch_transfers=False,
                            transfers_only_for_most_recent_season=True,
                            refresh_transfers=False,
                            refresh_active_only=True)

    Also exposed:
      fetch_league_id(country, league)
      fetch_players_for_league_season(league_id, season)    # profiles bulk with fallback
      fetch_missing_players(league_id=None, league_name=None, season=None, limit=None)
      fetch_player_by_id(player_id)
      fetch_transfers_for_player(player_id, overwrite=False)
      backfill_transfers(force_refresh=False, refresh_active_only=True, limit=None)
    """

    API_BASE = "https://v3.football.api-sports.io"
    REQUEST_TIMEOUT = 45
    REQUEST_SLEEP_SECS = 10.0
    BACKOFF_429_SECS = 2.0

    def __init__(self,
                 api_key: str,
                 *,
                 use_rapidapi: bool = False,
                 profiles_db: str = "player_profiles.db",
                 analysis_db: str = "analysis_results.db",
                 rate_limit_sec: float = 10.0,
                 max_player_pages: Optional[int] = 3,
                 use_squads_fallback: bool = False,
                 threshold_missing: int = 30,
                 verbose: bool = True,
                 debug: bool = False):
        if not api_key or api_key.strip() in ("YOUR_KEY", "YOUR_API_KEY_HERE"):
            raise ValueError("API key is not provided or invalid.")

        # HTTP
        self.api_key = api_key
        self.use_rapidapi = bool(use_rapidapi)
        self.session = requests.Session()
        self.headers = (
            {"x-rapidapi-host": "v3.football.api-sports.io", "x-rapidapi-key": self.api_key}
            if self.use_rapidapi else
            {"x-apisports-key": self.api_key}
        )

        # Behavior
        self.verbose = bool(verbose)
        self.debug = bool(debug)
        self.request_sleep = float(rate_limit_sec if rate_limit_sec is not None else self.REQUEST_SLEEP_SECS)
        self.backoff_429 = float(self.BACKOFF_429_SECS)
        self.max_player_pages = int(max_player_pages) if max_player_pages is not None else None
        self.use_squads_fallback = bool(use_squads_fallback)
        self.threshold_missing = int(threshold_missing)

        # DB
        self.analysis_db = analysis_db
        self.conn_profiles = sqlite3.connect(profiles_db)
        self._optimize_sqlite(self.conn_profiles)
        self._ensure_profiles_schema()
        self._ensure_transfers_schema()

        # Exit/Quota
        self.exit_mgr = GracefulExitManager()
        self.requests_remaining: Optional[int] = None
        self._start_tokens: Optional[int] = None

        # Context (set per job)
        self.country_name: Optional[str] = None
        self.league_name: Optional[str] = None
        self.season: Optional[int] = None
        self.api_url = self.API_BASE

        # Counters & paths
        self._dl_counts = {"teams": 0, "players_min": 0, "fixtures": 0}
        self.files = {
            "league": "league.json",
            "teams": "teams.json",
            "teams_status": "teams_status.json",
            "players": "players.json",
            "players_status": "players_status.json",
            "fixtures": "fixtures.json",
            "match_events": "match_events.json",
            "fixtures_status": "fixtures_status.json",
        }
        self._truncated_players_teams: List[str] = []  # for summary

    # -------------------------------------------------------------------
    # Prominent pipeline
    # -------------------------------------------------------------------
    def close(self) -> None:
        try:
            self.conn_profiles.close()
        except Exception:
            pass
    
    def __del__(self):
        try:
            if getattr(self, "conn_profiles", None):
                self.conn_profiles.close()
        except Exception:
            pass

    def process_league_season(self, country: str, league: str, season: int, *,
                               fetch_transfers: bool = False,
                               transfers_only_for_most_recent_season: bool = True,
                               refresh_transfers: bool = False,
                               refresh_active_only: bool = True) -> None:
        """
        Full pipeline for one (country, league, season):
        ...
        """
        try:
            self._set_context(country, league, season)
            self._vprint(f"⚽ Starting download for {self.country_name}/{self.league_name}/{self.season}")

            # --- Preflight quota ---
            self.requests_remaining = self.get_requests_left()
            self._vprint("🔎 Quota preflight", show_tokens=True)
            if self.requests_remaining is not None and self.requests_remaining <= 0:
                self.exit_mgr.exit("API quota exhausted (pre-check)", stage="preflight")

            # --- league id & access probe ---
            league_id = self.fetch_league_id(country, league)
            if not self._season_access_probe(league_id):
                self._vprint(f"⛔ Season {self.season} not available on current plan for {country}/{league}.")
                return

            # --- season JSONs ---
            teams_data, _ = self._fetch_teams_json(league_id)
            players_json, _ = self._fetch_players_minimal_json(teams_data)
            fixture_ids = self._fetch_fixtures_json(league_id)
            if fixture_ids:
                self._fetch_fixture_events_json(fixture_ids)
            
            
            
            # Profiles fetch/backfill
            self._profiles_fetch_or_backfill(league_id, players_json)

            # Transfers (optional)
            if fetch_transfers:
                # respect "most recent" switch
                if transfers_only_for_most_recent_season:
                    most_recent = self._get_most_recent_season() or int(self.season)
                    if int(self.season) != int(most_recent):
                        self._vprint(f"↪️ Skipping transfers: season {self.season} != most_recent {most_recent}.")
                    else:
                        try:
                            self.fetch_transfers_for_league(league_id, int(self.season),
                                                            force_refresh=refresh_transfers)
                        except GracefulExit:
                            self._vprint("Team-based transfer fetch hit a limit → fallback per-player.")
                            self.backfill_transfers(force_refresh=refresh_transfers,
                                                    refresh_active_only=True)
                else:
                    try:
                        self.fetch_transfers_for_league(league_id, int(self.season),
                                                        force_refresh=refresh_transfers)
                    except GracefulExit:
                        self._vprint("Team-based transfer fetch hit a limit → fallback per-player.")
                        self.backfill_transfers(force_refresh=refresh_transfers,
                                                refresh_active_only=True)

            # if fetch_transfers:
            #     if transfers_only_for_most_recent_season:
            #         most_recent = self._get_most_recent_season()
            #         most_recent = most_recent if most_recent is not None else int(self.season)
            #         if int(self.season) != int(most_recent):
            #             self._vprint(f"↪️ Skipping transfers: season {self.season} != most_recent {most_recent}.")
            #         else:
            #             try:
            #                 # Main approach: fetch transfers by team for the whole league
            #                 self.fetch_transfers_for_league(league_id, int(self.season),
            #                                                 force_refresh=refresh_transfers,
            #                                                 refresh_active_only=refresh_active_only)
            #             except GracefulExit:
            #                 # Fallback to player-by-player if team-based fetch hits a limit
            #                 self._vprint("Team-based transfer fetch failed; falling back to per-player transfers.")
            #                 self.backfill_transfers(force_refresh=refresh_transfers,
            #                                         refresh_active_only=refresh_active_only)
            #     else:
            #         try:
            #             self.fetch_transfers_for_league(league_id, int(self.season),
            #                                             force_refresh=refresh_transfers,
            #                                             refresh_active_only=refresh_active_only)
            #         except GracefulExit:
            #             self._vprint("Team-based transfer fetch failed; falling back to per-player transfers.")
            #             self.backfill_transfers(force_refresh=refresh_transfers,
            #                                     refresh_active_only=refresh_active_only)

            # Summary
            self._summarize()

        except GracefulExit:
            LOGGER.info("Graceful exit: %s", getattr(self.exit_mgr, "ctx", None))
            self._vprint(f"GracefulExit: {getattr(self.exit_mgr, 'ctx', None)}")
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            LOGGER.error("Unexpected error: %s", e)
            print(f"❌ Unexpected error: {e}")

    # -------------------------------------------------------------------
    # Season JSON stages
    # -------------------------------------------------------------------
    def fetch_league_id(self, country: str, league: str) -> int:
        """Resolve league_id; cache raw leagues response under <country>_<league>/league.json."""
        # Build a path *without* relying on self.save_country_dir
        save_country_dir = f"{country}_{league}"
        os.makedirs(save_country_dir, exist_ok=True)
        path = os.path.join(save_country_dir, self.files["league"])
    
        if os.path.exists(path):
            data = json.load(open(path, "r", encoding="utf-8"))
        else:
            data = self.api_get("/leagues", {"name": league, "country": country})
            if data.get("results", 0) == 0 or not data.get("response"):
                self.exit_mgr.exit("Invalid league or empty response", stage="leagues")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
        try:
            return int(data["response"][0]["league"]["id"])
        except Exception as e:
            self.exit_mgr.exit("Could not extract league_id", detail=str(e), stage="leagues")
            raise

    # def fetch_league_id(self, country: str, league: str) -> int:
    #     """Resolve league_id; cache raw leagues response in country dir."""
    #     path = os.path.join(self.save_country_dir, self.files["league"])
    #     if os.path.exists(path):
    #         data = json.load(open(path, "r", encoding="utf-8"))
    #     else:
    #         data = self.api_get("/leagues", {"name": league, "country": country})
    #         if data.get("results", 0) == 0 or not data.get("response"):
    #             self.exit_mgr.exit("Invalid league or empty response", stage="leagues")
    #         with open(path, "w", encoding="utf-8") as f:
    #             json.dump(data, f, ensure_ascii=False, indent=2)
    #     try:
    #         return int(data["response"][0]["league"]["id"])
    #     except Exception as e:
    #         self.exit_mgr.exit("Could not extract league_id", detail=str(e), stage="leagues")
    #         raise


    def _fetch_teams_json(self, league_id: int) -> Tuple[dict, dict]:
        self._vprint(f"\n==== PHASE: TEAMS | league_id={league_id} season={self.season} ====")
        teams_data = self._load_json(self.files["teams"])
        teams_status = self._load_json(self.files["teams_status"])
    
        if not teams_data:
            d = self.api_get("/teams", {"league": league_id, "season": self.season})
            if d.get("results", 0) == 0 or not d.get("response"):
                self.exit_mgr.exit("No team data (empty)", stage="teams")
            teams_data = d
            teams_status = {t["team"]["name"]: False for t in teams_data["response"]}
            self._save_json(self.files["teams"], teams_data)
            self._save_json(self.files["teams_status"], teams_status)
    
        teams = teams_data.get("response", [])
        remaining = [t for t in teams if not teams_status.get(t["team"]["name"], False)]
        if not remaining:
            self._vprint("✅ Teams already complete — skipping.")
            return teams_data, teams_status
    
        total_remaining = len(remaining)
        processed = 0
        for t in teams:
            n = t["team"]["name"]
            if teams_status.get(n):
                continue
            processed += 1
            self._vprint(f"📥 Team {processed}/{total_remaining}: {n}")
            teams_status[n] = True
            self._dl_counts["teams"] += 1
            self._save_json(self.files["teams_status"], teams_status)
            # NOTE: no sleep here — we don't call the API per team
    
        # Final save (idempotent)
        self._save_json(self.files["teams"], teams_data)
        self._save_json(self.files["teams_status"], teams_status)
        return teams_data, teams_status
    
    

    # def _fetch_teams_json(self, league_id: int) -> Tuple[dict, dict]:
    #     teams_data = self._load_json(self.files["teams"])
    #     teams_status = self._load_json(self.files["teams_status"])
    #     if not teams_data:
    #         d = self.api_get("/teams", {"league": league_id, "season": self.season})
    #         if d.get("results", 0) == 0 or not d.get("response"):
    #             self.exit_mgr.exit("No team data (empty)", stage="teams")
    #         teams_data = d
    #         teams_status = {t["team"]["name"]: False for t in teams_data["response"]}

    #     teams = teams_data.get("response", [])
    #     remaining = [t for t in teams if not teams_status.get(t["team"]["name"], False)]
    #     total_remaining = len(remaining)
    #     processed = 0

    #     for t in teams_data.get("response", []):
    #         n = t["team"]["name"]
    #         if teams_status.get(n):
    #             continue
    #         processed += 1
    #         self._vprint(f"📥 Team {processed}/{total_remaining}: {n}")

    #         teams_status[n] = True
    #         self._dl_counts["teams"] += 1
    #         self._save_json(self.files["teams_status"], teams_status)
    #         time.sleep(self.request_sleep)

    #     self._save_json(self.files["teams"], teams_data)
    #     self._save_json(self.files["teams_status"], teams_status)
    #     return teams_data, teams_status

    def _fetch_players_minimal_json(self, teams_data: dict) -> Tuple[dict, dict]:
        """Minimal per-team player list to JSON for season context."""
        players_data = self._load_json(self.files["players"])
        players_status = self._load_json(self.files["players_status"])
        if not players_status and teams_data:
            players_status = {t["team"]["name"]: False for t in teams_data.get("response", [])}


        team_list = [t for t in teams_data.get("response", []) if not players_status.get(t["team"]["name"], False)]
        total_teams = len(team_list)
        processed_teams = 0
        self._vprint(f"\n==== PHASE: PLAYERS (minimal) | season={self.season} teams={total_teams} ====")

        for t in team_list:
            name = t["team"]["name"]
            team_id = t["team"]["id"]
            processed_teams += 1
            self._vprint(f"📥 Downloading players (minimal) for {name} ({processed_teams}/{total_teams})")

            resp = self.api_get("/players", {"team": team_id, "season": self.season})
            if resp.get("results", 0) == 0 or not resp.get("response"):
                players_data[name] = []
                players_status[name] = True
                self._save_json(self.files["players"], players_data)
                self._save_json(self.files["players_status"], players_status)
                continue

            players_list, seen_ids = [], set()
            for p in resp.get("response", []):
                pid = p["player"]["id"]
                seen_ids.add(pid)
                players_list.append({
                    "id": pid,
                    "name": p["player"]["name"],
                    "age": p["player"].get("age"),
                    "position": (p.get("statistics") or [{}])[0].get("games", {}).get("position"),
                })

            total_pages = int((resp.get("paging", {}) or {}).get("total", 1) or 1)
            pages_to_fetch = total_pages if (self.max_player_pages is None) else min(total_pages, self.max_player_pages)
            if self.max_player_pages is not None and total_pages > self.max_player_pages:
                self._vprint(f"⚠️  Page-cap hit for {name}: {pages_to_fetch}/{total_pages}.")
                self._truncated_players_teams.append(name)

            for page in range(2, pages_to_fetch + 1):
                time.sleep(self.request_sleep)
                nxt = self.api_get("/players", {"team": team_id, "season": self.season, "page": page})
                for p in nxt.get("response", []):
                    pid = p["player"]["id"]
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    players_list.append({
                        "id": pid,
                        "name": p["player"]["name"],
                        "age": p["player"].get("age"),
                        "position": (p.get("statistics") or [{}])[0].get("games", {}).get("position"),
                    })

            if self.use_squads_fallback and (self.max_player_pages is not None) and (total_pages > self.max_player_pages):
                try:
                    time.sleep(self.request_sleep)
                    squad = self.api_get("/players/squads", {"team": team_id})
                    for team_blob in squad.get("response", []):
                        for p in team_blob.get("players", []):
                            pid = p.get("id")
                            if pid is None or pid in seen_ids:
                                continue
                            seen_ids.add(pid)
                            players_list.append({
                                "id": pid,
                                "name": p.get("name"),
                                "age": p.get("age"),
                                "position": p.get("position"),
                            })
                    self._vprint(f"ℹ️  {name}: added via squads fallback")
                except Exception:
                    if self.debug:
                        print(f"⚠️ squads fallback failed for {name}")

            players_data[name] = players_list
            players_status[name] = True
            self._save_json(self.files["players"], players_data)
            self._save_json(self.files["players_status"], players_status)
            self._dl_counts["players_min"] += len(players_list)
            time.sleep(self.request_sleep)
        
        
        self._vprint(f"✓ players(min) done — added {self._dl_counts['players_min']} entries so far.")
        return players_data, players_status

    def _fetch_fixtures_json(self, league_id: int) -> List[int]:
        self._vprint(f"[fixtures] Fetching fixture list for league_id={league_id}, season={self.season}")
        self._vprint(f"\n==== PHASE: FIXTURES | league_id={league_id} season={self.season} ====")
        fixtures_data = self._load_json(self.files["fixtures"])
        if not fixtures_data:
            d = self.api_get("/fixtures", {"league": league_id, "season": self.season})
            if d.get("results", 0) == 0 or not d.get("response"):
                self._save_json(self.files["fixtures"], d)
                return []
            self._save_json(self.files["fixtures"], d)
            fixtures_data = d
            
        self._vprint(f"[fixtures] Found {len([f['fixture']['id'] for f in fixtures_data.get('response', [])])} fixtures")
        
        
        return [f["fixture"]["id"] for f in fixtures_data.get("response", [])]

    def _fetch_fixture_events_json(self, fixture_ids: List[int]) -> Tuple[dict, dict]:
        
        match_events = self._load_json(self.files["match_events"])
        fixtures_status = self._load_json(self.files["fixtures_status"])
        fixtures_data = self._load_json(self.files["fixtures"])
        fixture_map = {}
        try:
            for f in fixtures_data.get("response", []):
                fid = f.get("fixture", {}).get("id")
                ht = (f.get("teams") or {}).get("home", {}).get("name")
                at = (f.get("teams") or {}).get("away", {}).get("name")
                if fid is not None:
                    fixture_map[int(fid)] = f"{ht} vs {at}" if ht and at else ""
        except Exception:
            fixture_map = {}


        to_fetch = []
        for fid in fixture_ids:
            key = str(fid)
            if key in match_events and match_events[key].get("lineups") and match_events[key].get("events"):
                continue
            to_fetch.append(fid)
        
        total_fx = len(to_fetch)
        self._vprint(f"\n==== PHASE: MATCH EVENTS | fixtures to fetch={total_fx} ====")
        for idx, fid in enumerate(to_fetch, start=1):
            key = str(fid)
            matchup = fixture_map.get(int(fid), "")
            self._vprint(f"📥 Fixture {idx}/{total_fx}: {fid} {('('+matchup+')' if matchup else '')}")
            lineup = self.api_get("/fixtures/lineups", {"fixture": fid})
            time.sleep(self.request_sleep)
            events = self.api_get("/fixtures/events", {"fixture": fid})
            time.sleep(self.request_sleep)


            match_events[key] = {
                "lineups": lineup.get("response", []),
                "events": events.get("response", []),
            }
            fixtures_status[key] = True
            self._dl_counts["fixtures"] += 1
            self._save_json(self.files["match_events"], match_events)
            self._save_json(self.files["fixtures_status"], fixtures_status)

        return match_events, fixtures_status

    # -------------------------------------------------------------------
    # Player profiles (SQLite): bulk with fallback + per-ID
    # -------------------------------------------------------------------
    def fetch_league_season_profiles(self, league_id: int, season: int) -> int:
        """
        Bulk fetch player profiles via /players?league=...&season=... (paginated)
        with robust free-plan fallback to team-wise fetch.
        Returns number of new profiles inserted.
        """
        self._vprint(f"[profiles] Bulk league-season fetch: league_id={league_id}, season={season}")

        new_profiles = 0
        page = 1
        expected_total = None
        empty_retries, MAX_EMPTY_RETRIES = 0, 3

        while True:
            self._vprint(f"[profiles] league={league_id}, season={season}: page {page}...")
            # Pre-empt free plan page-cap (3) if caller set max_player_pages >= 3
            if page == 4 and (self.max_player_pages is None or self.max_player_pages >= 3):
                self._vprint("  → Pre-empting at page 4 (free plan). Falling back to team-wise fetch.")
                return self._fetch_league_season_by_team(league_id, season)

            try:
                data = self.api_get(
                    "/players",
                    {"league": league_id, "season": season, "page": page},
                    tolerate_429=True
                )
            except GracefulExit:
                # Catch API 'errors' (plan cap), and fall back gracefully.
                ctx = getattr(self.exit_mgr, "ctx", None)
                detail = (getattr(ctx, "detail", "") or "").lower()
                if "maximum value of 3 for the page parameter" in detail:
                    self._vprint("  → Free plan page cap hit. Falling back to team-wise fetch.")
                    return self._fetch_league_season_by_team(league_id, season)
                raise  # Different error -> bubble up

            errs = data.get("errors") or {}
            if "plan" in errs and "maximum value of 3 for the Page parameter" in str(errs.get("plan", "")):
                self._vprint("  → Free plan page cap hit. Falling back to team-wise fetch.")
                return self._fetch_league_season_by_team(league_id, season)

            pg = data.get("paging") or {}
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
                    self._vprint(f"  Empty page; retry {empty_retries}/{MAX_EMPTY_RETRIES} after {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                self._vprint("  No more results (or retries exhausted).")
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
            time.sleep(self.request_sleep)

        return new_profiles

    def _fetch_league_season_by_team(self, league_id: int, season: int) -> int:
        """Team-wise fallback for player profiles."""
        self._vprint(f"[profiles] Team-wise fallback: league_id={league_id}, season={season}")

        teams = self._fetch_teams_simple(league_id, season)
        if not teams:
            self._vprint("  No teams returned for this league/season (fallback).")
            return 0
        new_profiles = 0
        for ti, team in enumerate(teams, 1):
            team_id, team_name = team["id"], team.get("name")
            self._vprint(f"  [{ti}/{len(teams)}] Team {team_id} – {team_name}")
            page = 1
            while page <= 3:  # free-plan safety
                data = self.api_get("/players", {"team": team_id, "season": season, "page": page}, tolerate_429=True)
                items = data.get("response") or []
                if not items:
                    break
                for item in items:
                    try:
                        pid, row = self._extract_profile(item)
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
                time.sleep(self.request_sleep)
        return new_profiles


    def _has_fetched_transfers_for_team(self, team_id: int, season: int | None) -> bool:
        c = self.conn_profiles.cursor()
        if season is None:
            # If no season provided, treat any logged season as fetched.
            c.execute("SELECT 1 FROM team_transfers_fetches WHERE team_id = ? LIMIT 1", (int(team_id),))
        else:
            c.execute(
                "SELECT 1 FROM team_transfers_fetches WHERE team_id = ? AND season = ? LIMIT 1",
                (int(team_id), int(season))
            )
        return c.fetchone() is not None
    
    def _mark_team_fetch_result(self, team_id: int, season: int | None, result: str) -> None:
        c = self.conn_profiles.cursor()
        # Upsert by (team_id, season)
        c.execute("""
            INSERT INTO team_transfers_fetches (team_id, season, last_fetched_at, last_result)
            VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'), ?)
            ON CONFLICT(team_id, season) DO UPDATE SET
              last_fetched_at=excluded.last_fetched_at,
              last_result=excluded.last_result
        """, (int(team_id), int(season) if season is not None else None, result[:200]))
        self.conn_profiles.commit()

    # Public wrapper
    def fetch_players_for_league_season(self, league_id: int, season: int) -> int:
        return self.fetch_league_season_profiles(league_id, season)

    # Per-ID and missing backfill
    def fetch_missing_players(self,
                              limit: Optional[int] = None,
                              league_id: Optional[int] = None,
                              league_name: Optional[str] = None,
                              season: Optional[int] = None) -> int:
        missing = self.find_missing_players_from_analysis_db(league_id=league_id,
                                                             league_name=league_name,
                                                             season=season)
        if not missing:
            self._vprint("[fallback] No missing players found.")
            return 0
        if limit is not None:
            missing = missing[:int(limit)]
        self._vprint(f"[fallback] Backfilling {len(missing)} players individually...")
        inserted = 0
        for pid in missing:
            try:
                inserted += self.fetch_player_by_id(pid, season=(season or self.season or self._get_most_recent_season()))

                #inserted += self.fetch_player_by_id(pid)
            except GracefulExit:
                raise
            except Exception as e:
                print(f"  error fetching player_id={pid}: {e}", file=sys.stderr)
            time.sleep(self.request_sleep)
        return inserted

    def fetch_player_by_id(self, player_id: int, season: Optional[int] = None) -> int:
        """
        Fetch a single player profile. API requires 'season' when using /players?id.
        Order of seasons to try:
          1) explicit 'season' arg if provided
          2) current job context self.season
          3) most recent season seen in player_profiles
          4) current year, then current year-1  (light fallback)
        """
        self._vprint(f"\n== fetching player_id={player_id} ==")
    
        # Build a short list of seasons to try (unique, in order)
        seasons_to_try: List[int] = []
        if season is not None:
            seasons_to_try.append(int(season))
        if getattr(self, "season", None) is not None:
            seasons_to_try.append(int(self.season))
        mr = self._get_most_recent_season()
        if mr is not None:
            seasons_to_try.append(int(mr))
        this_year = datetime.date.today().year
        seasons_to_try.extend([this_year, this_year - 1])
        # de-duplicate while preserving order
        seen = set()
        seasons_to_try = [s for s in seasons_to_try if not (s in seen or seen.add(s))]
    
        inserted = 0
        last_error = None
    
        for s in seasons_to_try:
            try:
                self._vprint(f"  → trying season={s}")
                data = self.api_get("/players", {"id": int(player_id), "season": int(s)}, tolerate_429=True)
            except GracefulExit as ge:
                # e.g., if the API complains again; try next season candidate
                last_error = ge
                continue
    
            items = data.get("response") or []
            if not items:
                self._vprint(f"  No data for player_id={player_id} in season={s}; checking another season…")
                time.sleep(self.request_sleep)
                continue
    
            for item in items:
                try:
                    pid, row = self._extract_profile(item)
                    if not self.profile_exists(pid):
                        self._insert_profile(row)
                        inserted += 1
                except Exception as e:
                    print(f"  error processing player {player_id}: {e}", file=sys.stderr)
            break  # success; stop trying other seasons
    
        if inserted == 0 and last_error:
            # propagate last API error if *all* attempts failed and we captured one
            raise last_error
        return inserted

    # -------------------------------------------------------------------
    # Transfers (SQLite)
    # -------------------------------------------------------------------
    
    def fetch_transfers_for_player(self, player_id: int, *, overwrite: bool = False) -> int:
        if overwrite:
            c = self.conn_profiles.cursor()
            c.execute("DELETE FROM player_transfers WHERE player_id = ?", (int(player_id),))
            self.conn_profiles.commit()

        data = self.api_get("/transfers", {"player": int(player_id)}, tolerate_429=True)
        resp = data.get("response") or []
        transfers: List[Dict[str, Any]] = []
        for item in resp:
            arr = item.get("transfers") or []
            if isinstance(arr, list):
                transfers.extend(arr)

        # Insert transfer records for this player
        inserted = self._upsert_transfers(player_id, transfers)

        # If no transfers, insert a placeholder record to mark as checked
        if not transfers:
            try:
                c = self.conn_profiles.cursor()
                c.execute("""
                    INSERT OR IGNORE INTO player_transfers (
                        player_id, transfer_date, type,
                        from_team_id, from_team_name,
                        to_team_id, to_team_name,
                        fee, season_hint, raw_json
                    )
                    VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, '')
                """, (int(player_id), "0000-00-00", "checked, no news"))
                self.conn_profiles.commit()
            except sqlite3.Error:
                pass

        # Record result in fetches table
        self._mark_fetch_result(player_id, "ok" if transfers else "checked, no news")
        return inserted
   
    def fetch_transfers_for_team(self, team_id: int, *, season: int | None = None,
                                 overwrite: bool = False) -> Tuple[int, bool]:
        """
        Fetch transfers for a team (incoming + outgoing).
        Returns (inserted_rows, was_skipped).
        Skips if already fetched for (team_id, season) and not overwriting.
        """
        # Skip if already fetched for this team-season and not overwriting
        if not overwrite and self._has_fetched_transfers_for_team(int(team_id), season):
            self._vprint(f"[transfers] ⏩ SKIP team {team_id} (season={season}): already fetched.")
            return 0, True
    
        # If overwrite, clear existing rows for this team (both directions)
        if overwrite:
            c = self.conn_profiles.cursor()
            try:
                c.execute("DELETE FROM player_transfers WHERE from_team_id=? OR to_team_id=?",
                          (int(team_id), int(team_id)))
                self.conn_profiles.commit()
            except sqlite3.Error as e:
                print(f"[warn] Could not delete existing transfers for team {team_id}: {e}", file=sys.stderr)
    
        # Hit the API (we'll sleep in here, not in the league loop)
        params: Dict[str, Any] = {"team": int(team_id)}
        data = self.api_get("/transfers", params, tolerate_429=True)
        resp = data.get("response") or []
    
        total_inserted = 0
        for item in resp:
            pid = item.get("player", {}).get("id")
            transfers_list = item.get("transfers") or []
            if pid is None:
                continue
    
            # Client-side season filter (API doesn't filter by season for team endpoint)
            if season is not None:
                y = str(int(season))
                filtered = []
                for tr in transfers_list:
                    dt = (tr.get("date") or tr.get("updated") or "")[:10]
                    if isinstance(dt, str) and dt.startswith(y):
                        filtered.append(tr)
                transfers_list = filtered
    
            inserted = self._upsert_transfers(int(pid), transfers_list)
            total_inserted += inserted
    
            # placeholder if none (prevents repeated re-fetches per player)
            if not transfers_list:
                try:
                    c = self.conn_profiles.cursor()
                    c.execute("""
                        INSERT OR IGNORE INTO player_transfers (
                            player_id, transfer_date, type,
                            from_team_id, from_team_name,
                            to_team_id, to_team_name,
                            fee, season_hint, raw_json
                        )
                        VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, '')
                    """, (int(pid), "0000-00-00", "checked, no news"))
                    self.conn_profiles.commit()
                except sqlite3.Error:
                    pass
    
            self._mark_fetch_result(int(pid), "ok" if transfers_list else "checked, no news")
    
        # Mark team-season as fetched (even if zero inserted; we still learned “no news”)
        self._mark_team_fetch_result(int(team_id), season, "ok" if total_inserted > 0 else "checked, no news")
    
        # Rate-limit sleep only when we actually called the API
        time.sleep(self.request_sleep)
    
        return total_inserted, False

    # def fetch_transfers_for_team(self, team_id: int, *, season: int | None = None,
    #                              overwrite: bool = False) -> int:
    #     """
    #     Fetch transfers involving a team (incoming & outgoing).
    #     If season is given but the API doesn't filter by season server-side,
    #     we filter client-side by transfer_date prefix 'YYYY-'.
    #     Uses team_transfers_fetches to avoid redundant API calls.
    #     Returns number of inserted rows.
    #     """
    #     # Skip if already fetched for this team-season and not overwriting
    #     if not overwrite and self._has_fetched_transfers_for_team(int(team_id), season):
    #         self._vprint(f"[transfers] Team {team_id} (season={season}) already fetched; skipping API call.")
    #         return 0
    
    #     # If overwrite, clear existing rows where this team is either from or to
    #     if overwrite:
    #         c = self.conn_profiles.cursor()
    #         try:
    #             c.execute("DELETE FROM player_transfers WHERE from_team_id=? OR to_team_id=?",
    #                       (int(team_id), int(team_id)))
    #         except sqlite3.Error as e:
    #             print(f"[warn] Could not delete existing transfers for team {team_id}: {e}", file=sys.stderr)
    #         else:
    #             self.conn_profiles.commit()
    
    #     # Fetch from API
    #     params: Dict[str, Any] = {"team": int(team_id)}
    #     data = self.api_get("/transfers", params, tolerate_429=True)
    #     resp = data.get("response") or []
    
    #     total_inserted = 0
    #     for item in resp:
    #         pid = item.get("player", {}).get("id")
    #         transfers_list = item.get("transfers") or []
    #         if pid is None:
    #             continue
    
    #         # Client-side season filter if requested
    #         if season is not None:
    #             y = str(int(season))
    #             filtered = []
    #             for tr in transfers_list:
    #                 dt = (tr.get("date") or tr.get("updated") or "")[:10]
    #                 if isinstance(dt, str) and dt.startswith(y):
    #                     filtered.append(tr)
    #             transfers_list = filtered
    
    #         inserted = self._upsert_transfers(int(pid), transfers_list)
    #         total_inserted += inserted
    
    #         # placeholder if none (prevents repeated re-fetches per player)
    #         if not transfers_list:
    #             try:
    #                 c = self.conn_profiles.cursor()
    #                 c.execute("""
    #                     INSERT OR IGNORE INTO player_transfers (
    #                         player_id, transfer_date, type,
    #                         from_team_id, from_team_name,
    #                         to_team_id, to_team_name,
    #                         fee, season_hint, raw_json
    #                     )
    #                     VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, '')
    #                 """, (int(pid), "0000-00-00", "checked, no news"))
    #                 self.conn_profiles.commit()
    #             except sqlite3.Error:
    #                 pass
    
    #         self._mark_fetch_result(int(pid), "ok" if transfers_list else "checked, no news")
    
    #     # Mark team-season as fetched (even if zero inserted; we still learned “no news”)
    #     self._mark_team_fetch_result(int(team_id), season, "ok" if total_inserted > 0 else "checked, no news")
    
    #     return total_inserted
 
    def fetch_transfers_for_league(self, league_id: int, season: int, *,
                                   force_refresh: bool = False) -> None:
        teams = self._fetch_teams_simple(league_id, season)
        if not teams:
            self._vprint(f"[transfers] No teams for league_id={league_id}, season={season}.")
            return
        self._vprint(f"\n==== PHASE: TRANSFERS | league_id={league_id} season={season} teams={len(teams)} ====")
        for i, t in enumerate(teams, 1):
            tid, tname = t["id"], t.get("name", "")
            self._vprint(f"  [{i}/{len(teams)}] team={tid} — {tname}")
            try:
                inserted, skipped = self.fetch_transfers_for_team(tid, season=season, overwrite=force_refresh)
                if skipped:
                    # no sleep — we did not call the API inside
                    continue
                # we already sleep inside fetch_transfers_for_team when it did call the API
            except GracefulExit:
                raise
            except Exception as e:
                print(f"    [error] team_id={tid}: {e}", file=sys.stderr)

    # def fetch_transfers_for_league(self, league_id: int, season: int, *,
    #                                force_refresh: bool = False) -> None:
    #     teams = self._fetch_teams_simple(league_id, season)
    #     if not teams:
    #         self._vprint(f"[transfers] No teams for league_id={league_id}, season={season}.")
    #         return
    #     self._vprint(f"[transfers] Team-based fetch for {len(teams)} teams…")
    #     for i, t in enumerate(teams, 1):
    #         tid, tname = t["id"], t.get("name", "")
    #         self._vprint(f"  [{i}/{len(teams)}] {tid} – {tname}")
    #         try:
    #             self.fetch_transfers_for_team(tid, season=season, overwrite=force_refresh)
    #         finally:
    #             time.sleep(self.request_sleep)
 
    def backfill_transfers(self,
                           *,
                           force_refresh: bool = False,
                           refresh_active_only: bool = True,
                           limit: Optional[int] = None) -> Tuple[int, int, int]:
        all_pids = self._get_all_player_ids()
        if limit is not None:
            all_pids = all_pids[:int(limit)]
        active = self._get_active_player_ids_in_most_recent_season() if refresh_active_only else set()
        current_season = self._get_most_recent_season()

        first_time, refreshed, skipped = 0, 0, 0
        self._vprint(f"[transfers] players total={len(all_pids)}, active_in_most_recent={len(active)}")
        for i, pid in enumerate(all_pids, 1):
            try:
                if self._has_transfers_for_later_season(pid, current_season):
                    skipped += 1
                    continue

                if not self._has_fetched_transfers(pid):
                    self._vprint(f"  [{i}/{len(all_pids)}] player_id={pid}: first-time fetch…")
                    _ = self.fetch_transfers_for_player(pid, overwrite=False)
                    first_time += 1
                else:
                    if force_refresh and ((not refresh_active_only) or (pid in active)):
                        self._vprint(f"  [{i}/{len(all_pids)}] player_id={pid}: refreshing…")
                        _ = self.fetch_transfers_for_player(pid, overwrite=True)
                        refreshed += 1
                    else:
                        skipped += 1
                        continue
            except GracefulExit:
                raise
            except Exception as e:
                print(f"    [error] player_id={pid}: {e}", file=sys.stderr)
            finally:
                time.sleep(self.request_sleep)

        self._vprint(f"[transfers] first_time={first_time}, refreshed={refreshed}, skipped={skipped}")
        return first_time, refreshed, skipped

    # -------------------------------------------------------------------
    # Missing players from analysis_results.db
    # -------------------------------------------------------------------
    def find_missing_players_from_analysis_db(self,
                                              league_id: Optional[int] = None,
                                              league_name: Optional[str] = None,
                                              season: Optional[int] = None) -> List[int]:
        if not self.analysis_db or not os.path.exists(self.analysis_db):
            self._vprint(f"[fallback] analysis DB not found: {self.analysis_db}")
            return []
        conn = sqlite3.connect(self.analysis_db)
        pids: Set[int] = set()
        try:
            cur = conn.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for t in tables:
                try:
                    cols = {r[1].lower(): r[1] for r in cur.execute(f'PRAGMA table_info("{t}")')}
                except sqlite3.Error:
                    continue
                if "player_id" not in cols:
                    continue
                q = f'SELECT DISTINCT player_id FROM "{t}"'
                filters, params = [], []
                if league_id is not None and "league_id" in cols:
                    filters.append(f'{cols["league_id"]} = ?'); params.append(int(league_id))
                elif league_name is not None and "league" in cols:
                    filters.append(f'{cols["league"]} = ?'); params.append(str(league_name))
                if season is not None and "season" in cols:
                    filters.append(f'{cols["season"]} = ?'); params.append(int(season))
                if filters:
                    q += " WHERE " + " AND ".join(filters)
                try:
                    for (val,) in cur.execute(q, params):
                        pid = self._coerce_int(val)
                        if pid is not None:
                            pids.add(pid)
                except sqlite3.Error:
                    continue
        finally:
            conn.close()

        # Remove those already present
        missing: Set[int] = set()
        cur_out = self.conn_profiles.cursor()
        for pid in pids:
            cur_out.execute("SELECT 1 FROM player_profiles WHERE player_id = ? LIMIT 1", (pid,))
            if cur_out.fetchone() is None:
                missing.add(pid)
        return sorted(missing)

    # -------------------------------------------------------------------
    # Internals: profile extraction & DB schema
    # -------------------------------------------------------------------
    @staticmethod
    def _optimize_sqlite(conn: sqlite3.Connection) -> None:
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

    def _ensure_transfers_schema(self) -> None:
        c = self.conn_profiles.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS player_transfers (
                player_id       INTEGER NOT NULL,
                transfer_date   TEXT,
                type            TEXT,
                from_team_id    INTEGER,
                from_team_name  TEXT,
                to_team_id      INTEGER,
                to_team_name    TEXT,
                fee             TEXT,
                season_hint     INTEGER,
                raw_json        TEXT,
                inserted_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                PRIMARY KEY (player_id, transfer_date, from_team_id, to_team_id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS player_transfers_fetches (
                player_id       INTEGER PRIMARY KEY,
                last_fetched_at TEXT,
                last_result     TEXT
            )
        """)
        self.conn_profiles.commit()
        
        
    def _ensure_transfers_schema(self) -> None:
        c = self.conn_profiles.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS player_transfers (
                player_id       INTEGER NOT NULL,
                transfer_date   TEXT,
                type            TEXT,
                from_team_id    INTEGER,
                from_team_name  TEXT,
                to_team_id      INTEGER,
                to_team_name    TEXT,
                fee             TEXT,
                season_hint     INTEGER,
                raw_json        TEXT,
                inserted_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                PRIMARY KEY (player_id, transfer_date, from_team_id, to_team_id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS player_transfers_fetches (
                player_id       INTEGER PRIMARY KEY,
                last_fetched_at TEXT,
                last_result     TEXT
            )
        """)
        # NEW: team-level fetch log (season-aware)
        c.execute("""
            CREATE TABLE IF NOT EXISTS team_transfers_fetches (
                team_id         INTEGER NOT NULL,
                season          INTEGER,
                last_fetched_at TEXT,
                last_result     TEXT,
                PRIMARY KEY (team_id, season)
            )
        """)
        self.conn_profiles.commit()


    def profile_exists(self, player_id: int) -> bool:
        cur = self.conn_profiles.cursor()
        cur.execute("SELECT 1 FROM player_profiles WHERE player_id = ? LIMIT 1", (player_id,))
        return cur.fetchone() is not None

    @staticmethod
    def _choose_best_stat(stats_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not stats_list:
            return None
        best = None
        best_min = -1
        for s in stats_list:
            gm = (s.get("games") or {})
            mins = gm.get("minutes") or 0
            try: mins = int(mins)
            except Exception: mins = 0
            if mins > best_min:
                best_min = mins
                best = s
        return best or (stats_list[0] if stats_list else None)

    @staticmethod
    def _extract_profile(item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        p = item.get("player") or {}
        stats_list = item.get("statistics") or []
        best = APIFootballBulkDownloader._choose_best_stat(stats_list) or {}
        games = best.get("games") or {}
        birth = p.get("birth") or {}
        lg = best.get("league") or {}
        tm = best.get("team") or {}

        row = {
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
        return int(p.get("id")), row

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

    # -------------------------------------------------------------------
    # Internals: transfers helpers
    # -------------------------------------------------------------------
    def _upsert_transfers(self, player_id: int, transfers: List[Dict[str, Any]]) -> int:
        if not transfers:
            return 0
        c = self.conn_profiles.cursor()
        inserted = 0
        for tr in transfers:
            date = (tr.get("date") or tr.get("updated") or "")[:10] or None
            ttype = tr.get("type") or None
            teams = tr.get("teams") or {}
            tin = teams.get("in") or {}
            tout = teams.get("out") or {}
            to_id, to_name = tin.get("id"), tin.get("name")
            fr_id, fr_name = tout.get("id"), tout.get("name")
            fee = tr.get("fee") or tr.get("price") or (tr.get("transfers") if isinstance(tr.get("transfers"), str) else None)

            season_hint = None
            for k in ("season", "league_season"):
                v = tr.get(k)
                if isinstance(v, int):
                    season_hint = v
                    break

            raw = json.dumps(tr, ensure_ascii=False)
            try:
                c.execute("""
                    INSERT OR IGNORE INTO player_transfers (
                        player_id, transfer_date, type,
                        from_team_id, from_team_name,
                        to_team_id, to_team_name,
                        fee, season_hint, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (player_id, date, ttype, fr_id, fr_name, to_id, to_name, fee, season_hint, raw))
                if c.rowcount > 0:
                    inserted += 1
            except sqlite3.Error as e:
                print(f"    [warn] upsert transfer failed for player {player_id}: {e}", file=sys.stderr)
        self.conn_profiles.commit()
        return inserted

    def _get_all_player_ids(self) -> List[int]:
        cur = self.conn_profiles.cursor()
        cur.execute("SELECT DISTINCT player_id FROM player_profiles ORDER BY player_id")
        return [int(r[0]) for r in cur.fetchall()]

    def _get_most_recent_season(self) -> Optional[int]:
        cur = self.conn_profiles.cursor()
        cur.execute("SELECT MAX(season) FROM player_profiles")
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def _get_active_player_ids_in_most_recent_season(self) -> Set[int]:
        most_recent = self._get_most_recent_season()
        if most_recent is None:
            return set()
        cur = self.conn_profiles.cursor()
        cur.execute("SELECT DISTINCT player_id FROM player_profiles WHERE season = ?", (most_recent,))
        return {int(r[0]) for r in cur.fetchall()}

    def _has_fetched_transfers(self, player_id: int) -> bool:
        cur = self.conn_profiles.cursor()
        cur.execute("SELECT 1 FROM player_transfers_fetches WHERE player_id = ? LIMIT 1", (player_id,))
        return cur.fetchone() is not None

    def _mark_fetch_result(self, player_id: int, result: str) -> None:
        c = self.conn_profiles.cursor()
        c.execute("""
            INSERT INTO player_transfers_fetches (player_id, last_fetched_at, last_result)
            VALUES (?, strftime('%Y-%m-%dT%H:%M:%SZ','now'), ?)
            ON CONFLICT(player_id) DO UPDATE SET
              last_fetched_at=excluded.last_fetched_at,
              last_result=excluded.last_result
        """, (int(player_id), result[:200]))
        self.conn_profiles.commit()

    def _has_transfers_for_later_season(self, player_id: int, current_season: Optional[int]) -> bool:
        """Skip if player's transfers in DB carry a season_hint > current_season."""
        if current_season is None:
            return False
        cur = self.conn_profiles.cursor()
        cur.execute("SELECT MAX(COALESCE(season_hint, 0)) FROM player_transfers WHERE player_id = ?", (int(player_id),))
        row = cur.fetchone()
        try:
            max_hint = int(row[0]) if row and row[0] is not None else 0
        except Exception:
            max_hint = 0
        return max_hint > int(current_season)

    # -------------------------------------------------------------------
    # Orchestration helpers
    # -------------------------------------------------------------------
    def _profiles_fetch_or_backfill(self, league_id: int, players_json: dict) -> None:
        # Gather expected from minimal JSON
        expected: Set[int] = set()
        for lst in (players_json or {}).values():
            for p in lst:
                pid = p.get("id")
                if pid is not None:
                    expected.add(int(pid))

        # Add any referenced in analysis_results.db
        extra = self.find_missing_players_from_analysis_db(league_id=league_id,
                                                           league_name=self.league_name,
                                                           season=self.season)
        expected.update(extra)

        # Determine which are missing from player_profiles
        missing: List[int] = []
        cur = self.conn_profiles.cursor()
        for pid in sorted(expected):
            cur.execute("SELECT 1 FROM player_profiles WHERE player_id = ? LIMIT 1", (pid,))
            if cur.fetchone() is None:
                missing.append(pid)

        if not missing:
            self._vprint("[profiles] Nothing missing; verifying and returning.")
            try:
                self.verify_league_season(league_id, int(self.season))
            except Exception:
                pass
            return

        if len(missing) <= self.threshold_missing:
            self._vprint(f"[profiles] Small gap ({len(missing)} <= {self.threshold_missing}): per-ID backfill.")
            inserted = 0
            for pid in missing:
                try:
                    inserted += self.fetch_player_by_id(pid, season=self.season)
                    #inserted += self.fetch_player_by_id(pid)
                except GracefulExit:
                    raise
                except Exception as e:
                    print(f"  error fetching player_id={pid}: {e}", file=sys.stderr)
                time.sleep(self.request_sleep)
            self._vprint(f"[profiles] per-ID inserted: {inserted}")
        else:
            self._vprint(f"[profiles] Large gap ({len(missing)} > {self.threshold_missing}): bulk league-season fetch.")
            ins = self.fetch_league_season_profiles(league_id, int(self.season))
            self._vprint(f"[profiles] bulk inserted: {ins}")

        try:
            self.verify_league_season(league_id, int(self.season))
        except Exception:
            pass

    def verify_league_season(self, league_id: int, season: int) -> None:
        c = self.conn_profiles.cursor()
        c.execute("""SELECT COUNT(DISTINCT player_id),
                            COUNT(1),
                            COUNT(DISTINCT team_id)
                     FROM player_profiles WHERE league_id=? AND season=?""", (league_id, season))
        dp, rows, teams = c.fetchone() or (0, 0, 0)
        self._vprint(f"[verify] league={league_id}, season={season} -> players={dp}, rows={rows}, teams={teams}")

    # season helpers
    def _season_access_probe(self, league_id: int) -> bool:
        try:
            _ = self.api_get("/teams", {"league": league_id, "season": self.season})
            return True
        except GracefulExit:
            if self.exit_mgr.ctx and "do not have access" in str(getattr(self.exit_mgr.ctx, "detail", "")):
                return False
            raise

    def _fetch_teams_simple(self, league_id: int, season: int) -> List[Dict[str, Any]]:
        data = self.api_get("/teams", {"league": league_id, "season": season})
        teams = []
        for item in data.get("response") or []:
            t = item.get("team") or {}
            if t.get("id"):
                teams.append({"id": int(t["id"]), "name": t.get("name")})
        return teams

    # -------------------------------------------------------------------
    # HTTP, quota, JSON IO, context, utilities
    # -------------------------------------------------------------------
    def get_requests_left(self) -> int:
        j = self.api_get("/status", params=None, raw=True)
        data = j
        if data.get("errors") and any(data["errors"].values()):
            self.exit_mgr.exit("API error on /status", detail=str(data["errors"]), stage="status")
        req_info = data["response"]["requests"]
        left = int(req_info["limit_day"] - req_info["current"])
        self.requests_remaining = left
        if self._start_tokens is None:
            self._start_tokens = left
        return left

    def api_get(self,
                endpoint: str,
                params: Optional[Dict[str, Any]] = None,
                raw: bool = False,
                tolerate_429: bool = False) -> Dict[str, Any]:
        """
        GET wrapper with:
          - unified headers
          - optional 429 backoff
          - ratelimit header tracking
          - raises GracefulExit on HTTP or API errors
        """
        if self.requests_remaining is not None and self.requests_remaining <= 0:
            self.exit_mgr.exit("API quota exhausted", stage=f"GET {endpoint}",
                               requests_remaining=self.requests_remaining)

        url = f"{self.api_url}{endpoint}"
        while True:
            try:
                resp = self.session.get(url, headers=self.headers, params=params, timeout=self.REQUEST_TIMEOUT)
                if tolerate_429 and resp.status_code == 429:
                    time.sleep(self.backoff_429)
                    continue
                resp.raise_for_status()
                break
            except requests.HTTPError as e:
                self.exit_mgr.exit("HTTP error", detail=str(e), stage=f"GET {endpoint}",
                                   requests_remaining=self.requests_remaining)
            except requests.RequestException as e:
                self.exit_mgr.exit("Connection error", detail=str(e), stage=f"GET {endpoint}",
                                   requests_remaining=self.requests_remaining)

        # Update quota info
        rem = resp.headers.get("x-ratelimit-requests-remaining")
        if rem is not None:
            try:
                self.requests_remaining = int(rem)
            except Exception:
                pass
        rem_min = resp.headers.get("x-ratelimit-remaining")
        if rem is not None or rem_min is not None:
            if self.verbose:
                self._vprint(f"quota remaining: day={rem or '?'} minute={rem_min or '?'}", show_tokens=False)

        data = resp.json()
        if raw:
            return data

        if data.get("errors") and any(data["errors"].values()):
            self.exit_mgr.exit("API responded with errors", detail=str(data["errors"]),
                               stage=f"GET {endpoint}", requests_remaining=self.requests_remaining)

        if self.verbose and not raw and endpoint != "/status":
            self._vprint(f"GET {endpoint} ✓")
        return data

    # JSON file IO
    def _load_json(self, filename: str) -> dict:
        path = self._path(filename)
        return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else {}

    def _save_json(self, filename: str, data: dict) -> None:
        path = self._path(filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # minor utilities
    def _set_context(self, country: str, league: str, season: int) -> None:
        self.country_name = str(country)
        self.league_name = str(league)
        self.season = int(season)
        self.save_country_dir = f"{self.country_name}_{self.league_name}"
        self.save_info_dir = os.path.join(self.save_country_dir, f"{self.season}")
        os.makedirs(self.save_country_dir, exist_ok=True)
        os.makedirs(self.save_info_dir, exist_ok=True)
        self._dl_counts = {"teams": 0, "players_min": 0, "fixtures": 0}
        self._truncated_players_teams = []

    def _path(self, filename: str) -> str:
        return os.path.join(self.save_info_dir, filename)

    def _vprint(self, msg: str, show_tokens: bool = True) -> None:
        if not self.verbose:
            return
        if show_tokens and self.requests_remaining is not None:
            print(f"{msg}  [tokens left: {self.requests_remaining}]")
        else:
            print(msg)

    @staticmethod
    def _coerce_int(val) -> Optional[int]:
        if val is None:
            return None
        m = re.search(r'\d+', str(val))
        return int(m.group(0)) if m else None

    def _summarize(self) -> None:
        trunc = (
            f" | players truncated for {len(self._truncated_players_teams)} teams"
            if self._truncated_players_teams else ""
        )
        if self.verbose:
            self._vprint(
                f"📊 Summary for {self.country_name}/{self.league_name}/{self.season}: "
                f"{self._dl_counts['teams']} teams, "
                f"{self._dl_counts['players_min']} players(min), "
                f"{self._dl_counts['fixtures']} fixtures downloaded{trunc}",
                show_tokens=True,
            )
            
            used = None
            if (self._start_tokens is not None) and (self.requests_remaining is not None):
                used = self._start_tokens - self.requests_remaining
            if used is not None:
                self._vprint(f"🏁 Tokens used: {used}", show_tokens=True)


# =======================================================================
#                              __main__ demos
# =======================================================================
if __name__ == "__main__":

    API_KEY = "427b1bc85aa3a6a81fc63b43df0dbd55"
        
        
    dl = APIFootballBulkDownloader(
        api_key=API_KEY,
        use_rapidapi=False,
        verbose=True,
        debug=True,
        rate_limit_sec=10.0,
        max_player_pages=3,          # free-plan safety
        use_squads_fallback=True,    # helpful for minimal players JSON
        threshold_missing=30
    )

    # 1) Single season (transfers only if this is the most recent in DB)
    dl.process_league_season(
        country="Germany",
        league="Bundesliga",
        season=2023,
        fetch_transfers=True,
        transfers_only_for_most_recent_season=True,
        refresh_transfers=False
    )






        # Add priority first:
    targets = [
        ("Germany", "U19 Bundesliga", 2020),
        ("Germany", "U19 Bundesliga", 2021),
        ("Germany", "U19 Bundesliga", 2022),
        ("Germany", "U19 Bundesliga", 2023),
    ]
    
    # Then all unfinished seasons from the audit (Germany first, then others)
    targets += [
        ("Germany", "Bundesliga", 2014),
        ("Germany", "Bundesliga", 2015),
        ("Germany", "Bundesliga", 2016),
        ("Germany", "Bundesliga", 2017),
        ("Germany", "Bundesliga", 2018),
        ("Germany", "Bundesliga", 2019),
        ("Germany", "Bundesliga", 2020),
        ("Germany", "2. Bundesliga", 2021),
        ("Germany", "2. Bundesliga", 2022),
        ("Germany", "2. Bundesliga", 2023),
        ("Germany", "2. Bundesliga", 2024),
        ("Germany", "Bundesliga", 2024),
    
        ("Austria", "Bundesliga", 2021),
        ("Austria", "Bundesliga", 2022),
        ("Austria", "Bundesliga", 2023),
        ("Austria", "Bundesliga", 2024),
    
        ("Belgium", "Jupiler Pro League", 2021),
        ("Belgium", "Jupiler Pro League", 2022),
        ("Belgium", "Jupiler Pro League", 2023),
        ("Belgium", "Jupiler Pro League", 2024),
    
        ("Denmark", "Superliga", 2021),
        ("Denmark", "Superliga", 2022),
        ("Denmark", "Superliga", 2023),
        ("Denmark", "Superliga", 2024),
    
        ("England", "Championship", 2021),
        ("England", "Championship", 2022),
        ("England", "Championship", 2023),
        ("England", "Championship", 2024),
        ("England", "Premier League", 2014),
        ("England", "Premier League", 2015),
        ("England", "Premier League", 2016),
        ("England", "Premier League", 2017),
        ("England", "Premier League", 2018),
        ("England", "Premier League", 2019),
        ("England", "Premier League", 2020),
        ("England", "Premier League", 2021),
        ("England", "Premier League", 2024),
    
        ("France", "Ligue 2", 2021),
        ("France", "Ligue 2", 2022),
        ("France", "Ligue 2", 2023),
        ("France", "Ligue 2", 2024),
    
        ("Italy", "Serie A", 2021),
        ("Italy", "Serie A", 2022),
        ("Italy", "Serie A", 2023),
        ("Italy", "Serie B", 2021),
        ("Italy", "Serie B", 2022),
        ("Italy", "Serie B", 2023),
        ("Italy", "Serie B", 2024),
    
        ("Mexico", "Liga MX", 2020),
        ("Mexico", "Liga MX", 2021),
        ("Mexico", "Liga MX", 2022),
        ("Mexico", "Liga MX", 2023),
        ("Mexico", "Liga MX", 2024),
    
        ("Netherlands", "Eredivisie", 2020),
        ("Netherlands", "Eredivisie", 2021),
        ("Netherlands", "Eredivisie", 2022),
        ("Netherlands", "Eredivisie", 2024),
    
        ("Norway", "Eliteserien", 2021),
        ("Norway", "Eliteserien", 2022),
        ("Norway", "Eliteserien", 2023),
        ("Norway", "Eliteserien", 2024),
    
        ("Portugal", "Primeira Liga", 2014),
        ("Portugal", "Primeira Liga", 2015),
        ("Portugal", "Primeira Liga", 2016),
        ("Portugal", "Primeira Liga", 2017),
        ("Portugal", "Primeira Liga", 2018),
        ("Portugal", "Primeira Liga", 2019),
        ("Portugal", "Primeira Liga", 2020),
        ("Portugal", "Primeira Liga", 2021),
        ("Portugal", "Primeira Liga", 2022),
        ("Portugal", "Primeira Liga", 2023),
        ("Portugal", "Primeira Liga", 2024),
    
        ("Spain", "La Liga", 2014),
        ("Spain", "La Liga", 2015),
        ("Spain", "La Liga", 2016),
        ("Spain", "La Liga", 2017),
        ("Spain", "La Liga", 2018),
        ("Spain", "La Liga", 2019),
        ("Spain", "La Liga", 2020),
        ("Spain", "La Liga", 2021),
        ("Spain", "La Liga", 2024),
        ("Spain", "Segunda División", 2021),
        ("Spain", "Segunda División", 2022),
        ("Spain", "Segunda División", 2023),
        ("Spain", "Segunda División", 2024),
    
        ("Sweden", "Allsvenskan", 2021),
        ("Sweden", "Allsvenskan", 2022),
        ("Sweden", "Allsvenskan", 2023),
        ("Sweden", "Allsvenskan", 2024),
    
        ("Switzerland", "Super League", 2021),
        ("Switzerland", "Super League", 2022),
        ("Switzerland", "Super League", 2023),
        ("Switzerland", "Super League", 2024),
    ]
    
    # Run them all
    for country, league, season in targets:
        print(f"\n=== Running {country} / {league} / {season} ===")
        dl.process_league_season(
            country=country,
            league=league,
            season=season,
            fetch_transfers=True,
            transfers_only_for_most_recent_season=True,
            refresh_transfers=False
        )

    '''
    # 2) Loop seasons (transfers on latest only)
    seasons = [2023, 2022, 2021]
    for s in seasons:
        dl.process_league_season(
            country="Italy",
            league="Serie A",
            season=s,
            fetch_transfers=(s == max(seasons)),
            transfers_only_for_most_recent_season=True,
            refresh_transfers=False
        )
        time.sleep(2)

    # 3) Players-only for a league/season
    dl._set_context("Switzerland", "Super League", 2023)  # sets storage paths
    lg_id = dl.fetch_league_id("Switzerland", "Super League")
    ins_bulk = dl.fetch_players_for_league_season(lg_id, 2023)
    print(f"[players-only] bulk inserted: {ins_bulk}")
    ins_backfill = dl.fetch_missing_players(league_id=lg_id, season=2023)
    print(f"[players-only] backfilled (per-ID): {ins_backfill}")

    # 4) Per-player quick actions
    # for pid in [276, 874, 12345]:  # replace with real IDs
    #     # dl.fetch_player_by_id(pid)
    #     dl.fetch_transfers_for_player(pid)
    # # for pid in [276, 874, 12345]:
    #     dl.fetch_player_by_id(pid, season=2023)   # or your current job season


    # 5) Refresh transfers for active players only
    dl.backfill_transfers(force_refresh=True, refresh_active_only=True, limit=None)


    # Cleanly close the SQLite connection at the end
    try:
        dl.close()
    except Exception:
        pass
    '''

        