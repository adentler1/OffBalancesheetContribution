#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified APIFootballBulkDownloader

Merges the functionality of:
  - SeasonFootballDataFetcher (season -> teams/players(list)/fixtures/events to JSON)
  - PlayerFootballDataFetcher (players -> SQLite; transfers -> SQLite, with fallback logic)

Key properties:
  - One prominent pipeline: process_league_season(country, league, season, ...)
  - Player fallback preserved: league-season bulk -> fallback to team-wise -> fallback to per-ID
  - Transfer fetch is optional; can be limited to the most recent season only (boolean)
  - Skips work already in DB (profiles/transfers). Extra skip: if transfers fetched for a later season, skip older.
  - Exposes targeted methods for per-season and per-player actions (profiles/transfers)
  - Default request sleep = 10s. Handles 429 backoff. Tracks quota.

External, subdued helpers (kept in separate modules as you requested):
  - GracefulExit, GracefulExitManager

Dependencies:
  pip install requests

SQLite files:
  - player_profiles.db   (table player_profiles)
  - player_transfers.*   (tables player_transfers, player_transfers_fetches)  [stored in same DB]

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
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# ----------------------------- Logging setup -----------------------------
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "football_fetcher.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("football")

# External helpers (subdued)
try:
    from GracefulExit import GracefulExit, GracefulExitManager
except Exception:  # Fallback if not present; pipeline still works with basic exceptions
    class GracefulExit(Exception):
        pass
    class GracefulExitManager:
        def __init__(self): self.ctx = None
        def exit(self, reason: str, **kw): 
            self.ctx = type("Ctx", (), {"reason": reason, **kw})
            raise GracefulExit(reason)
        def handle(self, save_progress_callable=None):
            if save_progress_callable: 
                try: save_progress_callable()
                except Exception: pass


class APIFootballBulkDownloader:
    """
    Unified downloader for API-FOOTBALL v3.

    Public, prominent pipeline:
      - process_league_season(country, league, season, fetch_transfers=..., ...)

    Also exposed, targeted methods:
      - fetch_league_id(country, league)
      - fetch_players_for_league_season(league_id, season)   # profiles -> DB (bulk with fallback)
      - fetch_missing_players(league_id=None, league_name=None, season=None, limit=None)
      - fetch_player_by_id(player_id)                        # profile -> DB
      - fetch_transfers_for_player(player_id, overwrite=False)
      - backfill_transfers(force_refresh=False, refresh_active_only=True, limit=None)

    Transfer fetch controls:
      - fetch_transfers (bool)
      - transfers_only_for_most_recent_season (bool): if True, only run transfers when 'season' is the DB's MAX(season)
      - refresh_transfers (bool): if True, re-fetch transfers for players (restricted by refresh_active_only)
      - Skip-by-default logic: if transfers exist and a later season is present for that player, skip.

    Default sleep between requests is 10 seconds.
    """

    # API config
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
        # transfers share the same DB file
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

        # Outputs
        self._dl_counts = {"teams": 0, "players_min": 0, "fixtures": 0}

        # Files
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

        self._truncated_players_teams: List[str] = []  # for reporting

    # ---------------------------------------------------------------------
    # Prominent public pipeline
    # ---------------------------------------------------------------------
    def process_league_season(self,
                              country: str,
                              league: str,
                              season: int,
                              *,
                              fetch_transfers: bool = False,
                              transfers_only_for_most_recent_season: bool = True,
                              refresh_transfers: bool = False,
                              refresh_active_only: bool = True) -> None:
        """
        Full pipeline for one (country, league, season):

          1) Season JSON: league_id -> teams -> (minimal) players per team -> fixtures -> events
          2) Player profiles -> SQLite (bulk with fallback, or per-ID if small missing set)
          3) Transfers -> SQLite (optional; can be limited to most recent season only)

        Skips DB duplicates. Player fallback preserved.
        """
        try:
            self._set_context(country, league, season)

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

            # --- player profiles (SQLite): missing -> bulk or per-id ---
            self._profiles_fetch_or_backfill(league_id, players_json)

            # --- transfers (SQLite) ---
            if fetch_transfers:
                if transfers_only_for_most_recent_season:
                    most_recent = self._get_most_recent_season()
                    most_recent = most_recent if most_recent is not None else int(self.season)
                    if int(self.season) != int(most_recent):
                        self._vprint(f"↪️ Skipping transfers: season {self.season} != most_recent {most_recent}.")
                    else:
                        self.backfill_transfers(force_refresh=refresh_transfers,
                                                refresh_active_only=refresh_active_only)
                else:
                    self.backfill_transfers(force_refresh=refresh_transfers,
                                            refresh_active_only=refresh_active_only)

            # --- Summary ---
            self._summarize()

        except GracefulExit:
            LOGGER.info("Graceful exit: %s", getattr(self.exit_mgr, "ctx", None))
            self._vprint(f"GracefulExit: {getattr(self.exit_mgr, 'ctx', None)}")
            # No Excel progress to update; nothing else to persist here.

        except Exception as e:
            if self.debug:
                traceback.print_exc()
            LOGGER.error("Unexpected error: %s", e)
            print(f"❌ Unexpected error: {e}")

    # ---------------------------------------------------------------------
    # Season: JSON pipeline (teams/players-min/fixtures/events)
    # ---------------------------------------------------------------------
    def fetch_league_id(self, country: str, league: str) -> int:
        """Resolve league_id from country+league names (cached to league.json in country dir)."""
        path = os.path.join(self.save_country_dir, self.files["league"])
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

    def _fetch_teams_json(self, league_id: int) -> Tuple[dict, dict]:
        teams_data = self._load_json(self.files["teams"])
        teams_status = self._load_json(self.files["teams_status"])

        if not teams_data:
            d = self.api_get("/teams", {"league": league_id, "season": self.season})
            if d.get("results", 0) == 0 or not d.get("response"):
                self.exit_mgr.exit("No team data (empty)", stage="teams")
            teams_data = d
            teams_status = {t["team"]["name"]: False for t in teams_data["response"]}

        for t in teams_data.get("response", []):
            n = t["team"]["name"]
            if teams_status.get(n):
                continue
            if self.verbose:
                self._vprint(f"📥 Downloaded team: {n}")
            teams_status[n] = True
            self._dl_counts["teams"] += 1
            self._save_json(self.files["teams_status"], teams_status)
            time.sleep(self.request_sleep)

        self._save_json(self.files["teams"], teams_data)
        self._save_json(self.files["teams_status"], teams_status)
        return teams_data, teams_status

    def _fetch_players_minimal_json(self, teams_data: dict) -> Tuple[dict, dict]:
        """Minimal per-team player list to JSON (for season context and completeness)."""
        players_data = self._load_json(self.files["players"])
        players_status = self._load_json(self.files["players_status"])
        if not players_status and teams_data:
            players_status = {t["team"]["name"]: False for t in teams_data["response"]}

        for t in teams_data.get("response", []):
            name = t["team"]["name"]
            team_id = t["team"]["id"]
            if players_status.get(name):
                continue
            if self.verbose:
                self._vprint(f"📥 Downloading players (minimal) for team: {name}")

            resp = self.api_get("/players", {"team": team_id, "season": self.season})
            if resp.get("results", 0) == 0 or not resp.get("response"):
                players_data[name] = []
                players_status[name] = True
                self._save_json(self.files["players"], players_data)
                self._save_json(self.files["players_status"], players_status)
                continue

            players_list, seen_ids = [], set()
            # page 1
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
            if self.max_player_pages is not None and total_pages > self.max_player_pages and self.verbose:
                self._vprint(f"⚠️  Page-cap hit for {name}: fetching only {pages_to_fetch}/{total_pages}.")
                self._truncated_players_teams.append(name)

            # next pages
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

            # optional squads fallback
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
                    if self.verbose:
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

        return players_data, players_status

    def _fetch_fixtures_json(self, league_id: int) -> List[int]:
        fixtures_data = self._load_json(self.files["fixtures"])
        if not fixtures_data:
            d = self.api_get("/fixtures", {"league": league_id, "season": self.season})
            if d.get("results", 0) == 0 or not d.get("response"):
                self._save_json(self.files["fixtures"], d)
                return []
            self._save_json(self.files["fixtures"], d)
            fixtures_data = d
        return [f["fixture"]["id"] for f in fixtures_data.get("response", [])]

    def _fetch_fixture_events_json(self, fixture_ids: List[int]) -> Tuple[dict, dict]:
        match_events = self._load_json(self.files["match_events"])
        fixtures_status = self._load_json(self.files["fixtures_status"])

        for fid in fixture_ids:
            key = str(fid)
            if key in match_events and match_events[key].get("lineups") and match_events[key].get("events"):
                continue

            lineup = self.api_get("/fixtures/lineups", {"fixture": fid})
            if self.verbose:
                self._vprint(f"📥 Downloading fixture events for fixture {fid}")
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

    # ---------------------------------------------------------------------
    # Player profiles (SQLite): bulk, fallback, per-id, and missing detection
    # ---------------------------------------------------------------------
    def fetch_league_season_profiles(self, league_id: int, season: int) -> int:
        """
        Bulk fetch player profiles via /players?league=...&season=... with fallback to team-wise fetch.
        Returns number of new profiles inserted.
        """
        new_profiles = 0
        page = 1
        expected_total = None
        empty_retries, MAX_EMPTY_RETRIES = 0, 3

        while True:
            self._vprint(f"[profiles] league={league_id}, season={season}: page {page}...")
            try:
                data = self.api_get("/players",
                    {"league": league_id, "season": season, "page": page},
                    tolerate_429=True, allow_api_errors=True)
                # data = self.api_get(
                #     "/players",
                #     {"league": league_id, "season": season, "page": page},
                #     tolerate_429=True
                # )
            except GracefulExit:
                # If it is the free-plan page-cap, fall back to team-wise fetch.
                ctx = getattr(self.exit_mgr, "ctx", None)
                detail = (getattr(ctx, "detail", "") or "").lower()
                if "maximum value of 3 for the page parameter" in detail:
                    self._vprint("  → Free plan page cap hit. Falling back to team-wise fetch.")
                    return self._fetch_league_season_by_team(league_id, season)
                raise  # not a plan-cap: bubble up
        
            errs = data.get("errors") or {}
            # Defensive: some tenants send it as a string in errors['plan'] without triggering HTTP error
            if "plan" in errs and "maximum value of 3 for the Page parameter" in str(errs.get("plan", "")):
                self._vprint("  → Free plan page cap hit. Falling back to team-wise fetch.")
                return self._fetch_league_season_by_team(league_id, season)
        
            pg = data.get("paging") or {}
            ...

            # free-plan cap -> fallback
            if "plan" in errs and "maximum value of 3 for the Page parameter" in str(errs.get("plan", "")):
                self._vprint("  → Free plan page cap hit. Falling back to team-wise fetch.")
                return self._fetch_league_season_by_team(league_id, season)

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

    # Public helpers
    def fetch_players_for_league_season(self, league_id: int, season: int) -> int:
        """Public wrapper to bulk-insert player profiles for (league_id, season) with fallback."""
        return self.fetch_league_season_profiles(league_id, season)

    def fetch_missing_players(self,
                              limit: Optional[int] = None,
                              league_id: Optional[int] = None,
                              league_name: Optional[str] = None,
                              season: Optional[int] = None) -> int:
        """Backfill players that appear in analysis_results.db but not in player_profiles.db."""
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
                inserted += self.fetch_player_by_id(pid)
            except GracefulExit:
                raise
            except Exception as e:
                print(f"  error fetching player_id={pid}: {e}", file=sys.stderr)
            time.sleep(self.request_sleep)
        return inserted

    def fetch_player_by_id(self, player_id: int) -> int:
        """Fetch a single player's profile by ID and insert if missing. Returns rows inserted (0/1+)."""
        self._vprint(f"\n== fetching player_id={player_id} ==")
        data = self.api_get("/players", {"id": int(player_id)}, tolerate_429=True)
        items = data.get("response") or []
        if not items:
            self._vprint("  No data for this player_id.")
            return 0
        inserted = 0
        for item in items:
            try:
                pid, row = self._extract_profile(item)
                if not self.profile_exists(pid):
                    self._insert_profile(row)
                    inserted += 1
            except Exception as e:
                print(f"  error processing player {player_id}: {e}", file=sys.stderr)
        return inserted

    # ---------------------------------------------------------------------
    # Transfers (SQLite): single-player and orchestrated backfill
    # ---------------------------------------------------------------------
    def fetch_transfers_for_player(self, player_id: int, *, overwrite: bool = False) -> int:
        """
        Fetch transfers for one player and upsert into DB.
        If overwrite=True, delete existing rows first.
        Returns #rows inserted.
        """
        if overwrite:
            c = self.conn_profiles.cursor()
            c.execute("DELETE FROM player_transfers WHERE player_id = ?", (int(player_id),))
            self.conn_profiles.commit()

        data = self.api_get("/transfers", {"player": int(player_id)}, tolerate_429=True)
        resp = data.get("response") or []
        transfers = []
        for item in resp:
            arr = item.get("transfers") or []
            if isinstance(arr, list):
                transfers.extend(arr)

        inserted = self._upsert_transfers(player_id, transfers)
        self._mark_fetch_result(player_id, "ok" if transfers else "empty")
        return inserted

    def backfill_transfers(self,
                           *,
                           force_refresh: bool = False,
                           refresh_active_only: bool = True,
                           limit: Optional[int] = None) -> Tuple[int, int, int]:
        """
        Fetch each player's transfer history once; refresh only if requested.
        Skip by default if:
          - transfers fetched already (player_transfers_fetches has a row), OR
          - player has transfers with season_hint later than the current season (avoid older-season duplication).
        Returns: (first_time, refreshed, skipped)
        """
        all_pids = self._get_all_player_ids()
        if limit is not None:
            all_pids = all_pids[:int(limit)]
        active = self._get_active_player_ids_in_most_recent_season() if refresh_active_only else set()
        current_season = self._get_most_recent_season()  # used for "later season" skip heuristic

        first_time, refreshed, skipped = 0, 0, 0
        self._vprint(f"[transfers] players total={len(all_pids)}, active_in_most_recent={len(active)}")

        for i, pid in enumerate(all_pids, 1):
            try:
                # Skip if transfers fetched for a later season already
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

    # ---------------------------------------------------------------------
    # Missing players logic (from analysis_results.db)
    # ---------------------------------------------------------------------
    def find_missing_players_from_analysis_db(self,
                                              league_id: Optional[int] = None,
                                              league_name: Optional[str] = None,
                                              season: Optional[int] = None) -> List[int]:
        """Scan analysis_results.db for player_id not present in player_profiles."""
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

        # remove those already present
        missing: Set[int] = set()
        cur_out = self.conn_profiles.cursor()
        for pid in pids:
            cur_out.execute("SELECT 1 FROM player_profiles WHERE player_id = ? LIMIT 1", (pid,))
            if cur_out.fetchone() is None:
                missing.add(pid)
        return sorted(missing)

    # ---------------------------------------------------------------------
    # Internals: profile extraction & DB schema
    # ---------------------------------------------------------------------
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
            try:
                mins = int(mins)
            except Exception:
                mins = 0
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

    # ---------------------------------------------------------------------
    # Internals: transfers table helpers
    # ---------------------------------------------------------------------
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

            # optional season hint if present
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
        """
        Skip heuristic: if we already have this player's transfers recorded with a season_hint
        greater than 'current_season', skip fetching for an older season run.
        """
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

    # ---------------------------------------------------------------------
    # Internals: orchestration helpers
    # ---------------------------------------------------------------------
    def _profiles_fetch_or_backfill(self, league_id: int, players_json: dict) -> None:
        """Decide bulk vs per-ID fetch based on missing profiles."""
        # gather expected from minimal JSON
        expected: Set[int] = set()
        for lst in (players_json or {}).values():
            for p in lst:
                pid = p.get("id")
                if pid is not None:
                    expected.add(int(pid))

        # add those referenced in analysis_results.db (filtered by this league/season)
        extra = self.find_missing_players_from_analysis_db(league_id=league_id,
                                                           league_name=self.league_name,
                                                           season=self.season)
        expected.update(extra)

        # find which are missing from player_profiles
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
                    inserted += self.fetch_player_by_id(pid)
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

    # ---------------------------------------------------------------------
    # HTTP, quota, JSON IO, context, utilities
    # ---------------------------------------------------------------------
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


    def api_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
                raw: bool = False, tolerate_429: bool = False,
                allow_api_errors: bool = False) -> Dict[str, Any]:
    # def api_get(self,
    #             endpoint: str,
    #             params: Optional[Dict[str, Any]] = None,
    #             raw: bool = False,
    #             tolerate_429: bool = False) -> Dict[str, Any]:
        """
        GET wrapper with:
          - unified headers (RapidAPI or direct)
          - 429 backoff (if tolerate_429=True)
          - ratelimit header tracking
          - GracefulExit on HTTP / API errors
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

        # Update quota info from headers if available
        rem = resp.headers.get("x-ratelimit-requests-remaining")
        if rem is not None:
            try:
                self.requests_remaining = int(rem)
            except Exception:
                pass
        # minute-based header (optional)
        rem_min = resp.headers.get("x-ratelimit-remaining")
        if rem is not None or rem_min is not None:
            if self.verbose:
                self._vprint(f"quota remaining: day={rem or '?'} minute={rem_min or '?'}", show_tokens=False)

        data = resp.json()
        if raw:
            return data
    
        if data.get("errors") and any(data["errors"].values()):
            if allow_api_errors:
                # let caller inspect data['errors'] for fallback decisions
                return data
            self.exit_mgr.exit("API responded with errors", detail=str(data["errors"]),
                               stage=f"GET {endpoint}", requests_remaining=self.requests_remaining)
        # data = resp.json()
        # if raw:
        #     return data

        # if data.get("errors") and any(data["errors"].values()):
        #     self.exit_mgr.exit("API responded with errors", detail=str(data["errors"]),
        #                        stage=f"GET {endpoint}", requests_remaining=self.requests_remaining)
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
                f"📊 Summary: {self._dl_counts['teams']} teams, "
                f"{self._dl_counts['players_min']} players(min), "
                f"{self._dl_counts['fixtures']} fixtures downloaded{trunc}",
                show_tokens=True,
            )
            used = None
            if (self._start_tokens is not None) and (self.requests_remaining is not None):
                used = self._start_tokens - self.requests_remaining
            if used is not None:
                self._vprint(f"🏁 Tokens used: {used}", show_tokens=True)



if __name__ == "__main__":
    import os
    API_KEY = "427b1bc85aa3a6a81fc63b43df0dbd55"

    # One reusable downloader instance (10s wait between calls by default)
    dl = APIFootballBulkDownloader(
        api_key=API_KEY,
        use_rapidapi=False,          # set True if you use RapidAPI headers
        verbose=True,
        debug=True,
        rate_limit_sec=10.0,         # default; keep conservative
        max_player_pages=3,          # free-plan safety
        use_squads_fallback=True,    # helpful when page-capped
        threshold_missing=30         # per-ID vs bulk pivot for profiles
    )

    # ------------------------------------------------------------------
    # 1) Single full-season pipeline (with transfers only if it's current)
    # ------------------------------------------------------------------
    dl.process_league_season(
        country="Germany",
        league="Bundesliga",
        season=2023,
        fetch_transfers=True,                         # attempt transfers
        transfers_only_for_most_recent_season=True,   # only if 2024 is DB's latest season
        refresh_transfers=False                       # don't force refresh
    )

    # ------------------------------------------------------------------
    # 2) Full-season *loop* for one league (transfers only on latest)
    # ------------------------------------------------------------------
    seasons = [2023, 2022, 2021]
    for s in seasons:
        dl.process_league_season(
            country="Italy",
            league="Serie A",
            season=s,
            fetch_transfers=(s == max(seasons)),          # run transfers only once on most recent
            transfers_only_for_most_recent_season=True,
            refresh_transfers=False
        )
        # optional small pause between seasons (API comfort)
        time.sleep(2)

    # ------------------------------------------------------------------
    # 3) Multiple leagues × seasons (common batch)
    #    Transfers only for the most recent season in each league.
    # ------------------------------------------------------------------
    jobs = [
        # already on your list
        ("Netherlands", "Eredivisie",   [2023, 2022]),
        ("Portugal", "Primeira Liga",   [2023, 2022]),
        ("Mexico", "Liga MX",   [2023, 2022]),
    
        # strong feeders (top tiers)
        ("Belgium", "Jupiler Pro League",   [2023, 2022]),        # African & South American pipeline
        ("Austria", "Bundesliga",   [2023, 2022]),                # RB Salzburg, German loans
        ("Switzerland", "Super League",   [2023, 2022]),          # stepping-stone for DACH/Balkans
        ("Denmark", "Superliga",   [2023, 2022]),                 # exports to Bundesliga & PL
        ("Norway", "Eliteserien",   [2023, 2022]),                # Ødegaard path, Premier League attention
        ("Sweden", "Allsvenskan",   [2023, 2022]),                # many Bundesliga/Netherlands links
    
        # important second leagues
        ("England", "Championship",   [2023, 2022]),              # extremely competitive, many loans from PL
        ("Spain", "Segunda División",   [2023, 2022]),            # La Liga clubs loan out prospects here
        ("Germany", "2. Bundesliga",   [2023, 2022]),             # crucial for Bundesliga player development
        ("Italy", "Serie B",   [2023, 2022]),                     # used for Serie A loanees and reclamations
        ("France", "Ligue 2",   [2023, 2022]),                    # big role in feeding Ligue 1 & exporting abroad
    ]
    # jobs = [
    #     ("Spain", "La Liga",   [2024, 2023]),
    #     ("Italy", "Serie A",   [2024, 2023]),
    # ]
    for country, league, seas in jobs:
        for s in seas:
            dl.process_league_season(
                country=country,
                league=league,
                season=s,
                fetch_transfers=False,#(s == max(seas)),
                transfers_only_for_most_recent_season=True,
                refresh_transfers=False
            )

    # ------------------------------------------------------------------
    # 4) Players-only for a league/season (profiles -> SQLite),
    #    plus per-ID backfill from analysis_results.db if needed.
    #    (Calls the bulk fetcher directly, keeping season JSON out.)
    # ------------------------------------------------------------------
    # NOTE: fetch_league_id() writes into a league.json path under the
    # current context; set context before calling it.
    dl._set_context("Switzerland", "Super League", 2023)  # internal but safe here
    lg_id = dl.fetch_league_id("Switzerland", "Super League")
    inserted_bulk = dl.fetch_players_for_league_season(lg_id, 2023)
    print(f"[players-only] bulk inserted: {inserted_bulk}")
    # Top up any stragglers referenced in analysis_results.db:
    inserted_backfill = dl.fetch_missing_players(league_id=lg_id, season=2023)
    print(f"[players-only] backfilled (per-ID): {inserted_backfill}")

    # ------------------------------------------------------------------
    # 5) Per-player actions (quick surgical ops)
    # ------------------------------------------------------------------
    sample_player_ids = [276, 874, 12345]  # replace with real IDs
    for pid in sample_player_ids:
        dl.fetch_player_by_id(pid)              # profile -> SQLite
        dl.fetch_transfers_for_player(pid)      # transfers -> SQLite

    # ------------------------------------------------------------------
    # 6) Refresh transfers for *active* players only (latest season),
    #    e.g., after a transfer window closed.
    # ------------------------------------------------------------------
    dl.backfill_transfers(
        force_refresh=True,          # re-download for eligible players
        refresh_active_only=True,    # limit to players seen in MAX(season)
        limit=None                   # or cap e.g. 500
    )
