#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored SeasonFootballDataFetcher with:
1) Graceful exits (clear reasons; logging; consistent cleanup)
2) XLS progress tracking (downloads_progress.xlsx)
3) Available catalog export (available_leagues.xlsx)

Dependencies:
    pip install requests pandas openpyxl

Environment:
    export API_FOOTBALL_KEY="YOUR_KEY"
    
Bulk player profiles + transfers downloader for API-FOOTBALL (v3).

- Profiles -> SQLite: player_profiles.db
- Transfers -> SQLite: player_transfers.db
- Source of league/season pairs: an Excel file "downloads_progress.xlsx"
  (rows with overall_status == 'Completed').

Implements:
  class APIFootballBulkDownloader
    - process_league_season(league_id, season, fetch_transfers=True, refresh_transfers=False)
    - run_for_pairs(pairs, fetch_transfers=True, refresh_transfers=False)

Author: you :)

Author: you + ChatGPT
Date: 2025-08-27
"""

import os
import sys
import time
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd



# ----------------------------- Logging setup -----------------------------
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "football_fetcher.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("football")

from GracefulExit import GracefulExit, GracefulExitManager
from ProgressTracker import ProgressTracker



# ----------------------------- Main Fetcher -------------------------------
class SeasonFootballDataFetcher:
    def __init__(self, api_key: str, league_name: str, country_name: str, season: int,
                 rate_limit_sec: float = 10.0, debug: bool = False, verbose: bool = True,
                 max_player_pages: Optional[int] = 3, use_squads_fallback: bool = False):
 
        self.api_key = api_key
        self.league_name = league_name
        self.country_name = country_name
        self.season = int(season)
        self.api_url = "https://v3.football.api-sports.io"
        self.requests_remaining: Optional[int] = None
        self.rate_delay = float(rate_limit_sec)
        self.debug = debug   # <-- new flag (default False)
        self.verbose = verbose
        self._dl_counts = {"teams": 0, "players": 0, "fixtures": 0}
        # --- add near other attrs in __init__ ---
        self._start_tokens: Optional[int] = None
        
        self.max_player_pages = int(max_player_pages) if max_player_pages is not None else None
        self.use_squads_fallback = bool(use_squads_fallback)
        self._truncated_players_teams = []   # track teams where we hit the cap
        
        self.save_country_dir = f"{self.country_name}_{self.league_name}"
        self.save_info_dir = os.path.join(self.save_country_dir, f"{self.season}")
        os.makedirs(self.save_country_dir, exist_ok=True)
        os.makedirs(self.save_info_dir, exist_ok=True)

        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "v3.football.api-sports.io",
        }

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

        # Progress tracking
        self.progress = ProgressTracker()
        self.progress.ensure_pending(self.country_name, self.league_name, self.season)

        # Exit manager
        self.exit_mgr = GracefulExitManager()

    # --------- Utility: file IO ----------
    # --- add this helper anywhere in the class ---
    def _vprint(self, msg: str, show_tokens: bool = True):
        """Verbose print with optional token suffix."""
        if not self.verbose:
            return
        if show_tokens and self.requests_remaining is not None:
            print(f"{msg}  [tokens left: {self.requests_remaining}]")
        else:
            print(msg)
    def _path(self, filename: str) -> str:
        return os.path.join(self.save_info_dir, filename)

    def load_json(self, filename: str) -> dict:
        path = self._path(filename)
        return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else {}

    def save_json(self, filename: str, data: dict):
        path = self._path(filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # --------- HTTP helpers ----------
    def get_requests_left(self) -> int:
        try:
            j = self.api_get("/status", params=None, raw=True)
            data = j
            if data.get("errors") and any(data["errors"].values()):
                self.exit_mgr.exit("API error on /status", detail=str(data["errors"]), stage="status")
            req_info = data["response"]["requests"]
            left = int(req_info["limit_day"] - req_info["current"])
            # set both current and baseline
            self.requests_remaining = left
            if self._start_tokens is None:
                self._start_tokens = left
            return left
        except requests.RequestException as e:
            self.exit_mgr.exit("Connection error on /status", detail=str(e), stage="status")
        except Exception as e:
            self.exit_mgr.exit("Unexpected response from /status", detail=str(e), stage="status")

    #         self.exit_mgr.exit("Unexpected response from /status", detail=str(e), stage="status")

    def api_get(self, endpoint: str, params: Optional[Dict] = None, raw: bool = False) -> Dict:
        if self.requests_remaining is not None and self.requests_remaining <= 0:
            self.exit_mgr.exit("API quota exhausted", stage=f"GET {endpoint}",
                               requests_remaining=self.requests_remaining)
    
        url = f"{self.api_url}{endpoint}"
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=45)
            resp.raise_for_status()
        except requests.HTTPError as e:
            self.exit_mgr.exit("HTTP error", detail=str(e), stage=f"GET {endpoint}",
                               requests_remaining=self.requests_remaining)
        except requests.RequestException as e:
            self.exit_mgr.exit("Connection error", detail=str(e), stage=f"GET {endpoint}",
                               requests_remaining=self.requests_remaining)
    
        # Update from headers if available
        rem = resp.headers.get("x-ratelimit-requests-remaining")
        if rem is not None:
            try:
                self.requests_remaining = int(rem)
            except Exception:
                pass
    
        # Light per-call verbose line (omit for /status to avoid duplication)
        if self.verbose and not raw:
            # keep it one short line to not flood logs
            self._vprint(f"GET {endpoint} ✓")
    
        data = resp.json()
        if raw:
            return data
    
        if data.get("errors") and any(data["errors"].values()):
            self.exit_mgr.exit("API responded with errors", detail=str(data["errors"]),
                               stage=f"GET {endpoint}", requests_remaining=self.requests_remaining)
    
        return data

    # --------- League helpers ----------
    def fetch_league_data(self) -> Optional[dict]:
        path = os.path.join(self.save_country_dir, self.files["league"])
        if os.path.exists(path):
            return json.load(open(path, "r", encoding="utf-8"))

        data = self.api_get("/leagues", {"name": self.league_name, "country": self.country_name})
        if data.get("results", 0) == 0 or not data.get("response"):
            self.exit_mgr.exit("Invalid league or empty response", stage="leagues")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    def fetch_league_id(self) -> int:
        data = self.fetch_league_data()
        try:
            return data["response"][0]["league"]["id"]
        except Exception as e:
            self.exit_mgr.exit("Could not extract league_id", detail=str(e), stage="leagues")

    # --------- Teams ----------
    def fetch_teams(self, league_id: int) -> Tuple[dict, dict]:
        teams_data = self.load_json(self.files["teams"])
        teams_status = self.load_json(self.files["teams_status"])

        if not teams_data:
            data = self.api_get("/teams", {"league": league_id, "season": self.season})
            if data.get("results", 0) == 0 or not data.get("response"):
                self.exit_mgr.exit("No team data (empty)", stage="teams")
            teams_data = data
            teams_status = {t["team"]["name"]: False for t in teams_data["response"]}

        # Simulate "processing" teams (your original code marked processed with sleeps)
        for t in teams_data.get("response", []):
            n = t["team"]["name"]
            if teams_status.get(n):
                continue
            
            if self.verbose:
                self._vprint(f"📥 Downloaded team: {n}")
            # if self.verbose:
            #     print(f"📥 Downloaded team: {n}")
            teams_status[n] = True
            self._dl_counts["teams"] += 1
            self.save_json(self.files["teams_status"], teams_status)
            time.sleep(self.rate_delay)

        self.save_json(self.files["teams"], teams_data)
        self.save_json(self.files["teams_status"], teams_status)
        return teams_data, teams_status

    # --------- Players ----------
    def fetch_players(self, teams_data: dict) -> Tuple[dict, dict]:
        players_data = self.load_json(self.files["players"])
        players_status = self.load_json(self.files["players_status"])
        if not players_status and teams_data:
            players_status = {t["team"]["name"]: False for t in teams_data["response"]}
    
        for t in teams_data["response"]:
            name = t["team"]["name"]
            team_id = t["team"]["id"]
            if players_status.get(name):
                continue            
            if self.verbose:
                self._vprint(f"📥 Downloading players for team: {name}")
            # if self.verbose:
                # print(f"📥 Downloading players for team: {name}")
    
            resp = self.api_get("/players", {"team": team_id, "season": self.season})
            if resp.get("results", 0) == 0 or not resp.get("response"):
                players_data[name] = []
                players_status[name] = True
                self.save_json(self.files["players"], players_data)
                self.save_json(self.files["players_status"], players_status)
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
    
            total_pages = int(resp.get("paging", {}).get("total", 1) or 1)
            pages_to_fetch = total_pages if (self.max_player_pages is None) else min(total_pages, self.max_player_pages)
            
            if self.max_player_pages is not None and total_pages > self.max_player_pages and self.verbose:
                self._vprint(f"⚠️  Plan page-cap hit for {name}: fetching only {pages_to_fetch}/{total_pages} pages.")
                self._truncated_players_teams.append(name)
    
            # next pages (capped)
            for page in range(2, pages_to_fetch + 1):
                time.sleep(self.rate_delay)
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
    
            # Optional soft fill: try current squad if truncated (note: not historical-accurate)
            if self.use_squads_fallback and (self.max_player_pages is not None) and (total_pages > self.max_player_pages):
                try:
                    time.sleep(self.rate_delay)
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
                    # in squads fallback success
                    if self.verbose:
                        self._vprint(f"ℹ️  {name}: added via squads fallback", show_tokens=True)
                except Exception:
                    if self.debug:
                        print(f"⚠️  squads fallback failed for {name}")
    
            players_data[name] = players_list
            players_status[name] = True
            self.save_json(self.files["players"], players_data)
            self.save_json(self.files["players_status"], players_status)
            time.sleep(self.rate_delay)
            self._dl_counts["players"] += len(players_list)
    
        return players_data, players_status
    
    # --------- Fixtures ----------
    def fetch_fixtures(self, league_id: int) -> List[int]:
        fixtures_data = self.load_json(self.files["fixtures"])
        if not fixtures_data:
            d = self.api_get("/fixtures", {"league": league_id, "season": self.season})
            if d.get("results", 0) == 0 or not d.get("response"):
                # No fixtures is not necessarily an error; mark and continue
                self.save_json(self.files["fixtures"], d)
                return []
            self.save_json(self.files["fixtures"], d)
            fixtures_data = d

        return [f["fixture"]["id"] for f in fixtures_data.get("response", [])]

    # --------- Fixture Events ----------
    def fetch_fixture_events(self, fixture_ids: List[int]) -> Tuple[dict, dict]:
        match_events = self.load_json(self.files["match_events"])
        fixtures_status = self.load_json(self.files["fixtures_status"])

        for fid in fixture_ids:
            key = str(fid)
            if key in match_events and match_events[key].get("lineups") and match_events[key].get("events"):
                continue

            lineup = self.api_get("/fixtures/lineups", {"fixture": fid})
            # in fetch_fixture_events
            if self.verbose:
                self._vprint(f"📥 Downloading fixture events for fixture {fid}")

            # if self.verbose:
            #     print(f"📥 Downloading fixture events for fixture {fid}")
            time.sleep(self.rate_delay)
            events = self.api_get("/fixtures/events", {"fixture": fid})
            time.sleep(self.rate_delay)

            match_events[key] = {
                "lineups": lineup.get("response", []),
                "events": events.get("response", []),
            }
            fixtures_status[key] = True
            self._dl_counts["fixtures"] += 1
            self.save_json(self.files["match_events"], match_events)
            self.save_json(self.files["fixtures_status"], fixtures_status)

        return match_events, fixtures_status

    # --------- Completion checks for ProgressTracker ----------
    def _teams_done(self) -> bool:
        st = self.load_json(self.files["teams_status"])
        return bool(st) and all(st.values())

    def _players_done(self) -> bool:
        st = self.load_json(self.files["players_status"])
        # if no teams, treat players as False
        if not st:
            return False
        return all(st.values())

    def _fixtures_done(self) -> bool:
        fx = self.load_json(self.files["fixtures"])
        # If fixtures response exists (even empty), consider fixture listing "done"
        return isinstance(fx, dict)

    def _events_done(self) -> bool:
        fx = self.load_json(self.files["fixtures"])
        if not fx or not fx.get("response"):
            # no fixtures means nothing to download; mark as True (nothing pending)
            return True
        st = self.load_json(self.files["fixtures_status"])
        ids = [str(f["fixture"]["id"]) for f in fx["response"]]
        return bool(st) and all(st.get(i) for i in ids)

    def _update_progress(self, last_reason: Optional[str] = None, last_stage: Optional[str] = None):
        self.progress.update_row(
            country=self.country_name,
            league=self.league_name,
            season=self.season,
            teams_done=self._teams_done(),
            players_done=self._players_done(),
            fixtures_done=self._fixtures_done(),
            events_done=self._events_done(),
            last_reason=last_reason or (self.exit_mgr.ctx.reason if self.exit_mgr.ctx else ""),
            last_stage=last_stage or (self.exit_mgr.ctx.stage if self.exit_mgr.ctx else ""),
            requests_remaining=self.requests_remaining,
        )



    def season_access_probe(self, league_id: int) -> bool:
        try:
            # light probe: ask for teams; will fail quickly if season blocked
            _ = self.api_get("/teams", {"league": league_id, "season": self.season})
            return True
        except GracefulExit:
            if self.exit_mgr.ctx and "do not have access" in str(self.exit_mgr.ctx.detail):
                if self.verbose:
                    print(f"⛔ Season {self.season} not available on current plan for {self.country_name}/{self.league_name}.")
                return False
            raise  # other errors bubble up
            
    # --------- Public: full pipeline ----------
    def run_fetch(self):
        """
        Full pipeline:
          - Preflight quota probe
          - League -> teams -> players -> fixtures -> events
          - Progress updates after each stage
          - Verbose reporting shows remaining tokens and final usage
        """
        try:
            # --- preflight: quota ---
            self.requests_remaining = self.get_requests_left()
            if getattr(self, "_start_tokens", None) is None:
                # in case get_requests_left() didn't set it (older version)
                self._start_tokens = self.requests_remaining
            self._vprint("🔎 Quota preflight", show_tokens=True)
    
            if self.requests_remaining is not None and self.requests_remaining <= 0:
                self.exit_mgr.exit(
                    "API quota exhausted (pre-check)",
                    stage="preflight",
                    requests_remaining=self.requests_remaining,
                )
    
            # --- league & access probe ---
            league_id = self.fetch_league_id()
    
            if not self.season_access_probe(league_id):
                # season not in plan; record and stop gracefully
                self._update_progress(last_reason="season_not_in_plan", last_stage="preflight")
                return
    
            # --- teams ---
            teams_data, _ = self.fetch_teams(league_id)
            self._update_progress(last_stage="teams")
    
            # --- players ---
            players_data, _ = self.fetch_players(teams_data)
            self._update_progress(last_stage="players")
    
            # --- fixtures list ---
            fixture_ids = self.fetch_fixtures(league_id)
            self._update_progress(last_stage="fixtures_list")
    
            # --- events (lineups + events) ---
            if fixture_ids:
                self.fetch_fixture_events(fixture_ids)
                self._update_progress(last_stage="fixtures_events")
    
            # --- summary / tokens accounting ---
            if self.verbose:
                trunc = (
                    f" | players truncated for {len(self._truncated_players_teams)} teams"
                    if getattr(self, "_truncated_players_teams", None)
                    else ""
                )
                self._vprint(
                    f"📊 Summary: {self._dl_counts['teams']} teams, "
                    f"{self._dl_counts['players']} players, "
                    f"{self._dl_counts['fixtures']} fixtures downloaded{trunc}",
                    show_tokens=True,
                )
    
                used = None
                if getattr(self, "_start_tokens", None) is not None and self.requests_remaining is not None:
                    used = self._start_tokens - self.requests_remaining
                if used is not None:
                    self._vprint(f"🏁 Tokens used: {used}", show_tokens=True)
    
            print(f"✅ Finished fetch: {self.country_name} / {self.league_name} / {self.season}")
    
        except GracefulExit:
            # Clean, expected exits
            LOGGER.info("Graceful exit: %s", self.exit_mgr.ctx)
            self._update_progress()
            self.exit_mgr.handle(save_progress_callable=lambda: self._update_progress())
    
        except Exception as e:
            # Unexpected errors
            if self.debug:
                print("❌ Full error traceback:")
                traceback.print_exc()
            else:
                print(f"❌ Unexpected error: {e}")
            self._update_progress(last_reason="unexpected_error", last_stage="runtime")
            self.exit_mgr.handle(save_progress_callable=lambda: self._update_progress())
            

# ----------------------------- Example CLI --------------------------------
if __name__ == "__main__":
    api_key = "427b1bc85aa3a6a81fc63b43df0dbd55"


    # Example batch (kept close to your original)
    jobs = [
        ("Netherlands", "Eredivisie"),
        ("Portugal", "Primeira Liga"),
        ("Mexico", "Liga MX"),
    ]
    jobs = [
        ("Germany", "Bundesliga"),
        ("Spain", "La Liga"),
        ("England", "Premier League"),
    ]
    jobs = [
        # already on your list
        ("Netherlands", "Eredivisie"),
        ("Portugal", "Primeira Liga"),
        ("Mexico", "Liga MX"),
    
        # strong feeders (top tiers)
        ("Belgium", "Jupiler Pro League"),        # African & South American pipeline
        ("Austria", "Bundesliga"),                # RB Salzburg, German loans
        ("Switzerland", "Super League"),          # stepping-stone for DACH/Balkans
        ("Denmark", "Superliga"),                 # exports to Bundesliga & PL
        ("Norway", "Eliteserien"),                # Ødegaard path, Premier League attention
        ("Sweden", "Allsvenskan"),                # many Bundesliga/Netherlands links
    
        # important second leagues
        ("England", "Championship"),              # extremely competitive, many loans from PL
        ("Spain", "Segunda División"),            # La Liga clubs loan out prospects here
        ("Germany", "2. Bundesliga"),             # crucial for Bundesliga player development
        ("Italy", "Serie B"),                     # used for Serie A loanees and reclamations
        ("France", "Ligue 2"),                    # big role in feeding Ligue 1 & exporting abroad
    ]

    for year in range(2023, 2020, -1):
        for country, league in jobs:
            fetcher = SeasonFootballDataFetcher(api_key, league, country, year,
                                          rate_limit_sec=10.0, 
                                          debug=True)
            fetcher.run_fetch()
            time.sleep(10)