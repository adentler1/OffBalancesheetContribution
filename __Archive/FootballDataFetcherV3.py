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



# # ----------------------------- Logging setup -----------------------------
# LOG_DIR = "./logs"
# os.makedirs(LOG_DIR, exist_ok=True)
# logging.basicConfig(
#     filename=os.path.join(LOG_DIR, "football_fetcher.log"),
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )
# LOGGER = logging.getLogger("football")

# from GracefulExit import GracefulExit, GracefulExitManager

# ----------------------------- Logging setup -----------------------------
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "football_fetcher.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("football")


# ---------------------- Graceful Exit / Error Handling --------------------
class GracefulExit(Exception):
    """
    # Raised to stop the pipeline cleanly with a clear reason.
    """


@dataclass
class ExitContext:
    reason: str
    detail: Optional[str] = None
    stage: Optional[str] = None
    requests_remaining: Optional[int] = None


class GracefulExitManager:
    """
    # Wraps a run sequence, records the last error/exit reason,
    # ensures progress is saved, and emits consistent messages.
    """
    def __init__(self):
        self.ctx: Optional[ExitContext] = None

    def exit(self, reason: str, detail: Optional[str] = None,
             stage: Optional[str] = None, requests_remaining: Optional[int] = None):
        self.ctx = ExitContext(reason=reason, detail=detail, stage=stage,
                               requests_remaining=requests_remaining)
        raise GracefulExit(reason)

    def handle(self, save_progress_callable=None):
        """
        # Use around your top-level run. Example:
        #     try:
        #         ... run pipeline ...
        #     except Exception:
        #          exit_mgr.handle(progress_tracker.save_now)
        """
        etype, value, tb = sys.exc_info()
        msg = f"{etype.__name__}: {value}"
        stack = "".join(traceback.format_tb(tb)) if tb else ""
        LOGGER.error("Unhandled exception:\n%s\n%s", msg, stack)

        # Ensure progress is saved if caller provides a callable
        if save_progress_callable:
            try:
                save_progress_callable()
            except Exception as e:
                LOGGER.error("Failed to save progress while handling exception: %s", e)

        # Emit a clean stdout message (short + useful)
        print("⚠️  Aborted cleanly.")
        if self.ctx:
            r = self.ctx
            print(f"Reason: {r.reason}")
            if r.stage:
                print(f"Stage: {r.stage}")
            if r.detail:
                print(f"Detail: {r.detail}")
            if r.requests_remaining is not None:
                print(f"Requests remaining (last known): {r.requests_remaining}")
        else:
            print("Reason: unexpected error (see log).")


# ----------------------------- Progress Tracker ---------------------------
class ProgressTracker:
    """
    Maintains downloads_progress.xlsx with rows keyed by (country, league, season).
    Status logic:
        - Completed: all modules present & complete
        - Partial: at least one module present but not all complete
        - Pending: defined in queue but not started (or nothing present)
    """
    FILEPATH = "downloads_progress.xlsx"
    COLUMNS = [
        "country", "league", "season",
        "teams_done", "players_done", "fixtures_done", "events_done",
        "overall_status", "last_reason", "last_stage", "requests_remaining",
        "updated_at"
    ]

    def __init__(self):
        # Lazy-load / create on first use
        if not os.path.exists(self.FILEPATH):
            df = pd.DataFrame(columns=self.COLUMNS)
            df.to_excel(self.FILEPATH, index=False)

    def _load(self) -> pd.DataFrame:
        return pd.read_excel(self.FILEPATH)

    def _save(self, df: pd.DataFrame):
        df = df[self.COLUMNS]
        df.to_excel(self.FILEPATH, index=False)

    @staticmethod
    def _compute_overall(teams: bool, players: bool, fixtures: bool, events: bool) -> str:
        vals = [teams, players, fixtures, events]
        if all(vals):
            return "Completed"
        if any(vals):
            return "Partial"
        return "Pending"

    def update_row(self,
                   country: str, league: str, season: int,
                   teams_done: bool, players_done: bool, fixtures_done: bool, events_done: bool,
                   last_reason: Optional[str] = None,
                   last_stage: Optional[str] = None,
                   requests_remaining: Optional[int] = None):
        df = self._load()
        mask = (df["country"] == country) & (df["league"] == league) & (df["season"] == season)
        overall = self._compute_overall(teams_done, players_done, fixtures_done, events_done)
        now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        row = {
            "country": country, "league": league, "season": season,
            "teams_done": bool(teams_done),
            "players_done": bool(players_done),
            "fixtures_done": bool(fixtures_done),
            "events_done": bool(events_done),
            "overall_status": overall,
            "last_reason": last_reason or "",
            "last_stage": last_stage or "",
            "requests_remaining": requests_remaining if requests_remaining is not None else "",
            "updated_at": now
        }

        if mask.any():
            df.loc[mask, :] = pd.DataFrame([row]).values
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        self._save(df)

    def ensure_pending(self, country: str, league: str, season: int):
        """Insert a Pending row if not present yet (marks 'next in line')."""
        df = self._load()
        mask = (df["country"] == country) & (df["league"] == league) & (df["season"] == season)
        if not mask.any():
            self.update_row(country, league, season, False, False, False, False)


# -------------------------- Available Catalog Export ----------------------
class AvailableCatalog:
    """
    Exports available (country, competition, season) to available_leagues.xlsx.

    Strategy:
      - /countries
      - /leagues?country=XX (or by country name) → seasons array → year + current + start/end
    """
    FILEPATH = "available_leagues.xlsx"

    def __init__(self, api_url: str, headers: Dict[str, str], rate_delay: float = 0.8):
        self.api_url = api_url
        self.headers = headers
        self.delay = rate_delay

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.api_url}{endpoint}"
        resp = requests.get(url, headers=self.headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def export_all(self):
        # 1) Countries
        countries = self._get("/countries").get("response", [])
        rows = []
        for c in countries:
            country_name = c.get("name")
            if not country_name:
                continue
            time.sleep(self.delay)

            # 2) Leagues for this country
            leagues = self._get("/leagues", {"country": country_name}).get("response", [])
            for L in leagues:
                league = L.get("league", {})
                comp_name = league.get("name")
                if not comp_name:
                    continue
                seasons = L.get("seasons", []) or []
                for s in seasons:
                    rows.append({
                        "country": country_name,
                        "competition": comp_name,
                        "season": s.get("year"),
                        "season_start": s.get("start"),
                        "season_end": s.get("end"),
                        "is_current": bool(s.get("current")),
                    })

        df = pd.DataFrame(rows, columns=["country", "competition", "season",
                                         "season_start", "season_end", "is_current"])
        df.sort_values(["country", "competition", "season"], inplace=True, ignore_index=True)
        df.to_excel(self.FILEPATH, index=False)
        LOGGER.info("Exported available leagues to %s (rows=%d)", self.FILEPATH, len(df))


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
                    
                    
                    # if self.verbose:
                    #     print(f"ℹ️  {name}: added {len(seen_ids) - len(players_list)} via squads fallback")
                except Exception as _:
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
            time.sleep(10)  # polite delay between league requests

 

    # Example: export available leagues (menu XLS)
    catalog = AvailableCatalog(api_url="https://v3.football.api-sports.io",
                               headers={"x-rapidapi-key": api_key,
                                        "x-rapidapi-host": "v3.football.api-sports.io"})
    # Comment/uncomment as you like:
    catalog.export_all()
    print("Exported available_leagues.xlsx")