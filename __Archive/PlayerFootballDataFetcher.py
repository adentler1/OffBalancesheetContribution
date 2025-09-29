#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Player profile downloader for API-FOOTBALL v3
- Reads completed league/season pairs from downloads_progress.xlsx
- Fills gaps from analysis_results.db into player_profiles.db
- Missing <= THRESHOLD -> fetch individually (/players?id=)
- Else -> bulk league-season fetch (auto-fallback to team-wise on free plan)
"""

import sqlite3
import pandas as pd
import requests
import json
import time
import sys
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Set#, Iterable

try:
    # Optional graceful exit (your tiny helper); safe if absent
    from GracefulExit import GracefulExit, GracefulExitManager
except Exception:
    class GracefulExit(Exception): ...
    class GracefulExitManager:
        def __init__(self): self.ctx=None
        def exit(self, *a, **k): raise GracefulExit("exit")
        def handle(self, *a, **k): print("Aborted cleanly.")

class PlayerFootballDataFetcher:
    """
    Fetch player profiles for league/season pairs and backfill missing players referenced
    in analysis_results.db. Stores rows in player_profiles.db, printing remaining quotas.
    """
    # API configuration
    API_BASE = "https://v3.football.api-sports.io"
    URL_PLAYERS = f"{API_BASE}/players"
    URL_TRANSFERS = f"{API_BASE}/transfers"
    REQUEST_TIMEOUT = 25
    REQUEST_SLEEP_SECS = 10.0
    BACKOFF_429_SECS = 2.0

    def __init__(self,
                 api_key: str,
                 use_rapidapi: bool = False,
                 profiles_db: str = "player_profiles.db",
                 analysis_db: str = "analysis_results.db",
                 excel_filename: str = "downloads_progress.xlsx",
                 threshold_missing: int = 30,
                 exit_manager: Optional[Any] = None):
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            if exit_manager:
                exit_manager.exit("API key is not provided or invalid", stage="Initialization")
            raise ValueError("API key is not provided or invalid.")

        self.api_key = api_key
        self.use_rapidapi = use_rapidapi
        self.exit_manager = exit_manager
        self.analysis_db = analysis_db
        self.excel_filename = excel_filename
        self.threshold_missing = int(threshold_missing)

        # HTTP session & headers
        self.session = requests.Session()
        self.headers = (
            {"x-rapidapi-host": "v3.football.api-sports.io", "x-rapidapi-key": self.api_key}
            if self.use_rapidapi else
            {"x-apisports-key": self.api_key}
        )

        # DB connect
        self.conn_profiles = sqlite3.connect(profiles_db)
        self._optimize_sqlite(self.conn_profiles)
        self._ensure_profiles_schema()

        # last-known quota (for printing)
        self.last_requests_remaining_day: Optional[int] = None
        self.last_requests_remaining_minute: Optional[int] = None

    # ---------- SQLite ----------
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

    # ---------- Helpers ----------
    @staticmethod
    def _coerce_int(val) -> Optional[int]:
        if val is None:
            return None
        m = re.search(r'\d+', str(val))
        return int(m.group(0)) if m else None

    @staticmethod
    def _parse_prefixed_pid(val) -> Optional[int]:
        if val is None:
            return None
        m = re.search(r'\d+', str(val))
        return int(m.group(0)) if m else None

    def profile_exists(self, player_id: int) -> bool:
        cur = self.conn_profiles.cursor()
        cur.execute("SELECT 1 FROM player_profiles WHERE player_id = ? LIMIT 1", (player_id,))
        return cur.fetchone() is not None

    def _get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        while True:
            r = self.session.get(url, headers=self.headers, params=params, timeout=self.REQUEST_TIMEOUT)
            if r.status_code == 429:
                time.sleep(self.BACKOFF_429_SECS)
                continue
            try:
                r.raise_for_status()
            except requests.RequestException as e:
                if self.exit_manager:
                    self.exit_manager.exit("API request failed",
                                           detail=f"{url} {params} -> {e}",
                                           stage="HTTP",
                                           requests_remaining=self.last_requests_remaining_day)
                raise
            data = r.json()
            day_left = r.headers.get("x-ratelimit-requests-remaining")
            min_left = r.headers.get("x-ratelimit-remaining")
            if day_left is not None and min_left is not None:
                try: self.last_requests_remaining_day = int(day_left)
                except: self.last_requests_remaining_day = None
                try: self.last_requests_remaining_minute = int(min_left)
                except: self.last_requests_remaining_minute = None
                print(f"    quota remaining: day={day_left}, minute={min_left}")
            return data

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
            except: mins = 0
            if mins > best_min:
                best_min = mins
                best = s
        return best or stats_list[0]

    @staticmethod
    def _extract_profile(item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        p = item.get("player") or {}
        stats_list = item.get("statistics") or []
        best = PlayerFootballDataFetcher._choose_best_stat(stats_list) or {}
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

    # ---------- Verification ----------
    def verify_league_season(self, league_id: int, season: int) -> None:
        c = self.conn_profiles.cursor()
        c.execute("""SELECT COUNT(DISTINCT player_id),
                            COUNT(1),
                            COUNT(DISTINCT team_id)
                     FROM player_profiles WHERE league_id=? AND season=?""", (league_id, season))
        dp, rows, teams = c.fetchone() or (0,0,0)
        print(f"[verify] league={league_id}, season={season} -> players={dp}, rows={rows}, teams={teams}")

    # ---------- Support fetchers ----------
    def _fetch_teams(self, league_id: int, season: int) -> List[Dict[str, Any]]:
        data = self._get_json(f"{self.API_BASE}/teams", {"league": league_id, "season": season})
        teams = []
        for item in data.get("response") or []:
            t = item.get("team") or {}
            if t.get("id"):
                teams.append({"id": t["id"], "name": t.get("name")})
        return teams

    # ---------- Missing player discovery ----------
    def find_missing_players_from_analysis_db(self,
                                          league_id: Optional[int] = None,
                                          league_name: Optional[str] = None,
                                          season: Optional[int] = None) -> List[int]:
        if not os.path.exists(self.analysis_db):
            print(f"[fallback] analysis DB not found: {self.analysis_db}")
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
                # prefer league_id if present, else league (name)
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

    def count_missing_players(self,
                              league_id: Optional[int] = None,
                              league_name: Optional[str] = None,
                              season: Optional[int] = None) -> int:
        missing = self.find_missing_players_from_analysis_db(
            league_id=league_id, league_name=league_name, season=season
        )
        print(f"[fallback] Missing players: {len(missing)}")
        return len(missing)

    # def count_missing_players(self,
    #                           league_id: Optional[int] = None,
    #                           season: Optional[int] = None) -> int:
    #     missing = self.find_missing_players_from_analysis_db(league_id=league_id, season=season)
    #     print(f"[fallback] Missing players: {len(missing)}")
    #     return len(missing)

    # ---------- Individual & bulk fetch ----------
    def fetch_player_by_id(self, player_id: int) -> int:
        print(f"\n== Fallback: fetching player_id={player_id} ==")
        data = self._get_json(self.URL_PLAYERS, {"id": player_id})
        items = data.get("response") or []
        if not items:
            print("  No data for this player_id.")
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
            while page <= 3:  # free-plan safety
                data = self._get_json(self.URL_PLAYERS, {"team": team_id, "season": season, "page": page})
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
            errs = data.get("errors") or {}
            pg = data.get("paging") or {}
            # plan cap -> fallback
            if "plan" in errs and "maximum value of 3 for the Page parameter" in str(errs.get("plan", "")):
                print("  → Free plan page cap hit. Falling back to team-wise fetch.")
                return self.fetch_league_season_by_team(league_id, season)

            if expected_total is None:
                try: expected_total = int(pg.get("total") or 1)
                except: expected_total = 1

            items = data.get("response") or []
            if not items:
                empty_retries += 1
                if empty_retries <= MAX_EMPTY_RETRIES and page <= (expected_total or page):
                    wait = 2.0 * empty_retries
                    print(f"  Empty page; retry {empty_retries}/{MAX_EMPTY_RETRIES} after {wait:.1f}s...")
                    time.sleep(wait); continue
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

    # ---------- Backfill loop ----------
    def fetch_missing_players(self, limit: Optional[int] = None,
                              league_id: Optional[int] = None,
                              season: Optional[int] = None) -> int:
        missing = self.find_missing_players_from_analysis_db(league_id=league_id, season=season)
        if not missing:
            print("[fallback] No missing players found.")
            return 0
        if limit is not None:
            missing = missing[:int(limit)]
        print(f"[fallback] Backfilling {len(missing)} players individually...")
        inserted = 0
        for pid in missing:
            try:
                inserted += self.fetch_player_by_id(pid)
            except GracefulExit:
                raise
            except Exception as e:
                print(f"  error fetching player_id={pid}: {e}", file=sys.stderr)
            time.sleep(self.REQUEST_SLEEP_SECS)
        return inserted

    # ---------- Excel reader & run orchestrator ----------
    def _resolve_excel_path(self) -> str:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        return os.path.join(base_dir, self.excel_filename)

    def read_league_season_pairs_from_excel(self, status_filter: str = "Completed") -> List[Tuple[int, int]]:
        xlsx_path = self._resolve_excel_path()
        if not os.path.exists(xlsx_path):
            raise FileNotFoundError(f"Excel file not found: {xlsx_path}")
        df = pd.read_excel(xlsx_path)
        cols = {c.lower().strip(): c for c in df.columns}
        status_col = next((cols[c] for c in ("overall_status", "status", "overallstatus") if c in cols), None)
        if status_col is None:
            raise KeyError("No 'overall_status' column in Excel.")
        mask = df[status_col].astype(str).str.strip().str.lower() == status_filter.strip().lower()
        df = df.loc[mask].copy()
        if df.empty: return []
        league_col = next((cols[c] for c in ("league_id","league","leagueid","league id") if c in cols), None)
        season_col = next((cols[c] for c in ("season","year","season_year","seasonyear") if c in cols), None)
        if league_col is None or season_col is None:
            raise KeyError("Missing 'league_id' or 'season' column in Excel.")
        pairs: List[Tuple[int,int]] = []
        for _, row in df[[league_col, season_col]].dropna().iterrows():
            lg = self._coerce_int(row[league_col]); ss = self._coerce_int(row[season_col])
            if lg is not None and ss is not None: pairs.append((lg, ss))
        # de-dup while preserving order
        seen, uniq = set(), []
        for p in pairs:
            if p not in seen:
                seen.add(p); uniq.append(p)
        return uniq

    def run_from_downloads_progress(self) -> Tuple[int,int]:
        """
        Drive the whole process:
          - for each (league,season) in downloads_progress.xlsx with status 'Completed':
              * count missing players (analysis_results -> not in profiles)
              * if missing == 0: skip
              * elif missing <= threshold: backfill individually
              * else: bulk fetch (with team-wise fallback on free plan)
              * verify after each pair
        Returns (total_bulk_inserted, total_backfilled)
        """
        pairs = self.read_league_season_pairs_from_excel(status_filter="Completed")
        if not pairs:
            print("No (league, season) pairs marked as Completed.")
            return (0,0)

        total_bulk, total_backfill = 0, 0
        for lg, ss in pairs:
            print(f"\n=== Processing league={lg}, season={ss} ===")
            missing = self.count_missing_players(league_id=lg, season=ss)
            if missing == 0:
                print(f"[{lg}-{ss}] Nothing missing; skipping.")
            elif missing <= self.threshold_missing:
                total_backfill += self.fetch_missing_players(limit=None, league_id=lg, season=ss)
            else:
                print(f"[{lg}-{ss}] {missing} missing > {self.threshold_missing}: running full fetch...")
                total_bulk += self.fetch_league_season(lg, ss)
            try:
                self.verify_league_season(lg, ss)
            except Exception as e:
                print(f"[{lg}-{ss}] verify step skipped/failed: {e}", file=sys.stderr)
        return (total_bulk, total_backfill)
  
    # ---------- Transfers: schema ----------
    def _ensure_transfers_schema(self) -> None:
        c = self.conn_profiles.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS player_transfers (
                player_id       INTEGER NOT NULL,
                transfer_date   TEXT,                 -- ISO date (YYYY-MM-DD)
                type            TEXT,                 -- e.g., 'Transfer', 'Loan', 'Free'
                from_team_id    INTEGER,
                from_team_name  TEXT,
                to_team_id      INTEGER,
                to_team_name    TEXT,
                fee             TEXT,                 -- fee string as delivered by API (may be None/'-')
                season_hint     INTEGER,              -- optional season hint if present in payload
                raw_json        TEXT,                 -- raw transfer JSON node (for audit)
                inserted_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                PRIMARY KEY (player_id, transfer_date, from_team_id, to_team_id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS player_transfers_fetches (
                player_id       INTEGER PRIMARY KEY,
                last_fetched_at TEXT,                 -- when we last fetched transfers for this player
                last_result     TEXT                  -- 'ok', 'empty', 'error:<msg>', etc.
            )
        """)
        self.conn_profiles.commit()

    # ---------- Transfers: helpers ----------
    def _get_all_player_ids(self) -> list[int]:
        cur = self.conn_profiles.cursor()
        cur.execute("SELECT DISTINCT player_id FROM player_profiles ORDER BY player_id")
        return [int(r[0]) for r in cur.fetchall()]

    def _get_most_recent_season(self) -> Optional[int]:
        cur = self.conn_profiles.cursor()
        cur.execute("SELECT MAX(season) FROM player_profiles")
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def _get_active_player_ids_in_most_recent_season(self) -> set[int]:
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
        """, (player_id, result[:200]))
        self.conn_profiles.commit()

    # ---------- Transfers: single-player fetch & upsert ----------
    def _upsert_transfers(self, player_id: int, transfers: list[dict]) -> int:
        """
        Insert transfers, de-duplicated by (player_id, date, from_team_id, to_team_id).
        Returns number of new rows inserted.
        """
        if not transfers:
            return 0
        c = self.conn_profiles.cursor()
        inserted = 0
        for tr in transfers:
            # API-Football shape: {'date': 'YYYY-MM-DD', 'type': 'Transfer'/'Loan'/..., 'teams': {'in': {...}, 'out': {...}}, 'fees': {'amount': '€x', ...}} varies by plan/version
            date = (tr.get("date") or tr.get("updated") or "")[:10] or None
            ttype = tr.get("type") or None
            teams = tr.get("teams") or {}
            tin  = teams.get("in")  or {}
            tout = teams.get("out") or {}
            to_id, to_name = tin.get("id"), tin.get("name")
            fr_id, fr_name = tout.get("id"), tout.get("name")
            fee = None
            # common variants: tr.get('type') may be 'Loan'/'Free', some payloads carry 'type' only; some have 'transfer' dict with 'date','type','teams','player'
            # fees: sometimes under 'teams'->'in'->'transfer' or a flat 'type' + fee string in another field; keep robust:
            fee = tr.get("fee") or tr.get("price") or (tr.get("transfers") if isinstance(tr.get("transfers"), str) else None)

            # optional season hint if present
            season_hint = None
            for k in ("season", "league_season"):
                v = tr.get(k)
                if isinstance(v, int):
                    season_hint = v; break

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

    def fetch_transfers_for_player(self, player_id: int, *, overwrite: bool = False) -> int:
        """
        Fetch transfers for one player from API-Football and upsert into DB.
        If overwrite=True, we delete existing rows for this player first.
        Returns #rows inserted (after optional overwrite).
        """
        self._ensure_transfers_schema()
        if overwrite:
            c = self.conn_profiles.cursor()
            c.execute("DELETE FROM player_transfers WHERE player_id = ?", (player_id,))
            self.conn_profiles.commit()

        try:
            # standard call: /transfers?player={id}
            data = self._get_json(self.URL_TRANSFERS, {"player": int(player_id)})
        except Exception as e:
            self._mark_fetch_result(player_id, f"error:{e}")
            raise

        resp = data.get("response") or []
        # response format: usually a list with one item per player containing {'player': {...}, 'transfers': [ ... ] }
        transfers = []
        for item in resp:
            arr = item.get("transfers") or []
            if isinstance(arr, list):
                transfers.extend(arr)

        inserted = self._upsert_transfers(player_id, transfers)
        self._mark_fetch_result(player_id, "ok" if transfers else "empty")
        return inserted

    # ---------- Transfers: orchestrator ----------
    def backfill_transfers(
        self,
        *,
        force_refresh: bool = False,
        refresh_active_only: bool = True,
        limit: Optional[int] = None
    ) -> tuple[int, int, int]:
        """
        Download each player's transfer history ONCE.
        Re-download ONLY if:
          (a) force_refresh=True  AND
          (b) player appears in the most recent season (else assumed retired).

        refresh_active_only=True restricts refreshes to players seen in MAX(season).
        limit: optional cap on number of players processed.

        Returns: (num_fetched_first_time, num_refreshed, num_skipped)
        """
        self._ensure_transfers_schema()

        all_pids = self._get_all_player_ids()
        active = self._get_active_player_ids_in_most_recent_season() if refresh_active_only else set()

        if limit is not None:
            all_pids = all_pids[:int(limit)]

        first_time, refreshed, skipped = 0, 0, 0
        print(f"[transfers] players total={len(all_pids)}, active_in_most_recent={len(active)}")
        for i, pid in enumerate(all_pids, 1):
            try:
                if not self._has_fetched_transfers(pid):
                    print(f"  [{i}/{len(all_pids)}] player_id={pid}: first-time fetch…")
                    ins = self.fetch_transfers_for_player(pid, overwrite=False)
                    first_time += 1
                else:
                    if force_refresh and ((not refresh_active_only) or (pid in active)):
                        print(f"  [{i}/{len(all_pids)}] player_id={pid}: refreshing…")
                        ins = self.fetch_transfers_for_player(pid, overwrite=True)
                        refreshed += 1
                    else:
                        skipped += 1
                        continue
            except GracefulExit:
                raise
            except Exception as e:
                print(f"    [error] player_id={pid}: {e}", file=sys.stderr)
            finally:
                time.sleep(self.REQUEST_SLEEP_SECS)
        print(f"[transfers] first_time={first_time}, refreshed={refreshed}, skipped={skipped}")
        return first_time, refreshed, skipped

# ----------------------------- Run module -----------------------------
if __name__ == "__main__":
    # **** EDIT YOUR KEY ****
    API_KEY = "427b1bc85aa3a6a81fc63b43df0dbd55"
    USE_RAPIDAPI = False

    exit_mgr = GracefulExitManager()
    fetcher = PlayerFootballDataFetcher(
        api_key=API_KEY,
        use_rapidapi=USE_RAPIDAPI,
        profiles_db="player_profiles.db",
        analysis_db="analysis_results.db",
        excel_filename="downloads_progress.xlsx",
        threshold_missing=30,
        exit_manager=exit_mgr
    )
    
    fetcher.count_missing_players()
    if True:
        try:
            total_bulk, total_backfill = fetcher.run_from_downloads_progress()
            
            fetcher.backfill_transfers(force_refresh=False, refresh_active_only=True)
            
            
            print(f"\nDone. Bulk inserts: {total_bulk} | Backfilled individually: {total_backfill}")
        except GracefulExit:
            exit_mgr.handle()
        finally:
            try: fetcher.conn_profiles.close()
            except: pass
