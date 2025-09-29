#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
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
"""

import os
import sys
import time
import json
import re
import sqlite3
from typing import Dict, Any, List, Optional, Tuple, Iterable, Set

import requests
import pandas as pd


# ==========================
# CONFIG — EDIT THESE VALUES
# ==========================
API_KEY = "427b1bc85aa3a6a81fc63b43df0dbd55"
# API_KEY = os.getenv("APIFOOTBALL_KEY", "YOUR_API_KEY_HERE")  # <-- put your API-FOOTBALL key here or set env var
USE_RAPIDAPI = False                                          # True if using RapidAPI instead of direct API-Sports
RAPIDAPI_HOST = "v3.football.api-sports.io"

PROFILES_DB  = "player_profiles.db"
TRANSFERS_DB = "player_transfers.db"

REQUEST_TIMEOUT     = 25
REQUEST_SLEEP_SECS  = 0.30   # gentle throttling between requests
BACKOFF_429_SECS    = 2.0    # simple backoff if rate-limited

EXCEL_FILENAME      = "downloads_progress.xlsx"  # read from the same folder as this script
STATUS_COMPLETED    = "Completed"                # rows to include from the Excel file

# Endpoints (v3)
API_BASE       = "https://v3.football.api-sports.io"
URL_PLAYERS    = f"{API_BASE}/players"
URL_TRANSFERS  = f"{API_BASE}/transfers"
# ==========================


def _coerce_int(val) -> Optional[int]:
    """Extract an integer from mixed input like '123', 123, ' L-123 ' -> 123; None if no digits."""
    if val is None:
        return None
    m = re.search(r'\d+', str(val))
    return int(m.group(0)) if m else None


class APIFootballBulkDownloader:
    """
    Download player profiles (via /players?league=&season=&page=) and transfers (via /transfers?player=)
    and persist to SQLite databases, skipping what's already saved.
    """

    def __init__(
        self,
        api_key: str,
        use_rapidapi: bool = USE_RAPIDAPI,
        profiles_db: str = PROFILES_DB,
        transfers_db: str = TRANSFERS_DB,
        request_timeout: int = REQUEST_TIMEOUT,
        sleep_secs: float = REQUEST_SLEEP_SECS,
        backoff_429_secs: float = BACKOFF_429_SECS,
    ) -> None:
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Please set API_KEY (or APIFOOTBALL_KEY env var).")

        self.api_key = api_key
        self.use_rapidapi = use_rapidapi
        self.request_timeout = request_timeout
        self.sleep_secs = sleep_secs
        self.backoff_429_secs = backoff_429_secs

        # HTTP session
        self.session = requests.Session()
        self.headers = (
            {"x-rapidapi-host": RAPIDAPI_HOST, "x-rapidapi-key": self.api_key}
            if self.use_rapidapi
            else {"x-apisports-key": self.api_key}
        )

        # SQLite connections
        self.conn_profiles = sqlite3.connect(profiles_db)
        self.conn_transfers = sqlite3.connect(transfers_db)

        self._optimize_sqlite(self.conn_profiles)
        self._optimize_sqlite(self.conn_transfers)
        self._ensure_profiles_schema()
        self._ensure_transfers_schema()

    # ---------- SQLite schema & pragmas ----------
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
                player_id     INTEGER PRIMARY KEY,
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
                fetched_at    TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            )
        """)
        self.conn_profiles.commit()

    def _ensure_transfers_schema(self) -> None:
        c = self.conn_transfers.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS player_transfers (
                player_id        INTEGER NOT NULL,
                date             TEXT,
                type             TEXT,
                team_in_id       INTEGER,
                team_in_name     TEXT,
                team_in_country  TEXT,
                team_in_logo     TEXT,
                team_out_id      INTEGER,
                team_out_name    TEXT,
                team_out_country TEXT,
                team_out_logo    TEXT,
                raw_json         TEXT,
                fetched_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                PRIMARY KEY (player_id, date, team_in_id, team_out_id, type)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_transfers_player ON player_transfers (player_id)")
        self.conn_transfers.commit()

    # ---------- Existence checks ----------
    def profile_exists(self, player_id: int) -> bool:
        cur = self.conn_profiles.cursor()
        cur.execute("SELECT 1 FROM player_profiles WHERE player_id = ? LIMIT 1", (player_id,))
        return cur.fetchone() is not None

    def any_transfer_exists(self, player_id: int) -> bool:
        cur = self.conn_transfers.cursor()
        cur.execute("SELECT 1 FROM player_transfers WHERE player_id = ? LIMIT 1", (player_id,))
        return cur.fetchone() is not None

    # ---------- HTTP helpers ----------
    def _get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        while True:
            r = self.session.get(url, headers=self.headers, params=params, timeout=self.request_timeout)
            if r.status_code == 429:
                time.sleep(self.backoff_429_secs)
                continue
            r.raise_for_status()
            # optional quota log
            day_left = r.headers.get("x-ratelimit-requests-remaining")
            min_left = r.headers.get("x-ratelimit-remaining")
            if day_left and min_left:
                print(f"    quota remaining: day={day_left}, minute={min_left}")
            return r.json()

    # ---------- Extraction ----------
    @staticmethod
    def _choose_best_stat(stat_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Pick the statistics block with most minutes played (fallback: first)."""
        if not stat_list:
            return None
        best = None
        best_min = -1
        for s in stat_list:
            g = (s.get("games") or {})
            mins = g.get("minutes") or 0
            try:
                mins = int(mins)
            except Exception:
                mins = 0
            if mins > best_min:
                best_min = mins
                best = s
        return best or stat_list[0]

    @staticmethod
    def _extract_profile_from_players_item(item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        From /players item:
          { "player": {...}, "statistics": [ {...}, ... ] }
        Return (player_id, row_dict) matching player_profiles schema.
        """
        p = item.get("player") or {}
        stats = item.get("statistics") or []
        best = APIFootballBulkDownloader._choose_best_stat(stats) or {}
        games = best.get("games") or {}

        birth = p.get("birth") or {}
        row = {
            "player_id": p.get("id"),
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

    # ---------- Inserts ----------
    def _insert_profile(self, row: Dict[str, Any]) -> None:
        c = self.conn_profiles.cursor()
        c.execute("""
            INSERT OR IGNORE INTO player_profiles (
                player_id, name, firstname, lastname, age,
                birth_date, birth_place, birth_country,
                nationality, height, weight, number, position, photo, raw_json
            )
            VALUES (:player_id, :name, :firstname, :lastname, :age,
                    :birth_date, :birth_place, :birth_country,
                    :nationality, :height, :weight, :number, :position, :photo, :raw_json)
        """, row)
        self.conn_profiles.commit()

    @staticmethod
    def _parse_transfer_row(player_id: int, tr: Dict[str, Any]) -> Tuple:
        date = tr.get("date")
        typ = tr.get("type")
        teams = tr.get("teams") or {}
        tin = teams.get("in") or {}
        tout = teams.get("out") or {}
        return (
            player_id,
            date,
            typ,
            tin.get("id"),  tin.get("name"),  tin.get("country"),  tin.get("logo"),
            tout.get("id"), tout.get("name"), tout.get("country"), tout.get("logo"),
            json.dumps(tr, ensure_ascii=False),
        )

    def _insert_transfers(self, player_id: int, transfers: List[Dict[str, Any]]) -> int:
        if not transfers:
            return 0
        rows = [self._parse_transfer_row(player_id, t) for t in transfers]
        cur = self.conn_transfers.cursor()
        cur.executemany("""
            INSERT OR IGNORE INTO player_transfers (
                player_id, date, type,
                team_in_id, team_in_name, team_in_country, team_in_logo,
                team_out_id, team_out_name, team_out_country, team_out_logo,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        self.conn_transfers.commit()
        return cur.rowcount or 0

    # ---------- Fetchers ----------
    def _fetch_players_page(self, league_id: int, season: int, page: int) -> Dict[str, Any]:
        params = {"league": league_id, "season": season, "page": page}
        return self._get_json(URL_PLAYERS, params)

    def _fetch_transfers_for_player(self, player_id: int) -> List[Dict[str, Any]]:
        data = self._get_json(URL_TRANSFERS, {"player": player_id})
        out: List[Dict[str, Any]] = []
        for block in data.get("response") or []:
            for tr in block.get("transfers") or []:
                out.append(tr)
        return out

    # ---------- Orchestration ----------
    def process_league_season(
        self,
        league_id: int,
        season: int,
        fetch_transfers: bool = True,
        refresh_transfers: bool = False,
    ) -> Tuple[int, int]:
        """
        Download all players for (league_id, season) and optionally their transfers.
        Returns (new_profiles_inserted, new_transfer_rows_inserted).
        """
        print(f"\n== League {league_id}, Season {season} ==")
        page = 1
        new_profiles = 0
        new_transfer_rows = 0

        while True:
            print(f"  fetching page {page} ...")
            try:
                data = self._fetch_players_page(league_id, season, page)
            except requests.RequestException as e:
                print(f"  HTTP error: {e}", file=sys.stderr)
                break

            resp = data.get("response") or []
            paging = data.get("paging") or {}

            if not resp:
                print("  no more results.")
                break

            for item in resp:
                try:
                    pid, row = self._extract_profile_from_players_item(item)
                    if not self.profile_exists(pid):
                        self._insert_profile(row)
                        new_profiles += 1

                    if fetch_transfers:
                        already = self.any_transfer_exists(pid)
                        if already and not refresh_transfers:
                            pass
                        else:
                            trs = self._fetch_transfers_for_player(pid)
                            new_transfer_rows += max(0, self._insert_transfers(pid, trs))

                except Exception as e:
                    print(f"  error processing player item: {e}", file=sys.stderr)

                time.sleep(self.sleep_secs)

            cur = _coerce_int(paging.get("current")) or page
            tot = _coerce_int(paging.get("total")) or page
            if int(cur) >= int(tot):
                break
            page += 1

        return new_profiles, new_transfer_rows

    def run_for_pairs(
        self,
        pairs: Iterable[Tuple[int, int]],
        fetch_transfers: bool = True,
        refresh_transfers: bool = False,
    ) -> Tuple[int, int]:
        """Run process_league_season for all (league_id, season) pairs; returns cumulative counts."""
        total_profiles = 0
        total_transfers = 0
        seen: Set[Tuple[int, int]] = set()

        for lg, ssn in pairs:
            lg_i = _coerce_int(lg)
            ssn_i = _coerce_int(ssn)
            if lg_i is None or ssn_i is None:
                continue
            if (lg_i, ssn_i) in seen:
                continue
            seen.add((lg_i, ssn_i))

            p_new, t_new = self.process_league_season(
                lg_i, ssn_i,
                fetch_transfers=fetch_transfers,
                refresh_transfers=refresh_transfers
            )
            total_profiles += p_new
            total_transfers += t_new

        return total_profiles, total_transfers

    # ---------- Lifecycle ----------
    def close(self) -> None:
        try:
            self.conn_profiles.close()
        finally:
            try:
                self.conn_transfers.close()
            except Exception:
                pass


def _read_league_season_pairs_from_excel(xlsx_path: str) -> List[Tuple[int, int]]:
    """
    Read (league_id, season) pairs from an Excel file, keeping only rows
    where overall_status == 'Completed' (case-insensitive).
    Tries common column names for league & season.
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    # Normalize columns
    cols = {c.lower().strip(): c for c in df.columns}

    # Find status column
    status_col = None
    for cand in ("overall_status", "status", "overallstatus"):
        if cand in cols:
            status_col = cols[cand]
            break
    if status_col is None:
        raise KeyError("No 'overall_status' (or similar) column found in Excel.")

    # Filter to Completed
    mask = df[status_col].astype(str).str.strip().str.lower() == STATUS_COMPLETED.lower()
    df = df.loc[mask].copy()
    if df.empty:
        return []

    # Find league and season columns (be generous)
    league_col = None
    for cand in ("league_id", "league", "leagueid", "league id"):
        if cand in cols:
            league_col = cols[cand]
            break
    if league_col is None:
        raise KeyError("No 'league_id' (or similar) column found in Excel.")

    season_col = None
    for cand in ("season", "year", "season_year", "seasonyear"):
        if cand in cols:
            season_col = cols[cand]
            break
    if season_col is None:
        raise KeyError("No 'season' (or similar) column found in Excel.")

    pairs: List[Tuple[int, int]] = []
    for _, row in df[[league_col, season_col]].dropna().iterrows():
        lg = _coerce_int(row[league_col])
        ss = _coerce_int(row[season_col])
        if lg is not None and ss is not None:
            pairs.append((lg, ss))

    # Deduplicate while preserving order
    seen: Set[Tuple[int, int]] = set()
    out: List[Tuple[int, int]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


if __name__ == "__main__":
    # Resolve the Excel path next to this script (works well in Spyder too)
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback if __file__ is not defined (e.g., interactive)
        base_dir = os.getcwd()
    excel_path = os.path.join(base_dir, EXCEL_FILENAME)

    # Read league-season pairs from Excel where overall_status == 'Completed'
    pairs = _read_league_season_pairs_from_excel(excel_path)
    if not pairs:
        print("No (league_id, season) pairs with overall_status == 'Completed' found.")
        sys.exit(0)

    # Run downloader: fetch profiles + transfers; skip transfers if already saved
    downloader = APIFootballBulkDownloader(api_key=API_KEY, use_rapidapi=USE_RAPIDAPI)
    try:
        total_profiles, total_transfers = downloader.run_for_pairs(
            pairs,
            fetch_transfers=True,
            refresh_transfers=False   # <- do not redownload transfers if player already has rows
        )
        print("\nDone.")
        print(f"New profiles inserted:  {total_profiles}")
        print(f"New transfer rows:      {total_transfers}")
    finally:
        downloader.close()
