#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch player profiles and transfer history from API-FOOTBALL and persist to SQLite.

What it does
------------
1) Reads distinct `player_id`s from *all tables* in analysis_results.db that contain a `player_id` column.
2) Fetches player *profiles* from the API-FOOTBALL v3 endpoint `players/profiles`.
3) Fetches player *transfer history* from the endpoint `transfers`.
4) Writes profiles to player_profiles.db and transfers to player_transfers.db.
5) Skips work that is already saved (idempotent). Transfers can be optionally refreshed.

Docs notes
----------
- Base URL: https://v3.football.api-sports.io
- Headers: Either direct `x-apisports-key` or RapidAPI pair (`x-rapidapi-host`, `x-rapidapi-key`).
- `players/profiles` accepts `player` and `search` and returns a `player` object (biographical profile).
- `transfers` accepts `player` and returns transfer items (date, type, teams in/out).

Edit CONFIG below and run in Spyder.
@author: alexander
"""
import re
import os
import sys
import time
import json
import sqlite3
from typing import Dict, Any, Iterable, List, Set, Tuple, Optional

import requests

# ==========================
# CONFIG — EDIT THESE VALUES
# ==========================
api_key = "427b1bc85aa3a6a81fc63b43df0dbd55"
# api_key = "YOUR_API_KEY_HERE"       # <-- put your API-FOOTBALL key here
USE_RAPIDAPI = False                # True if you use RapidAPI instead of direct API-Sports
RAPIDAPI_HOST = "v3.football.api-sports.io"

INPUT_DB = "analysis_results.db"    # source DB (same folder as this script)
PROFILES_DB = "player_profiles.db"  # output DB for player profiles
TRANSFERS_DB = "player_transfers.db"  # output DB for transfers

REQUEST_TIMEOUT = 20
REQUEST_SLEEP_SECS = 0.35           # gentle throttling
ALWAYS_REFRESH_TRANSFERS = False    # if False, skip fetching transfers when we already have rows for that player

# Endpoints (v3)
API_BASE = "https://v3.football.api-sports.io"
URL_PLAYERS_PROFILES = f"{API_BASE}/players/profiles"
URL_TRANSFERS = f"{API_BASE}/transfers"


# ==========================
# Helpers
# ==========================
def parse_player_id_value(val) -> Optional[int]:
    """
    Accepts 'p276', '276', 276, ' p000276 ' -> 276; returns None if no digits found.
    """
    if val is None:
        return None
    m = re.search(r'\d+', str(val))
    return int(m.group(0)) if m else None

def build_headers() -> Dict[str, str]:
    """Build request headers for either direct API-SPORTS or RapidAPI."""
    if USE_RAPIDAPI:
        return {
            "x-rapidapi-host": RAPIDAPI_HOST,
            "x-rapidapi-key": api_key,
        }
    else:
        return {
            "x-apisports-key": api_key,
        }


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None


def ensure_profiles_schema(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
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
    conn.commit()


def ensure_transfers_schema(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
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
    conn.commit()


def sqlite_tables(conn: sqlite3.Connection) -> List[str]:
    c = conn.cursor()
    return [row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def table_has_player_id(conn: sqlite3.Connection, table: str) -> bool:
    c = conn.cursor()
    cols = [row[1].lower() for row in c.execute(f'PRAGMA table_info("{table}")')]
    return "player_id" in cols


def load_player_ids(input_db: str) -> List[int]:
    """Scan all tables in input_db for a `player_id` column and return distinct ids."""
    if not os.path.exists(input_db):
        raise FileNotFoundError(f"Input DB not found: {input_db}")

    conn = sqlite3.connect(input_db)
    try:
        ids: Set[int] = set()
        for tbl in sqlite_tables(conn):
            if not table_has_player_id(conn, tbl):
                continue
            cur = conn.cursor()
            try:
                for (pid,) in cur.execute(f'SELECT DISTINCT player_id FROM "{tbl}" WHERE player_id IS NOT NULL'):
                    i = parse_player_id_value(pid)
                    if i is not None:
                        ids.add(i)

                # for (pid,) in cur.execute(f'SELECT DISTINCT player_id FROM "{tbl}" WHERE player_id IS NOT NULL'):
                #     i = safe_int(pid)
                #     if i is not None:
                #         ids.add(i)
            except sqlite3.Error:
                # If a table is weirdly quoted or has a view-like behavior, skip quietly
                continue
        return sorted(ids)
    finally:
        conn.close()


def profile_exists(conn: sqlite3.Connection, player_id: int) -> bool:
    c = conn.cursor()
    c.execute("SELECT 1 FROM player_profiles WHERE player_id = ? LIMIT 1", (player_id,))
    return c.fetchone() is not None


def any_transfer_exists(conn: sqlite3.Connection, player_id: int) -> bool:
    c = conn.cursor()
    c.execute("SELECT 1 FROM player_transfers WHERE player_id = ? LIMIT 1", (player_id,))
    return c.fetchone() is not None


def fetch_profile(session: requests.Session, headers: Dict[str, str], player_id: int) -> Optional[Dict[str, Any]]:
    """
    Call players/profiles?player=ID.
    Returns the `player` dict or None if not found.
    """
    params = {"player": player_id}
    r = session.get(URL_PLAYERS_PROFILES, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    # Expected shape: {"response": [ {"player": {...}} , ... ]}
    resp = data.get("response") or []
    if not resp:
        return None
    item = resp[0]  # when filtering by a specific player id, first item is the profile
    pl = item.get("player") if isinstance(item, dict) else None
    return pl or item  # some SDKs expose directly the player payload


def insert_profile(conn: sqlite3.Connection, player: Dict[str, Any]) -> None:
    """Insert a player profile row if not already present."""
    c = conn.cursor()
    birth = player.get("birth") or {}
    vals = (
        safe_int(player.get("id")),
        player.get("name"),
        player.get("firstname"),
        player.get("lastname"),
        safe_int(player.get("age")),
        (birth.get("date") if isinstance(birth, dict) else None),
        (birth.get("place") if isinstance(birth, dict) else None),
        (birth.get("country") if isinstance(birth, dict) else None),
        player.get("nationality"),
        player.get("height"),
        player.get("weight"),
        safe_int(player.get("number")),
        player.get("position"),
        player.get("photo"),
        json.dumps(player, ensure_ascii=False),
    )
    c.execute("""
        INSERT OR IGNORE INTO player_profiles (
            player_id, name, firstname, lastname, age,
            birth_date, birth_place, birth_country,
            nationality, height, weight, number, position, photo, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, vals)
    conn.commit()


def fetch_transfers(session: requests.Session, headers: Dict[str, str], player_id: int) -> List[Dict[str, Any]]:
    """
    Call transfers?player=ID.
    Returns a list of transfer dicts (we store raw_json too).
    """
    params = {"player": player_id}
    r = session.get(URL_TRANSFERS, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    # Expected shape: {"response": [ {"transfers": [ {...}, ... ], "player": {...}}, ... ] }
    out: List[Dict[str, Any]] = []
    for block in data.get("response") or []:
        transfers = block.get("transfers") or []
        for tr in transfers:
            out.append(tr)
    return out


def parse_transfer_row(player_id: int, tr: Dict[str, Any]) -> Tuple:
    date = tr.get("date")
    typ = tr.get("type")
    teams = tr.get("teams") or {}
    tin = teams.get("in") or {}
    tout = teams.get("out") or {}
    return (
        player_id,
        date,
        typ,
        safe_int(tin.get("id")),
        tin.get("name"),
        tin.get("country"),
        tin.get("logo"),
        safe_int(tout.get("id")),
        tout.get("name"),
        tout.get("country"),
        tout.get("logo"),
        json.dumps(tr, ensure_ascii=False),
    )


def insert_transfers(conn: sqlite3.Connection, player_id: int, transfers: List[Dict[str, Any]]) -> int:
    """Insert transfer rows (ignoring duplicates). Returns number of rows inserted."""
    if not transfers:
        return 0
    c = conn.cursor()
    rows = [parse_transfer_row(player_id, t) for t in transfers]
    c.executemany("""
        INSERT OR IGNORE INTO player_transfers (
            player_id, date, type,
            team_in_id, team_in_name, team_in_country, team_in_logo,
            team_out_id, team_out_name, team_out_country, team_out_logo,
            raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    return c.rowcount or 0


def run() -> None:
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("Please set api_key in the CONFIG section.", file=sys.stderr)
        sys.exit(1)

    # Load player IDs from input DB
    player_ids = load_player_ids(INPUT_DB)
    if not player_ids:
        print("No player_id values found in input DB.")
        return

    # Prepare output DBs
    conn_profiles = sqlite3.connect(PROFILES_DB)
    conn_transfers = sqlite3.connect(TRANSFERS_DB)
    try:
        ensure_profiles_schema(conn_profiles)
        ensure_transfers_schema(conn_transfers)

        headers = build_headers()
        session = requests.Session()

        new_profiles = 0
        new_transfer_rows = 0

        for idx, pid in enumerate(player_ids, 1):
            print(f"[{idx}/{len(player_ids)}] player_id={pid}")

            # PROFILE
            if profile_exists(conn_profiles, pid):
                print("  • profile: already saved")
            else:
                try:
                    prof = fetch_profile(session, headers, pid)
                    if prof:
                        insert_profile(conn_profiles, prof)
                        new_profiles += 1
                        print("  • profile: saved")
                    else:
                        print("  • profile: not found")
                except requests.RequestException as e:
                    print(f"  • profile: HTTP error: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"  • profile: error: {e}", file=sys.stderr)

            # TRANSFERS
            have_transfers = any_transfer_exists(conn_transfers, pid)
            if have_transfers and not ALWAYS_REFRESH_TRANSFERS:
                print("  • transfers: already saved (skip)")
            else:
                try:
                    transfers = fetch_transfers(session, headers, pid)
                    inserted = insert_transfers(conn_transfers, pid, transfers)
                    new_transfer_rows += max(0, inserted)
                    msg = f"saved {inserted}" if inserted else "none"
                    print(f"  • transfers: {msg}")
                except requests.RequestException as e:
                    print(f"  • transfers: HTTP error: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"  • transfers: error: {e}", file=sys.stderr)

            time.sleep(REQUEST_SLEEP_SECS)

        print("\nDone.")
        print(f"New profiles: {new_profiles}")
        print(f"New transfer rows: {new_transfer_rows}")

    finally:
        conn_profiles.close()
        conn_transfers.close()


if __name__ == "__main__":
    # Spyder-friendly: edit CONFIG above and just run this file.
    run()
