#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FBref backfill (slow & polite) → distinct output tree to avoid collisions.

Output tree:
  fbref_data/<Country>_<League>/<SeasonEndYear>/
      fixtures_fbref.json
      match_events_fbref.json
      players_fbref.json
    [optional analyzer copies]
      fixtures.json
      match_events.json
      players.json

Usage examples:
  python fbref_backfill_runner.py --season 2025 --leagues "Germany:Bundesliga" "Spain:La Liga"
  python fbref_backfill_runner.py --season 2025 --emit-compat
  python fbref_backfill_runner.py --season 2025 --sleep-min 12 --sleep-max 25 --batch 12

Requires:
  pip install soccerdata pandas
"""

import os, json, time, math, random, argparse, traceback
from typing import Dict, List, Tuple
import pandas as pd

try:
    import soccerdata as sd
except Exception as e:
    raise SystemExit(
        "soccerdata is required. Install with: pip install soccerdata pandas\n"
        f"Import error: {e}"
    )

# -------------------- Politeness --------------------
def polite_sleep(low: float, high: float):
    time.sleep(random.uniform(low, high))

# -------------------- ID helpers --------------------
def stable_int_id(s: str) -> int:
    """Stable pseudo-ID for strings (team/player) when FBref ID absent."""
    return abs(hash(str(s))) % (10**9)

def coerce_int(x):
    try:
        return int(x)
    except Exception:
        return None

# -------------------- Schema builders --------------------
def fixtures_to_api_shape(schedule_df: pd.DataFrame) -> Dict:
    """Build API-Sports-like fixtures payload from FBref schedule (minimal fields you need)."""
    out = []
    for _, r in schedule_df.iterrows():
        mid = coerce_int(r.get("match_id"))
        if mid is None:
            # fallback to stable hashed id
            mid = stable_int_id(f"{r.get('home_team')} vs {r.get('away_team')} @ {r.get('date')}")
        round_str = str(r.get("round") or "")
        home = str(r.get("home_team") or "")
        away = str(r.get("away_team") or "")
        out.append({
            "fixture": {"id": int(mid)},
            "league": {"round": round_str},
            "teams": {
                "home": {"id": stable_int_id(home), "name": home},
                "away": {"id": stable_int_id(away), "name": away},
            }
        })
    return {"response": out}

def lineups_to_api_shape(lineups_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Build match_events dict shells with 'lineups' filled.
    Expected FBref columns include (names can vary by soccerdata version):
      match_id, team, player, position, started, player_id(?)
    """
    match_dict: Dict[str, Dict] = {}
    if not isinstance(lineups_df, pd.DataFrame) or lineups_df.empty:
        return match_dict

    # normalize columns
    cols = {c.lower(): c for c in lineups_df.columns}
    col_match = cols.get("match_id", "match_id")
    col_team  = cols.get("team", "team")
    col_player= cols.get("player", "player")
    col_pos   = cols.get("position", "position")
    col_started = cols.get("started", "started")
    col_player_id = cols.get("player_id", "player_id")

    for mid, dfm in lineups_df.groupby(col_match):
        key = str(int(coerce_int(mid) or stable_int_id(mid)))
        lineups = []
        for team, tdf in dfm.groupby(col_team):
            team_name = str(team or "")
            team_id   = stable_int_id(team_name)
            # starters only
            starters = tdf[tdf[col_started] == True] if col_started in tdf else tdf
            startXI = []
            for _, row in starters.iterrows():
                pid_raw = row.get(col_player_id)
                pid = coerce_int(pid_raw)
                if pid is None:
                    pid = stable_int_id(row.get(col_player))
                startXI.append({
                    "player": {
                        "id": int(pid),
                        "name": row.get(col_player),
                        "position": row.get(col_pos) or row.get("pos")
                    }
                })
            lineups.append({"team": {"id": team_id, "name": team_name}, "startXI": startXI})
        match_dict.setdefault(key, {"lineups": [], "events": []})
        match_dict[key]["lineups"] = lineups
    return match_dict

def map_event_type(t: str) -> str:
    """Map FBref/soccerdata event types to the minimal set SeasonAnalyzer expects."""
    t0 = (str(t) or "").lower()
    if "sub" in t0:
        return "subst"
    if "goal" in t0:
        return "Goal"
    # cards, fouls, etc. are ignored by current analyzer
    return t or ""

def append_events_to_matches(match_events: Dict[str, Dict], events_df: pd.DataFrame):
    """
    Append events into match_events dict under 'events' list.
    Expected FBref columns include:
      match_id, team, player, assist, minute, minute_stoppage_time, event_type/event, event_description
    """
    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return

    cols = {c.lower(): c for c in events_df.columns}
    col_match = cols.get("match_id", "match_id")
    col_team  = cols.get("team", "team")
    col_player= cols.get("player", "player")
    col_assist= cols.get("assist", "assist")
    col_min   = cols.get("minute", "minute")
    col_stopp = cols.get("minute_stoppage_time", "minute_stoppage_time")
    col_type1 = cols.get("event_type", "event_type")
    col_type2 = cols.get("event", "event")
    col_desc  = cols.get("event_description", "event_description")

    for mid, dfm in events_df.groupby(col_match):
        key = str(int(coerce_int(mid) or stable_int_id(mid)))
        recs = []
        for _, row in dfm.iterrows():
            etype = row.get(col_type1)
            if pd.isna(etype):
                etype = row.get(col_type2)
            etype = map_event_type(etype)
            # Only 'Goal' and 'subst' are used downstream
            minute = int(coerce_int(row.get(col_min)) or 0)
            extra  = int(coerce_int(row.get(col_stopp)) or 0)
            team_name = row.get(col_team)
            player = row.get(col_player)
            assist = row.get(col_assist)

            recs.append({
                "time": {"elapsed": minute, "extra": extra},
                "type": etype,
                "team": {"id": stable_int_id(team_name), "name": team_name},
                "player": {"id": stable_int_id(player), "name": player},
                "assist": {"id": stable_int_id(assist), "name": assist},
                "detail": row.get(col_desc)
            })
        match_events.setdefault(key, {"lineups": [], "events": []})
        match_events[key]["events"] = recs

# -------------------- Main backfill --------------------
def backfill_fbref(country: str,
                   league: str,
                   season_end_year: int,
                   out_root: str = "fbref_data",
                   batch_size: int = 15,
                   sleep_min: float = 10.0,
                   sleep_max: float = 20.0,
                   emit_compat: bool = False,
                   debug: bool = False) -> Dict:
    """
    Backfill one league-season (slow & polite).
    Returns a small report dict.
    """
    league_key = f"{country}-{league}"
    out_dir = os.path.join(out_root, f"{country}_{league}", str(int(season_end_year)))
    os.makedirs(out_dir, exist_ok=True)

    fb = sd.FBref(leagues=[league_key], seasons=season_end_year)

    # Schedule → fixtures
    schedule = fb.read_schedule()
    if not isinstance(schedule, pd.DataFrame) or schedule.empty:
        return {"status": "empty_schedule", "out_dir": out_dir}

    match_ids = []
    for mid in schedule["match_id"]:
        mid_int = coerce_int(mid)
        match_ids.append(int(mid_int if mid_int is not None else stable_int_id(mid)))

    fixtures_payload = fixtures_to_api_shape(schedule)

    # Resume support: load existing outputs
    fixtures_path = os.path.join(out_dir, "fixtures_fbref.json")
    events_path   = os.path.join(out_dir, "match_events_fbref.json")
    players_path  = os.path.join(out_dir, "players_fbref.json")

    if os.path.exists(events_path):
        with open(events_path, "r", encoding="utf-8") as f:
            match_events = json.load(f)
    else:
        match_events = {}

    # players map is built from lineups
    if os.path.exists(players_path):
        with open(players_path, "r", encoding="utf-8") as f:
            players_map = json.load(f)
    else:
        players_map = {}  # {team_name: [ {id,name,age,position}, ... ]}

    # Process in polite batches
    total = len(match_ids)
    processed = 0
    for i in range(0, total, batch_size):
        chunk = match_ids[i:i + batch_size]
        try:
            lineups_df = fb.read_lineup(match_id=chunk, force_cache=True)
        except Exception as e:
            if debug: print("[WARN] read_lineup failed:", e)
            lineups_df = pd.DataFrame()

        try:
            events_df  = fb.read_events(match_id=chunk,  force_cache=True)
        except Exception as e:
            if debug: print("[WARN] read_events failed:", e)
            events_df = pd.DataFrame()

        # Merge lineups
        lu_dict = lineups_to_api_shape(lineups_df)
        for k, v in lu_dict.items():
            match_events.setdefault(k, {"lineups": [], "events": []})
            match_events[k]["lineups"] = v["lineups"]
            # build/update team rosters
            for li in v["lineups"]:
                tname = li["team"]["name"]
                players_map.setdefault(tname, [])
                seen = {(p["id"], p["name"]) for p in players_map[tname]}
                for px in li.get("startXI", []):
                    pid = px["player"]["id"]
                    pname = px["player"]["name"]
                    pos = px["player"].get("position")
                    if (pid, pname) not in seen:
                        players_map[tname].append({"id": pid, "name": pname, "age": None, "position": pos})

        # Merge events
        append_events_to_matches(match_events, events_df)

        processed += len(chunk)
        # Save partials every batch (safe resume)
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump(match_events, f, ensure_ascii=False, indent=2)
        with open(players_path, "w", encoding="utf-8") as f:
            json.dump(players_map, f, ensure_ascii=False, indent=2)

        if debug:
            print(f"[{league_key} {season_end_year}] processed {processed}/{total}")
        polite_sleep(sleep_min, sleep_max)

    # Save fixtures last
    with open(fixtures_path, "w", encoding="utf-8") as f:
        json.dump(fixtures_payload, f, ensure_ascii=False, indent=2)

    # Completeness metrics
    # A match is "have" if it has both lineups and events lists present (non-empty)
    keys = {str(int(coerce_int(m) or stable_int_id(m))) for m in schedule["match_id"]}
    have = sum(
        1 for k in keys
        if (k in match_events and
            isinstance(match_events[k].get("lineups"), list) and len(match_events[k]["lineups"]) > 0 and
            isinstance(match_events[k].get("events"), list) and len(match_events[k]["events"]) > 0)
    )
    missing = len(keys) - have

    # Optional analyzer-compatible copies (safe: inside fbref_data root)
    if emit_compat:
        with open(os.path.join(out_dir, "fixtures.json"), "w", encoding="utf-8") as f:
            json.dump(fixtures_payload, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "match_events.json"), "w", encoding="utf-8") as f:
            json.dump(match_events, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "players.json"), "w", encoding="utf-8") as f:
            json.dump(players_map, f, ensure_ascii=False, indent=2)

    # Small report files
    report = {
        "country": country,
        "league": league,
        "season_end_year": int(season_end_year),
        "total_matches": int(len(keys)),
        "downloaded": int(have),
        "missing": int(missing),
        "out_dir": out_dir,
        "files": {
            "fixtures_fbref": fixtures_path,
            "match_events_fbref": events_path,
            "players_fbref": players_path
        }
    }
    with open(os.path.join(out_dir, "backfill_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    pd.DataFrame([report]).to_csv(os.path.join(out_dir, "backfill_report.csv"), index=False)
    return report

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="FBref backfill to distinct fbref_data tree.")
    p.add_argument("--season", type=int, required=True,
                   help="Season end year (e.g., 2025 for 2024/25).")
    p.add_argument("--leagues", nargs="+", default=["Germany:Bundesliga", "Spain:La Liga"],
                   help='Space-separated list like "Germany:Bundesliga" "Spain:La Liga".')
    p.add_argument("--out-root", default="fbref_data", help="Root folder for FBref outputs.")
    p.add_argument("--batch", type=int, default=15, help="Batch size for match_id requests.")
    p.add_argument("--sleep-min", type=float, default=10.0, help="Min sleep between batches (sec).")
    p.add_argument("--sleep-max", type=float, default=20.0, help="Max sleep between batches (sec).")
    p.add_argument("--emit-compat", action="store_true",
                   help="Also write analyzer-compatible copies (fixtures.json, etc.) into the FBref folder.")
    p.add_argument("--debug", action="store_true", help="Print progress and warnings.")
    return p.parse_args()

def main():
    args = parse_args()
    reports = []
    for item in args.leagues:
        try:
            country, league = [x.strip() for x in item.split(":", 1)]
        except ValueError:
            print(f"[WARN] Bad --leagues entry (expected Country:League): {item}")
            continue
        try:
            rep = backfill_fbref(
                country=country,
                league=league,
                season_end_year=args.season,
                out_root=args.out_root,
                batch_size=args.batch,
                sleep_min=args.sleep_min,
                sleep_max=args.sleep_max,
                emit_compat=args.emit_compat,
                debug=args.debug,
            )
            if args.debug:
                print(rep)
            reports.append(rep)
        except Exception as e:
            print(f"[ERROR] {country} / {league} / {args.season}")
            if args.debug:
                print("".join(traceback.format_exception(type(e), e, e.__traceback__))[:4000])
            else:
                print(f"{type(e).__name__}: {e}")

    # summary file at root
    if reports:
        root = args.out_root
        os.makedirs(root, exist_ok=True)
        pd.DataFrame(reports).to_csv(os.path.join(root, "fbref_backfill_summary.csv"), index=False)
        with open(os.path.join(root, "fbref_backfill_summary.json"), "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
