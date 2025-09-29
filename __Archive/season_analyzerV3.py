#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeasonAnalyzer: build segment-level plus/minus, WLS & Lasso impacts.

Key changes vs your draft
-------------------------
- Keeps `self.player_cols` and `self.all_player_ids`.
- Avoids per-row DataFrame concat (collect dicts -> single DataFrame).
- Fixes helper ordering, guards, types, and NaN handling.
- Separates meta vs player matrices cleanly; consistent column ordering.
- Makes save paths explicit; fewer silent assumptions.
- Adds light logging prints (can swap to logging module easily).
"""

from __future__ import annotations
import os, json, sqlite3
from typing import Dict, List, Set, Optional, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from threadpoolctl import threadpool_limits
from sklearn.linear_model import Lasso, Ridge

import collections

class SeasonAnalyzer:   
    def __init__(
        self,
        league_name: str = "Bundesliga",
        country_name: str = "Germany",
        season: int = 2023,
        lasso_alphas: List[float] | float = (0.01, 0.001),
        ridge_alphas: List[float] | float = (0.1, 1.0),  # NEW: ridge grid
        save_intermediate: bool = False,
        base_dir: Optional[str] = None,
        date_cutoff: Optional[str] = None,
    ):
        # Config
        self.LEAGUE_NAME = league_name
        self.COUNTRY_NAME = country_name
        self.SEASON = season
        self.SAVE_DIR = os.path.join(base_dir or "", f"{country_name}_{league_name}", str(season))
        self.FIXTURES_FILE = "fixtures.json"
        self.EVENTS_FILE = "match_events.json"
        self.PLAYER_FILE = "players.json"
        self.lasso_alphas = list(lasso_alphas) if isinstance(lasso_alphas, (list, tuple, set)) else [float(lasso_alphas)]
        # NEW: ridge alphas
        self.ridge_alphas = list(ridge_alphas) if isinstance(ridge_alphas, (list, tuple, set)) else [float(ridge_alphas)]

        # self.minutes_equiv_tolerance = 0
    
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.DB_PATH = os.path.join(script_dir, "analysis_results.db")
    
        # Data containers
        self.fixtures_data: List[Dict[str, Any]] | None = None
        self.match_events: Dict[str, Dict[str, Any]] | None = None
        self.players: Dict[str, Any] | None = None
        # NEW: extra containers produced by the one-pass prep module
        self.players_meta_df: pd.DataFrame | None = None
        self.team_presence_signed_df: pd.DataFrame | None = None
        self.team_presence_abs_df: pd.DataFrame | None = None
        
        # NEW: UTC-aware cutoff (day-level logic applied downstream)
        self.date_cutoff_utc = pd.to_datetime(date_cutoff, utc=True) if date_cutoff else None

    
        self.team_players: Dict[int, Set[int]] = {}     # {team_id: set(player_ids)}
        self.player_id_to_name: Dict[int, str] = {}
        self.player_id_to_position: Dict[int, str] = {}
        self.player_id_to_age: Dict[int, Optional[int]] = {}
    
        # Matrices / tables
        self.segment_df: pd.DataFrame | None = None      # meta + player indicators
        self.match_summary_df: pd.DataFrame | None = None
        self.team_summary_df: pd.DataFrame | None = None
        self.coef_df_final: pd.DataFrame | None = None
        self.coef_df_pooled: pd.DataFrame | None = None  # NEW: collapsed view (pooled)
        self.player_game_metrics: Dict[int, Dict[str, float | int]] = {}
    
        # Accounting
        self.total_regular_games = 0
        self.valid_regular_games = 0
        self.missing_games_count = 0
        self.valid_fixture_ids: Set[str] = set()
    
        # Columns for players (persisted)
        self.all_player_ids: List[int] = []
        self.player_cols: List[str] = []  # e.g., ["p123", "p456", ...]
    
        self.save_intermediate = save_intermediate
    
        # NEW: cutoff timestamp or None
        # self.date_cutoff = pd.to_datetime(date_cutoff) if date_cutoff else None
        # NEW: cutoff timestamp (tz-aware UTC) or None
        self.date_cutoff_utc = pd.to_datetime(date_cutoff, utc=True) if date_cutoff else None

    # ---------- IO ----------
    def _p(self, filename: str) -> str:
        return os.path.join(self.SAVE_DIR, filename)
    
    def _to_utc(self, dt_like) -> Optional[pd.Timestamp]:
        """Parse a date/time string (or Timestamp) to a tz-aware UTC Timestamp; NaT -> None."""
        if dt_like is None:
            return None
        ts = pd.to_datetime(dt_like, utc=True, errors="coerce")
        return None if pd.isna(ts) else ts


    def load_data(self) -> None:
        """Load fixtures, events, and players data from JSON files."""
        with open(self._p(self.FIXTURES_FILE), "r", encoding="utf-8") as f:
            self.fixtures_data = json.load(f)["response"]
        with open(self._p(self.EVENTS_FILE), "r", encoding="utf-8") as f:
            self.match_events = json.load(f)
        with open(self._p(self.PLAYER_FILE), "r", encoding="utf-8") as f:
            self.players = json.load(f)

    # ---------- helpers ----------
    @staticmethod
    def _get_minute(ev: Dict[str, Any]) -> int:
        ti = ev.get("time", {}) or {}
        extra = ti.get("extra") or 0
        elapsed = ti.get("elapsed") or 0
        return int(extra) + int(elapsed)

    @staticmethod
    def _is_regular(fix: Dict[str, Any]) -> bool:
        round_str = (fix.get("league", {}) or {}).get("round", "") or ""
        return round_str.startswith("Regular Season")
    
    def _to_utc(self, ts_like) -> Optional[pd.Timestamp]:
        """Parse date/time to tz-aware UTC pandas Timestamp; return None on failure."""
        if ts_like is None:
            return None
        ts = pd.to_datetime(ts_like, utc=True, errors="coerce")
        return None if pd.isna(ts) else ts


    def _fixture_has_required_data(self, fid: str) -> bool:
        ev = (self.match_events or {}).get(fid, {})
        return bool(ev.get("lineups")) and bool(ev.get("events"))

    # ---------- pipeline steps ----------
    
    def prep_data_up_to_cutoff(self, *, keep_player_after_red: bool = False) -> None:
        """
        One-pass season preparation up to (and including) self.date_cutoff_utc.day():
          - Build segment matrix (players) with meta and goal_diff per segment
          - Build team presence matrices (signed and absolute) per segment
          - Update per-player meta (starts, subs on/off, full games, minutes, FTE)
          - Tally goals/assists, penalties (scored/missed), own goals, yellow cards
          - Build match_summary and team_summary
        Skips fixtures with missing lineups or events (prints a short note).
        Segment end minute = last event minute (not forced to 90).
        Presence dtypes = int8.
        """
        assert self.fixtures_data is not None and self.match_events is not None and self.players is not None, \
            "Call load_data() first."
    
        # Helpers
        def allowed_fixture(fix) -> bool:
            if not self._is_regular(fix):
                return False
            if self.date_cutoff_utc is None:
                return True
            fdt = self._to_utc(fix.get("fixture", {}).get("date"))
            return (fdt is not None) and (fdt.date() <= self.date_cutoff_utc.date())
    
        def minute(ev) -> int:
            ti = ev.get("time", {}) or {}
            return int(ti.get("elapsed") or 0) + int(ti.get("extra") or 0)
    
        def is_red(ev) -> bool:
            if (ev.get("type") or "").lower() != "card":
                return False
            d = (ev.get("detail") or "").lower()
            return ("red" in d) or ("second yellow" in d)
    
        def is_yellow(ev) -> bool:
            if (ev.get("type") or "").lower() != "card":
                return False
            return "yellow" in (ev.get("detail") or "").lower()
    
        def goal_class(ev) -> str:
            if (ev.get("type") or "").lower() != "goal":
                return ""
            d = (ev.get("detail") or "").lower()
            if "own" in d:
                return "own"
            if "pen" in d and "miss" not in d:
                return "pen_scored"
            return "open"
    
        def is_pen_missed(ev) -> bool:
            t = (ev.get("type") or "").lower()
            d = (ev.get("detail") or "").lower()
            return ("miss" in d and "pen" in d) or (t == "missed penalty")
    
        # Seed players_meta from players.json (name/position only; no age)
        # Robust to different shapes; we also add players seen only in fixtures on the fly
        self.players_meta_df = pd.DataFrame(columns=[
            "player_id","player_name","position",
            "primary_team_id","primary_team_name",
            "games_started","full_games_played","games_subbed_on","games_subbed_off",
            "minutes_played","FTE_games_played",
            "goals","assists","pen_goals","pen_assists","pen_missed","own_goals","yellow_cards",
            "first_seen_utc","last_seen_utc"
        ]).set_index("player_id", drop=False)
    
        def add_or_update_player(pid:int, name:str="Unknown", pos:str="Unknown"):
            if pid not in self.players_meta_df.index:
                self.players_meta_df.loc[pid, :] = [
                    pid, str(name), str(pos), np.nan, np.nan,
                    0, 0, 0, 0, 0.0, 0.0,
                    0, 0, 0, 0, 0, 0, 0,    # goals, assists, pen_goals, pen_assists, pen_missed, own_goals, yellow_cards
                    pd.NaT, pd.NaT
                ]
    
        # Preload from self.players (best-effort)
        if isinstance(self.players, dict):
            # common shapes: {team_id: {...}} or {"players":[...]} or {"response":[...]}
            if "players" in self.players and isinstance(self.players["players"], list):
                for p in self.players["players"]:
                    pid = p.get("id") or p.get("player_id") or (p.get("player") or {}).get("id")
                    if pid is None: continue
                    try: pid = int(pid)
                    except: continue
                    name = p.get("name") or (p.get("player") or {}).get("name") or "Unknown"
                    pos  = (p.get("player") or {}).get("pos", (p.get("player") or {}).get("position","Unknown"))
                    add_or_update_player(pid, name, pos)
            else:
                for _, block in self.players.items():
                    if not isinstance(block, dict): continue
                    plist = block.get("players")
                    if isinstance(plist, list):
                        for p in plist:
                            pid = p.get("id") or p.get("player_id") or (p.get("player") or {}).get("id")
                            if pid is None: continue
                            try: pid = int(pid)
                            except: continue
                            name = p.get("name") or (p.get("player") or {}).get("name") or "Unknown"
                            pos  = (p.get("player") or {}).get("pos", (p.get("player") or {}).get("position","Unknown"))
                            add_or_update_player(pid, name, pos)
        elif isinstance(self.players, list):
            for item in self.players:
                if not isinstance(item, dict): continue
                team_players = item.get("players")
                if isinstance(team_players, list):
                    for p in team_players:
                        pid = p.get("id") or p.get("player_id") or (p.get("player") or {}).get("id")
                        if pid is None: continue
                        try: pid = int(pid)
                        except: continue
                        name = p.get("name") or (p.get("player") or {}).get("name") or "Unknown"
                        pos  = (p.get("player") or {}).get("pos", (p.get("player") or {}).get("position","Unknown"))
                        add_or_update_player(pid, name, pos)
    
        # Accumulators
        meta_rows: List[Dict[str, Any]] = []
        player_rows: List[Dict[str,int]] = []
        team_rows_signed: List[Dict[str,int]] = []
        team_rows_abs: List[Dict[str,int]] = []
    
        player_ids_seen: Set[int] = set()
        team_ids_seen: Set[int] = set()
        id_to_teamname: Dict[int,str] = {}
    
        # For match/team summaries and player minutes/appearances
        match_rows: List[Dict[str,Any]] = []
        minutes_by_player_total: Dict[int,float] = collections.defaultdict(float)
        first_seen: Dict[int,pd.Timestamp] = {}
        last_seen: Dict[int,pd.Timestamp] = {}
        games_started: Dict[int,int] = collections.defaultdict(int)
        full_games: Dict[int,int] = collections.defaultdict(int)
        subs_on: Dict[int,int] = collections.defaultdict(int)
        subs_off: Dict[int,int] = collections.defaultdict(int)
    
        # For primary team decision: minutes per (player, team) and last-seen per (player, team)
        minutes_by_player_team: Dict[int, Dict[int,float]] = collections.defaultdict(lambda: collections.defaultdict(float))
        last_seen_by_player_team: Dict[int, Dict[int,pd.Timestamp]] = collections.defaultdict(dict)
    
        # Scoring tallies
        goals: Dict[int,int] = collections.defaultdict(int)
        assists: Dict[int,int] = collections.defaultdict(int)
        pen_goals: Dict[int,int] = collections.defaultdict(int)
        pen_assists: Dict[int,int] = collections.defaultdict(int)
        pen_missed: Dict[int,int] = collections.defaultdict(int)
        own_goals: Dict[int,int] = collections.defaultdict(int)
        yellow_cards: Dict[int,int] = collections.defaultdict(int)
    
        # Accounting
        reg_fixture_ids: List[str] = []
        skipped_missing = 0
    
        # Pass over fixtures
        for fix in self.fixtures_data:
            if not allowed_fixture(fix):
                continue
    
            fid = str(fix["fixture"]["id"])
            evpack = self.match_events.get(fid, {})
            if not (evpack and evpack.get("lineups") and evpack.get("events")):
                skipped_missing += 1
                continue
    
            reg_fixture_ids.append(fid)
            home_id = fix["teams"]["home"]["id"]; away_id = fix["teams"]["away"]["id"]
            home_team = fix["teams"]["home"]["name"]; away_team = fix["teams"]["away"]["name"]
            team_ids_seen.update([home_id, away_id])
            id_to_teamname[home_id] = home_team; id_to_teamname[away_id] = away_team
    
            fdt_utc = self._to_utc(fix.get("fixture", {}).get("date"))
    
            events = evpack.get("events", []) or []
            match_end_min = max([minute(e) for e in events])
    
            # Starters / started flags (also ensure players exist in meta)
            starters: Set[int] = set()
            for lu in evpack.get("lineups", []) or []:
                sign = 1 if lu["team"]["id"] == home_id else -1
                for px in lu.get("startXI", []) or []:
                    pid = int(px["player"]["id"])
                    starters.add(pid)
                    games_started[pid] += 1
                    add_or_update_player(pid, px["player"].get("name","Unknown"),
                                         px["player"].get("pos", px["player"].get("position","Unknown")))
                    self.player_id_to_name[pid] = px["player"].get("name","Unknown")
                    self.player_id_to_position[pid] = px["player"].get("pos", px["player"].get("position","Unknown"))
    
            # Substitution & red moments
            subs = [e for e in events if (e.get("type") or "").lower() == "subst"]
            subs_by_min = collections.defaultdict(list)
            for e in subs:
                subs_by_min[minute(e)].append(e)
    
            reds_by_min: Dict[int, List[int]] = collections.defaultdict(list)
            if not keep_player_after_red:
                for e in events:
                    if is_red(e):
                        m = minute(e)
                        pid_rc = (e.get("player") or {}).get("id")
                        if pid_rc:
                            reds_by_min[m].append(int(pid_rc))
    
            boundaries = sorted(set(list(subs_by_min.keys()) + list(reds_by_min.keys()) + [match_end_min]))
    
            # Initialize on-field from starting lineups
            current: Dict[int,int] = {}
            for lu in evpack.get("lineups", []) or []:
                sign = 1 if lu["team"]["id"] == home_id else -1
                for px in lu.get("startXI", []) or []:
                    pid = int(px["player"]["id"])
                    current[pid] = sign
                    player_ids_seen.add(pid)
    
            # Build segments and accumulate minutes
            seg_start = 0
            # Also collect match goal totals and unchanged starters
            home_goal_total = 0; away_goal_total = 0
            subs_out_set: Set[int] = set()
    
            for seg_end in boundaries:
                if seg_end <= seg_start:
                    continue
                dur = int(seg_end - seg_start)
    
                # goals within (seg_start, seg_end]
                hg = 0; ag = 0
                for e in events:
                    if (e.get("type") or "").lower() != "goal":
                        continue
                    m = minute(e)
                    if not (seg_start < m <= seg_end):
                        continue
                    if (e.get("team") or {}).get("id") == home_id:
                        hg += 1
                    elif (e.get("team") or {}).get("id") == away_id:
                        ag += 1
                home_goal_total += hg; away_goal_total += ag
    
                # sparse presence rows
                prow: Dict[str,int] = {f"p{pid}": (1 if sign > 0 else -1) for pid, sign in current.items()}
                player_rows.append(prow)
                meta_rows.append({
                    "fixture_id": fid,
                    "game_date_utc": fdt_utc,
                    "home_id": home_id, "away_id": away_id,
                    "home_team": home_team, "away_team": away_team,
                    "start": int(seg_start), "end": int(seg_end),
                    "duration": dur, "goal_diff": int(hg - ag)
                })
                team_rows_signed.append({f"t{home_id}": 1, f"t{away_id}": -1})
                team_rows_abs.append({f"t{home_id}": 1, f"t{away_id}": 1})
    
                # accumulate minutes per player and per team for primary team logic
                for pid, sign in current.items():
                    minutes_by_player_total[pid] += dur
                    player_ids_seen.add(pid)
                    # first/last seen
                    if pid not in first_seen or fdt_utc < first_seen[pid]:
                        first_seen[pid] = fdt_utc
                    if pid not in last_seen or fdt_utc > last_seen[pid]:
                        last_seen[pid] = fdt_utc
                    # attribute minutes to team by sign
                    if sign > 0:
                        minutes_by_player_team[pid][home_id] += dur
                        last_seen_by_player_team[pid][home_id] = fdt_utc
                    else:
                        minutes_by_player_team[pid][away_id] += dur
                        last_seen_by_player_team[pid][away_id] = fdt_utc
    
                # apply subs at boundary
                for e in subs_by_min.get(seg_end, []):
                    pid_out = (e.get("player") or {}).get("id")
                    pid_in  = (e.get("assist") or {}).get("id")
                    tm_id   = (e.get("team") or {}).get("id")
                    if pid_out:
                        pid_out = int(pid_out)
                        subs_off[pid_out] += 1
                        subs_out_set.add(pid_out)
                        current.pop(pid_out, None)
                    if pid_in:
                        pid_in = int(pid_in)
                        subs_on[pid_in] += 1
                        current[pid_in] = 1 if tm_id == home_id else -1
                        # ensure metadata exists
                        add_or_update_player(pid_in)
    
                # apply reds at boundary
                if not keep_player_after_red and seg_end in reds_by_min:
                    for pid_rc in reds_by_min[seg_end]:
                        current.pop(int(pid_rc), None)
    
                seg_start = seg_end
    
            # full games = starters not subbed off
            for pid in (starters - subs_out_set):
                full_games[pid] += 1
    
            # match summary (unchanged starters, sub moments)
            subs_moments = {( (e.get("team") or {}).get("id"), minute(e) ) for e in subs}
            unchanged = len(starters - set([ (e.get("player") or {}).get("id") for e in subs if (e.get("player") or {}).get("id") ]))
            # safer unchanged: based on subs_out_set
            unchanged = len(starters - subs_out_set)
    
            round_str = (fix.get("league", {}) or {}).get("round", "")
            try:
                matchday = int(str(round_str).split("-")[-1].strip())
            except Exception:
                matchday = None
    
            match_rows.append({
                "matchday": matchday,
                "fixture_id": fid,
                "home_team": home_team, "away_team": away_team,
                "home_team_id": home_id, "away_team_id": away_id,
                "home_goals": int(home_goal_total), "away_goals": int(away_goal_total),
                "sub_events": int(len(subs_moments)), "unchanged_players": int(unchanged)
            })
    
            # Scoring tallies (whole match)
            for ev in events:
                # yellow cards
                if is_yellow(ev):
                    pid = (ev.get("player") or {}).get("id")
                    if pid: yellow_cards[int(pid)] += 1
                # missed pens
                if is_pen_missed(ev):
                    pid = (ev.get("player") or {}).get("id")
                    if pid: pen_missed[int(pid)] += 1
                # goals
                if (ev.get("type") or "").lower() == "goal":
                    who = (ev.get("player") or {}).get("id")
                    ast = (ev.get("assist") or {}).get("id")
                    cls = goal_class(ev)
                    if who:
                        who = int(who)
                        add_or_update_player(who)
                        if cls == "own":
                            own_goals[who] += 1
                        elif cls == "pen_scored":
                            pen_goals[who] += 1
                            goals[who] += 1
                            if ast:
                                ast = int(ast); pen_assists[ast] += 1; assists[ast] += 1
                        else:
                            goals[who] += 1
                            if ast:
                                assists[int(ast)] += 1
    
            # update seen team names
            id_to_teamname[home_id] = home_team
            id_to_teamname[away_id] = away_team
    
        # Accounting for total/valid/missing games
        self.total_regular_games = sum(1 for f in self.fixtures_data if self._is_regular(f) and allowed_fixture(f))
        self.valid_fixture_ids = set(reg_fixture_ids)
        self.valid_regular_games = len(self.valid_fixture_ids)
        self.missing_games_count = self.total_regular_games - self.valid_regular_games
        if skipped_missing:
            print(f"[info] Skipped {skipped_missing} fixtures (missing lineups or events).")
    
        # Build DataFrames for segments and teams
        if not meta_rows:
            # No data — initialize empty frames consistent with your attributes
            self.segment_df = pd.DataFrame(columns=[
                "fixture_id","game_date_utc","home_id","away_id","home_team","away_team",
                "start","end","duration","goal_diff"
            ])
            self.team_presence_signed_df = pd.DataFrame()
            self.team_presence_abs_df = pd.DataFrame()
            self.match_summary_df = pd.DataFrame()
            self.team_summary_df = pd.DataFrame()
            # still ensure player_cols
            self.all_player_ids = sorted(list(player_ids_seen))
            self.player_cols = [f"p{pid}" for pid in self.all_player_ids]
            return
    
        meta_df = pd.DataFrame(meta_rows)
        self.all_player_ids = sorted(list(player_ids_seen))
        self.player_cols = [f"p{pid}" for pid in self.all_player_ids]
        players_mat_df = pd.DataFrame(player_rows).reindex(columns=self.player_cols, fill_value=0).fillna(0)
        players_mat_df = players_mat_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype("int8")
    
        team_cols = [f"t{tid}" for tid in sorted(list(team_ids_seen))]
        teams_signed_df = pd.DataFrame(team_rows_signed).reindex(columns=team_cols, fill_value=0).fillna(0)
        teams_signed_df = teams_signed_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype("int8")
        teams_abs_df = pd.DataFrame(team_rows_abs).reindex(columns=team_cols, fill_value=0).fillna(0)
        teams_abs_df = teams_abs_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype("int8")
    
        self.segment_df = pd.concat([meta_df.reset_index(drop=True), players_mat_df.reset_index(drop=True)], axis=1)
        self.team_presence_signed_df = teams_signed_df
        self.team_presence_abs_df = teams_abs_df
    
        # Update per-player meta dataframe
        # Ensure presence for players seen only in fixtures
        for pid in self.all_player_ids:
            add_or_update_player(pid, self.player_id_to_name.get(pid,"Unknown"),
                                 self.player_id_to_position.get(pid,"Unknown"))
    
        # Minutes & appearances
        self.players_meta_df.loc[self.all_player_ids, "minutes_played"] = [
            float(minutes_by_player_total.get(pid, 0.0)) for pid in self.all_player_ids
        ]
        self.players_meta_df.loc[self.all_player_ids, "FTE_games_played"] = (
            self.players_meta_df.loc[self.all_player_ids, "minutes_played"] / 90.0
        ).astype(float).round(3)
        self.players_meta_df.loc[self.all_player_ids, "games_started"] = [int(games_started.get(pid,0)) for pid in self.all_player_ids]
        self.players_meta_df.loc[self.all_player_ids, "full_games_played"] = [int(full_games.get(pid,0)) for pid in self.all_player_ids]
        self.players_meta_df.loc[self.all_player_ids, "games_subbed_on"] = [int(subs_on.get(pid,0)) for pid in self.all_player_ids]
        self.players_meta_df.loc[self.all_player_ids, "games_subbed_off"] = [int(subs_off.get(pid,0)) for pid in self.all_player_ids]
        self.players_meta_df.loc[self.all_player_ids, "first_seen_utc"] = [first_seen.get(pid, pd.NaT) for pid in self.all_player_ids]
        self.players_meta_df.loc[self.all_player_ids, "last_seen_utc"]  = [last_seen.get(pid, pd.NaT) for pid in self.all_player_ids]
    
        # Scoring/cards
        for col, d in [
            ("goals", goals), ("assists", assists),
            ("pen_goals", pen_goals), ("pen_assists", pen_assists),
            ("pen_missed", pen_missed), ("own_goals", own_goals),
            ("yellow_cards", yellow_cards),
        ]:
            self.players_meta_df[col] = self.players_meta_df.index.to_series().map(pd.Series(d)).fillna(0).astype(int)
    
        # Primary team (single scalar id/name)
        prim_ids = []
        prim_names = []
        for pid in self.players_meta_df.index:
            tmins = minutes_by_player_team.get(pid, {})
            if not tmins:
                prim_ids.append(np.nan); prim_names.append(np.nan); continue
            maxm = max(tmins.values())
            cands = [tid for tid, m in tmins.items() if m == maxm]
            if len(cands) == 1:
                choice = cands[0]
            else:
                # tie-break: most recent appearance then lowest id
                last_map = last_seen_by_player_team.get(pid, {})
                latest = None; choice = None
                for tid in cands:
                    dt = last_map.get(tid, pd.Timestamp(0, tz="UTC"))
                    if latest is None or dt > latest or (dt == latest and (choice is None or tid < choice)):
                        latest = dt; choice = tid
            prim_ids.append(choice)
            prim_names.append(id_to_teamname.get(choice, np.nan))
        self.players_meta_df["primary_team_id"] = prim_ids
        self.players_meta_df["primary_team_name"] = prim_names
    
        # Build match summary dataframe
        self.match_summary_df = pd.DataFrame(match_rows)
    
        # Build team summary from match summary and segments
        team_rows_out: List[Dict[str,Any]] = []
        total_minutes = float(self.segment_df["duration"].sum())
        if not self.match_summary_df.empty and total_minutes > 0:
            tm = self.match_summary_df
            team_ids_all = sorted(set(tm["home_team_id"]).union(tm["away_team_id"]))
            for team_id in team_ids_all:
                mask_home = (tm["home_team_id"] == team_id)
                mask_away = (tm["away_team_id"] == team_id)
                wins   = (mask_home & (tm["home_goals"] > tm["away_goals"])).sum() + (mask_away & (tm["away_goals"] > tm["home_goals"])).sum()
                draws  = (tm["home_goals"] == tm["away_goals"]).sum()
                losses = len(tm) - wins - draws
                goals_scored   = (mask_home * tm["home_goals"] + mask_away * tm["away_goals"]).sum()
                goals_conceded = (mask_home * tm["away_goals"] + mask_away * tm["home_goals"]).sum()
    
                # total_players_used and always_playing based on presence
                pcols = self.player_cols
                total_players_used = 0
                always_playing = 0
                if pcols:
                    df = self.segment_df[["home_id","away_id","duration"] + pcols]
                    mins_by_player = collections.defaultdict(float)
                    for _, r in df.iterrows():
                        sign_row = 1 if r["home_id"] == team_id else (-1 if r["away_id"] == team_id else 0)
                        if sign_row == 0: continue
                        dur = float(r["duration"])
                        for pc in pcols:
                            if int(r[pc]) == sign_row:
                                mins_by_player[int(pc[1:])] += dur
                    for pid, m in mins_by_player.items():
                        if m > 0: total_players_used += 1
                        if m == total_minutes: always_playing += 1
    
                team_rows_out.append({
                    "team_id": int(team_id),
                    "wins": int(wins), "draws": int(draws), "losses": int(losses),
                    "goals_scored": int(goals_scored), "goals_conceded": int(goals_conceded),
                    "total_players_used": int(total_players_used),
                    "always_playing": int(always_playing),
                })
            self.team_summary_df = pd.DataFrame(team_rows_out)
            if not self.team_summary_df.empty:
                self.team_summary_df["points"] = (self.team_summary_df["wins"] * 3 + self.team_summary_df["draws"])
                self.team_summary_df.sort_values(by=["points","goals_scored"], ascending=[False, False], inplace=True)
                self.team_summary_df.reset_index(drop=True, inplace=True)
                self.team_summary_df["league_position"] = self.team_summary_df.index + 1
        else:
            self.team_summary_df = pd.DataFrame(columns=[
                "team_id","wins","draws","losses","goals_scored","goals_conceded",
                "total_players_used","always_playing","points","league_position"
            ])
 

    def add_goals_assists(self) -> None:
        """Add total goals and assists for each player (valid fixtures only)."""
        assert self.fixtures_data is not None and self.match_events is not None and self.coef_df_final is not None

        goals: Dict[int, int] = {}
        assists: Dict[int, int] = {}

        for fix in self.fixtures_data:
            fid = str(fix["fixture"]["id"])
            if fid not in self.valid_fixture_ids:
                continue
            for ev in (self.match_events[fid].get("events") or []):
                if ev.get("type") == "Goal":
                    detail = ev.get("detail", "") or ""
                    if isinstance(detail, str) and "Own Goal" in detail:
                        continue
                    scorer = (ev.get("player") or {}).get("id")
                    helper = (ev.get("assist") or {}).get("id")
                    if scorer: goals[scorer] = goals.get(scorer, 0) + 1
                    if helper: assists[helper] = assists.get(helper, 0) + 1

        coef_df = self.coef_df_final
        coef_df["goals"] = coef_df["player_id"].apply(lambda pid: goals.get(int(pid[1:]), 0) if pid.startswith("p") else 0)
        coef_df["assists"] = coef_df["player_id"].apply(lambda pid: assists.get(int(pid[1:]), 0) if pid.startswith("p") else 0)
        self.coef_df_final = coef_df

    def compute_impacts(self) -> None:
        """Compute WLS plus/minus and Lasso regularized impacts + pooling flags."""
        assert self.segment_df is not None
    
        # Target and weights
        y = pd.to_numeric(self.segment_df["goal_diff"], errors="coerce").fillna(0.0).astype(float)
        w = pd.to_numeric(self.segment_df["duration"], errors="coerce").fillna(1.0).astype(float)
    
        # Design matrix X = players only
        X_df = self.segment_df[self.player_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    
        # --- WLS
        def fit_wls(X: pd.DataFrame, y: pd.Series, w: pd.Series) -> tuple[pd.Series, pd.Series]:
            Xc = sm.add_constant(X, has_constant="add")
            y_arr = y.to_numpy(dtype=np.float64)
            w_arr = w.to_numpy(dtype=np.float64)
            X_arr = Xc.to_numpy(dtype=np.float64)
            valid = np.isfinite(y_arr) & np.isfinite(w_arr) & np.isfinite(X_arr).all(axis=1)
            if not np.any(valid):
                raise ValueError("No valid data points for WLS regression.")
            with threadpool_limits(1):  # determinism & stability
                model = sm.WLS(y_arr[valid], X_arr[valid], weights=w_arr[valid]).fit(method="pinv")
            params = pd.Series(model.params, index=Xc.columns)
            bse = pd.Series(model.bse, index=Xc.columns)
            return params.drop("const", errors="ignore"), bse.drop("const", errors="ignore")
    
        coef, err = fit_wls(X_df, y, w)
    
        # Results frame
        coef_df = pd.DataFrame({
            "player_id": coef.index,              # "p123", ...
            "impact": coef.values,
            "impact_std": err.reindex(coef.index).values
        })
     
        # Minutes played from segment presence (NumPy path to avoid Pandas nanops/BLAS issues)
        presence_np = X_df.abs().to_numpy(dtype=np.float64, copy=False)
        dur = pd.to_numeric(self.segment_df["duration"], errors="coerce").fillna(0).to_numpy(dtype=np.float64, copy=False)
        if presence_np.shape[0] != dur.shape[0]:
            raise ValueError(
                f"Shape mismatch: presence rows {presence_np.shape[0]} vs duration {dur.shape[0]}"
            )
        # single-thread for safety/determinism
        with threadpool_limits(1):
            mp = (presence_np * dur[:, None]).sum(axis=0)
        minutes_played = pd.Series(mp, index=X_df.columns)
        coef_df["minutes_played"] = (
            coef_df["player_id"].map(minutes_played.to_dict()).fillna(0.0).astype(float)
        )

        # presence_matrix = X_df.abs()
        # minutes_played = presence_matrix.mul(self.segment_df["duration"].to_numpy(), axis=0).sum()
        # coef_df["minutes_played"] = coef_df["player_id"].map(minutes_played.to_dict()).fillna(0.0).astype(float)
    
        # Observation flag (any minutes > 0)
        coef_df["no_observations"] = coef_df["minutes_played"].le(0.0)
    
        # Map player names & positions
        def _name_from_pid(pid_str: str) -> str:
            if not pid_str.startswith("p"): return pid_str
            try:
                return self.player_id_to_name.get(int(pid_str[1:]), "Unknown")
            except Exception:
                return "Unknown"
    
        def _pos_from_pid(pid_str: str) -> str:
            if not pid_str.startswith("p"): return "Unknown"
            try:
                return self.player_id_to_position.get(int(pid_str[1:]), "Unknown")
            except Exception:
                return "Unknown"
    
        coef_df["player_name"] = coef_df["player_id"].apply(_name_from_pid)
        coef_df["position"] = coef_df["player_id"].apply(_pos_from_pid)
    
        # Team(s) from segments (home=+1, away=-1)
        def _teams_from_pid(pid_str: str) -> Optional[str]:
            if self.segment_df is None or pid_str not in self.segment_df:
                return None
            homes = self.segment_df.loc[self.segment_df[pid_str] == 1, "home_team"].unique().tolist()
            aways = self.segment_df.loc[self.segment_df[pid_str] == -1, "away_team"].unique().tolist()
            teams = sorted(set(homes + aways))
            if len(teams) == 1: return teams[0]
            if len(teams) > 1:  return f"Multiple Teams ({', '.join(teams)})"
            return None
    
        coef_df["team(s)"] = coef_df["player_id"].apply(_teams_from_pid)
    
        # League positions map
        id_to_name = {}
        for fix in (self.fixtures_data or []):
            id_to_name[fix["teams"]["home"]["id"]] = fix["teams"]["home"]["name"]
            id_to_name[fix["teams"]["away"]["id"]] = fix["teams"]["away"]["name"]
    
        team_position_map: Dict[str, int] = {}
        if self.team_summary_df is not None and len(self.team_summary_df):
            for _, row in self.team_summary_df.iterrows():
                nm = id_to_name.get(int(row["team_id"]), "")
                team_position_map[nm] = int(row["league_position"])
    
        def _league_pos(team: Optional[str]) -> Optional[int | str]:
            if team is None: return None
            if str(team).startswith("Multiple"): return "Multiple Teams"
            return team_position_map.get(str(team))
    
        coef_df["league_position"] = coef_df["team(s)"].apply(_league_pos)
    
        # --- pooling membership detection (identical presence signature)
        presence_int = X_df.astype("int8")
        sig_to_players: Dict[tuple, List[str]] = {}
        for col in presence_int.columns:
            signature = tuple(presence_int[col].tolist())
            sig_to_players.setdefault(signature, []).append(col)
    
        def _peers(pid_str: str) -> str:
            if pid_str not in presence_int.columns:
                return ""
            sig = tuple(presence_int[pid_str].tolist())
            peers = [c for c in sig_to_players.get(sig, []) if c != pid_str]
            return ", ".join(self.player_id_to_name.get(int(p[1:]), "Unknown") for p in peers)
    
        # group key: '123+456' if pooled, else None
        group_key_map: Dict[str, Optional[str]] = {}
        for sig, cols in sig_to_players.items():
            if len(cols) > 1:
                ids = "+".join(sorted(c[1:] for c in cols))  # numeric, sorted
                for c in cols:
                    group_key_map[c] = ids
            else:
                group_key_map[cols[0]] = None
    
        coef_df["other_players_same_minutes"] = coef_df["player_id"].apply(_peers)
        coef_df["pooled_group_key"] = coef_df["player_id"].map(group_key_map)
        coef_df["is_pooled_member"] = coef_df["pooled_group_key"].notna()
    
        # --- Lasso
        valid_idx = np.isfinite(y) & np.isfinite(X_df).all(axis=1)
        X_lasso = X_df.loc[valid_idx]; y_lasso = y.loc[valid_idx]
    
        for alpha in self.lasso_alphas:
            a = float(alpha)
            a_str = str(a).replace(".", "_")
            with threadpool_limits(1):
                model = Lasso(alpha=a, fit_intercept=True, max_iter=10000)
                model.fit(X_lasso, y_lasso)
            coef_series = pd.Series(model.coef_, index=X_lasso.columns)
            coef_df[f"lasso_impact_alpha_{a_str}"] = coef_df["player_id"].map(coef_series.to_dict())
            coef_df[f"lasso_std_alpha_{a_str}"] = float(coef_series.std())
            
        # --- Ridge (L2) over specified alphas (weighted via sqrt(w))
        if len(self.ridge_alphas) > 0:
            # restrict to valid rows (reuse valid_idx from Lasso section)
            X_ridge = X_df.loc[valid_idx]
            y_ridge = y.loc[valid_idx]
            w_ridge = w.loc[valid_idx].to_numpy(dtype=float)
            sw = np.sqrt(w_ridge)
            Xw = X_ridge.to_numpy(dtype=float) * sw[:, None]
            yw = y_ridge.to_numpy(dtype=float) * sw
    
            for alpha in self.ridge_alphas:
                a = float(alpha)
                a_str = str(a).replace(".", "_")
                with threadpool_limits(1):
                    ridge = Ridge(alpha=a, fit_intercept=True)
                    ridge.fit(Xw, yw)
                coef_series = pd.Series(ridge.coef_, index=X_ridge.columns)
                coef_df[f"ridge_impact_alpha_{a_str}"] = coef_df["player_id"].map(coef_series.to_dict())

    
        # Appearance metrics
        for metric_key in ["games_started", "full_games_played", "games_subbed_on", "games_subbed_off"]:
            coef_df[metric_key] = coef_df["player_id"].apply(
                lambda pid, k=metric_key: (self.player_game_metrics.get(int(pid[1:]), {}).get(k, 0)
                                           if pid.startswith("p") else 0)
            )
        coef_df["FTE_games_played"] = (coef_df["minutes_played"] / 90).round(2)
    
        # Context columns
        coef_df["country"] = self.COUNTRY_NAME
        coef_df["league"] = self.LEAGUE_NAME
        coef_df["season"] = self.SEASON
        coef_df["total_regular_games"] = self.total_regular_games
        coef_df["valid_regular_games"] = self.valid_regular_games
        coef_df["missing_games_count"] = self.missing_games_count
    
        coef_df.sort_values(by="impact", ascending=False, inplace=True)
        coef_df.reset_index(drop=True, inplace=True)
        self.coef_df_final = coef_df
        

    def build_pooled_results(self) -> None:
        """
        Build a collapsed (pooled) results DataFrame:
        - Identifies groups of players with identical presence signature.
        - For groups of size >1, aggregates minutes, games, goals/assists, and sums impacts.
        - Approximates pooled std via root-sum-of-squares of individual stds.
        - Marks is_pooled_estimate=True and lists pooled members.
        Result stored in self.coef_df_pooled (does not replace coef_df_final).
        """
        assert self.coef_df_final is not None and self.segment_df is not None
    
        # signature → player columns
        presence_int = self.segment_df[self.player_cols].astype("int8")
        sig_to_players: Dict[tuple, List[str]] = {}
        for col in presence_int.columns:
            signature = tuple(presence_int[col].tolist())
            sig_to_players.setdefault(signature, []).append(col)
    
        df = self.coef_df_final.copy()
    
        # helper to get numeric id from "p123"
        def _num(pid_str: str) -> int:
            return int(pid_str[1:]) if isinstance(pid_str, str) and pid_str.startswith("p") else int(pid_str)
    
        pooled_rows: List[Dict[str, Any]] = []
        used: Set[str] = set()
    
        # columns that should be summed if pooled
        sum_cols = [
            "minutes_played","games_started","full_games_played",
            "games_subbed_on","games_subbed_off","FTE_games_played",
            "goals","assists"
        ]
        # lasso columns to sum
        lasso_cols = [c for c in df.columns if c.startswith("lasso_impact_alpha_")]
        lasso_std_cols = [c for c in df.columns if c.startswith("lasso_std_alpha_")]
    
        for sig, cols in sig_to_players.items():
            if len(cols) == 1:
                col = cols[0]
                row = df.loc[df["player_id"] == col].copy()
                if row.empty:
                    continue
                out = row.iloc[0].to_dict()
                out["player_id"] = _num(col)  # numeric id
                out["is_pooled_estimate"] = False
                out["pooled_members"] = str(_num(col))
                pooled_rows.append(out)
                continue
    
            # group > 1 → aggregate
            sub = df[df["player_id"].isin(cols)].copy()
            if sub.empty:
                continue
    
            member_ids = [str(_num(c)) for c in cols]
            member_names = sub["player_name"].tolist()
            # teams union
            teams = set()
            for t in sub["team(s)"].dropna().tolist():
                if isinstance(t, str) and t.startswith("Multiple Teams ("):
                    inner = t[len("Multiple Teams ("):-1]
                    teams.update([x.strip() for x in inner.split(",") if x.strip()])
                elif t:
                    teams.add(str(t))
            team_repr = None
            if len(teams) == 1:
                team_repr = list(teams)[0]
            elif len(teams) > 1:
                team_repr = f"Multiple Teams ({', '.join(sorted(teams))})"
    
            # league pos
            league_pos_vals = sub["league_position"].dropna().unique().tolist()
            league_pos_repr = league_pos_vals[0] if len(league_pos_vals) == 1 else ("Multiple Teams" if len(league_pos_vals) > 1 else None)
    
            out: Dict[str, Any] = {
                "player_id": "+".join(member_ids),               # composite id (string) in pooled view
                "player_name": " + ".join(member_names),
                "position": "/".join(sorted(set(sub["position"].dropna().tolist()))) if not sub["position"].isna().all() else "Unknown",
                "team(s)": team_repr,
                "league_position": league_pos_repr,
                "impact": sub["impact"].sum(),
                "impact_std": float(np.sqrt(np.nansum(np.square(sub["impact_std"].to_numpy(dtype=float)))) if "impact_std" in sub else np.nan),
                "is_pooled_estimate": True,
                "pooled_members": ", ".join(f"{i}:{n}" for i, n in zip(member_ids, member_names)),
                "country": self.COUNTRY_NAME,
                "league": self.LEAGUE_NAME,
                "season": self.SEASON,
                "total_regular_games": self.total_regular_games,
                "valid_regular_games": self.valid_regular_games,
                "missing_games_count": self.missing_games_count,
            }
            for c in sum_cols:
                if c in sub:
                    out[c] = float(sub[c].sum()) if c == "minutes_played" else int(sub[c].sum())
            # lasso impacts sum; std keep first (reference)
            for c in lasso_cols:
                out[c] = float(sub[c].sum(skipna=True)) if c in sub else np.nan
            for c in lasso_std_cols:
                out[c] = float(sub[c].iloc[0]) if (c in sub and not sub[c].isna().all()) else np.nan
    
            pooled_rows.append(out)
            used.update(cols)
    
        pooled_df = pd.DataFrame(pooled_rows)
        # stable sort by impact desc
        if not pooled_df.empty:
            pooled_df.sort_values(by="impact", ascending=False, inplace=True)
            pooled_df.reset_index(drop=True, inplace=True)
        self.coef_df_pooled = pooled_df
        
    def run_analysis(self, until_date: Optional[str] = None, group_pooled: bool = True) -> None:
        """End-to-end analysis (no I/O besides optional intermediate pickle)."""
        if until_date is not None:
            self.date_cutoff_utc = pd.to_datetime(until_date, utc=True)
    
        print("[INFO] Building season data (one pass)...")
        self.prep_data_up_to_cutoff(keep_player_after_red=False)  # default: drop after red
    
        if self.save_intermediate and self.segment_df is not None:
            try:
                intermediate_path = self._p("regression_data.pkl")
                self.segment_df.to_pickle(intermediate_path)
                print(f"[INFO] Saved intermediate regression data -> {intermediate_path}")
            except Exception as e:
                print(f"[WARN] Could not save intermediate data: {e}")
    
        print("[INFO] Computing impacts (WLS + Lasso)...")
        self.compute_impacts()
    
        print("[INFO] Adding goals & assists...")
        # self.players_meta_df already has tallies; keep your existing column add for coef_df:
        self.add_goals_assists()
    
        if group_pooled:
            print("[INFO] Building pooled results...")
            self.build_pooled_results()


    def save_results(self, output_path: Optional[str] = None) -> None:
        """
        Save per-player (and pooled) results to Excel/CSV.
        - Estimation columns are formatted to 3 significant digits.
        - Column order is standardized for readability.
        - If a date cutoff is active, results are NOT written to the general SQLite DB.
          Instead, files are saved next to the script (main folder) to keep ad-hoc runs separate.
        - Without a cutoff, behavior stays: write season Excel and upsert per-player table to SQLite.
        """
        assert self.coef_df_final is not None, "No results to save. Did you run run_analysis()?"
    
        # -------- helpers --------
        def _round_sig(x: Any, sig: int = 3):
            try:
                if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                    return x
                if isinstance(x, (int, np.integer)):
                    return int(x)
                # numeric -> 3 significant digits
                x = float(x)
                if x == 0.0:
                    return 0.0
                from math import log10, floor
                return round(x, int(sig - 1 - floor(log10(abs(x)))))
            except Exception:
                return x
    
        def _apply_sig(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].map(lambda v: _round_sig(v, 3))
            return df
    
        # -------- base frames --------
        # Per-player (drop 'p' prefix to numeric player_id for outputs)
        out_df = self.coef_df_final.copy()
        out_df["player_id"] = out_df["player_id"].apply(lambda pid: int(pid[1:]) if isinstance(pid, str) and pid.startswith("p") else pid)
    
        # Optional pooled view (may be None/empty)
        pooled_df = self.coef_df_pooled.copy() if getattr(self, "coef_df_pooled", None) is not None else pd.DataFrame()
    
        # -------- columns: order + 3-sig on estimation --------
        # Identify column groups
        est_cols = ["impact", "impact_std"]
        est_cols += [c for c in out_df.columns if c.startswith("ridge_impact_alpha_")]
        est_cols += [c for c in out_df.columns if c.startswith("lasso_impact_alpha_")]
        est_cols += [c for c in out_df.columns if c.startswith("lasso_std_alpha_")]
    
        # sensible column order
        meta_cols = [
            "player_id", "player_name", "position", "team(s)", "league_position",
            "country", "league", "season"
        ]
        minutes_cols = ["minutes_played", "FTE_games_played"]
        appearances_cols = ["games_started", "full_games_played", "games_subbed_on", "games_subbed_off"]
        scoring_cols = ["goals", "assists", "pen_goals", "pen_assists", "pen_missed", "own_goals", "yellow_cards"]
        pooling_cols = ["is_pooled_member", "pooled_group_key", "other_players_same_minutes", "pooled", "pooled_members"]
        dataset_cols = ["total_regular_games", "valid_regular_games", "missing_games_count"]
    
        # keep only those that exist, in that order, then tack on any remainders at end
        def _reorder(df: pd.DataFrame) -> pd.DataFrame:
            ordered = (
                [c for c in meta_cols if c in df.columns] +
                [c for c in minutes_cols if c in df.columns] +
                [c for c in appearances_cols if c in df.columns] +
                [c for c in scoring_cols if c in df.columns] +
                [c for c in est_cols if c in df.columns] +
                [c for c in pooling_cols if c in df.columns] +
                [c for c in dataset_cols if c in df.columns]
            )
            remainder = [c for c in df.columns if c not in ordered]
            return df[ordered + remainder]
    
        # apply 3-sig to estimation columns
        out_df = _apply_sig(out_df, est_cols)
        out_df = _reorder(out_df).sort_values(by="impact", ascending=False)
    
        if not pooled_df.empty:
            # pooled may have composite IDs in 'player_id' — leave as-is for readability
            pooled_est_cols = [c for c in pooled_df.columns if c in est_cols]
            pooled_df = _apply_sig(pooled_df, pooled_est_cols)
            pooled_df = _reorder(pooled_df).sort_values(by="impact", ascending=False)
    
        # -------- choose save targets (cutoff runs go to main folder; full runs to season folder + DB) --------
        script_dir = os.path.dirname(os.path.abspath(__file__))
        is_cutoff_run = getattr(self, "date_cutoff_utc", None) is not None
    
        if output_path is None:
            if is_cutoff_run:
                # place in main folder with a clear suffix
                cutoff_tag = self.date_cutoff_utc.date().isoformat()
                base_name = f"output_cutoff_{self.COUNTRY_NAME}_{self.LEAGUE_NAME}_{self.SEASON}_{cutoff_tag}"
                output_path = os.path.join(script_dir, f"{base_name}.xlsx")
                csv_path = os.path.join(script_dir, f"{base_name}.csv")
                pooled_csv_path = os.path.join(script_dir, f"{base_name}_pooled.csv")
            else:
                # standard season path
                output_path = self._p("output.xlsx")
                csv_path = self._p("output.csv")
                pooled_csv_path = self._p("output_pooled.csv")
    
        # -------- write Excel (per-player + lasso/ridge sheets + pooled if present) --------
        # Per-player sheet
        out_df.to_excel(output_path, sheet_name="OLS", index=False)
    
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            # Lasso sheets
            for alpha in self.lasso_alphas:
                a = float(alpha); a_str = str(a).replace(".", "_")
                col = f"lasso_impact_alpha_{a_str}"
                if col in out_df:
                    df_sub = out_df[out_df[col].notna()].copy()
                    df_sub.sort_values(by=col, ascending=False, inplace=True)
                    df_sub.to_excel(writer, sheet_name=f"Lasso_{a_str}", index=False)
    
            # Ridge sheets
            for alpha in getattr(self, "ridge_alphas", []):
                a = float(alpha); a_str = str(a).replace(".", "_")
                col = f"ridge_impact_alpha_{a_str}"
                if col in out_df:
                    df_sub = out_df[out_df[col].notna()].copy()
                    df_sub.sort_values(by=col, ascending=False, inplace=True)
                    df_sub.to_excel(writer, sheet_name=f"Ridge_{a_str}", index=False)
    
            # Pooled sheets
            if not pooled_df.empty:
                pooled_df.to_excel(writer, sheet_name="OLS_POOLED", index=False)
                for alpha in self.lasso_alphas:
                    a = float(alpha); a_str = str(a).replace(".", "_")
                    col = f"lasso_impact_alpha_{a_str}"
                    if col in pooled_df:
                        dfp = pooled_df[pooled_df[col].notna()].copy()
                        dfp.sort_values(by=col, ascending=False, inplace=True)
                        dfp.to_excel(writer, sheet_name=f"Lasso_{a_str}_POOLED", index=False)
                for alpha in getattr(self, "ridge_alphas", []):
                    a = float(alpha); a_str = str(a).replace(".", "_")
                    col = f"ridge_impact_alpha_{a_str}"
                    if col in pooled_df:
                        dfp = pooled_df[pooled_df[col].notna()].copy()
                        dfp.sort_values(by=col, ascending=False, inplace=True)
                        dfp.to_excel(writer, sheet_name=f"Ridge_{a_str}_POOLED", index=False)
    
        # -------- write CSVs for quick pandas consumption --------
        out_df.to_csv(csv_path, index=False)
        if not pooled_df.empty:
            pooled_df.to_csv(pooled_csv_path, index=False)
    
        # -------- database writes (only for full, non-cutoff runs) --------
        if not is_cutoff_run:
            conn = sqlite3.connect(self.DB_PATH); cur = conn.cursor()
            try:
                table_exists = cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results';"
                ).fetchone() is not None
    
                # per-player: use numeric player_id frame (out_df)
                if table_exists:
                    existing_cols = {c[1] for c in cur.execute("PRAGMA table_info(analysis_results);").fetchall()}
                    new_cols = set(out_df.columns)
                    if existing_cols != new_cols:
                        out_df.to_sql("analysis_results_new", conn, if_exists="replace", index=False)
                    else:
                        cur.execute(
                            "DELETE FROM analysis_results WHERE country=? AND league=? AND season=?;",
                            (self.COUNTRY_NAME, self.LEAGUE_NAME, self.SEASON),
                        )
                        conn.commit()
                        out_df.to_sql("analysis_results", conn, if_exists="append", index=False)
                else:
                    out_df.to_sql("analysis_results", conn, if_exists="replace", index=False)
    
                # pooled: optional separate table (string IDs ok)
                if not pooled_df.empty:
                    pooled_exists = cur.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results_pooled';"
                    ).fetchone() is not None
                    if pooled_exists:
                        cur.execute(
                            "DELETE FROM analysis_results_pooled WHERE country=? AND league=? AND season=?;",
                            (self.COUNTRY_NAME, self.LEAGUE_NAME, self.SEASON),
                        )
                        conn.commit()
                    pooled_df.to_sql("analysis_results_pooled", conn, if_exists="append", index=False)
    
            finally:
                conn.close()
    
            # export per-player slice from DB for convenience
            conn = sqlite3.connect(self.DB_PATH)
            try:
                query = """
                    SELECT * FROM analysis_results
                     WHERE country = ? AND league = ? AND season = ?;
                """
                df_from_db = pd.read_sql_query(query, conn, params=[self.COUNTRY_NAME, self.LEAGUE_NAME, self.SEASON])
            finally:
                conn.close()
    
            df_from_db.to_excel(self._p("analysis_from_db.xlsx"), index=False)
            df_from_db.to_csv(self._p("analysis_from_db.csv"), index=False)
    
        else:
            # Cutoff run: be explicit about the non-DB behavior
            tag = self.date_cutoff_utc.date().isoformat()
            print(f"[INFO] Cutoff run ({tag}): saved Excel/CSV to {os.path.dirname(output_path)} and skipped DB write.")


    def run(self, until_date: str = None, group_pooled: bool = True) -> None:
        """Convenience: load → analyze(with cutoff) → save."""
        self.load_data()
        self.run_analysis(until_date=until_date, group_pooled=group_pooled)
        self.save_results()

# ---------------- main: load progress + run completed downloads (no argparse) ----------------
if __name__ == "__main__":
    # Run all rows that are downloaded and overall_status == "Completed",
    # unless already processed (output.xlsx or DB has rows).
    runner = SeasonAnalyzer(
        country_name="Germany",
        league_name="Bundesliga",
        season=2023,
        lasso_alphas=(0.01, 0.001),
        ridge_alphas=(0.1, 1.0, 10.0),  # example
    )

    runner.run()
     