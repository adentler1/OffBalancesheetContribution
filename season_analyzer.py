#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeasonAnalyzer: build segment-level plus/minus, OLS/WLS & regularized contributions,
including team-partialled variants, with complete dispersion columns.

Key notes:
- FTE_games_played is normalized by each fixture's actual duration
  (sum of segment minutes per fixture), so it never exceeds matches played.
- All estimate columns (including *_std and 'best' alpha columns) are rounded
  to 3 significant digits for outputs.
- Team coefficients are stored with player-style naming:
    team_contribution_ols, team_contribution_ols_std
- League position is stored as integer in team tables.
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
        ridge_alphas: List[float] | float = (1.0, 10.0),
        auto_select_lasso_alpha: bool = True,
        select_alpha_by: str = "ic",          # "ic" (information criterion) or "none"
        info_criterion: str = "AIC",          # "AIC", "AICc", "BIC", "HQ"
        ic_alpha_bounds: tuple = (1e-6, 1e1), # search bounds for α
        ic_tol: float = 1e-2,                 # tolerance on log(α) span
        ic_max_iter: int = 60,                # golden-section steps

        lasso_alpha_grid: Optional[List[float]] = None,
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
        self.TEAMS_FILE = "teams.json"   # optional reference; names already come from fixtures

        self.lasso_alphas = list(lasso_alphas) if isinstance(lasso_alphas, (list, tuple, set)) else [float(lasso_alphas)]
        self.ridge_alphas = list(ridge_alphas) if isinstance(ridge_alphas, (list, tuple, set)) else [float(ridge_alphas)]
        self.auto_select_lasso_alpha = bool(auto_select_lasso_alpha)
        self.select_alpha_by = str(select_alpha_by).lower()
        self.info_criterion = str(info_criterion).upper()
        self.ic_alpha_bounds = ic_alpha_bounds
        self.ic_tol = float(ic_tol)
        self.ic_max_iter = int(ic_max_iter)

        # default grid: spans tighter around your 0.001–0.01 judgment
        self.lasso_alpha_grid = (
            list(lasso_alpha_grid) if lasso_alpha_grid is not None else
            [0.00001, 0.00003, 0.00005, 0.0001, 0.0003, 0.0005, 0.001]
        )
        self.lasso_best_alpha: Optional[float] = None

        os.makedirs(self.SAVE_DIR, exist_ok=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.DB_PATH = os.path.join(script_dir, "analysis_results.db")
    
        # Data containers
        self.fixtures_data: List[Dict[str, Any]] | None = None
        self.match_events: Dict[str, Dict[str, Any]] | None = None
        self.players: Dict[str, Any] | None = None
        # Matrices prepared by pipeline
        self.players_meta_df: pd.DataFrame | None = None
        self.team_presence_signed_df: pd.DataFrame | None = None
        self.team_presence_abs_df: pd.DataFrame | None = None
        
        self.date_cutoff_utc = pd.to_datetime(date_cutoff, utc=True) if date_cutoff else None

        self.team_players: Dict[int, Set[int]] = {}
        self.player_id_to_name: Dict[int, str] = {}
        self.player_id_to_position: Dict[int, str] = {}
        self.player_id_to_age: Dict[int, Optional[int]] = {}
    
        # Matrices / tables
        self.segment_df: pd.DataFrame | None = None
        self.match_summary_df: pd.DataFrame | None = None
        self.team_summary_df: pd.DataFrame | None = None
        self.coef_df_final: pd.DataFrame | None = None
        self.coef_df_pooled: pd.DataFrame | None = None
        self.player_game_metrics: Dict[int, Dict[str, float | int]] = {}
        self.fte_games_played_by_player: Dict[int, float] = {}
    
        # Accounting
        self.total_regular_games = 0
        self.valid_regular_games = 0
        self.missing_games_count = 0
        self.valid_fixture_ids: Set[str] = set()
    
        # Columns for players (persisted)
        self.all_player_ids: List[int] = []
        self.player_cols: List[str] = []  # ["p123", ...]

        self.save_intermediate = save_intermediate
        self.date_cutoff_utc = pd.to_datetime(date_cutoff, utc=True) if date_cutoff else None

    # ---------- IO ----------
    def _p(self, filename: str) -> str:
        return os.path.join(self.SAVE_DIR, filename)
    
    def _to_utc(self, dt_like) -> Optional[pd.Timestamp]:
        if dt_like is None:
            return None
        ts = pd.to_datetime(dt_like, utc=True, errors="coerce")
        return None if pd.isna(ts) else ts

    def load_data(self) -> None:
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
          - Update per-player meta (starts, subs on/off, full games, minutes, FTE normalized by fixture duration)
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
                    0, 0, 0, 0, 0, 0, 0,
                    pd.NaT, pd.NaT
                ]
    
        # Preload from self.players (best-effort)
        if isinstance(self.players, dict):
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
                        add_or_update_player(pid_in)
    
                # apply reds at boundary
                if not keep_player_after_red and seg_end in reds_by_min:
                    for pid_rc in reds_by_min[seg_end]:
                        current.pop(int(pid_rc), None)
    
                seg_start = seg_end
    
            # full games = starters not subbed off
            for pid in (starters - subs_out_set):
                full_games[pid] += 1
    
            # match summary
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
                "sub_events": int(len([e for e in events if (e.get("type") or "").lower()=="subst"])),
                "unchanged_players": int(len(starters - subs_out_set))
            })
    
            # Scoring tallies (whole match)
            for ev in events:
                if is_yellow(ev):
                    pid = (ev.get("player") or {}).get("id")
                    if pid: yellow_cards[int(pid)] += 1
                if is_pen_missed(ev):
                    pid = (ev.get("player") or {}).get("id")
                    if pid: pen_missed[int(pid)] += 1
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
    
            id_to_teamname[home_id] = home_team
            id_to_teamname[away_id] = away_team
    
        # finalize appearance metrics
        self.player_game_metrics = {}
        for pid in player_ids_seen:
            self.player_game_metrics[pid] = {
                "games_started":   int(games_started.get(pid, 0)),
                "full_games_played": int(full_games.get(pid, 0)),
                "games_subbed_on":  int(subs_on.get(pid, 0)),
                "games_subbed_off": int(subs_off.get(pid, 0)),
            }
            
        self.total_regular_games = sum(1 for f in self.fixtures_data if self._is_regular(f) and allowed_fixture(f))
        self.valid_fixture_ids = set(reg_fixture_ids)
        self.valid_regular_games = len(self.valid_fixture_ids)
        self.missing_games_count = self.total_regular_games - self.valid_regular_games
        if skipped_missing:
            print(f"[info] Skipped {skipped_missing} fixtures (missing lineups or events).")
    
        # Build DataFrames for segments and teams
        if not meta_rows:
            self.segment_df = pd.DataFrame(columns=[
                "fixture_id","game_date_utc","home_id","away_id","home_team","away_team",
                "start","end","duration","goal_diff"
            ])
            self.team_presence_signed_df = pd.DataFrame()
            self.team_presence_abs_df = pd.DataFrame()
            self.match_summary_df = pd.DataFrame()
            self.team_summary_df = pd.DataFrame()
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
        for pid in self.all_player_ids:
            add_or_update_player(pid, self.player_id_to_name.get(pid,"Unknown"),
                                 self.player_id_to_position.get(pid,"Unknown"))
    
        # Minutes & appearances
        self.players_meta_df.loc[self.all_player_ids, "minutes_played"] = [
            float(minutes_by_player_total.get(pid, 0.0)) for pid in self.all_player_ids
        ]

        # --- FTE normalized by fixture actual duration ---
        # Build per-fixture durations and per-player minutes within fixture
        seg = self.segment_df[["fixture_id","duration"] + self.player_cols].copy()
        seg[self.player_cols] = seg[self.player_cols].abs().multiply(seg["duration"], axis=0)
        player_minutes_by_fixture = seg.groupby("fixture_id")[self.player_cols].sum()
        fixture_duration = seg.groupby("fixture_id")["duration"].sum()
        # fractions per fixture, clipped to 1
        frac = player_minutes_by_fixture.divide(fixture_duration, axis=0).clip(upper=1.0).fillna(0.0)
        fte_series = frac.sum(axis=0)  # index = "p{pid}"
        fte_map = {int(col[1:]): float(val) for col, val in fte_series.items()}
        self.fte_games_played_by_player = fte_map

        self.players_meta_df.loc[self.all_player_ids, "FTE_games_played"] = [
            round(float(self.fte_games_played_by_player.get(pid, 0.0)), 3) for pid in self.all_player_ids
        ]

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
    
        # Build team summary (kept for compatibility; not used for league_position anymore)
        team_rows_out: List[Dict[str,Any]] = []
        total_minutes = float(self.segment_df["duration"].sum())
        if not self.match_summary_df.empty and total_minutes > 0:
            tm = self.match_summary_df
            team_ids_all = sorted(set(tm["home_team_id"]).union(tm["away_team_id"]))
            for team_id in team_ids_all:
                mask_home = (tm["home_team_id"] == team_id)
                mask_away = (tm["away_team_id"] == team_id)
                wins   = (mask_home & (tm["home_goals"] > tm["away_goals"])).sum() + (mask_away & (tm["away_goals"] > tm["home_goals"])).sum()
                # NOTE: draws here are overall draws; we no longer rely on this frame for positions.
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
                if (ev.get("type") or "").lower() == "goal":
                    detail = (ev.get("detail") or "").lower()
                    if "own" in detail:
                        continue
                    scorer = (ev.get("player") or {}).get("id")
                    helper = (ev.get("assist") or {}).get("id")
                    if scorer: goals[scorer] = goals.get(scorer, 0) + 1
                    if helper: assists[helper] = assists.get(helper, 0) + 1

        coef_df = self.coef_df_final
        coef_df["goals"] = coef_df["player_id"].apply(lambda pid: goals.get(int(pid[1:]), 0) if isinstance(pid,str) and pid.startswith("p") else 0)
        coef_df["assists"] = coef_df["player_id"].apply(lambda pid: assists.get(int(pid[1:]), 0) if isinstance(pid,str) and pid.startswith("p") else 0)
        self.coef_df_final = coef_df

    # ---------- math helpers ----------
    @staticmethod
    def _ridge_se_diagonal_from_transformed(Xw: np.ndarray, rw: np.ndarray, alpha: float) -> np.ndarray:
        """
        Approximate coefficient std errors for ridge with transformed (weighted) design.
        Var(beta_ridge) ≈ sigma^2 * (X'X + αI)^(-1) X'X (X'X + αI)^(-1)
        where X here is Xw (sqrt(weights)*X), sigma^2 = RSS/(n - p) as a rough df.
        Returns sqrt(diag(Var)).
        """
        n, p = Xw.shape
        XtX = Xw.T @ Xw
        A = XtX + alpha * np.eye(p, dtype=float)
        A_inv = np.linalg.pinv(A)
        beta = A_inv @ (Xw.T @ rw)
        resid = rw - Xw @ beta
        df = max(n - p, 1)
        sigma2 = float((resid @ resid) / df)
        M = A_inv @ XtX @ A_inv
        var_diag = sigma2 * np.clip(np.diag(M), 0.0, np.inf)
        return np.sqrt(var_diag)
    
    def _ic_value_for_lasso(self, alpha: float, X: np.ndarray, y: np.ndarray, w: np.ndarray, ic: str) -> float:
        # Fit lasso at given alpha (weighted via sample_weight)
        model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=10000)
        model.fit(X, y, sample_weight=w)
        yhat = model.predict(X)
        resid = y - yhat
        # Weighted RSS (equivalent to transformed WLS)
        wrss = float(np.sum(w * (resid ** 2)))
        n_eff = int(np.sum(np.isfinite(y)))  # effective n (segments)
        # DoF ≈ nonzeros + intercept
        df = int(np.count_nonzero(model.coef_)) + 1
        # Guard against degenerate small-n situations
        if n_eff <= df + 1:
            return 1e12 + (df - n_eff + 1) * 1e10
        sigma2 = wrss / max(n_eff, 1)
        ll_term = n_eff * np.log(max(sigma2, 1e-12))  # −2 log L up to constant
    
        ic_u = ic.upper()
        if ic_u == "AIC":
            return ll_term + 2 * df
        if ic_u == "AICC":
            return ll_term + 2 * df + (2 * df * (df + 1)) / max(n_eff - df - 1, 1)
        if ic_u == "BIC":
            return ll_term + df * np.log(max(n_eff, 2))
        if ic_u == "HQ":
            return ll_term + 2 * df * np.log(np.log(max(n_eff, 3)))
        # Fallback: AIC
        return ll_term + 2 * df
    
    def _select_lasso_alpha_ic(
        self,
        X_df: pd.DataFrame,
        y: pd.Series,
        w: pd.Series,
        *,
        ic: str = "AIC",
        bounds: tuple = (1e-6, 1e1),
        tol: float = 1e-2,
        max_iter: int = 60
    ) -> float:
        """Golden-section search on log(alpha) to minimize the chosen information criterion."""
        # Prepare arrays
        valid = np.isfinite(y) & np.isfinite(X_df).all(axis=1)
        X = X_df.loc[valid].to_numpy(dtype=float, copy=False)
        yy = y.loc[valid].to_numpy(dtype=float, copy=False)
        ww = w.loc[valid].to_numpy(dtype=float, copy=False)
        # Work in log-space for scale invariance
        lo, hi = float(bounds[0]), float(bounds[1])
        lo = max(lo, 1e-12)
        import math
        a, b = math.log(lo), math.log(hi)
        phi = (math.sqrt(5) - 1) / 2  # ~0.618
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc = self._ic_value_for_lasso(math.exp(c), X, yy, ww, ic)
        fd = self._ic_value_for_lasso(math.exp(d), X, yy, ww, ic)
        it = 0
        while (b - a) > tol and it < max_iter:
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = self._ic_value_for_lasso(math.exp(c), X, yy, ww, ic)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = self._ic_value_for_lasso(math.exp(d), X, yy, ww, ic)
            it += 1
        return float(math.exp((a + b) / 2.0))


    def _select_lasso_alpha(
        self,
        X_df: pd.DataFrame,
        y: pd.Series,
        w: pd.Series,
        team_position_map: Dict[str, int],
        *,
        subsamples: int = 5,
        sparsity_range: tuple[float, float] = (0.03, 0.35),  # 3%–35% nonzeros
        stability_weight: float = 0.3,                       # coherence gets 0.7
        random_state: int = 42
    ) -> float:
        """
        Choose alpha by balancing:
          1) Team coherence (non-partialled): more negative corr(team-mean, league_position) is better.
          2) Coefficient stability under subsampling (Spearman vs full fit).
          3) Reasonable sparsity (penalize if outside range).
        Returns best alpha. Falls back to 0.001 if ties/NaNs.
        """
        rng = np.random.default_rng(random_state)
        valid = np.isfinite(y) & np.isfinite(X_df).all(axis=1)
        X = X_df.loc[valid].to_numpy(dtype=float)
        yy = y.loc[valid].to_numpy(dtype=float)
        ww = w.loc[valid].to_numpy(dtype=float)
    
        # Quick map: player column -> single team for coherence (skip multi-team)
        col_to_team = {}
        for col in X_df.columns:
            if col not in self.segment_df:
                col_to_team[col] = None
                continue
            homes = self.segment_df.loc[self.segment_df[col] == 1, "home_team"].unique().tolist()
            aways = self.segment_df.loc[self.segment_df[col] == -1, "away_team"].unique().tolist()
            teams = sorted(set(homes + aways))
            col_to_team[col] = teams[0] if len(teams) == 1 else None
    
        def team_coherence(coefs: np.ndarray) -> float:
            dfc = pd.DataFrame({"col": X_df.columns, "coef": coefs, "team": [col_to_team[c] for c in X_df.columns]})
            dfc = dfc.dropna(subset=["team"])
            if dfc.empty:
                return np.nan
            g = dfc.groupby("team")["coef"].mean().reset_index()
            g["league_pos"] = g["team"].map(team_position_map)
            g = g.dropna(subset=["league_pos"])
            if len(g) < 4:
                return np.nan
            return float(g["coef"].corr(g["league_pos"]))  # expect negative
    
        def coef_stability(full_coef: np.ndarray, alpha: float) -> float:
            from scipy.stats import spearmanr
            if subsamples <= 0:
                return np.nan
            idx = np.arange(X.shape[0])
            reps = []
            for _ in range(subsamples):
                sub_idx = rng.choice(idx, size=int(0.75 * len(idx)), replace=False)
                Xs, ys, ws = X[sub_idx], yy[sub_idx], ww[sub_idx]
                model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
                model.fit(Xs, ys, sample_weight=ws)
                r, _ = spearmanr(full_coef, model.coef_)
                if np.isfinite(r):
                    reps.append(r)
            return float(np.nanmedian(reps)) if reps else np.nan
    
        best_alpha = None
        best_score = np.inf
    
        # Fit a single full model per alpha, then score
        for alpha in self.lasso_alpha_grid:
            try:
                model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=10000)
                model.fit(X, yy, sample_weight=ww)
                coefs = model.coef_
                # sparsity
                nz_frac = (np.abs(coefs) > 0).mean()
                penalty = 0.0
                lo, hi = sparsity_range
                if not (lo <= nz_frac <= hi):
                    penalty = (lo - nz_frac) ** 2 if nz_frac < lo else (nz_frac - hi) ** 2
                # coherence (want negative corr) → objective uses (-corr)
                corr = team_coherence(coefs)
                coherence_term = -corr if np.isfinite(corr) else 10.0
                # stability
                stab = coef_stability(coefs, float(alpha))
                stab_term = 1.0 - (stab if np.isfinite(stab) else 0.0)
                # weighted score
                score = (0.7 * coherence_term) + (stability_weight * stab_term) + (0.3 * penalty)
            except Exception:
                score = np.inf
    
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
    
        return float(best_alpha if best_alpha is not None else 0.001)

    def compute_impacts(self) -> None:
        """Compute baseline (non-partialled) contributions and team-partialled contributions, with stds."""
        assert self.segment_df is not None
    
        # Target and weights
        y = pd.to_numeric(self.segment_df["goal_diff"], errors="coerce").fillna(0.0).astype(float)
        w = pd.to_numeric(self.segment_df["duration"], errors="coerce").fillna(1.0).astype(float)
    
        # Design matrix X = players only
        X_df = self.segment_df[self.player_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

        # --- WLS helper (baseline & partialled OLS)
        def fit_wls(X: pd.DataFrame, y_: pd.Series, w_: pd.Series) -> tuple[pd.Series, pd.Series]:
            Xc = sm.add_constant(X, has_constant="add")
            y_arr = y_.to_numpy(dtype=np.float64)
            w_arr = w_.to_numpy(dtype=np.float64)
            X_arr = Xc.to_numpy(dtype=np.float64)
            valid = np.isfinite(y_arr) & np.isfinite(w_arr) & np.isfinite(X_arr).all(axis=1)
            if not np.any(valid):
                raise ValueError("No valid data points for WLS regression.")
            with threadpool_limits(1):
                model = sm.WLS(y_arr[valid], X_arr[valid], weights=w_arr[valid]).fit(method="pinv")
            params = pd.Series(model.params, index=Xc.columns)
            bse = pd.Series(model.bse, index=Xc.columns)
            return params.drop("const", errors="ignore"), bse.drop("const", errors="ignore")

        # ---------- BASELINE (no partialling) ----------
        ols_coef, ols_bse = fit_wls(X_df, y, w)

        coef_df = pd.DataFrame({
            "player_id": ols_coef.index,
            "contribution_ols": ols_coef.values,
            "contribution_ols_std": ols_bse.reindex(ols_coef.index).values
        })

        # Minutes played
        presence_np = X_df.abs().to_numpy(dtype=np.float64, copy=False)
        dur_arr = pd.to_numeric(self.segment_df["duration"], errors="coerce").fillna(0).to_numpy(dtype=np.float64, copy=False)
        if presence_np.shape[0] != dur_arr.shape[0]:
            raise ValueError(f"Shape mismatch: presence rows {presence_np.shape[0]} vs duration {dur_arr.shape[0]}")
        with threadpool_limits(1):
            mp = (presence_np * dur_arr[:, None]).sum(axis=0)
        minutes_played = pd.Series(mp, index=X_df.columns)
        coef_df["minutes_played"] = coef_df["player_id"].map(minutes_played.to_dict()).fillna(0.0).astype(float)
        coef_df["no_observations"] = coef_df["minutes_played"].le(0.0)

        # Names & positions
        def _name_from_pid(pid_str: str) -> str:
            if not isinstance(pid_str, str) or not pid_str.startswith("p"):
                return str(pid_str)
            try:
                return self.player_id_to_name.get(int(pid_str[1:]), "Unknown")
            except Exception:
                return "Unknown"
        def _pos_from_pid(pid_str: str) -> str:
            if not isinstance(pid_str, str) or not pid_str.startswith("p"):
                return "Unknown"
            try:
                return self.player_id_to_position.get(int(pid_str[1:]), "Unknown")
            except Exception:
                return "Unknown"
        coef_df["player_name"] = coef_df["player_id"].apply(_name_from_pid)
        coef_df["position"] = coef_df["player_id"].apply(_pos_from_pid)

        # Teams per player from segments
        def _teams_from_pid(pid_str: str) -> Optional[str]:
            if self.segment_df is None or pid_str not in self.segment_df:
                return None
            homes = self.segment_df.loc[self.segment_df[pid_str] == 1, "home_team"].unique().tolist()
            aways = self.segment_df.loc[self.segment_df[pid_str] == -1, "away_team"].unique().tolist()
            teams = sorted(set(homes + aways))
            if len(teams) == 1:
                return teams[0]
            if len(teams) > 1:
                return f"Multiple Teams ({', '.join(teams)})"
            return None
        coef_df["team(s)"] = coef_df["player_id"].apply(_teams_from_pid)

        # League position mapping (for players; may be "Multiple Teams")
        id_to_name: Dict[int, str] = {}
        for fix in (self.fixtures_data or []):
            id_to_name[fix["teams"]["home"]["id"]] = fix["teams"]["home"]["name"]
            id_to_name[fix["teams"]["away"]["id"]] = fix["teams"]["away"]["name"]
        team_position_map: Dict[str, int] = {}
        if self.match_summary_df is not None and len(self.match_summary_df):
            # We no longer trust team_summary_df for ordering; player-facing league pos is best-effort from match summary.
            tm = self.match_summary_df.copy()
            # Build a quick table
            team_ids = sorted(set(tm["home_team_id"]).union(tm["away_team_id"]))
            tmp_rows = []
            for tid in team_ids:
                mh = (tm["home_team_id"] == tid); ma = (tm["away_team_id"] == tid); mt = (mh | ma)
                wins = int((mh & (tm["home_goals"] > tm["away_goals"])).sum() + (ma & (tm["away_goals"] > tm["home_goals"])).sum())
                draws = int((mt & (tm["home_goals"] == tm["away_goals"])).sum())
                pts = wins*3 + draws
                gf = int((mh * tm["home_goals"] + ma * tm["away_goals"]).sum())
                tmp_rows.append({"team_id": int(tid), "points": int(pts), "gf": gf})
            td = pd.DataFrame(tmp_rows).sort_values(["points","gf"], ascending=[False, False]).reset_index(drop=True)
            for i, r in td.iterrows():
                team_position_map[id_to_name.get(int(r.team_id), "")] = i+1

        def _league_pos(team: Optional[str]) -> Optional[int | str]:
            if team is None:
                return None
            if str(team).startswith("Multiple"):
                return "Multiple Teams"
            return team_position_map.get(str(team))
        coef_df["league_position"] = coef_df["team(s)"].apply(_league_pos)

        # Pooling detection
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
        group_key_map: Dict[str, Optional[str]] = {}
        for sig, cols in sig_to_players.items():
            if len(cols) > 1:
                ids = "+".join(sorted(c[1:] for c in cols))
                for c in cols: group_key_map[c] = ids
            else:
                group_key_map[cols[0]] = None
        coef_df["other_players_same_minutes"] = coef_df["player_id"].apply(_peers)
        coef_df["pooled_group_key"] = coef_df["player_id"].map(group_key_map)
        coef_df["is_pooled_member"] = coef_df["pooled_group_key"].notna()

        # --- Lasso (baseline) ---
        valid_idx = np.isfinite(y) & np.isfinite(X_df).all(axis=1)
        X_lasso = X_df.loc[valid_idx]
        y_lasso = y.loc[valid_idx]
        w_lasso = w.loc[valid_idx]

        
        # >>> REPLACE YOUR EXISTING "alpha selection" BLOCK WITH THIS <<<
        chosen_alpha = None
        if self.auto_select_lasso_alpha and getattr(self, "select_alpha_by", "ic") == "ic":
            chosen_alpha = self._select_lasso_alpha_ic(
                X_lasso, y_lasso, w_lasso,
                ic=getattr(self, "info_criterion", "AIC"),
                bounds=getattr(self, "ic_alpha_bounds", (1e-6, 1e1)),
                tol=float(getattr(self, "ic_tol", 1e-2)),
                max_iter=int(getattr(self, "ic_max_iter", 60)),
            )
            self.lasso_best_alpha = float(chosen_alpha)
        # <<< END REPLACEMENT >>>
        
        # Fit requested alphas + the chosen one (dedup)
        alphas_to_fit = set(self.lasso_alphas)
        if chosen_alpha is not None:
            alphas_to_fit.add(float(chosen_alpha))

        for alpha in sorted(alphas_to_fit):
            a = float(alpha)
            a_tag = ("best" if (chosen_alpha is not None and abs(a - chosen_alpha) < 1e-12)
                     else str(a).replace(".", "_"))
            with threadpool_limits(1):
                model = Lasso(alpha=a, fit_intercept=True, max_iter=10000)
                model.fit(X_lasso, y_lasso, sample_weight=w_lasso)
            coef_series = pd.Series(model.coef_, index=X_lasso.columns)
            coef_df[f"lasso_contribution_alpha_{a_tag}"] = coef_df["player_id"].map(coef_series.to_dict())
            coef_df[f"lasso_contribution_alpha_{a_tag}_std"] = float(coef_series.std())

        # --- Ridge (baseline; weighted via sqrt(w)) ---
        if len(self.ridge_alphas) > 0:
            X_ridge = X_df.loc[valid_idx]
            y_ridge = y.loc[valid_idx]
            w_ridge_arr = w.loc[valid_idx].to_numpy(dtype=float)
            sw = np.sqrt(w_ridge_arr)
            Xw = X_ridge.to_numpy(dtype=float) * sw[:, None]
            yw = y_ridge.to_numpy(dtype=float) * sw
            for alpha in self.ridge_alphas:
                a = float(alpha)
                a_str = str(a).replace(".", "_")
                with threadpool_limits(1):
                    ridge = Ridge(alpha=a, fit_intercept=True)
                    ridge.fit(Xw, yw)
                coef_series = pd.Series(ridge.coef_, index=X_ridge.columns)
                coef_df[f"ridge_contribution_alpha_{a_str}"] = coef_df["player_id"].map(coef_series.to_dict())
                ridge_se = self._ridge_se_diagonal_from_transformed(Xw, yw, a)
                se_map = {col: float(s) for col, s in zip(X_ridge.columns, ridge_se)}
                coef_df[f"ridge_contribution_alpha_{a_str}_std"] = coef_df["player_id"].map(se_map)

        # ---------- TEAM-PARTIALLED ----------
        assert self.team_presence_signed_df is not None, "Team presence data is missing."
        X_team_df = self.team_presence_signed_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        X_team_c = sm.add_constant(X_team_df, has_constant="add")
        y_arr = y.to_numpy(dtype=np.float64)
        w_arr = w.to_numpy(dtype=np.float64)
        X_team_arr = X_team_c.to_numpy(dtype=np.float64)
        valid_team = np.isfinite(y_arr) & np.isfinite(w_arr) & np.isfinite(X_team_arr).all(axis=1)
        with threadpool_limits(1):
            model_team = sm.WLS(y_arr[valid_team], X_team_arr[valid_team], weights=w_arr[valid_team]).fit(method="pinv")
        
        # Team effects (player-style naming)
        team_params = pd.Series(model_team.params, index=X_team_c.columns).drop("const", errors="ignore")
        team_bse    = pd.Series(model_team.bse,    index=X_team_c.columns).drop("const", errors="ignore")
        def _col_to_team_id(col: str) -> int:
            if col.startswith("t"):
                return int(col[1:])
            raise ValueError(f"Unexpected team column: {col}")
        team_ids = [_col_to_team_id(c) for c in team_params.index]
        id_to_name = {}
        for fix in (self.fixtures_data or []):
            id_to_name[fix["teams"]["home"]["id"]] = fix["teams"]["home"]["name"]
            id_to_name[fix["teams"]["away"]["id"]] = fix["teams"]["away"]["name"]
        self.team_coef_df = pd.DataFrame({
            "team_id": team_ids,
            "team_name": [id_to_name.get(tid, None) for tid in team_ids],
            "team_contribution_ols": team_params.values.astype(float),
            "team_contribution_ols_std": team_bse.reindex(team_params.index).to_numpy(dtype=float, copy=False),
        })
            
        y_pred_team = model_team.predict(X_team_arr)
        residual = y - pd.Series(y_pred_team, index=y.index)

        # OLS on residuals
        p_ols_coef, p_ols_bse = fit_wls(X_df, residual, w)
        coef_df["partialled_contribution_ols"] = coef_df["player_id"].map(p_ols_coef.to_dict()).fillna(0.0)
        coef_df["partialled_contribution_ols_std"] = coef_df["player_id"].map(p_ols_bse.to_dict()).fillna(np.nan)

        # Lasso on residuals (reuse chosen alpha for comparability)
        valid_idx_res = np.isfinite(residual) & np.isfinite(X_df).all(axis=1)
        X_lasso_res = X_df.loc[valid_idx_res]
        r_residual = residual.loc[valid_idx_res]

        part_alphas = set(self.lasso_alphas)
        if self.lasso_best_alpha is not None:
            part_alphas.add(float(self.lasso_best_alpha))

        for alpha in sorted(part_alphas):
            a = float(alpha)
            a_tag = ("best" if (self.lasso_best_alpha is not None and abs(a - self.lasso_best_alpha) < 1e-12)
                     else str(a).replace(".", "_"))
            with threadpool_limits(1):
                model = Lasso(alpha=a, fit_intercept=True, max_iter=10000)
                model.fit(X_lasso_res, r_residual)
            coef_series = pd.Series(model.coef_, index=X_lasso_res.columns)
            coef_df[f"partialled_lasso_contribution_alpha_{a_tag}"] = coef_df["player_id"].map(coef_series.to_dict()).fillna(0.0)
            coef_df[f"partialled_lasso_contribution_alpha_{a_tag}_std"] = float(coef_series.std())

        # Ridge on residuals (weighted)
        if len(self.ridge_alphas) > 0:
            X_ridge_res = X_df.loc[valid_idx_res]
            r_ridge = residual.loc[valid_idx_res]
            w_ridge_res = w.loc[valid_idx_res].to_numpy(dtype=float)
            sw_res = np.sqrt(w_ridge_res)
            Xw_res = X_ridge_res.to_numpy(dtype=float) * sw_res[:, None]
            rw = r_ridge.to_numpy(dtype=float) * sw_res
            for alpha in self.ridge_alphas:
                a = float(alpha)
                a_str = str(a).replace(".", "_")
                with threadpool_limits(1):
                    ridge = Ridge(alpha=a, fit_intercept=True)
                    ridge.fit(Xw_res, rw)
                coef_series = pd.Series(ridge.coef_, index=X_ridge_res.columns)
                coef_df[f"partialled_ridge_contribution_alpha_{a_str}"] = coef_df["player_id"].map(coef_series.to_dict()).fillna(0.0)
                ridge_se = self._ridge_se_diagonal_from_transformed(Xw_res, rw, a)
                se_map = {col: float(s) for col, s in zip(X_ridge_res.columns, ridge_se)}
                coef_df[f"partialled_ridge_contribution_alpha_{a_str}_std"] = coef_df["player_id"].map(se_map)

        # Appearance metrics (from precomputed maps/meta)
        def _appearance(pid_str: str, key: str) -> int:
            if not (isinstance(pid_str, str) and pid_str.startswith("p")):
                return 0
            pid = int(pid_str[1:])
            val = (self.player_game_metrics or {}).get(pid, {}).get(key)
            if val is not None:
                return int(val)
            if self.players_meta_df is not None and pid in self.players_meta_df.index and key in self.players_meta_df.columns:
                return int(self.players_meta_df.at[pid, key] or 0)
            return 0
        
        for metric_key in ["games_started", "full_games_played", "games_subbed_on", "games_subbed_off"]:
            coef_df[metric_key] = coef_df["player_id"].map(lambda p: _appearance(p, metric_key))

        # FTE from normalized map (not minutes/90)
        def _fte_from_map(pid_str: str) -> float:
            if not (isinstance(pid_str, str) and pid_str.startswith("p")):
                return 0.0
            return float(self.fte_games_played_by_player.get(int(pid_str[1:]), 0.0))
        coef_df["FTE_games_played"] = coef_df["player_id"].map(_fte_from_map).round(3)

        # Context
        coef_df["country"] = self.COUNTRY_NAME
        coef_df["league"] = self.LEAGUE_NAME
        coef_df["season"] = self.SEASON
        coef_df["total_regular_games"] = self.total_regular_games
        coef_df["valid_regular_games"] = self.valid_regular_games
        coef_df["missing_games_count"] = self.missing_games_count

        # Finalize (sort by baseline OLS)
        coef_df.sort_values(by="contribution_ols", ascending=False, inplace=True)
        coef_df.reset_index(drop=True, inplace=True)
        self.coef_df_final = coef_df

    def build_pooled_results(self) -> None:
        """
        Build a collapsed (pooled) results DataFrame, aggregating across identical presence signatures.
        Std aggregation: root-sum-of-squares where meaningful; lasso_std proxies copied from first member.
        """
        assert self.coef_df_final is not None and self.segment_df is not None
        presence_int = self.segment_df[self.player_cols].astype("int8")
        sig_to_players: Dict[tuple, List[str]] = {}
        for col in presence_int.columns:
            signature = tuple(presence_int[col].tolist())
            sig_to_players.setdefault(signature, []).append(col)
        df = self.coef_df_final.copy()

        def _num(pid_str: str) -> int:
            return int(pid_str[1:]) if isinstance(pid_str, str) and pid_str.startswith("p") else int(pid_str)

        pooled_rows: List[Dict[str, Any]] = []

        sum_cols = [
            "minutes_played","games_started","full_games_played",
            "games_subbed_on","games_subbed_off","FTE_games_played",
            "goals","assists"
        ]
        est_point_cols = [c for c in df.columns if (
            c == "contribution_ols" or
            c.startswith("ridge_contribution_alpha_") or
            c.startswith("lasso_contribution_alpha_") or
            c == "partialled_contribution_ols" or
            c.startswith("partialled_ridge_contribution_alpha_") or
            c.startswith("partialled_lasso_contribution_alpha_")
        ) and not c.endswith("_std")]
        est_std_cols = [c for c in df.columns if c.endswith("_std")]

        for sig, cols in sig_to_players.items():
            if len(cols) == 1:
                col = cols[0]
                row = df.loc[df["player_id"] == col].copy()
                if row.empty:
                    continue
                out = row.iloc[0].to_dict()
                out["player_id"] = _num(col)
                out["is_pooled_estimate"] = False
                out["pooled_members"] = str(_num(col))
                pooled_rows.append(out)
                continue

            sub = df[df["player_id"].isin(cols)].copy()
            if sub.empty:
                continue

            member_ids = [str(_num(c)) for c in cols]
            member_names = sub["player_name"].tolist()
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
            league_pos_vals = sub["league_position"].dropna().unique().tolist()
            league_pos_repr = league_pos_vals[0] if len(league_pos_vals) == 1 else ("Multiple Teams" if len(league_pos_vals) > 1 else None)

            out: Dict[str, Any] = {
                "player_id": "+".join(member_ids),
                "player_name": " + ".join(member_names),
                "position": "/".join(sorted(set(sub["position"].dropna().tolist()))) if not sub["position"].isna().all() else "Unknown",
                "team(s)": team_repr,
                "league_position": league_pos_repr,
                "is_pooled_estimate": True,
                "pooled_members": ", ".join(f"{i}:{n}" for i, n in zip(member_ids, member_names)),
                "country": self.COUNTRY_NAME,
                "league": self.LEAGUE_NAME,
                "season": self.SEASON,
                "total_regular_games": self.total_regular_games,
                "valid_regular_games": self.valid_regular_games,
                "missing_games_count": self.missing_games_count,
            }
            out["contribution_ols"] = float(sub["contribution_ols"].sum()) if "contribution_ols" in sub else np.nan
            out["contribution_ols_std"] = float(np.sqrt(np.nansum(np.square(sub["contribution_ols_std"].to_numpy(dtype=float))))) if "contribution_ols_std" in sub else np.nan

            for c in est_point_cols:
                out[c] = float(sub[c].sum(skipna=True)) if c in sub else np.nan
            for c in est_std_cols:
                if c in sub and not sub[c].isna().all():
                    try:
                        out[c] = float(np.sqrt(np.nansum(np.square(sub[c].to_numpy(dtype=float)))))
                    except Exception:
                        out[c] = float(sub[c].iloc[0])

            for c in sum_cols:
                if c in sub:
                    out[c] = float(sub[c].sum()) if c == "minutes_played" else int(sub[c].sum())

            pooled_rows.append(out)

        pooled_df = pd.DataFrame(pooled_rows)
        if not pooled_df.empty:
            pooled_df.sort_values(by="contribution_ols", ascending=False, inplace=True)
            pooled_df.reset_index(drop=True, inplace=True)
        self.coef_df_pooled = pooled_df
        
    # ---------- relegation/playoff helpers ----------
    def _is_playoff_like_round(self, round_str: str) -> bool:
        if not round_str:
            return False
        s = str(round_str).lower()
        keys = ["relegation", "promotion", "play-off", "playoff", "play-offs", "barrage", "qualifying"]
        return any(k in s for k in keys)

    def _cutoff_allows(self, fix: dict) -> bool:
        if self.date_cutoff_utc is None:
            return True
        fdt = self._to_utc(((fix.get("fixture") or {}).get("date")))
        return (fdt is not None) and (fdt.date() <= self.date_cutoff_utc.date())

    def _build_regular_team_table(self) -> pd.DataFrame:
        """
        Regular-season only, cutoff-aware. Computes W/D/L, points, GD per team
        and derives league_position directly from this table (integer).
        """
        if self.match_summary_df is None or self.match_summary_df.empty:
            return pd.DataFrame(columns=[
                "team_id","matches_played","won","drawn","lost",
                "goals_scored","goals_conceded","league_position","points","goal_difference"
            ])

        tm = self.match_summary_df.copy()
        team_ids = sorted(set(tm["home_team_id"]).union(tm["away_team_id"]))
        rows = []
        for tid in team_ids:
            mh = (tm["home_team_id"] == tid)
            ma = (tm["away_team_id"] == tid)
            mt = (mh | ma)

            wins_home = (mh & (tm["home_goals"] > tm["away_goals"])).sum()
            wins_away = (ma & (tm["away_goals"] > tm["home_goals"])).sum()
            wins = int(wins_home + wins_away)

            draws = int((mt & (tm["home_goals"] == tm["away_goals"])).sum())

            losses_home = (mh & (tm["home_goals"] < tm["away_goals"])).sum()
            losses_away = (ma & (tm["away_goals"] < tm["home_goals"])).sum()
            losses = int(losses_home + losses_away)

            matches_played = int(mt.sum())

            gf = int((mh * tm["home_goals"] + ma * tm["away_goals"]).sum())
            ga = int((mh * tm["away_goals"] + ma * tm["home_goals"]).sum())

            points = int(wins * 3 + draws)
            gd = int(gf - ga)

            rows.append({
                "team_id": int(tid),
                "matches_played": matches_played,
                "won": wins, "drawn": draws, "lost": losses,
                "goals_scored": gf, "goals_conceded": ga,
                "points": points, "goal_difference": gd,
            })

        df = pd.DataFrame(rows)

        # league position from this df directly (points, then goals_scored as tiebreaker for stability)
        df_sorted = df.sort_values(by=["points","goals_scored"], ascending=[False, False]).reset_index(drop=True)
        pos_map = {int(tid): i+1 for i, tid in enumerate(df_sorted["team_id"].tolist())}
        df["league_position"] = df["team_id"].map(pos_map).astype("Int64")

        # sanity check
        bad = df[(df["matches_played"] != df["won"] + df["drawn"] + df["lost"])]
        if not bad.empty:
            print("[WARN] W/D/L do not sum to matches_played for team_ids:", bad["team_id"].tolist())

        return df
    
    def _goals_from_fixture_or_events(self, fix: dict, evpack: dict) -> tuple[int, int]:
        """
        Prefer event parsing; if missing/incomplete, fall back to fixture score fields.
        Returns (home_goals, away_goals) as integers.
        """
        events = (evpack or {}).get("events") or []
        if events:
            hg = ag = 0
            for e in events:
                t = (e.get("type") or "").lower()
                if t != "goal":
                    continue
                tm_id = (e.get("team") or {}).get("id")
                if tm_id == fix["teams"]["home"]["id"]:
                    hg += 1
                elif tm_id == fix["teams"]["away"]["id"]:
                    ag += 1
            return int(hg), int(ag)
        
        gblk = (fix.get("goals") or {})
        if gblk and gblk.get("home") is not None and gblk.get("away") is not None:
            return int(gblk["home"] or 0), int(gblk["away"] or 0)
        
        sblk = (fix.get("score") or {})
        ft = (sblk.get("fulltime") or {})
        if ft and ft.get("home") is not None and ft.get("away") is not None:
            return int(ft["home"] or 0), int(ft["away"] or 0)
        
        return 0, 0
        
    def _build_playoff_team_table(self) -> pd.DataFrame:
        """
        Relegation/promotion/playoff matches only (cutoff-aware).
        These do NOT affect league_position, but will be merged into totals.
        """
        if not self.fixtures_data:
            return pd.DataFrame(columns=[
                "team_id","matches_played_playoff","won_playoff","drawn_playoff","lost_playoff",
                "goals_scored_playoff","goals_conceded_playoff","points_playoff","goal_difference_playoff"
            ])
        
        rows = collections.defaultdict(lambda: {
            "matches_played_playoff": 0, "won_playoff": 0, "drawn_playoff": 0, "lost_playoff": 0,
            "goals_scored_playoff": 0, "goals_conceded_playoff": 0
        })
        
        for fix in self.fixtures_data:
            round_str = (fix.get("league", {}) or {}).get("round", "") or ""
            if self._is_regular(fix) or not self._is_playoff_like_round(round_str):
                continue
            if not self._cutoff_allows(fix):
                continue
        
            fid = str(fix["fixture"]["id"])
            evpack = (self.match_events or {}).get(fid) or {}
        
            home_id = int(fix["teams"]["home"]["id"])
            away_id = int(fix["teams"]["away"]["id"])
        
            hg, ag = self._goals_from_fixture_or_events(fix, evpack)
        
            rows[home_id]["matches_played_playoff"] += 1
            rows[home_id]["goals_scored_playoff"] += hg
            rows[home_id]["goals_conceded_playoff"] += ag
            if hg > ag:
                rows[home_id]["won_playoff"] += 1
            elif hg == ag:
                rows[home_id]["drawn_playoff"] += 1
            else:
                rows[home_id]["lost_playoff"] += 1
        
            rows[away_id]["matches_played_playoff"] += 1
            rows[away_id]["goals_scored_playoff"] += ag
            rows[away_id]["goals_conceded_playoff"] += hg
            if ag > hg:
                rows[away_id]["won_playoff"] += 1
            elif ag == hg:
                rows[away_id]["drawn_playoff"] += 1
            else:
                rows[away_id]["lost_playoff"] += 1
        
        if not rows:
            return pd.DataFrame(columns=[
                "team_id","matches_played_playoff","won_playoff","drawn_playoff","lost_playoff",
                "goals_scored_playoff","goals_conceded_playoff","points_playoff","goal_difference_playoff"
            ])
        
        df = pd.DataFrame([{"team_id": tid, **stats} for tid, stats in rows.items()])
        df["points_playoff"] = (df["won_playoff"] * 3 + df["drawn_playoff"]).astype(int)
        df["goal_difference_playoff"] = (df["goals_scored_playoff"] - df["goals_conceded_playoff"]).astype(int)
        
        bad = df[(df["matches_played_playoff"] != df["won_playoff"] + df["drawn_playoff"] + df["lost_playoff"])]
        if not bad.empty:
            print("[WARN] Playoff W/D/L mismatch for team_ids:", bad["team_id"].tolist())
        
        return df

    def build_team_coefficients_df(self) -> pd.DataFrame:
        """
        Merge: team coefficients + regular-season totals (+ playoff add-on) into one table.
        Totals include playoffs; league_position stays regular-season only (integer).
        """
        assert getattr(self, "team_coef_df", None) is not None, "team_coef_df missing. Did compute_impacts() run?"
    
        reg = self._build_regular_team_table()
        po  = self._build_playoff_team_table()
    
        merged = pd.merge(reg, po, on="team_id", how="outer")
    
        # fill missing playoff columns
        for c in ["matches_played_playoff","won_playoff","drawn_playoff","lost_playoff",
                  "goals_scored_playoff","goals_conceded_playoff","points_playoff","goal_difference_playoff"]:
            if c not in merged: merged[c] = 0
            merged[c] = merged[c].fillna(0).astype(int)
    
        # totals = regular + playoffs
        merged["matches_played"]   = merged["matches_played"].fillna(0).astype(int)   + merged["matches_played_playoff"]
        merged["won"]              = merged["won"].fillna(0).astype(int)              + merged["won_playoff"]
        merged["drawn"]            = merged["drawn"].fillna(0).astype(int)            + merged["drawn_playoff"]
        merged["lost"]             = merged["lost"].fillna(0).astype(int)             + merged["lost_playoff"]
        merged["goals_scored"]     = merged["goals_scored"].fillna(0).astype(int)     + merged["goals_scored_playoff"]
        merged["goals_conceded"]   = merged["goals_conceded"].fillna(0).astype(int)   + merged["goals_conceded_playoff"]
        merged["points"]           = merged["points"].fillna(0).astype(int)           + merged["points_playoff"]
        merged["goal_difference"]  = merged["goal_difference"].fillna(0).astype(int)  + merged["goal_difference_playoff"]
    
        # attach coefficients & names
        df = pd.merge(self.team_coef_df.copy(), merged, on="team_id", how="left")
    
        # context
        df["country"] = self.COUNTRY_NAME
        df["league"]  = self.LEAGUE_NAME
        df["season"]  = self.SEASON
        df["cutoff_date"] = (self.date_cutoff_utc.date().isoformat()
                             if getattr(self, "date_cutoff_utc", None) is not None else None)

        # ensure integer league_position (nullable)
        if "league_position" in df.columns:
            try:
                df["league_position"] = df["league_position"].astype("Int64")
            except Exception:
                pass
    
        # Final column order
        ordered = [
            "country","league","season","cutoff_date",
            "team_id","team_name","league_position",
            "matches_played","won","drawn","lost","goals_scored","goals_conceded",
            "points","goal_difference",
            "team_contribution_ols","team_contribution_ols_std",
            # playoff audit
            "matches_played_playoff","won_playoff","drawn_playoff","lost_playoff",
            "goals_scored_playoff","goals_conceded_playoff","points_playoff","goal_difference_playoff",
        ]
        for c in ordered:
            if c not in df.columns:
                df[c] = np.nan
    
        # sort by coefficient desc (like players)
        return df[ordered].sort_values(
            ["season","league","team_contribution_ols"], ascending=[True, True, False]
        ).reset_index(drop=True)
        
    def run_analysis(self, until_date: Optional[str] = None, group_pooled: bool = True) -> None:
        if until_date is not None:
            self.date_cutoff_utc = pd.to_datetime(until_date, utc=True)
        print("[INFO] Building season data (one pass)...")
        self.prep_data_up_to_cutoff(keep_player_after_red=False)
        if self.save_intermediate and self.segment_df is not None:
            try:
                intermediate_path = self._p("regression_data.pkl")
                self.segment_df.to_pickle(intermediate_path)
                print(f"[INFO] Saved intermediate regression data -> {intermediate_path}")
            except Exception as e:
                print(f"[WARN] Could not save intermediate data: {e}")
        print("[INFO] Computing contributions (baseline + team-partialled)...")
        self.compute_impacts()
        print("[INFO] Adding goals & assists...")
        self.add_goals_assists()
        if group_pooled:
            print("[INFO] Building pooled results...")
            self.build_pooled_results()
            
            
    def save_results(self, output_path: Optional[str] = None) -> None:
        """
        Save per-player (and pooled) results to Excel/CSV and to SQLite (full runs).
        Column order respects:
            non-partialled (OLS → Ridge → Lasso; point then std)
            THEN partialled (OLS → Ridge → Lasso; point then std)

        Additionally, persist team coefficients & team totals (regular-season table + playoff breakdown)
        into SQLite table `team_coefficients`. For cutoff runs, this table is written with a cutoff_date
        key so you can keep multiple snapshots per season.
        """
        assert self.coef_df_final is not None, "No results to save. Did you run run_analysis()?"

        def _round_sig(x: Any, sig: int = 3):
            try:
                if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                    return x
                if isinstance(x, (int, np.integer)):
                    return int(x)
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

        out_df = self.coef_df_final.copy()
        # convert player_id "p123" → 123 in outputs
        out_df["player_id"] = out_df["player_id"].apply(lambda pid: int(pid[1:]) if isinstance(pid, str) and pid.startswith("p") else pid)
        pooled_df = self.coef_df_pooled.copy() if getattr(self, "coef_df_pooled", None) is not None else pd.DataFrame()

        # ----- Build ordered estimation columns -----
        est_cols = []
        # OLS
        if "contribution_ols" in out_df.columns:
            est_cols += ["contribution_ols"]
        if "contribution_ols_std" in out_df.columns:
            est_cols += ["contribution_ols_std"]
        # Ridge
        for alpha in self.ridge_alphas:
            a_str = str(float(alpha)).replace(".", "_")
            pt = f"ridge_contribution_alpha_{a_str}"
            sd = f"ridge_contribution_alpha_{a_str}_std"
            if pt in out_df.columns: est_cols.append(pt)
            if sd in out_df.columns: est_cols.append(sd)
        # Lasso (fixed alphas)
        for alpha in self.lasso_alphas:
            a_str = str(float(alpha)).replace(".", "_")
            pt = f"lasso_contribution_alpha_{a_str}"
            sd = f"lasso_contribution_alpha_{a_str}_std"
            if pt in out_df.columns: est_cols.append(pt)
            if sd in out_df.columns: est_cols.append(sd)
        # Lasso "best" alpha columns
        for col in ["lasso_contribution_alpha_best", "lasso_contribution_alpha_best_std",
                    "partialled_lasso_contribution_alpha_best", "partialled_lasso_contribution_alpha_best_std"]:
            if col in out_df.columns:
                est_cols.append(col)

        if "lasso_contribution_alpha_best" in out_df.columns:
            out_df["lasso_alpha_selected"] = self.lasso_best_alpha

        # THEN partialled: OLS → Ridge → Lasso
        if "partialled_contribution_ols" in out_df.columns:
            est_cols += ["partialled_contribution_ols"]
        if "partialled_contribution_ols_std" in out_df.columns:
            est_cols += ["partialled_contribution_ols_std"]
        for alpha in self.ridge_alphas:
            a_str = str(float(alpha)).replace(".", "_")
            pt = f"partialled_ridge_contribution_alpha_{a_str}"
            sd = f"partialled_ridge_contribution_alpha_{a_str}_std"
            if pt in out_df.columns: est_cols.append(pt)
            if sd in out_df.columns: est_cols.append(sd)
        for alpha in self.lasso_alphas:
            a_str = str(float(alpha)).replace(".", "_")
            pt = f"partialled_lasso_contribution_alpha_{a_str}"
            sd = f"partialled_lasso_contribution_alpha_{a_str}_std"
            if pt in out_df.columns: est_cols.append(pt)
            if sd in out_df.columns: est_cols.append(sd)

        if "partialled_lasso_contribution_alpha_best" in out_df.columns:
            out_df["partialled_lasso_alpha_selected"] = self.lasso_best_alpha

        meta_cols = ["player_id", "player_name", "position", "team(s)", "league_position", "country", "league", "season"]
        minutes_cols = ["minutes_played", "FTE_games_played"]
        appearances_cols = ["games_started", "full_games_played", "games_subbed_on", "games_subbed_off"]
        scoring_cols = ["goals", "assists", "pen_goals", "pen_assists", "pen_missed", "own_goals", "yellow_cards"]
        pooling_cols = ["is_pooled_member", "pooled_group_key", "other_players_same_minutes", "pooled", "pooled_members"]
        dataset_cols = ["total_regular_games", "valid_regular_games", "missing_games_count"]

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

        # Apply 3-sig formatting to estimates & stds (incl. "best")
        out_df = _apply_sig(out_df, est_cols)
        out_df = _reorder(out_df).sort_values(by="contribution_ols", ascending=False)

        if not pooled_df.empty:
            pooled_est_cols = [c for c in pooled_df.columns if c in est_cols]
            pooled_df = _apply_sig(pooled_df, pooled_est_cols)
            pooled_df = _reorder(pooled_df).sort_values(by="contribution_ols", ascending=False)

        # Paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        is_cutoff_run = getattr(self, "date_cutoff_utc", None) is not None
        if output_path is None:
            if is_cutoff_run:
                cutoff_tag = self.date_cutoff_utc.date().isoformat() if self.date_cutoff_utc else "cutoff"
                base_name = f"output_cutoff_{self.COUNTRY_NAME}_{self.LEAGUE_NAME}_{self.SEASON}_{cutoff_tag}"
                output_path = os.path.join(script_dir, f"{base_name}.xlsx")
                csv_path = os.path.join(script_dir, f"{base_name}.csv")
                pooled_csv_path = os.path.join(script_dir, f"{base_name}_pooled.csv")
            else:
                output_path = self._p("output.xlsx")
                csv_path = self._p("output.csv")
                pooled_csv_path = self._p("output_pooled.csv")

        # Excel
        out_df.to_excel(output_path, sheet_name="OLS", index=False)
        from openpyxl import Workbook  # ensure engine available
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            # Baseline sheets
            for alpha in self.lasso_alphas:
                a_str = str(float(alpha)).replace(".", "_")
                col = f"lasso_contribution_alpha_{a_str}"
                if col in out_df:
                    df_sub = out_df[out_df[col].notna()].copy()
                    df_sub.sort_values(by=col, ascending=False, inplace=True)
                    df_sub.to_excel(writer, sheet_name=f"Lasso_{a_str}", index=False)
            if "lasso_contribution_alpha_best" in out_df:
                df_sub = out_df[out_df["lasso_contribution_alpha_best"].notna()].copy()
                df_sub.sort_values(by="lasso_contribution_alpha_best", ascending=False, inplace=True)
                df_sub.to_excel(writer, sheet_name="Lasso_best", index=False)

            for alpha in self.ridge_alphas:
                a_str = str(float(alpha)).replace(".", "_")
                col = f"ridge_contribution_alpha_{a_str}"
                if col in out_df:
                    df_sub = out_df[out_df[col].notna()].copy()
                    df_sub.sort_values(by=col, ascending=False, inplace=True)
                    df_sub.to_excel(writer, sheet_name=f"Ridge_{a_str}", index=False)

            # Partialled sheets
            if "partialled_contribution_ols" in out_df:
                df_sub = out_df.copy().sort_values(by="partialled_contribution_ols", ascending=False)
                df_sub.to_excel(writer, sheet_name="OLS_PARTIALLED", index=False)

            for alpha in self.lasso_alphas:
                a_str = str(float(alpha)).replace(".", "_")
                col = f"partialled_lasso_contribution_alpha_{a_str}"
                if col in out_df:
                    df_sub = out_df[out_df[col].notna()].copy()
                    df_sub.sort_values(by=col, ascending=False, inplace=True)
                    df_sub.to_excel(writer, sheet_name=f"Lasso_{a_str}_PARTIALLED", index=False)
            if "partialled_lasso_contribution_alpha_best" in out_df:
                df_sub = out_df[out_df["partialled_lasso_contribution_alpha_best"].notna()].copy()
                df_sub.sort_values(by="partialled_lasso_contribution_alpha_best", ascending=False, inplace=True)
                df_sub.to_excel(writer, sheet_name="Lasso_best_PARTIALLED", index=False)

            for alpha in self.ridge_alphas:
                a_str = str(float(alpha)).replace(".", "_")
                col = f"partialled_ridge_contribution_alpha_{a_str}"
                if col in out_df:
                    df_sub = out_df[out_df[col].notna()].copy()
                    df_sub.sort_values(by=col, ascending=False, inplace=True)
                    df_sub.to_excel(writer, sheet_name=f"Ridge_{a_str}_PARTIALLED", index=False)

            # Pooled sheets
            if not pooled_df.empty:
                pooled_df.to_excel(writer, sheet_name="OLS_POOLED", index=False)
                for alpha in self.lasso_alphas:
                    a_str = str(float(alpha)).replace(".", "_")
                    col = f"lasso_contribution_alpha_{a_str}"
                    if col in pooled_df:
                        dfp = pooled_df[pooled_df[col].notna()].copy()
                        dfp.sort_values(by=col, ascending=False, inplace=True)
                        dfp.to_excel(writer, sheet_name=f"Lasso_{a_str}_POOLED", index=False)
                if "lasso_contribution_alpha_best" in pooled_df:
                    dfp = pooled_df[pooled_df["lasso_contribution_alpha_best"].notna()].copy()
                    dfp.sort_values(by="lasso_contribution_alpha_best", ascending=False, inplace=True)
                    dfp.to_excel(writer, sheet_name="Lasso_best_POOLED", index=False)
                for alpha in self.ridge_alphas:
                    a_str = str(float(alpha)).replace(".", "_")
                    col = f"ridge_contribution_alpha_{a_str}"
                    if col in pooled_df:
                        dfp = pooled_df[pooled_df[col].notna()].copy()
                        dfp.sort_values(by=col, ascending=False, inplace=True)
                        dfp.to_excel(writer, sheet_name=f"Ridge_{a_str}_POOLED", index=False)

        # CSV
        out_df.to_csv(csv_path, index=False)
        if not pooled_df.empty:
            pooled_df.to_csv(pooled_csv_path, index=False)

        # ---- Build team coefficients DF (apply rounding + integer league_position) ----
        team_df_available = getattr(self, "team_coef_df", None) is not None
        full_team_df = None
        if team_df_available:
            try:
                full_team_df = self.build_team_coefficients_df()
                # apply 3-sig rule to coefficients & stds
                for c in ["team_contribution_ols","team_contribution_ols_std"]:
                    if c in full_team_df.columns:
                        full_team_df[c] = full_team_df[c].map(lambda v: _round_sig(v, 3))
                if "league_position" in full_team_df.columns:
                    try:
                        full_team_df["league_position"] = full_team_df["league_position"].astype("Int64")
                    except Exception:
                        pass
            except Exception as e:
                print(f"[WARN] Failed to build team_coefficients DF: {e}")

        # DB writes (full runs only) for analysis_results / pooled
        if not is_cutoff_run:
            conn = sqlite3.connect(self.DB_PATH); cur = conn.cursor()
            try:
                table_exists = cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results';"
                ).fetchone() is not None
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

                if not pooled_df.empty:
                    pooled_exists = cur.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results_pooled';"
                    ).fetchone() is not None
                    if pooled_exists:
                        cur.execute(
                            "DELETE FROM analysis_results_pooled WHERE country=? AND league=? AND season=?;",
                            (self.COUNTRY_NAME, self.LEAGUE_NAME, self.SEASON)
                        )
                        conn.commit()
                    pooled_df.to_sql("analysis_results_pooled", conn, if_exists="append", index=False)
            finally:
                conn.close()

            # Mirror-to-disk check
            conn = sqlite3.connect(self.DB_PATH)
            try:
                df_from_db = pd.read_sql_query(
                    "SELECT * FROM analysis_results WHERE country=? AND league=? AND season=?;",
                    conn, params=[self.COUNTRY_NAME, self.LEAGUE_NAME, self.SEASON]
                )
            finally:
                conn.close()
            df_from_db.to_excel(self._p("analysis_from_db.xlsx"), index=False)
            df_from_db.to_csv(self._p("analysis_from_db.csv"), index=False)
        else:
            tag = self.date_cutoff_utc.date().isoformat() if self.date_cutoff_utc else "cutoff"
            print(f"[INFO] Cutoff run ({tag}): saved Excel/CSV to {os.path.dirname(output_path)} and skipped DB write for per-player tables.")

        # ---- write team_coefficients table (also for cutoff runs) ----
        if full_team_df is not None and not full_team_df.empty:
            conn = sqlite3.connect(self.DB_PATH); cur = conn.cursor()
            try:
                table_name = "team_coefficients"
                exists = cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
                ).fetchone() is not None

                def _cols(conn, name):
                    rows = conn.execute(f"PRAGMA table_info({name});").fetchall()
                    return [r[1] for r in rows]

                if exists:
                    existing_cols = _cols(conn, table_name)
                    if set(existing_cols) != set(full_team_df.columns.tolist()):
                        full_team_df.to_sql(table_name, conn, if_exists="replace", index=False)
                    else:
                        if is_cutoff_run:
                            cur.execute(
                                f"DELETE FROM {table_name} WHERE country=? AND league=? AND season=? AND cutoff_date=?;",
                                (self.COUNTRY_NAME, self.LEAGUE_NAME, self.SEASON,
                                 self.date_cutoff_utc.date().isoformat())
                            )
                        else:
                            cur.execute(
                                f"DELETE FROM {table_name} WHERE country=? AND league=? AND season=? AND cutoff_date IS NULL;",
                                (self.COUNTRY_NAME, self.LEAGUE_NAME, self.SEASON)
                            )
                        conn.commit()
                        full_team_df.to_sql(table_name, conn, if_exists="append", index=False)
                else:
                    full_team_df.to_sql(table_name, conn, if_exists="replace", index=False)
            finally:
                conn.close()

            if is_cutoff_run:
                print("[INFO] team_coefficients written for cutoff snapshot.")
        elif team_df_available and (full_team_df is None or full_team_df.empty):
            print("[WARN] team_coefficients not written: empty dataframe.")

    def run(self, until_date: str = None, group_pooled: bool = True) -> None:
        self.load_data()
        self.run_analysis(until_date=until_date, group_pooled=group_pooled)
        self.save_results()

# ---------------- main ----------------
if __name__ == "__main__":
    runner = SeasonAnalyzer(
        country_name="Germany",
        league_name="Bundesliga",
        season=2022,
        lasso_alphas=(0.001),  # 0.01,
        ridge_alphas=(10.0),   # 1.0,
    )
    runner.run()
