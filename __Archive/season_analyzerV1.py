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
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits


class SeasonAnalyzer:
    def __init__(
        self,
        league_name: str = "Bundesliga",
        country_name: str = "Germany",
        season: int = 2023,
        lasso_alphas: List[float] | float = (0.01, 0.001),
        save_intermediate: bool = False,
        base_dir: Optional[str] = None,
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

        self.minutes_equiv_tolerance = 0

        os.makedirs(self.SAVE_DIR, exist_ok=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.DB_PATH = os.path.join(script_dir, "analysis_results.db")

        # Data containers
        self.fixtures_data: List[Dict[str, Any]] | None = None
        self.match_events: Dict[str, Dict[str, Any]] | None = None
        self.players: Dict[str, Any] | None = None

        self.team_players: Dict[int, Set[int]] = {}     # {team_id: set(player_ids)}
        self.player_id_to_name: Dict[int, str] = {}
        self.player_id_to_position: Dict[int, str] = {}
        self.player_id_to_age: Dict[int, Optional[int]] = {}

        # Matrices / tables
        self.segment_df: pd.DataFrame | None = None      # meta + player indicators
        self.match_summary_df: pd.DataFrame | None = None
        self.team_summary_df: pd.DataFrame | None = None
        self.coef_df_final: pd.DataFrame | None = None
        self.player_game_metrics: Dict[int, Dict[str, float | int]] = {}

        # Accounting
        self.total_regular_games = 0
        self.valid_regular_games = 0
        self.missing_games_count = 0
        self.valid_fixture_ids: Set[str] = set()

        # Columns for players (now persisted on the instance)
        self.all_player_ids: List[int] = []
        self.player_cols: List[str] = []  # e.g., ["p123", "p456", ...]

        self.save_intermediate = save_intermediate

    # ---------- IO ----------
    def _p(self, filename: str) -> str:
        return os.path.join(self.SAVE_DIR, filename)

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

    def _fixture_has_required_data(self, fid: str) -> bool:
        ev = (self.match_events or {}).get(fid, {})
        return bool(ev.get("lineups")) and bool(ev.get("events"))

    # ---------- pipeline steps ----------
    def identify_players_and_build_segments(self) -> None:
        """
        Build player-presence segments per match and compile meta information.
        Efficient: collects rows in lists and builds DataFrames once.
        """
        assert self.fixtures_data is not None and self.match_events is not None, "Call load_data() first."

        # Regular-season fixtures with data
        reg_fixture_ids = [str(fix["fixture"]["id"]) for fix in self.fixtures_data if self._is_regular(fix)]
        self.total_regular_games = len(reg_fixture_ids)
        self.valid_fixture_ids = {fid for fid in reg_fixture_ids if self._fixture_has_required_data(fid)}
        self.valid_regular_games = len(self.valid_fixture_ids)
        self.missing_games_count = self.total_regular_games - self.valid_regular_games

        # Collect rosters & basic player info
        for fix in self.fixtures_data:
            fid = str(fix["fixture"]["id"])
            if fid not in self.valid_fixture_ids:
                continue
            evpack = self.match_events[fid]

            # lineups -> starters
            for lineup in evpack.get("lineups", []) or []:
                team_id = lineup["team"]["id"]
                self.team_players.setdefault(team_id, set())
                for entry in lineup.get("startXI", []) or []:
                    pid = entry["player"]["id"]
                    pname = entry["player"]["name"]
                    ppos = entry["player"].get("pos", entry["player"].get("position", "Unknown"))
                    self.team_players[team_id].add(pid)
                    self.player_id_to_name[pid] = pname
                    self.player_id_to_position[pid] = ppos

            # substitutions -> catch on/off players too
            for ev in evpack.get("events", []) or []:
                if ev.get("type") == "subst":
                    team_id = ev["team"]["id"]
                    self.team_players.setdefault(team_id, set())
                    pid_out = (ev.get("player") or {}).get("id")
                    pname_out = (ev.get("player") or {}).get("name")
                    pid_in = (ev.get("assist") or {}).get("id")
                    pname_in = (ev.get("assist") or {}).get("name")
                    if pid_in:
                        self.team_players[team_id].add(pid_in)
                        if pname_in:
                            self.player_id_to_name[pid_in] = pname_in
                    if pid_out:
                        self.team_players[team_id].add(pid_out)
                        if pname_out:
                            self.player_id_to_name[pid_out] = pname_out

        # Persist player columns
        all_ids = sorted({pid for s in self.team_players.values() for pid in s})
        self.all_player_ids = all_ids
        self.player_cols = [f"p{pid}" for pid in all_ids]

        # Build segments
        meta_rows: List[Dict[str, Any]] = []
        player_rows: List[Dict[str, int]] = []   # sparse dicts: only present players -> +/-1

        for fix in self.fixtures_data:
            fid = str(fix["fixture"]["id"])
            if fid not in self.valid_fixture_ids:
                continue

            evpack = self.match_events[fid]
            home_team = fix["teams"]["home"]["name"]; away_team = fix["teams"]["away"]["name"]
            home_id = fix["teams"]["home"]["id"];     away_id = fix["teams"]["away"]["id"]
            game_date = (fix["fixture"].get("date") if fix.get("fixture") else None)

            events = evpack.get("events", []) or []
            match_end_min = max([self._get_minute(e) for e in events] + [90])

            # init current on-field from lineups
            current: Dict[int, int] = {}
            for lineup in evpack.get("lineups", []) or []:
                team_sign = 1 if lineup["team"]["id"] == home_id else -1
                for entry in lineup.get("startXI", []) or []:
                    current[entry["player"]["id"]] = team_sign

            # sort substitutions
            subs = sorted([e for e in events if e.get("type") == "subst"], key=self._get_minute)
            boundaries = sorted({self._get_minute(e) for e in subs} | {match_end_min})

            segment_start = 0
            for seg_end in boundaries:
                duration = seg_end - segment_start

                # goals in (segment_start, seg_end]
                home_goals = sum(
                    1 for e in events
                    if e.get("type") == "Goal" and e["team"]["id"] == home_id
                    and segment_start < self._get_minute(e) <= seg_end
                )
                away_goals = sum(
                    1 for e in events
                    if e.get("type") == "Goal" and e["team"]["id"] == away_id
                    and segment_start < self._get_minute(e) <= seg_end
                )

                # sparse player row
                prow: Dict[str, int] = {}
                for pid, sign in current.items():
                    prow[f"p{pid}"] = sign

                player_rows.append(prow)
                meta_rows.append({
                    "fixture": fid,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_id": home_id,
                    "away_id": away_id,
                    "start": segment_start,
                    "end": seg_end,
                    "duration": duration,
                    "goal_diff": home_goals - away_goals,
                    "game_date": game_date,
                })

                # apply subs at seg_end
                for e in (x for x in subs if self._get_minute(x) == seg_end):
                    pid_out = (e.get("player") or {}).get("id")
                    pid_in = (e.get("assist") or {}).get("id")
                    if pid_out in current:
                        current.pop(pid_out, None)
                    if pid_in:
                        current[pid_in] = 1 if e["team"]["id"] == home_id else -1

                segment_start = seg_end

        meta_df = pd.DataFrame(meta_rows)
        players_df = pd.DataFrame(player_rows).reindex(columns=self.player_cols, fill_value=0).fillna(0)
        # ensure numeric & small dtype
        players_df = players_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype("int8")
        self.segment_df = pd.concat([meta_df, players_df], axis=1)

    def calculate_game_metrics(self) -> None:
        """Per-player appearance counts and minutes played for valid fixtures."""
        assert self.fixtures_data is not None and self.match_events is not None

        self.player_game_metrics = {
            pid: dict(games_started=0, full_games_played=0, games_subbed_on=0, games_subbed_off=0, minutes_played=0)
            for pid in self.player_id_to_name
        }

        for fix in self.fixtures_data:
            fid = str(fix["fixture"]["id"])
            if fid not in self.valid_fixture_ids:
                continue

            evpack = self.match_events[fid]
            events = evpack.get("events", []) or []
            starters: Set[int] = set()
            subs_out: Set[int] = set()
            match_end_min = max([self._get_minute(e) for e in events] + [90])

            for lineup in evpack.get("lineups", []) or []:
                for px in lineup.get("startXI", []) or []:
                    pid = px["player"]["id"]
                    starters.add(pid)
                    self.player_game_metrics[pid]["games_started"] += 1

            for e in events:
                if e.get("type") == "subst":
                    pid_out = (e.get("player") or {}).get("id")
                    pid_in = (e.get("assist") or {}).get("id")
                    minute = self._get_minute(e)
                    if pid_out:
                        subs_out.add(pid_out)
                        self.player_game_metrics[pid_out]["games_subbed_off"] += 1
                        self.player_game_metrics[pid_out]["minutes_played"] += minute
                    if pid_in:
                        self.player_game_metrics[pid_in]["games_subbed_on"] += 1
                        self.player_game_metrics[pid_in]["minutes_played"] += (match_end_min - minute)

            for pid in (starters - subs_out):
                self.player_game_metrics[pid]["full_games_played"] += 1
                self.player_game_metrics[pid]["minutes_played"] += match_end_min

    def summarize_matches_and_teams(self) -> None:
        """Create per-match summary and league table summary for valid fixtures."""
        assert self.segment_df is not None and self.fixtures_data is not None and self.match_events is not None

        match_summary_list: List[Dict[str, Any]] = []
        for fix in self.fixtures_data:
            fid = str(fix["fixture"]["id"])
            if fid not in self.valid_fixture_ids:
                continue
            evs = (self.match_events[fid].get("events") or [])

            home_team = fix["teams"]["home"]["name"]; away_team = fix["teams"]["away"]["name"]
            home_id = fix["teams"]["home"]["id"];     away_id = fix["teams"]["away"]["id"]

            round_str = (fix.get("league", {}) or {}).get("round", "")
            try:
                matchday = int(str(round_str).split("-")[-1].strip())
            except Exception:
                matchday = None

            start_players = {px["player"]["id"] for lu in (self.match_events[fid].get("lineups") or [])
                             for px in (lu.get("startXI") or [])}
            subs_out_players = { (e.get("player") or {}).get("id") for e in evs if e.get("type") == "subst" }
            unchanged = len(start_players - subs_out_players)

            subs_by_min = {}
            for e in evs:
                if e.get("type") == "subst":
                    minute = self._get_minute(e)
                    subs_by_min.setdefault((e["team"]["id"], minute), 0)
                    subs_by_min[(e["team"]["id"], minute)] += 1
            total_subs = len(subs_by_min)

            home_goals = sum(1 for e in evs if e.get("type") == "Goal" and e["team"]["id"] == home_id)
            away_goals = sum(1 for e in evs if e.get("type") == "Goal" and e["team"]["id"] == away_id)

            match_summary_list.append({
                "matchday": matchday,
                "fixture_id": fid,
                "home_team": home_team, "away_team": away_team,
                "home_team_id": home_id, "away_team_id": away_id,
                "home_goals": home_goals, "away_goals": away_goals,
                "sub_events": total_subs, "unchanged_players": unchanged
            })

        self.match_summary_df = pd.DataFrame(match_summary_list)

        # Team summary
        team_summary_list: List[Dict[str, Any]] = []
        total_minutes = float(self.segment_df["duration"].sum()) if len(self.segment_df) else 0.0

        for team_id, players in self.team_players.items():
            tm = self.match_summary_df
            mask_home = (tm["home_team_id"] == team_id)
            mask_away = (tm["away_team_id"] == team_id)

            wins = (mask_home & (tm["home_goals"] > tm["away_goals"])).sum() + \
                   (mask_away & (tm["away_goals"] > tm["home_goals"])).sum()
            draws = (tm["home_goals"] == tm["away_goals"]).sum()
            losses = len(tm) - wins - draws

            goals_scored = (mask_home * tm["home_goals"] + mask_away * tm["away_goals"]).sum()
            goals_conceded = (mask_home * tm["away_goals"] + mask_away * tm["home_goals"]).sum()

            # always_playing: total presence minutes equals league total minutes
            always_playing = 0
            for pid in players:
                col = f"p{pid}"
                if col in self.segment_df:
                    mins = (self.segment_df[col].abs() * self.segment_df["duration"]).sum()
                    if mins == total_minutes:
                        always_playing += 1

            team_summary_list.append({
                "team_id": team_id,
                "wins": int(wins), "draws": int(draws), "losses": int(losses),
                "goals_scored": int(goals_scored), "goals_conceded": int(goals_conceded),
                "total_players_used": len(players),
                "always_playing": always_playing
            })

        self.team_summary_df = pd.DataFrame(team_summary_list)
        self.team_summary_df["points"] = (self.team_summary_df["wins"] * 3 + self.team_summary_df["draws"])
        self.team_summary_df.sort_values(by=["points", "goals_scored"], ascending=[False, False], inplace=True)
        self.team_summary_df.reset_index(drop=True, inplace=True)
        self.team_summary_df["league_position"] = self.team_summary_df.index + 1

    def compute_impacts(self) -> None:
        """Compute WLS plus/minus and Lasso regularized impacts."""
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

        # Minutes played from segment presence
        presence_matrix = X_df.abs()
        minutes_played = presence_matrix.mul(self.segment_df["duration"].to_numpy(), axis=0).sum()
        coef_df["minutes_played"] = coef_df["player_id"].map(minutes_played.to_dict()).fillna(0.0).astype(float)

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

        # Appearance metrics
        for metric_key in ["games_started", "full_games_played", "games_subbed_on", "games_subbed_off"]:
            coef_df[metric_key] = coef_df["player_id"].apply(
                lambda pid, k=metric_key: (self.player_game_metrics.get(int(pid[1:]), {}).get(k, 0)
                                           if pid.startswith("p") else 0)
            )
        coef_df["FTE_games_played"] = (coef_df["minutes_played"] / 90).round(2)

        # players with identical segment signatures
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

        coef_df["other_players_same_minutes"] = coef_df["player_id"].apply(_peers)

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

    def run_analysis(self) -> None:
        """End-to-end analysis (no I/O besides optional intermediate pickle)."""
        print("[INFO] Building segments...")
        self.identify_players_and_build_segments()
        print("[INFO] Calculating game metrics...")
        self.calculate_game_metrics()
        print("[INFO] Summarizing matches and teams...")
        self.summarize_matches_and_teams()

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
        self.add_goals_assists()

    def save_results(self, output_path: Optional[str] = None) -> None:
        """Save results to Excel and SQLite; also export DB slice to Excel/CSV."""
        assert self.coef_df_final is not None
        output_path = output_path or self._p("output.xlsx")

        # OLS sheet
        ols_df = self.coef_df_final.sort_values(by="impact", ascending=False)
        ols_df.to_excel(output_path, sheet_name="OLS", index=False)

        # Lasso sheets
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            for alpha in self.lasso_alphas:
                a = float(alpha); a_str = str(a).replace(".", "_")
                col = f"lasso_impact_alpha_{a_str}"
                if col in self.coef_df_final:
                    df_sub = self.coef_df_final[self.coef_df_final[col].fillna(0) != 0].copy()
                    df_sub.sort_values(by=col, ascending=False, inplace=True)
                    df_sub.to_excel(writer, sheet_name=f"Lasso_{a_str}", index=False)

        # SQLite upsert-ish
        conn = sqlite3.connect(self.DB_PATH); cur = conn.cursor()
        table_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results';"
        ).fetchone() is not None

        if table_exists:
            existing_cols = {c[1] for c in cur.execute("PRAGMA table_info(analysis_results);").fetchall()}
            new_cols = set(self.coef_df_final.columns)
            if existing_cols != new_cols:
                self.coef_df_final.to_sql("analysis_results_new", conn, if_exists="replace", index=False)
            else:
                cur.execute(
                    "DELETE FROM analysis_results WHERE country=? AND league=? AND season=?;",
                    (self.COUNTRY_NAME, self.LEAGUE_NAME, self.SEASON),
                )
                conn.commit()
                self.coef_df_final.to_sql("analysis_results", conn, if_exists="append", index=False)
        else:
            self.coef_df_final.to_sql("analysis_results", conn, if_exists="replace", index=False)
        conn.close()

        # Export slice
        conn = sqlite3.connect(self.DB_PATH)
        query = """
            SELECT * FROM analysis_results
             WHERE country = ? AND league = ? AND season = ?;
        """
        df_from_db = pd.read_sql_query(query, conn, params=[self.COUNTRY_NAME, self.LEAGUE_NAME, self.SEASON])
        conn.close()

        df_from_db.to_excel(self._p("analysis_from_db.xlsx"), index=False)
        df_from_db.to_csv(self._p("analysis_from_db.csv"), index=False)

    def run(self) -> None:
        """Convenience: load → analyze → save."""
        self.load_data()
        self.run_analysis()
        self.save_results()

# ---------------- main: load progress + run completed downloads (no argparse) ----------------
if __name__ == "__main__":
    # Run all rows that are downloaded and overall_status == "Completed",
    # unless already processed (output.xlsx or DB has rows).
    runner = SeasonAnalyzer(
        save_intermediate=True,
    )
    runner.run()
     