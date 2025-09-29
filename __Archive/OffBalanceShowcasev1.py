
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-balance-sheet contributions: DB-driven showcase (estimator-aware, with comparison support)

- Loads per-player results for a (country, league, season) triple from analysis_results.db
- Lets you choose OLS ("impact"), or Ridge/Lasso via alpha (or auto-pick first available)
- Optional: load a comparison dataset (e.g., previous season or another league)
- Saves figures under a central folder: save_root/<country>_<league>_<season>/<estimator_tag>/figs/
- show_plots=True will display figures in-IDE (and still save them)

Expected core columns in analysis_results:
  - player_id, player_name, position, minutes_played, FTE_games_played, goals, assists,
    impact (OLS), country, league, season, team(s)
Optional:
  - ridge_impact_alpha_*, lasso_impact_alpha_*, league_position, games_* appearance columns
"""

from __future__ import annotations
import os, sqlite3
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class OffBalanceShowcase:
    
    def __init__(
        self,
        *,
        db_path: str = "analysis_results.db",
        country: str,
        league: str,
        season: int,
        # estimator selection
        estimator: str = "ols",
        ridge_alpha: Optional[float] = None,
        lasso_alpha: Optional[float] = None,
        # optional comparison target
        comp_country: Optional[str] = None,
        comp_league: Optional[str] = None,
        comp_season: Optional[int] = None,
        # IO & viz
        save_root: str = "showcase_outputs",
        show_plots: bool = False,
        topk: int = 20,
        position_map: Optional[Dict[str, Iterable[str]]] = None,
        # ---------- NEW: filters ----------
        drop_unknowns: bool = True,          # drop players named/position "Unknown"
        min_minutes: float = 45.0,           # drop players with minutes_played < this
        unknown_name_tokens: Tuple[str, ...] = ("unknown", "n/a", ""),
        unknown_position_tokens: Tuple[str, ...] = ("unknown", ""),
    ):
        self.db_path = db_path
        self.country = country
        self.league = league
        self.season = int(season)

        # estimator config
        self.estimator = estimator.strip().lower()
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self._imp_col = None  # resolved in load()

        # optional comparison triplet
        self.comp_country = comp_country
        self.comp_league = comp_league
        self.comp_season = int(comp_season) if comp_season is not None else None

        # output
        self.show_plots = bool(show_plots)
        self.topk = int(topk)

        self.position_map = position_map or {
            "defender":  ["df", "def", "cb", "rb", "lb", "rwb", "lwb", "centre-back", "full-back", "defender"],
            "midfielder":["mf", "mid", "cm", "dm", "am", "rm", "lm", "wing-back", "midfielder"],
            "attacker":  ["fw", "st", "lw", "rw", "cf", "striker", "forward", "attacker"],
            "goalkeeper":["gk", "goalkeeper", "keeper"],
        }
        
        # filters
        self.drop_unknowns = bool(drop_unknowns)
        self.min_minutes = float(min_minutes)
        self.unknown_name_tokens = tuple(t.strip().lower() for t in unknown_name_tokens)
        self.unknown_position_tokens = tuple(t.strip().lower() for t in unknown_position_tokens)


        # central folder + estimator tag
        est_tag = self._estimator_tag()
        tag = f"{self.country}_{self.league}_{self.season}"
        self.base_dir = os.path.join(save_root, tag, est_tag)
        self.fig_dir = os.path.join(self.base_dir, "figs")
        os.makedirs(self.fig_dir, exist_ok=True)

        # holders
        self.players_df: pd.DataFrame = pd.DataFrame()
        self.pooled_df: pd.DataFrame = pd.DataFrame()
        self.players_df_comp: pd.DataFrame = pd.DataFrame()  # optional comparison

    # ---------------------- helpers ----------------------


    def _is_unknown_token(self, s: Optional[str], bag: Tuple[str, ...]) -> bool:
        if s is None:
            return True
        txt = str(s).strip().lower()
        return txt in bag

    def _estimator_tag(self) -> str:
        if self.estimator == "ols":
            return "ols"
        if self.estimator == "ridge":
            return f"ridge_{str(float(self.ridge_alpha)).replace('.','_')}" if self.ridge_alpha is not None else "ridge_auto"
        if self.estimator == "lasso":
            return f"lasso_{str(float(self.lasso_alpha)).replace('.','_')}" if self.lasso_alpha is not None else "lasso_auto"
        return "ols"

    @staticmethod
    def _col_for_ridge(df: pd.DataFrame, alpha: Optional[float]) -> str:
        if alpha is None:
            cols = [c for c in df.columns if c.startswith("ridge_impact_alpha_")]
            if not cols:
                raise ValueError("No ridge columns found in DB for this dataset.")
            return cols[0]
        col = f"ridge_impact_alpha_{str(float(alpha)).replace('.','_')}"
        if col not in df.columns:
            raise ValueError(f"{col} not found in DB columns.")
        return col

    @staticmethod
    def _col_for_lasso(df: pd.DataFrame, alpha: Optional[float]) -> str:
        if alpha is None:
            cols = [c for c in df.columns if c.startswith("lasso_impact_alpha_")]
            if not cols:
                raise ValueError("No lasso columns found in DB for this dataset.")
            return cols[0]
        col = f"lasso_impact_alpha_{str(float(alpha)).replace('.','_')}"
        if col not in df.columns:
            raise ValueError(f"{col} not found in DB columns.")
        return col

    def _resolve_imp_col(self, df: pd.DataFrame) -> str:
        if self.estimator == "ols":
            if "impact" not in df.columns:
                raise ValueError("'impact' column (OLS) not found in DB.")
            return "impact"
        if self.estimator == "ridge":
            return self._col_for_ridge(df, self.ridge_alpha)
        if self.estimator == "lasso":
            return self._col_for_lasso(df, self.lasso_alpha)
        # fallback
        return "impact"

    def _pos_bucket(self, pos: str) -> str:
        p = (pos or "").strip().lower()
        for k, keys in self.position_map.items():
            if any(token in p for token in keys):
                return k
        return "other"

    def _finish(self, fig: plt.Figure, name: str) -> str:
        out = os.path.join(self.fig_dir, f"{name}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        return out

    def _need(self, df: pd.DataFrame, msg="Required DataFrame is empty"):
        if df is None or df.empty:
            raise ValueError(msg)

    # ------------------------ IO ------------------------

    def load(self) -> None:
        """Load primary dataset; resolve impact column according to estimator; load pooled if present; optionally load comparison dataset."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"DB not found: {self.db_path}")

        with sqlite3.connect(self.db_path) as conn:
            q = """SELECT * FROM analysis_results WHERE country=? AND league=? AND season=?"""
            self.players_df = pd.read_sql_query(q, conn, params=[self.country, self.league, self.season])
            try:
                q2 = """SELECT * FROM analysis_results_pooled WHERE country=? AND league=? AND season=?"""
                self.pooled_df = pd.read_sql_query(q2, conn, params=[self.country, self.league, self.season])
            except Exception:
                self.pooled_df = pd.DataFrame()

        if self.players_df.empty:
            raise ValueError(f"No rows in analysis_results for ({self.country}, {self.league}, {self.season}).")
        
        
        # ---------- NEW: filtering ----------
        # Ensure minutes is numeric (missing -> 0)
        self.players_df["minutes_played"] = pd.to_numeric(self.players_df.get("minutes_played", 0), errors="coerce").fillna(0.0)
        
        mask = pd.Series(True, index=self.players_df.index)
        
        if self.drop_unknowns:
            # Drop rows where player_name is in unknown tokens
            if "player_name" in self.players_df.columns:
                name_unknown = self.players_df["player_name"].astype(str).str.strip().str.lower().isin(self.unknown_name_tokens)
                mask &= ~name_unknown
            # Drop rows where position is unknown
            if "position" in self.players_df.columns:
                pos_unknown = self.players_df["position"].astype(str).str.strip().str.lower().isin(self.unknown_position_tokens)
                mask &= ~pos_unknown
        
        # Drop rows below minute threshold
        mask &= self.players_df["minutes_played"] >= self.min_minutes
        
        # Apply mask
        before = len(self.players_df)
        self.players_df = self.players_df.loc[mask].copy()
        # Optional: keep an eye on what was filtered (comment out if you want zero console noise)
        # print(f"[filter] kept {len(self.players_df)}/{before} rows after unknown/minutes filters (min_minutes={self.min_minutes})")

        # normalize numeric
        for c in ["minutes_played", "FTE_games_played"]:
            if c in self.players_df:
                self.players_df[c] = pd.to_numeric(self.players_df[c], errors="coerce")

        # numeric player_id when possible
        if "player_id" in self.players_df:
            try: self.players_df["player_id"] = self.players_df["player_id"].astype(int)
            except Exception: pass

        # resolve impact column
        self._imp_col = self._resolve_imp_col(self.players_df)

        # optional comparison dataset
        if self.comp_country and self.comp_league and self.comp_season is not None:
            with sqlite3.connect(self.db_path) as conn:
                qc = """SELECT * FROM analysis_results WHERE country=? AND league=? AND season=?"""
                self.players_df_comp = pd.read_sql_query(qc, conn, params=[self.comp_country, self.comp_league, int(self.comp_season)])
                if not self.players_df_comp.empty:
                    for c in ["minutes_played", "FTE_games_played"]:
                        if c in self.players_df_comp:
                            self.players_df_comp[c] = pd.to_numeric(self.players_df_comp[c], errors="coerce")
                    try: self.players_df_comp["player_id"] = self.players_df_comp["player_id"].astype(int)
                    except Exception: pass
                # impact col for comparison
                if not self.players_df_comp.empty:
                    _ = self._resolve_imp_col(self.players_df_comp)  # validate existence

    # ---------------------- INTRO ----------------------

    def intro(self) -> Dict[str, str]:
        """
        Estimator-aware intro:
          1) Distribution of chosen impact
          2) Minutes vs Impact (size ∝ FTE) with a clean legend
          3) Tornado chart: Top vs Bottom impacts (names centered, no boxes)
          4) Pooled group sizes (explicitly say if none)
        """
        self._need(self.players_df)
        col = self._imp_col
        tag = f"{self.country} – {self.league} {self.season} [{self._estimator_tag()}]"
        out: Dict[str, str] = {}
    
        # ---------- (1) Distribution ----------
        vals = pd.to_numeric(self.players_df[col], errors="coerce").dropna().to_numpy()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.hist(vals, bins=30)
        mu = np.nanmean(vals) if vals.size else 0.0
        sd = np.nanstd(vals, ddof=1) if vals.size > 1 else 0.0
        ax.axvline(mu, linestyle="--")
        ax.text(0.02, 0.95, f"μ={mu:.3g}, σ={sd:.3g}, n={len(vals)}", transform=ax.transAxes,
                ha="left", va="top")
        ax.set_title(f"Distribution of {col} — {tag}")
        ax.set_xlabel(col); ax.set_ylabel("Players")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        out["impact_distribution"] = self._finish(fig, "intro_impact_distribution")
    
        # ---------- (2) Minutes vs Impact (clean legend) ----------
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        df = self.players_df.copy()
        df["pos_bucket"] = df.get("position", "").map(self._pos_bucket)
    
        shown_labels = set()
        any_non_other = False
        for bucket in ["attacker", "midfielder", "defender", "goalkeeper", "other"]:
            sub = df[df["pos_bucket"] == bucket]
            if sub.empty:
                continue
            if bucket != "other":
                any_non_other = True
            fte = pd.to_numeric(sub.get("FTE_games_played", 0), errors="coerce").fillna(0).to_numpy()
            sizes = np.clip(fte * 5, 10, 80)
            label = bucket if bucket not in shown_labels else None
            ax.scatter(
                pd.to_numeric(sub["minutes_played"], errors="coerce"),
                pd.to_numeric(sub[col], errors="coerce"),
                s=sizes, alpha=0.75, label=label,
            )
            shown_labels.add(bucket)
    
        ax.set_title(f"{col} vs Minutes — {tag}")
        ax.set_xlabel("Minutes played"); ax.set_ylabel(col)
        ax.grid(True, alpha=0.25, linewidth=0.6)
    
        # Only show legend if there is more than just "other"
        if any_non_other:
            ax.legend(frameon=False, ncol=3, title="Position bucket")
        else:
            ax.text(0.98, 0.02, "All classified as 'other' (source positions unavailable)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    
        out["impact_vs_minutes"] = self._finish(fig, "intro_impact_vs_minutes")
    
        # ---------- (3) Tornado: Top vs Bottom impacts (names centered) ----------
        # Filter out Unknown names for readability
        clean = self.players_df.copy()
        clean["player_name"] = clean["player_name"].fillna("Unknown")
        clean = clean[clean["player_name"].str.strip().str.lower() != "unknown"].copy()
    
        # Build Top(+K) and Bottom(−K)
        top = (clean.nlargest(self.topk, col)[["player_name", col]]
                     .sort_values(col, ascending=False))
        bot = (clean.nsmallest(self.topk, col)[["player_name", col]]
                     .sort_values(col, ascending=False))  # still store descending for nice order
    
        # Align by the max length list; names in the middle (use top list order)
        names = list(top["player_name"])
        pos_vals = list(top[col].astype(float))
        # For bottom, we take abs values (magnitudes) so bars grow left; sort to match the same number of rows
        bot_vals = list(bot[col].astype(float))
        # if different lengths, pad the shorter with zeros and placeholder names
        L = max(len(names), len(bot_vals))
        names += [""] * (L - len(names))
        pos_vals += [np.nan] * (L - len(pos_vals))
        bot_vals += [np.nan] * (L - len(bot_vals))
    
        y = np.arange(L)
    
        fig, ax = plt.subplots(figsize=(10, 8))
        # Draw right (Top positives)
        ax.barh(y, [v if (not pd.isna(v) and v > 0) else 0 for v in pos_vals], align="center")
        # Draw left (Bottom negatives), mirrored
        ax.barh(y, [(-v) if (not pd.isna(v) and v < 0) else 0 for v in bot_vals], align="center")
    
        # Center names as y tick labels
        ax.set_yticks(y, names)
        # Remove boxes (spines)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.axvline(0, linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    
        ax.set_title(f"Player {col} — {tag}\nTop (right) vs Bottom (left); ordered from top down")
        ax.set_xlabel(col)
        out["top_bottom_impacts"] = self._finish(fig, "intro_top_bottom_tornado")
    
        # ---------- (4) Pooled groups (explicit message if none) ----------
        if self.pooled_df is not None and not self.pooled_df.empty:
            # compute group sizes from composite player_id like "123+456+..."
            sizes = []
            for pid in self.pooled_df["player_id"].astype(str):
                sizes.append(1 if "+" not in pid else pid.count("+") + 1)
    
            if max(sizes) <= 1:
                # all unique (no pooling)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.axis("off")
                ax.text(0.5, 0.55, "No pooled groups detected",
                        ha="center", va="center", fontsize=14)
                ax.text(0.5, 0.42, "Every player has a unique on-field timeline.",
                        ha="center", va="center", fontsize=11)
                ax.set_title(f"Pooled Groups — {tag}")
                out["pooled_group_sizes"] = self._finish(fig, "intro_pooled_groups_none")
            else:
                sizes_np = np.array(sizes, dtype=int)
                pooled_mask = sizes_np > 1
                n_groups = int(pooled_mask.sum())
                n_players_in_pooled = int(sizes_np[pooled_mask].sum())
    
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(sizes_np[pooled_mask], bins=np.arange(1.5, sizes_np.max()+1.5, 1))
                ax.set_title(f"Pooled Groups — {tag}")
                ax.set_xlabel("Group size (>1 only)"); ax.set_ylabel("Count")
                ax.grid(True, alpha=0.25, linewidth=0.6)
                ax.text(0.98, 0.95, f"groups>1 = {n_groups}\nplayers in pooled = {n_players_in_pooled}",
                        transform=ax.transAxes, ha="right", va="top")
                out["pooled_group_sizes"] = self._finish(fig, "intro_pooled_group_sizes")
        else:
            # no pooled table
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axis("off")
            ax.set_title(f"Pooled Groups — {tag}")
            ax.text(0.5, 0.5, "No pooled table available in DB.", ha="center", va="center", fontsize=12)
            out["pooled_group_sizes"] = self._finish(fig, "intro_pooled_groups_absent")
    
        return out 

    # -------------- Showcase modules (estimator-aware) --------------

    def mod_goals_vs_impact(self) -> str:
        """Chosen impact vs goal involvement (goals+assists)."""
        self._need(self.players_df)
        col = self._imp_col
        df = self.players_df.copy()
        df["gi"] = pd.to_numeric(df.get("goals",0), errors="coerce").fillna(0) + \
                   pd.to_numeric(df.get("assists",0), errors="coerce").fillna(0)
        fig, ax = plt.subplots(figsize=(7.5,4.5))
        ax.scatter(df["gi"], df[col], alpha=0.7)
        ax.set_title(f"{col} vs Goal Involvements"); ax.set_xlabel("Goals+Assists"); ax.set_ylabel(col)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        return self._finish(fig, "mod_goals_vs_impact")

    def mod_position_buckets(self) -> str:
        """Boxplot of chosen impact by position buckets."""
        self._need(self.players_df)
        col = self._imp_col
        df = self.players_df.copy()
        df["pos_bucket"] = df.get("position","").map(self._pos_bucket)
        groups = [pd.to_numeric(df[df["pos_bucket"]==k][col], errors="coerce").dropna().to_numpy()
                  for k in ["goalkeeper","defender","midfielder","attacker","other"]]
        labels = ["GK","DEF","MID","ATT","OTH"]
        fig, ax = plt.subplots(figsize=(7,4))
        ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_title(f"{col} by Position Bucket"); ax.set_ylabel(col)
        ax.grid(True, axis="y", alpha=0.25)
        return self._finish(fig, "mod_position_buckets")

    def mod_minutes_efficiency_frontier(self) -> str:
        """Minutes (x) vs (impact × minutes) using chosen impact."""
        self._need(self.players_df)
        col = self._imp_col
        df = self.players_df.copy()
        df["cum_contrib"] = pd.to_numeric(df[col], errors="coerce").fillna(0) * \
                            pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0)
        fig, ax = plt.subplots(figsize=(7.5,4.5))
        ax.scatter(df["minutes_played"], df["cum_contrib"], alpha=0.7)
        ax.set_title("Minutes-Efficiency Frontier"); ax.set_xlabel("Minutes played"); ax.set_ylabel(f"{col} × Minutes")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        return self._finish(fig, "mod_minutes_efficiency_frontier")

    def mod_team_averages(self) -> str:
        """Average chosen impact by team."""
        self._need(self.players_df)
        col = self._imp_col
        if "team(s)" not in self.players_df.columns:
            raise ValueError("Column 'team(s)' not found in DB.")
        g = (self.players_df.groupby("team(s)", dropna=True)[col]
                          .mean().sort_values(ascending=False).head(20))
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(g.index[::-1], g.values[::-1])
        ax.set_title(f"Average {col} by Team (Top 20)"); ax.set_xlabel(col)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
        return self._finish(fig, "mod_team_averages")

    def mod_under_the_radar(self, topn: int = 20) -> str:
        """Players with high impact but low goals+assists (off-balance signal)."""
        self._need(self.players_df)
        col = self._imp_col
        df = self.players_df.copy()
        df["gi"] = pd.to_numeric(df.get("goals",0), errors="coerce").fillna(0) + \
                   pd.to_numeric(df.get("assists",0), errors="coerce").fillna(0)
        df["r_imp"] = pd.to_numeric(df[col], errors="coerce").rank(ascending=False, method="min")
        df["r_gi"] = df["gi"].rank(ascending=True, method="min")
        df["off_balance_score"] = df["r_gi"] - df["r_imp"]
        top = df.nlargest(topn, "off_balance_score")[["player_name", col, "gi", "off_balance_score"]]
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(top["player_name"][::-1], top["off_balance_score"][::-1])
        ax.set_title("Under-the-Radar (High Impact, Low Goals/Assists)"); ax.set_xlabel("Score ↑")
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
        return self._finish(fig, "mod_under_the_radar")

    def mod_stability_ridge_vs_ols(self, ridge_alpha: Optional[float] = None) -> str:
        """Scatter: Ridge(alpha) vs OLS impact for stability illustration."""
        self._need(self.players_df)
        if "impact" not in self.players_df:
            raise ValueError("OLS 'impact' column not found.")
        col_r = self._col_for_ridge(self.players_df, ridge_alpha)
        fig, ax = plt.subplots(figsize=(7.5,4.5))
        ax.scatter(pd.to_numeric(self.players_df["impact"], errors="coerce"),
                   pd.to_numeric(self.players_df[col_r], errors="coerce"),
                   alpha=0.65)
        ax.set_title("Ridge vs OLS"); ax.set_xlabel("OLS Impact"); ax.set_ylabel(f"Ridge {col_r.split('alpha_')[-1].replace('_','.')}")
        ax.grid(True, alpha=0.25, linewidth=0.6); ax.axhline(0,lw=0.7); ax.axvline(0,lw=0.7)
        return self._finish(fig, "mod_stability_ridge_vs_ols")

    def mod_minutes_concentration_curve(self) -> str:
        """Lorenz-style curve: cumulative minutes vs cumulative (impact × minutes)."""
        self._need(self.players_df)
        col = self._imp_col
        df = self.players_df.copy()
        w = pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0).to_numpy()
        y = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        idx = np.argsort(-w)
        w_sorted, y_sorted = w[idx], y[idx]
        if w_sorted.sum() <= 0:
            raise ValueError("No minutes in dataset.")
        w_cum = np.cumsum(w_sorted) / w_sorted.sum()
        y_contrib = y_sorted * w_sorted
        denom = y_contrib.sum() if y_contrib.sum() != 0 else 1.0
        y_cum = np.cumsum(y_contrib) / denom
        fig, ax = plt.subplots(figsize=(7.5,4.5))
        ax.plot(w_cum, y_cum); ax.plot([0,1],[0,1], linestyle="--", linewidth=0.8)
        ax.set_title(f"Minutes vs {self._imp_col}×Minutes (Concentration)")
        ax.set_xlabel("Cumulative minutes share"); ax.set_ylabel(f"Cumulative ({self._imp_col}×minutes) share")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        return self._finish(fig, "mod_minutes_concentration_curve")

    def mod_fte_distribution(self) -> str:
        """Histogram of FTE games played (minutes/90)."""
        self._need(self.players_df)
        fte = pd.to_numeric(self.players_df.get("FTE_games_played", 0), errors="coerce").dropna().to_numpy()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.hist(fte, bins=30)
        ax.set_title("FTE Games Played (Distribution)"); ax.set_xlabel("FTE games"); ax.set_ylabel("Players")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        return self._finish(fig, "mod_fte_distribution")

    def mod_apps_breakdown(self) -> str:
        """Total appearances breakdown (starts/full/sub on/off) — uses whichever columns exist."""
        self._need(self.players_df)
        cols = [c for c in ["games_started","full_games_played","games_subbed_on","games_subbed_off"]
                if c in self.players_df.columns]
        if not cols:
            raise ValueError("No appearance columns found in DB.")
        sums = self.players_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7.5,4.5))
        ax.bar(sums.index, sums.values)
        ax.set_title("Appearances Breakdown (Total)"); ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        return self._finish(fig, "mod_apps_breakdown")

    # --------- Comparison-aware modules (need comp_* loaded) ---------

    def _need_comp(self):
        if self.players_df_comp is None or self.players_df_comp.empty:
            raise ValueError("Comparison dataset not loaded. Provide comp_country/comp_league/comp_season and call load().")

    def mod_season_shift(self, topn: int = 20) -> str:
        """
        Season-over-season change: merge on player_name (or player_id if consistent) and show largest deltas in chosen impact.
        Useful for 'previous season vs current'.
        """
        self._need(self.players_df); self._need_comp()
        col = self._imp_col
        left = self.players_df[["player_name","player_id",col]].copy().rename(columns={col:"imp_now"})
        right = self.players_df_comp[["player_name","player_id"] + [self._resolve_imp_col(self.players_df_comp)]].copy()
        right = right.rename(columns={self._resolve_imp_col(self.players_df_comp): "imp_prev"})

        # merge by player_name first (safer across leagues), fallback to id where available
        merged = pd.merge(left, right, on="player_name", how="inner", suffixes=("","_y"))
        if merged.empty and "player_id" in left and "player_id" in right:
            merged = pd.merge(left, right, on="player_id", how="inner", suffixes=("","_y"))
        if merged.empty:
            raise ValueError("Could not align players between seasons/leagues (name and id merges both empty).")

        merged["delta"] = pd.to_numeric(merged["imp_now"], errors="coerce") - pd.to_numeric(merged["imp_prev"], errors="coerce")
        top = merged.nlargest(topn, "delta")[["player_name","imp_prev","imp_now","delta"]]
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(top["player_name"][::-1], top["delta"][::-1])
        ax.set_title(f"Change in {col}: {self.comp_league or self.league} {self.comp_season} → {self.league} {self.season}")
        ax.set_xlabel(f"Δ {col} (now - prev)"); ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
        return self._finish(fig, "mod_season_shift")

    def mod_cross_league_positions(self, which_positions: Optional[List[str]] = None, normalize_by_league: bool = True) -> str:
        """
        Compare chosen impact by broad positions across current vs comparison league/season.
        """
        self._need(self.players_df); self._need_comp()
        col = self._imp_col
        cur = self.players_df.copy(); cur["pos_bucket"] = cur.get("position","").map(self._pos_bucket)
        prv = self.players_df_comp.copy(); prv["pos_bucket"] = prv.get("position","").map(self._pos_bucket)
        if which_positions:
            cur = cur[cur["pos_bucket"].isin(which_positions)]
            prv = prv[prv["pos_bucket"].isin(which_positions)]
        g1 = cur.groupby("pos_bucket")[col].mean()
        g2 = prv.groupby("pos_bucket")[self._resolve_imp_col(prv)].mean()

        if normalize_by_league:
            # center by league mean for comparability
            g1 = g1 - g1.mean()
            g2 = g2 - g2.mean()

        cats = sorted(set(g1.index).union(g2.index))
        v1 = [g1.get(k, np.nan) for k in cats]
        v2 = [g2.get(k, np.nan) for k in cats]

        x = np.arange(len(cats))
        fig, ax = plt.subplots(figsize=(9,5))
        ax.bar(x - 0.2, v2, width=0.4, label=f"{self.comp_league or self.league} {self.comp_season}")
        ax.bar(x + 0.2, v1, width=0.4, label=f"{self.league} {self.season}")
        ax.set_xticks(x, cats); ax.set_title(f"{col} by Position — Cross-League/Season")
        ax.set_ylabel(col); ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        return self._finish(fig, "mod_cross_league_positions")


# ----------------------------- Run module -----------------------------
if __name__ == "__main__":
    # from season_analyzer import SeasonAnalyzer
    # from off_balance_showcase import OffBalanceShowcase 
    country_name="Germany"
    league_name="Bundesliga"
    season=2023 
    
    
    # # Example 1: OLS, single dataset, show plots in IDE
    # show = OffBalanceShowcase(
    #     db_path="analysis_results.db",
    #     country=country_name, league=league_name, season=season,        #estimator="ols",
    #     estimator="ridge", ridge_alpha=1.0,
    #     save_root="showcase_outputs",
    #     show_plots=True,   # show figures in IDE
    #     topk=15,
    # )
    # show.load()
    # show.intro()
    # show.mod_goals_vs_impact()
    
    # Example 2: Ridge(α=1.0), compare to previous season, silent save
    show = OffBalanceShowcase(
        db_path="analysis_results.db",
        country=country_name, league=league_name, season=season,
        estimator="ridge", ridge_alpha=1.0,
        comp_country=country_name, comp_league=league_name, comp_season=2022,#comp_country="England", comp_league="Premier League", comp_season=2022,
        save_root="showcase_outputs",
        show_plots=True,
    )
    show.load()
    show.intro()
    show.mod_season_shift(topn=25)
    show.mod_cross_league_positions(which_positions=["defender","midfielder","attacker"])

    
    show.load()            # <— checks DB and brings data into memory
    paths = show.intro()   # saves to showcase_outputs/Netherlands_Eredivisie_2023/figs/
    print(paths)
    
    # any of the modules:
    show.mod_goals_vs_impact()
    show.mod_position_buckets()
    show.mod_minutes_efficiency_frontier()
    show.mod_team_averages()
    show.mod_under_the_radar(topn=25)
    show.mod_stability_ridge_vs_ols()           # uses first available ridge column
    show.mod_position_heatmap_like()
    show.mod_minutes_concentration_curve()
    show.mod_fte_distribution()
    show.mod_apps_breakdown()
