#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convergence tracker for SeasonAnalyzer.

Run impacts at multiple date cutoffs, normalize by the final (full-season) value,
save trajectories, and plot spaghetti lines.

Usage (from Python):
--------------------
from convergence_tracker import run_convergence

df_long, df_wide = run_convergence(
    country="Germany", league="Bundesliga", season=2023,
    checkpoints=["2023-08-31","2023-10-31","2023-12-31","2024-03-31","2024-05-18"],
    base_dir=None, normalize=True, top_n_minutes=40, save_prefix="bundesliga_2023"
)

This writes:
- <SAVE_DIR>/bundesliga_2023_trajectory_long.csv|.xlsx (long)
- <SAVE_DIR>/bundesliga_2023_trajectory_wide.csv|.xlsx (wide)
- <SAVE_DIR>/bundesliga_2023_spaghetti.png
"""

from __future__ import annotations
import os
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from season_analyzer import SeasonAnalyzer  # same folder

def _impact_series(an: SeasonAnalyzer) -> pd.Series:
    """Return per-player impact series indexed by numeric player id (no pooling)."""
    df = an.coef_df_final.copy()
    # numeric id
    ids = df["player_id"].apply(lambda pid: int(pid[1:]) if isinstance(pid, str) and pid.startswith("p") else int(pid))
    s = pd.Series(df["impact"].to_numpy(), index=ids)
    return s

def _minutes_series(an: SeasonAnalyzer) -> pd.Series:
    df = an.coef_df_final.copy()
    ids = df["player_id"].apply(lambda pid: int(pid[1:]) if isinstance(pid, str) and pid.startswith("p") else int(pid))
    s = pd.Series(df["minutes_played"].to_numpy(), index=ids)
    return s

def run_convergence(
    country: str,
    league: str,
    season: int,
    checkpoints: List[str],
    base_dir: Optional[str] = None,
    normalize: bool = True,
    top_n_minutes: Optional[int] = None,
    save_prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_long, df_wide).
      df_long columns: [date_cutoff, player_id, impact, norm_impact]
      df_wide index: date_cutoff; columns: player_id; values: norm_impact or impact
    """
    # Make analyzer and load data once (we reuse object to avoid file I/O churn)
    an = SeasonAnalyzer(
        league_name=league, country_name=country, season=season, base_dir=base_dir
    )
    an.load_data()

    # full-season for normalization baseline
    an.run_analysis(until_date=None, group_pooled=False)
    final_impacts = _impact_series(an)
    final_minutes = _minutes_series(an)

    # Optionally select top-N by minutes for the spaghetti plot and columns
    selected_ids = None
    if top_n_minutes is not None and len(final_minutes) > 0:
        selected_ids = set(final_minutes.sort_values(ascending=False).head(int(top_n_minutes)).index.tolist())

    rows = []
    for dt in checkpoints:
        an.run_analysis(until_date=dt, group_pooled=False)
        cur_imp = _impact_series(an)
        # align to union so that missing ids become 0 (no obs yet)
        all_ids = sorted(set(final_impacts.index).union(cur_imp.index))
        cur_imp = cur_imp.reindex(all_ids).fillna(0.0)
        fin_imp = final_impacts.reindex(all_ids).fillna(0.0)

        # normalize (avoid divide by zero)
        if normalize:
            norm = np.where(fin_imp.to_numpy() != 0, cur_imp.to_numpy() / fin_imp.to_numpy(), np.nan)
            for pid, imp, nimp in zip(all_ids, cur_imp.to_numpy(), norm):
                if selected_ids is None or pid in selected_ids:
                    # rows.append({"date_cutoff": pd.to_datetime(dt), "player_id": int(pid), "impact": float(imp), "norm_impact": float(nimp)})
                    rows.append({"date_cutoff": pd.to_datetime(dt, utc=True), "player_id": int(pid), "impact": float(imp), "norm_impact": float(nimp)})

        else:
            for pid, imp in zip(all_ids, cur_imp.to_numpy()):
                if selected_ids is None or pid in selected_ids:
                    # rows.append({"date_cutoff": pd.to_datetime(dt), "player_id": int(pid), "impact": float(imp), "norm_impact": np.nan})
                    rows.append({"date_cutoff": pd.to_datetime(dt, utc=True), "player_id": int(pid), "impact": float(imp), "norm_impact": np.nan})


    df_long = pd.DataFrame(rows).sort_values(["date_cutoff","player_id"]).reset_index(drop=True)

    # wide (use normalized if available else raw)
    value_col = "norm_impact" if normalize else "impact"
    df_wide = df_long.pivot(index="date_cutoff", columns="player_id", values=value_col).sort_index()
    df_wide = df_wide.astype(float)

    # Save
    save_dir = os.path.join(base_dir or "", f"{country}_{league}", str(season))
    os.makedirs(save_dir, exist_ok=True)
    prefix = save_prefix or f"{league.lower()}_{season}"

    df_long.to_csv(os.path.join(save_dir, f"{prefix}_trajectory_long.csv"), index=False)
    df_wide.to_csv(os.path.join(save_dir, f"{prefix}_trajectory_wide.csv"), index=True)
    df_long.to_excel(os.path.join(save_dir, f"{prefix}_trajectory_long.xlsx"), index=False)
    df_wide.to_excel(os.path.join(save_dir, f"{prefix}_trajectory_wide.xlsx"))

    # Plot spaghetti lines (normalized by default)
    plt.figure(figsize=(10, 6))
    for pid in df_wide.columns:
        plt.plot(df_wide.index, df_wide[pid], label=str(pid))
    if normalize:
        plt.axhline(1.0)
    plt.xlabel("Date cutoff")
    plt.ylabel("Normalized impact" if normalize else "Impact")
    plt.title(f"Convergence of player impacts: {country} {league} {season}")
    # Do not force legend for many lines; keep plot clean
    out_png = os.path.join(save_dir, f"{prefix}_spaghetti.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    # plt.show()  # optional

    return df_long, df_wide

if __name__ == "__main__":
    
    
    from season_analyzer import SeasonAnalyzer
    from DataAnalysis import SeasonBatchRunner

    an = SeasonAnalyzer(country_name="Germany", league_name="Bundesliga", season=2023, save_intermediate=True)
    an.run()  # full season
    # Outputs:
    #   <country>_<league>/<season>/output.xlsx (sheets: OLS, Lasso_*, OLS_POOLED, Lasso_*_POOLED)
    #   analysis_results.db (tables: analysis_results, analysis_results_pooled)

    an = SeasonAnalyzer(country_name="Germany", league_name="Bundesliga", season=2023)
    an.run(until_date="2023-12-31", group_pooled=True)   # exclude all matches after Dec 31

   
    SeasonBatchRunner().run_all()
    # Prints: [OK] Country / League / Season


    # from convergence_tracker import run_convergence
    
    df_long, df_wide = run_convergence(
        country="Germany", league="Bundesliga", season=2023,
        checkpoints=["2023-08-31","2023-10-31","2023-12-31","2024-03-31","2024-05-18"],
        base_dir=None, normalize=True, top_n_minutes=40, save_prefix="bundesliga_2023"
    )
