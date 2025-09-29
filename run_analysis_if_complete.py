#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 10:23:32 2025

@author: alexander
"""
 
# run_analysis_if_complete.py
from __future__ import annotations

from typing import List, Tuple, Type#Iterable, 

from season_download_audit import SeasonDownloadAuditor, SeasonAuditRow

# Lazy import so importing this file in Spyder never fails
def _default_analyzer_cls():
    from season_analyzer import SeasonAnalyzer  # adjust module path if needed
    return SeasonAnalyzer


def run_analysis_if_complete(
    country_name: str,
    league_name: str,
    season: int,
    *,
    lasso_alphas: Tuple[float, float] = (0.01, 0.001),
    ridge_alphas: Tuple[float, float] = (1.0, 10.0),
    root_dir: str = ".",
    analyzer_cls: Type | None = None,
) -> bool:
    """Run SeasonAnalyzer for a single (country, league, season) only if status is 'complete'."""
    auditor = SeasonDownloadAuditor(root_dir=root_dir)
    row = auditor.status_for(country_name, league_name, season)

    if row.status != "complete":
        print(f"[skip] {country_name}/{league_name}/{season} -> {row.status}: {row.details}")
        print(f"       folder: {row.folder}")
        return False

    if analyzer_cls is None:
        analyzer_cls = _default_analyzer_cls()

    print(
        f"[run] {country_name}/{league_name}/{season} is complete. "
        f"teams {row.teams_done}/{row.teams_total}, "
        f"players {row.players_done}/{row.players_total}, "
        f"fixtures {row.fixtures_done}/{row.fixtures_total}"
    )
    runner = analyzer_cls(
        country_name=country_name,
        league_name=league_name,
        season=int(season),
        lasso_alphas=lasso_alphas,
        ridge_alphas=ridge_alphas,
    )
    runner.run()
    return True


def list_completed_seasons(
    *,
    root_dir: str = ".",
    country_filter: str | None = None,
    league_filter: str | None = None,
) -> List[SeasonAuditRow]:
    """
    Return all 'complete' rows (already sorted by the auditor: complete → incomplete → not_started).
    Optional filters for country and league.
    """
    auditor = SeasonDownloadAuditor(root_dir=root_dir)
    rows = auditor.scan_all()
    completed = [r for r in rows if r.status == "complete"]
    if country_filter:
        completed = [r for r in completed if r.country == country_filter]
    if league_filter:
        completed = [r for r in completed if r.league == league_filter]
    return completed


def run_all_completed(
    *,
    root_dir: str = ".",
    country_filter: str | None = None,
    league_filter: str | None = None,
    lasso_alphas: Tuple[float, float] = (0.01, 0.001),
    ridge_alphas: Tuple[float, float] = (1.0, 10.0),
    analyzer_cls: Type | None = None,
    max_runs: int | None = None,
) -> List[Tuple[str, str, int, bool]]:
    """
    Iterate over all completed seasons (optionally filtered) and run SeasonAnalyzer for each.
    Returns a list of (country, league, season, ran) tuples.
    """
    completed = list_completed_seasons(root_dir=root_dir, country_filter=country_filter, league_filter=league_filter)
    if analyzer_cls is None:
        analyzer_cls = _default_analyzer_cls()

    results: List[Tuple[str, str, int, bool]] = []
    for i, r in enumerate(completed, start=1):
        if (max_runs is not None) and (i > max_runs):
            break
        print(f"[{i}/{len(completed)}] Evaluating {r.country}/{r.league}/{r.season} …")
        runner = analyzer_cls(
            country_name=r.country,
            league_name=r.league,
            season=int(r.season),
            lasso_alphas=lasso_alphas,
            ridge_alphas=ridge_alphas,
        )
        try:
            runner.run()
            results.append((r.country, r.league, r.season, True))
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append((r.country, r.league, r.season, False))
    return results


# =======================================================================
#                              __main__ demos
# =======================================================================
if __name__ == "__main__":

    # from run_analysis_if_complete import run_analysis_if_complete, list_completed_seasons, run_all_completed
    
    # 1) Single season (only if complete)
    # run_analysis_if_complete(
    #     "Germany", "Bundesliga", 2022,
    #     lasso_alphas=(0.001),  # 0.01,
    #     ridge_alphas=(10.0),   # 1.0,
    #     root_dir=".",
    # )
    
    # 2) See all completed seasons (sorted, complete first)
    completed = list_completed_seasons(root_dir=".")
    completed[:5]
    
    # 3) Run all completed seasons, optionally filter, cap runs
    results = run_all_completed(
        root_dir=".",
        country_filter=None,        # e.g., "Germany"
        league_filter=None,         # e.g., "Bundesliga"
        lasso_alphas=(0.001),  # 0.01,
        ridge_alphas=(10.0),   # 1.0,
        max_runs=None               # e.g., 3 to test
    )
    results
