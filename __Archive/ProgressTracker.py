#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 08:40:44 2025

@author: alexander
"""

import os
# import sys
# import time
# import json
import logging
# import traceback
# from dataclasses import dataclass, field
from typing import  Optional#, Tuple, Dict, List

# import requests
import pandas as pd



# ----------------------------- Logging setup -----------------------------
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "football_fetcher.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("football")

# from GracefulExit import GracefulExit, GracefulExitManager
 
# ----------------------------- Progress Tracker ---------------------------
class ProgressTracker:
    """
    Maintains downloads_progress.xlsx with rows keyed by (country, league, season).
    Status logic:
        - Completed: all modules present & complete
        - Partial: at least one module present but not all complete
        - Pending: defined in queue but not started (or nothing present)
    """
    FILEPATH = "downloads_progress.xlsx"
    COLUMNS = [
        "country", "league", "season",
        "teams_done", "players_done", "fixtures_done", "events_done",
        "overall_status", "last_reason", "last_stage", "requests_remaining",
        "updated_at"
    ]

    def __init__(self):
        # Lazy-load / create on first use
        if not os.path.exists(self.FILEPATH):
            df = pd.DataFrame(columns=self.COLUMNS)
            df.to_excel(self.FILEPATH, index=False)

    def _load(self) -> pd.DataFrame:
        return pd.read_excel(self.FILEPATH)

    def _save(self, df: pd.DataFrame):
        df = df[self.COLUMNS]
        df.to_excel(self.FILEPATH, index=False)

    @staticmethod
    def _compute_overall(teams: bool, players: bool, fixtures: bool, events: bool) -> str:
        vals = [teams, players, fixtures, events]
        if all(vals):
            return "Completed"
        if any(vals):
            return "Partial"
        return "Pending"

    def update_row(self,
                   country: str, league: str, season: int,
                   teams_done: bool, players_done: bool, fixtures_done: bool, events_done: bool,
                   last_reason: Optional[str] = None,
                   last_stage: Optional[str] = None,
                   requests_remaining: Optional[int] = None):
        df = self._load()
        mask = (df["country"] == country) & (df["league"] == league) & (df["season"] == season)
        overall = self._compute_overall(teams_done, players_done, fixtures_done, events_done)
        now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        row = {
            "country": country, "league": league, "season": season,
            "teams_done": bool(teams_done),
            "players_done": bool(players_done),
            "fixtures_done": bool(fixtures_done),
            "events_done": bool(events_done),
            "overall_status": overall,
            "last_reason": last_reason or "",
            "last_stage": last_stage or "",
            "requests_remaining": requests_remaining if requests_remaining is not None else "",
            "updated_at": now
        }

        if mask.any():
            df.loc[mask, :] = pd.DataFrame([row]).values
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        self._save(df)

    def ensure_pending(self, country: str, league: str, season: int):
        """Insert a Pending row if not present yet (marks 'next in line')."""
        df = self._load()
        mask = (df["country"] == country) & (df["league"] == league) & (df["season"] == season)
        if not mask.any():
            self.update_row(country, league, season, False, False, False, False)