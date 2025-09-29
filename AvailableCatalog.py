#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 08:36:05 2025

@author: alexander
"""

import time
from typing import Dict, Optional
import logging
import os
import requests
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
# -------------------------- Available Catalog Export ----------------------
class AvailableCatalog:
    """
    Exports available (country, competition, season) to available_leagues.xlsx.

    Strategy:
      - /countries
      - /leagues?country=XX (or by country name) → seasons array → year + current + start/end
    """
    FILEPATH = "available_leagues.xlsx"

    def __init__(self, api_url: str, headers: Dict[str, str], rate_delay: float = 0.8):
        self.api_url = api_url
        self.headers = headers
        self.delay = rate_delay

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.api_url}{endpoint}"
        resp = requests.get(url, headers=self.headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def export_all(self):
        # 1) Countries
        countries = self._get("/countries").get("response", [])
        rows = []
        for c in countries:
            country_name = c.get("name")
            if not country_name:
                continue
            time.sleep(self.delay)

            # 2) Leagues for this country
            leagues = self._get("/leagues", {"country": country_name}).get("response", [])
            for L in leagues:
                league = L.get("league", {})
                comp_name = league.get("name")
                if not comp_name:
                    continue
                seasons = L.get("seasons", []) or []
                for s in seasons:
                    rows.append({
                        "country": country_name,
                        "competition": comp_name,
                        "season": s.get("year"),
                        "season_start": s.get("start"),
                        "season_end": s.get("end"),
                        "is_current": bool(s.get("current")),
                    })

        df = pd.DataFrame(rows, columns=["country", "competition", "season",
                                         "season_start", "season_end", "is_current"])
        df.sort_values(["country", "competition", "season"], inplace=True, ignore_index=True)
        df.to_excel(self.FILEPATH, index=False)
        LOGGER.info("Exported available leagues to %s (rows=%d)", self.FILEPATH, len(df))
if __name__ == "__main__":
    api_key = "427b1bc85aa3a6a81fc63b43df0dbd55"

 
         
    
    # Example: export available leagues (menu XLS)
    catalog = AvailableCatalog(api_url="https://v3.football.api-sports.io",
                               headers={"x-rapidapi-key": api_key,
                                        "x-rapidapi-host": "v3.football.api-sports.io"})
    # Comment/uncomment as you like:
    catalog.export_all()
    print("Exported available_leagues.xlsx")