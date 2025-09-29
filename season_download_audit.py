#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 10:21:53 2025

@author: alexander
"""

# season_download_audit.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class SeasonAuditRow:
    country: str
    league: str
    season: int
    status: str  # 'not_started' | 'incomplete' | 'complete'
    details: str
    teams_done: int
    teams_total: int
    players_done: int
    players_total: int
    fixtures_done: int
    fixtures_total: int
    folder: str


class SeasonDownloadAuditor:
    """
    Classifies each Country_League/<season>/ as:
      - not_started: season folder exists and is completely empty
      - incomplete: has files but any *_status.json not finished (or missing)
      - complete: all *_status.json indicate done; fixtures also have lineups/events

    Results are returned/saved sorted as:
      complete → incomplete → not_started, then by country, league, season.
    """

    FILES = {
        "league": "league.json",
        "teams": "teams.json",
        "teams_status": "teams_status.json",
        "players": "players.json",
        "players_status": "players_status.json",
        "fixtures": "fixtures.json",
        "match_events": "match_events.json",
        "fixtures_status": "fixtures_status.json",
    }

    # sorting order: complete(0) → incomplete(1) → not_started(2)
    STATUS_ORDER = {"complete": 0, "incomplete": 1, "not_started": 2}

    def __init__(self, root_dir: str | Path = "."):
        self.root = Path(root_dir).resolve()

    # -------- Public API --------

    def scan_all(self) -> List[SeasonAuditRow]:
        """Scan all folders and return rows already sorted (complete → incomplete → not_started)."""
        rows: List[SeasonAuditRow] = []
        for country_league_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            season_dirs = [d for d in country_league_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if not season_dirs:
                continue
            country, league = self._split_country_league(country_league_dir.name)
            for season_dir in sorted(season_dirs, key=lambda p: int(p.name)):
                row = self._assess_season_dir(country, league, season_dir)
                rows.append(row)
        return self._sorted_rows(rows)

    def status_for(self, country: str, league: str, season: int) -> SeasonAuditRow:
        cl_dir = self.root / f"{country}_{league}"
        s_dir = cl_dir / str(season)
        if not s_dir.exists():
            return SeasonAuditRow(country, league, int(season), "not_started", "season folder missing",
                                  0, 0, 0, 0, 0, 0, str(s_dir))
        return self._assess_season_dir(country, league, s_dir)

    def to_excel(self, rows: List[SeasonAuditRow], out_path: str | Path = "season_audit.xlsx") -> Path:
        """Writes a sorted audit (complete → incomplete → not_started) to Excel."""
        rows = self._sorted_rows(rows)
        df = pd.DataFrame([asdict(r) for r in rows])
        out = Path(out_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        engine = None
        for candidate in ("openpyxl", "xlsxwriter"):
            try:
                __import__(candidate)
                engine = candidate
                break
            except Exception:
                pass
        if engine is None:
            raise RuntimeError("Install 'openpyxl' or 'xlsxwriter' to write .xlsx")

        with pd.ExcelWriter(out, engine=engine) as writer:
            df.to_excel(writer, index=False, sheet_name="audit")
        return out

    # -------- Internals --------

    def _sorted_rows(self, rows: List[SeasonAuditRow]) -> List[SeasonAuditRow]:
        """Sort by status (complete → incomplete → not_started), then country, league, season."""
        return sorted(
            rows,
            key=lambda r: (
                self.STATUS_ORDER.get(r.status, 99),
                r.country,
                r.league,
                r.season,
            ),
        )

    def _assess_season_dir(self, country: str, league: str, season_dir: Path) -> SeasonAuditRow:
        season = int(season_dir.name)

        # a) not_started = directory is completely empty
        try:
            is_empty = next(season_dir.iterdir(), None) is None
        except PermissionError:
            is_empty = False  # unreadable -> treat as not empty
        if is_empty:
            return SeasonAuditRow(country, league, season, "not_started", "empty folder",
                                  0, 0, 0, 0, 0, 0, str(season_dir))

        # Load JSONs (silently default to {})
        load = lambda name: self._load_json(season_dir / self.FILES[name])
        teams_data      = load("teams")
        teams_status    = load("teams_status")
        players_status  = load("players_status")
        fixtures_data   = load("fixtures")
        fixtures_status = load("fixtures_status")
        match_events    = load("match_events")

        team_names  = self._extract_team_names(teams_data)
        fixture_ids = self._extract_fixture_ids(fixtures_data)

        teams_done, teams_total, teams_ok = self._eval_name_status(team_names, teams_status)
        players_done, players_total, players_ok = self._eval_name_status(team_names, players_status)
        fixtures_done, fixtures_total, fixtures_ok = self._eval_fixtures(fixture_ids, fixtures_status, match_events)

        missing_status = self._missing_status_files(season_dir)
        bits: List[str] = []
        if missing_status:
            bits.append(f"missing: {', '.join(missing_status)}")
        if not teams_ok:
            bits.append(f"teams_status.json ({teams_done}/{teams_total})")
        if not players_ok:
            bits.append(f"players_status.json ({players_done}/{players_total})")
        if not fixtures_ok:
            bits.append(f"fixtures_status.json ({fixtures_done}/{fixtures_total})")

        if missing_status or (not teams_ok) or (not players_ok) or (not fixtures_ok):
            status = "incomplete"
            details = "; ".join(bits) if bits else "incomplete"
        else:
            status = "complete"
            details = "ok"

        return SeasonAuditRow(
            country=country, league=league, season=season, status=status, details=details,
            teams_done=teams_done, teams_total=teams_total,
            players_done=players_done, players_total=players_total,
            fixtures_done=fixtures_done, fixtures_total=fixtures_total,
            folder=str(season_dir),
        )

    @staticmethod
    def _split_country_league(dirname: str) -> Tuple[str, str]:
        # Only the first underscore splits country and league (league may contain spaces)
        if "_" in dirname:
            a, b = dirname.split("_", 1)
            return a, b
        return dirname, ""

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    @staticmethod
    def _extract_team_names(teams_data: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for item in teams_data.get("response", []) or []:
            team = (item or {}).get("team") or {}
            name = team.get("name")
            if name:
                out.append(str(name))
        return out

    @staticmethod
    def _extract_fixture_ids(fixtures_data: Dict[str, Any]) -> List[int]:
        ids: List[int] = []
        for item in fixtures_data.get("response", []) or []:
            fid = ((item or {}).get("fixture") or {}).get("id")
            if isinstance(fid, int):
                ids.append(fid)
        return ids

    @staticmethod
    def _eval_name_status(names: List[str], status_map: Dict[str, Any]) -> Tuple[int, int, bool]:
        total = len(names)
        if total == 0:
            return 0, 0, False
        done = sum(1 for n in names if status_map.get(n) is True)
        return done, total, (done == total)

    @staticmethod
    def _eval_fixtures(
        fixture_ids: List[int],
        fixtures_status: Dict[str, Any],
        match_events: Dict[str, Any],
    ) -> Tuple[int, int, bool]:
        total = len(fixture_ids)
        if total == 0:
            return 0, 0, True  # nothing to fetch -> OK

        def has_events(fid: int) -> bool:
            key = str(fid)
            entry = match_events.get(key) or {}
            return ("lineups" in entry) and ("events" in entry)

        done = 0
        for fid in fixture_ids:
            if fixtures_status.get(str(fid)) is True and has_events(fid):
                done += 1
        return done, total, (done == total)

    def _missing_status_files(self, season_dir: Path) -> List[str]:
        needed = [
            self.FILES["teams_status"],
            self.FILES["players_status"],
            self.FILES["fixtures_status"],
        ]
        return [nm for nm in needed if not (season_dir / nm).exists()]

    
    
# =======================================================================
#                              __main__ demos
# =======================================================================
if __name__ == "__main__":
    
    # from season_download_audit import SeasonDownloadAuditor

    auditor = SeasonDownloadAuditor(root_dir=".")
    rows = auditor.scan_all()
    xlsx_path = auditor.to_excel(rows, out_path="season_audit.xlsx")
    print(f"Wrote: {xlsx_path}")
    
    # Look up a specific season quickly:
    row = auditor.status_for("Germany", "Bundesliga", 2022)
    row

