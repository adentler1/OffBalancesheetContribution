#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeasonAnalyzer — player-level impacts with robust missing-game handling and DB→Excel/CSV export
"""
import os
# --- unify threading & OpenMP runtime to avoid double-free ---
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")     # <-- key when libgomp is present
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")       # stabilize forking in kernels
# Optional: temporary diagnostic escape hatch (remove once stable)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
 

# import sys
# import traceback
# import time
# import pandas as pd

# import json, sqlite3
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from sklearn.linear_model import Lasso

# After all scientific imports:
from threadpoolctl import threadpool_limits, threadpool_info
# Force single-thread compute in all linked BLAS/OpenMP backends
threadpool_limits(1)
try:
    import mkl
    mkl.set_num_threads(1)  # belt-and-suspenders for MKL
except Exception:
    pass
try:
    print("[Threadpools]", ", ".join(f"{b.get('internal_api','?')}:{b.get('num_threads','?')}" for b in threadpool_info()))
except Exception:
    pass

try:
    backends = threadpool_info()
    print("[Threadpools]", ", ".join(f"{b.get('internal_api','?')}:{b.get('num_threads','?')}" for b in backends))
except Exception:
    pass
 
 
"""
"""
# ---------------- SeasonBatchRunner (no argparse, filter straight from XLSX) ----------------

class SeasonBatchRunner:
    """
    Reads `downloads_progress.xlsx`, selects rows with downloaded==True and overall_status=='Completed',
    and runs SeasonAnalyzer per row in a subprocess. Skips seasons already processed.

    Produces: season_batch_report.csv and season_batch_report.xlsx next to the progress file.
    """

    def __init__(
        self,
        progress_filename: str = "downloads_progress.xlsx",
        minutes_equiv_tolerance: int = 0,
        lasso_alphas=(0.01, 0.001),
        debug: bool = False,
    ):
        import os, pandas as pd

        self.debug = debug
        self.lasso_alphas = list(lasso_alphas)
        self.minutes_equiv_tolerance = int(minutes_equiv_tolerance)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.script_path = os.path.abspath(__file__)
        self.progress_path = os.path.join(self.script_dir, progress_filename)

        if not os.path.exists(self.progress_path):
            raise FileNotFoundError(f"Could not find {self.progress_path}")

        df = pd.read_excel(self.progress_path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        self.progress_df = df

        # Required ID columns
        self.col_country = self._first_of(("country",))
        self.col_league  = self._first_of(("league",))
        self.col_season  = self._first_of(("season",))
        missing = [c for c in [self.col_country, self.col_league, self.col_season] if c is None]
        if missing:
            raise ValueError(f"Progress file missing required columns: {missing}")

        # Filter columns (from XLSX)
        self.col_downloaded     = self._first_of(("downloaded", "is_downloaded", "download_status"))
        self.col_overall_status = self._first_of(("overall_status", "status"))

        if self.col_overall_status is None:
            raise ValueError("Progress file must include an 'overall_status' (or 'status') column.")
        if self.col_downloaded is None:
            # Not fatal, but we'll warn and rely on overall_status only
            print("[WARN] No 'downloaded' column found; using only overall_status=='Completed' to select rows.")

    def _first_of(self, candidates):
        for c in candidates:
            if c in self.progress_df.columns:
                return c
        return None

    @staticmethod
    def _truthy(val) -> bool:
        import pandas as pd
        if isinstance(val, bool): return val
        if pd.isna(val): return False
        if isinstance(val, (int, float)): return val != 0
        return str(val).strip().lower() in {"true", "t", "yes", "y", "1", "downloaded", "done", "ok"}

    def _json_inputs_exist(self, country, league, season) -> bool:
        import os
        save_dir = os.path.join(self.script_dir, f"{country}_{league}", str(int(season)))
        needed = [
            os.path.join(save_dir, "fixtures.json"),
            os.path.join(save_dir, "match_events.json"),
            os.path.join(save_dir, "players.json"),
        ]
        return all(os.path.exists(p) for p in needed)

    def _preflight_validate_json(self, country, league, season):
        import os, json
        save_dir = os.path.join(self.script_dir, f"{country}_{league}", str(int(season)))

        with open(os.path.join(save_dir, "fixtures.json"), "r", encoding="utf-8") as f:
            fixtures = json.load(f)
        with open(os.path.join(save_dir, "match_events.json"), "r", encoding="utf-8") as f:
            events = json.load(f)
        with open(os.path.join(save_dir, "players.json"), "r", encoding="utf-8") as f:
            players = json.load(f)

        if not isinstance(fixtures, dict) or "response" not in fixtures or not isinstance(fixtures["response"], list):
            raise ValueError("fixtures.json malformed (missing 'response' list)")
        if not isinstance(events, dict):
            raise ValueError("match_events.json malformed (expected dict keyed by fixture_id)")
        if not isinstance(players, dict):
            raise ValueError("players.json malformed (expected dict keyed by team_id)")
        if len(fixtures["response"]) == 0:
            raise ValueError("fixtures.json has empty 'response'")

    def _already_done(self, country: str, league: str, season: int) -> bool:
        """
        Consider a season 'done' if:
          1) <SAVE_DIR>/output.xlsx exists, OR
          2) analysis_results has rows for (country, league, season)
        """
        import os, sqlite3, pandas as pd

        save_dir = os.path.join(self.script_dir, f"{country}_{league}", str(int(season)))
        out_path = os.path.join(save_dir, "output.xlsx")
        if os.path.isfile(out_path):
            return True

        db_path = os.path.join(self.script_dir, "analysis_results.db")
        if not os.path.isfile(db_path):
            return False

        try:
            conn = sqlite3.connect(db_path)
            q = """
                SELECT COUNT(1) AS n
                FROM analysis_results
                WHERE country=? AND league=? AND season=?
            """
            cdf = pd.read_sql_query(q, conn, params=[country, league, int(season)])
            conn.close()
            return (not cdf.empty) and int(cdf["n"].iloc[0]) > 0
        except Exception:
            return False

    def _run_single_season_subprocess(self, country, league, season) -> tuple[int, str]:
        import os, sys, subprocess, json

        cfg = {
            "country": country,
            "league": league,
            "season": int(season),
            "lasso_alphas": self.lasso_alphas,
            "minutes_equiv_tolerance": self.minutes_equiv_tolerance,
        }

        env = os.environ.copy()
        env["BATCH_JOB_CFG"] = json.dumps(cfg)
        env["SCRIPT_PATH"] = self.script_path

        # BLAS/OpenMP stability
        env.pop("KMP_DUPLICATE_LIB_OK", None)
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")

        code = r"""
import os, json, runpy
ns = runpy.run_path(os.environ["SCRIPT_PATH"])
SA = ns["SeasonAnalyzer"]
cfg = json.loads(os.environ["BATCH_JOB_CFG"])
an = SA(
    league_name=cfg["league"],
    country_name=cfg["country"],
    season=int(cfg["season"]),
    lasso_alphas=cfg["lasso_alphas"],
)
if hasattr(an, "minutes_equiv_tolerance"):
    an.minutes_equiv_tolerance = int(cfg.get("minutes_equiv_tolerance", 0))
an.run()
"""
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        tail = (p.stderr or p.stdout or "")[-2000:]
        return p.returncode, tail

    def run_all(self, limit: int | None = None):
        import time, pandas as pd, os, traceback

        df = self.progress_df.copy()

        # --- direct XLSX filter: downloaded==True AND overall_status=="Completed" ---
        if self.col_downloaded is not None:
            mask_downloaded = df[self.col_downloaded].apply(self._truthy)
        else:
            mask_downloaded = True  # if no column, don't block

        mask_completed = df[self.col_overall_status].astype(str).str.strip().str.lower() == "completed"
        df = df[mask_downloaded & mask_completed]

        if limit is not None:
            df = df.head(int(limit))

        rows = []
        for _, row in df.iterrows():
            country = str(row[self.col_country]).strip()
            league  = str(row[self.col_league]).strip()
            season  = int(row[self.col_season])

            started_at = time.time()
            status, message = "skipped", ""
            outputs = {}

            try:
                if self._already_done(country, league, season):
                    status = "already_done"
                    message = "output.xlsx or DB rows found"
                elif not self._json_inputs_exist(country, league, season):
                    status = "missing_inputs"
                    message = "fixtures/events/players JSON missing"
                else:
                    try:
                        self._preflight_validate_json(country, league, season)
                    except Exception as ve:
                        status = "invalid_dataset"
                        message = f"Validation fail: {ve}"
                        raise RuntimeError("preflight_stop")

                    rc, tail = self._run_single_season_subprocess(country, league, season)
                    if rc == 0:
                        status = "ok"
                    else:
                        status = "error_subprocess"
                        message = f"rc={rc}. tail:\n{tail}"

            except RuntimeError as rt:
                if str(rt) != "preflight_stop":
                    status = "error"
                    message = f"RuntimeError: {rt}"
            except Exception as e:
                status = "error"
                if self.debug:
                    message = "".join(traceback.format_exception(type(e), e, e.__traceback__))[:4000]
                    print(f"[ERROR] {country} / {league} / {season}\n{message}")
                else:
                    message = f"{type(e).__name__}: {e}"

            elapsed = round(time.time() - started_at, 2)
            rows.append({
                "country": country,
                "league": league,
                "season": season,
                "status": status,
                "message": message,
                "elapsed_s": elapsed,
                **outputs,
            })

        report = pd.DataFrame(rows)
        out_csv  = os.path.join(self.script_dir, "season_batch_report.csv")
        out_xlsx = os.path.join(self.script_dir, "season_batch_report.xlsx")
        report.to_csv(out_csv, index=False)
        report.to_excel(out_xlsx, index=False)
        print(f"[Batch] Saved: {out_csv}\n[Batch] Saved: {out_xlsx}")
        return report


# ---------------- main: load progress + run completed downloads (no argparse) ----------------
if __name__ == "__main__":
    # Run all rows that are downloaded and overall_status == "Completed",
    # unless already processed (output.xlsx or DB has rows).
    runner = SeasonBatchRunner(
        progress_filename="downloads_progress.xlsx",
        minutes_equiv_tolerance=0,
        lasso_alphas=(0.01, 0.001),
        debug=True,
    )
    runner.run_all()


# if __name__ == "__main__":
#     # Defaults: require all *_done flags; set debug=True for full traces
#     runner = SeasonBatchRunner(
#         progress_filename="downloads_progress.xlsx",
#         filter_mode="all_done",       # or "events_only"
#         minutes_equiv_tolerance=0,    # 0 for exact minutes; >0 for bucketing
#         lasso_alphas=(0.01, 0.001),
#         debug=True,                   # show full error tracebacks in console
#     )
#     # runner.run_all()
    
    
#     smallrunner = SeasonAnalyzer(league_name="Bundesliga", 
#                                  country_name="Germany", season=2022, 
#                                  lasso_alphas=[0.01, 0.001],
#                                  save_intermediate=False)
#     smallrunner.run()
    
    
    # jobs = [
    #     ("Germany", "Bundesliga"),
    #     ("Spain", "La Liga"),
    #     ("England", "Premier League"),
    # ]
    
    # # smallrunner = SeasonAnalyzer(league_name="Bundesliga", country_name="Germany", season=2022, lasso_alphas=[0.01, 0.001])
   
    # for year in range(2024, 2013, -1):
    #     for country, league in jobs:
            
    #         smallrunner = SeasonAnalyzer(league_name= league, country_name=country, season=year, lasso_alphas=[0.01, 0.001])
    #         smallrunner.run()
