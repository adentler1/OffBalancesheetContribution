# --- top-of-file imports (preferred) ---
import os
import sqlite3
# import time
# import traceback
from typing import Optional, List
import pandas as pd

from season_analyzer import SeasonAnalyzer

# ---------------- SeasonBatchRunner (wrapper over SeasonAnalyzer; top-level imports) ----------------
class SeasonBatchRunner:
    """
    Reads `downloads_progress.xlsx`, selects rows where overall_status == 'Completed',
    and runs SeasonAnalyzer per row. Skips seasons already processed.
    """
    def __init__(
        self,
        progress_filename: str = "downloads_progress.xlsx",
        # minutes_equiv_tolerance: int = 0,
        lasso_alphas: List[float] | float = (0.01, 0.001),
        ridge_alphas: List[float] | float = (1.0, 10.0),  # NEW: ridge grid
        skip_already_done: bool = True,
        debug: bool = False,
    ):
        self.debug = bool(debug)
        self.skip_already_done = bool(skip_already_done)
        self.lasso_alphas = list(lasso_alphas)
        self.ridge_alphas = list(ridge_alphas)
        # self.minutes_equiv_tolerance = int(minutes_equiv_tolerance)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.progress_path = os.path.join(self.script_dir, progress_filename)

        if not os.path.exists(self.progress_path):
            raise FileNotFoundError(f"Could not find {self.progress_path}")

        df = pd.read_excel(self.progress_path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        self.progress_df = df

        # required columns
        for col in ("country", "league", "season", "overall_status"):
            if col not in self.progress_df.columns:
                raise ValueError(f"Progress file missing required column '{col}'")

    def _json_inputs_exist(self, country: str, league: str, season: int) -> bool:
        save_dir = os.path.join(self.script_dir, f"{country}_{league}", str(int(season)))
        needed = [
            os.path.join(save_dir, "fixtures.json"),
            os.path.join(save_dir, "match_events.json"),
            os.path.join(save_dir, "players.json"),
        ]
        return all(os.path.exists(p) for p in needed)

    def _already_done(self, country: str, league: str, season: int) -> bool:
        """Done if output.xlsx exists OR analysis_results has rows for (country, league, season)."""
        
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

    def run_all(self, limit: Optional[int] = None):
        df = self.progress_df.copy()
        mask_completed = df["overall_status"].astype(str).str.strip().str.lower() == "completed"
        df = df[mask_completed]
        if limit is not None:
            df = df.head(int(limit))

        # rows = []
        for _, row in df.iterrows():
            country = str(row["country"]).strip()
            league  = str(row["league"]).strip()
            season  = int(row["season"])

            try:
                if self.skip_already_done and self._already_done(country, league, season):
                    pass
                    # status = "already_done"
                    # message = "output.xlsx or DB rows found"
                elif not self._json_inputs_exist(country, league, season):
                    pass
                    # status = "missing_inputs"
                    # message = "fixtures/events/players JSON missing"
                else:
                    an = SeasonAnalyzer(
                        league_name=league,
                        country_name=country,
                        season=int(season),
                        lasso_alphas=self.lasso_alphas,
                        ridge_alphas=self.ridge_alphas,
                        base_dir=self.script_dir,
                        date_cutoff=None,  # full season in batch by default
                    )
                    # an.minutes_equiv_tolerance = self.minutes_equiv_tolerance
                    an.run()  # load → analyze → save
                    # status = "ok"
                    print(f"[OK] {country} / {league} / {season}")  # requested one-liner

            except Exception:
                pass
                # status = "error"
                # message = f"{type(e).__name__}: {e}" if not self.debug else "".join(
                #     traceback.format_exception(type(e), e, e.__traceback__)
                # )[:4000]

            # elapsed = round(time.time() - started, 2)
            # rows.append({
            #     "country": country,
            #     "league": league,
            #     "season": season,
            #     "status": status,
            #     "message": message,
            #     "elapsed_s": elapsed,
            # })

        # report = pd.DataFrame(rows)
        # out_csv  = os.path.join(self.script_dir, "season_batch_report.csv")
        # out_xlsx = os.path.join(self.script_dir, "season_batch_report.xlsx")
        # report.to_csv(out_csv, index=False)
        # report.to_excel(out_xlsx, index=False)
        # print(f"[Batch] Saved: {out_csv}\n[Batch] Saved: {out_xlsx}")
        # return report

# ---------------- main: run completed rows via SeasonAnalyzer (no argparse) ----------------
if __name__ == "__main__":
    
    # Batch: run rows where overall_status == "Completed", unless already processed.
    runner = SeasonBatchRunner(
        progress_filename="downloads_progress.xlsx",
        lasso_alphas=(0.01, 0.001),
        ridge_alphas=(1.0, 10.0),  # example
        skip_already_done=True,
        debug=False,
    )
    runner.run_all()