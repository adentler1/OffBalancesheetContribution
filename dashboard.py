#!/usr/bin/env python3
"""
Simple Streamlit Dashboard for Soccer Analytics.

=== HOW TO RUN (copy-paste) ===

# Activate environment first:
cd ~/Nextcloud/Projects/AthleteComparison/Soccer && source ~/.venvs/athlete-soccer/bin/activate

# Then run one of these:

# 1. Default: Launch dashboard with Streamlit (auto-refreshes data):
streamlit run dashboard.py

# 2. Alternative: Run directly with Python (auto-launches streamlit):
python dashboard.py

# 3. Headless mode (for servers):
streamlit run dashboard.py --server.headless true

# 4. Custom port:
streamlit run dashboard.py --server.port 8502

# 5. Allow external access:
streamlit run dashboard.py --server.address 0.0.0.0

# 6. Combined server options:
streamlit run dashboard.py --server.headless true --server.port 8501 --server.address 0.0.0.0

=== NOTES ===

- Dashboard auto-refreshes data from analysis_results.db on startup
- Data is exported to parquet files in artefacts/shared/ for fast loading
- Works in "bundled mode" when deployed without full src/ structure

===============================
"""

import sys
import os

# Auto-launch with streamlit if run directly with python
if __name__ == "__main__" and "streamlit" not in sys.modules:
    import subprocess
    # Re-run this script with streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", __file__,
        "--server.headless", "true"
    ])
    sys.exit(0)

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import uniform_filter1d

# Try to import DuckDB for fast parquet queries
try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

# Detect bundled mode (running from shared folder without full src structure)
_BUNDLED_MODE = not (Path(__file__).parent / "src" / "utils").exists()

if not _BUNDLED_MODE:
    from src.utils.config_loader import config, get_absolute_path, get_data_dir, get_db_path
    from src.utils.league_classifier import get_classifier
else:
    # In bundled mode, we use parquet files in the same folder
    config = None
    get_absolute_path = None
    get_classifier = None

# =============================================================================
#                     AUTO-REFRESH DASHBOARD DATABASE
# =============================================================================

# Automatically export/refresh data on startup (only in dev mode, not bundled)
@st.cache_resource(ttl=300)  # Cache for 5 minutes to avoid re-running on every rerun
def _refresh_dashboard_db():
    """Export latest data on startup (to shared folder for GitHub sync)."""
    if _BUNDLED_MODE:
        return None  # Skip refresh in bundled mode - use existing files
    try:
        from export_dashboard_db import export_dashboard_db
        base_dir = Path(__file__).parent
        stats = export_dashboard_db(
            analysis_db=str(get_db_path("analysis_db")),
            league_quality_db=str(base_dir / "league_quality.db"),
            profiles_db=str(get_db_path("profiles_db")),
            verbose=False
        )
        return stats
    except Exception as e:
        # If export fails, dashboard will fall back to existing data
        print(f"[WARN] Dashboard export failed: {e}")
        return None

# Run the refresh on import
_refresh_dashboard_db()

# =============================================================================
#                          CONFIG & DATA SOURCE DETECTION
# =============================================================================

# Detect data source: parquet (preferred) or SQLite (fallback)
_LOCAL_DIR = Path(__file__).parent
_SHARED_DIR = Path(__file__).parent.parent / "artefacts" / "shared"

# Check for parquet files (new format)
_LOCAL_PARQUET = _LOCAL_DIR / "players_recent.parquet"
_SHARED_PARQUET = _SHARED_DIR / "players_recent.parquet"
_LOCAL_SQLITE = _LOCAL_DIR / "dashboard.db"
_SHARED_SQLITE = _SHARED_DIR / "dashboard.db"

# Determine data source and path
if _LOCAL_PARQUET.exists() and _HAS_DUCKDB:
    # Bundled mode with parquet (preferred)
    _DATA_DIR = _LOCAL_DIR
    _DATA_FORMAT = "parquet"
    _USING_DASHBOARD_DB = True
elif _SHARED_PARQUET.exists() and _HAS_DUCKDB:
    # Dev mode with shared parquet
    _DATA_DIR = _SHARED_DIR
    _DATA_FORMAT = "parquet"
    _USING_DASHBOARD_DB = True
elif _LOCAL_SQLITE.exists():
    # Bundled mode with SQLite fallback
    _DATA_DIR = _LOCAL_DIR
    _DATA_FORMAT = "sqlite"
    DB_PATH = str(_LOCAL_SQLITE)
    PROFILES_DB_PATH = str(_LOCAL_SQLITE)
    _USING_DASHBOARD_DB = True
elif _SHARED_SQLITE.exists():
    # Dev mode with shared SQLite
    _DATA_DIR = _SHARED_DIR
    _DATA_FORMAT = "sqlite"
    DB_PATH = str(_SHARED_SQLITE)
    PROFILES_DB_PATH = str(_SHARED_SQLITE)
    _USING_DASHBOARD_DB = True
elif not _BUNDLED_MODE:
    # Full development mode - use source databases
    _DATA_DIR = None
    _DATA_FORMAT = "sqlite"
    DB_PATH = str(get_db_path("analysis_db"))
    PROFILES_DB_PATH = str(get_db_path("profiles_db"))
    _USING_DASHBOARD_DB = False
else:
    # Bundled mode but no data found - error state
    _DATA_DIR = _LOCAL_DIR
    _DATA_FORMAT = "sqlite"
    DB_PATH = str(_LOCAL_SQLITE)
    PROFILES_DB_PATH = str(_LOCAL_SQLITE)
    _USING_DASHBOARD_DB = True

# For parquet mode, set paths
if _DATA_FORMAT == "parquet":
    DB_PATH = str(_DATA_DIR / "players_recent.parquet")  # For compatibility
    PROFILES_DB_PATH = str(_DATA_DIR / "profiles.parquet")

CONTRIB_COL = "lasso_contribution_alpha_best"  # Best alpha auto-selected via IC

# =============================================================================
#                          DUCKDB / PARQUET HELPERS
# =============================================================================

@st.cache_resource
def _get_duckdb_connection():
    """Get a DuckDB connection for querying parquet files."""
    if not _HAS_DUCKDB:
        return None
    return duckdb.connect(":memory:")


def _query_parquet(query: str, params: tuple = None) -> pd.DataFrame:
    """Execute a DuckDB query against parquet files. Returns DataFrame."""
    if not _HAS_DUCKDB or _DATA_FORMAT != "parquet":
        return pd.DataFrame()

    conn = _get_duckdb_connection()
    try:
        if params:
            return conn.execute(query, params).fetchdf()
        return conn.execute(query).fetchdf()
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()


def _get_parquet_path(name: str) -> str:
    """Get full path to a parquet file."""
    return str(_DATA_DIR / name)


def _load_parquet(name: str) -> pd.DataFrame:
    """Load a parquet file directly as DataFrame."""
    path = _DATA_DIR / name
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def _load_players_parquet(include_historical: bool = False) -> pd.DataFrame:
    """
    Load players data from parquet files.
    By default, only loads recent (hot) data for faster startup.
    Set include_historical=True to load all data.
    """
    if _DATA_FORMAT != "parquet":
        return pd.DataFrame()

    recent_path = _DATA_DIR / "players_recent.parquet"
    if not recent_path.exists():
        return pd.DataFrame()

    # Load recent data (always)
    recent_df = pd.read_parquet(recent_path)

    if include_historical:
        # Also load historical data
        hist_path = _DATA_DIR / "players_historical.parquet"
        if hist_path.exists():
            hist_df = pd.read_parquet(hist_path)
            return pd.concat([recent_df, hist_df], ignore_index=True)

    return recent_df


@st.cache_data(ttl=600, show_spinner=False)
def _load_summaries() -> dict:
    """Load pre-aggregated summary data."""
    summaries = {}
    summaries_dir = _DATA_DIR / "summaries" if _DATA_DIR else None

    if summaries_dir and summaries_dir.exists():
        for name in ["top_players", "top_clubs", "top_leagues"]:
            path = summaries_dir / f"{name}.parquet"
            if path.exists():
                summaries[name] = pd.read_parquet(path)

    return summaries


def get_data_info() -> dict:
    """Get information about the current data source."""
    info = {
        "format": _DATA_FORMAT,
        "has_duckdb": _HAS_DUCKDB,
        "bundled_mode": _BUNDLED_MODE,
        "data_dir": str(_DATA_DIR) if _DATA_DIR else None,
    }

    # Load metadata if available
    if _DATA_DIR:
        metadata_path = _DATA_DIR / "_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                info["metadata"] = json.load(f)

    return info


# Cache for pre-enrichment status (loaded once)
_PRE_ENRICHED_STATUS = None


def _is_data_pre_enriched() -> bool:
    """Check if parquet data is pre-enriched (transforms already applied at export time)."""
    global _PRE_ENRICHED_STATUS
    if _PRE_ENRICHED_STATUS is not None:
        return _PRE_ENRICHED_STATUS

    _PRE_ENRICHED_STATUS = False
    if _DATA_DIR:
        metadata_path = _DATA_DIR / "_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    _PRE_ENRICHED_STATUS = metadata.get("pre_enriched", False)
            except Exception:
                pass

    return _PRE_ENRICHED_STATUS


# Color-blind friendly palette (IBM Design / Wong palette)
# These colors are distinguishable for most forms of color blindness
POSITION_COLORS = {
    "Goalkeeper": "#648FFF",   # Blue (perceptually distinct)
    "Defender": "#FFB000",     # Amber/Orange
    "Midfielder": "#785EF0",   # Purple
    "Forward": "#DC267F",      # Magenta/Pink
    "Unknown": "#9E9E9E",      # Gray
}

# Chart color sequence for general use (color-blind safe)
CB_PALETTE = ["#648FFF", "#FFB000", "#DC267F", "#785EF0", "#FE6100", "#9E9E9E"]

# Position name mapping (API abbreviations to full names)
POSITION_MAP = {
    "G": "Goalkeeper",
    "D": "Defender",
    "M": "Midfielder",
    "F": "Forward",
    "": "Unknown",
    "-": "Unknown",
    "Unknown": "Unknown",
}

# Profile position mapping (API full names to our standard names)
PROFILE_POSITION_MAP = {
    "Goalkeeper": "Goalkeeper",
    "Defender": "Defender",
    "Midfielder": "Midfielder",
    "Attacker": "Forward",
    "Forward": "Forward",
}

# Team name normalization (standardize variations to canonical names)
# Maps variations to preferred canonical form
TEAM_NAME_ALIASES = {
    # German teams - prefer German umlauts
    "Bayern Munich": "Bayern München",
    "Borussia Monchengladbach": "Borussia Mönchengladbach",
    "1.FC Köln": "1. FC Köln",
    "FC Nurnberg": "FC Nürnberg",
    "Fortuna Dusseldorf": "Fortuna Düsseldorf",
    "SpVgg Greuther Furth": "SpVgg Greuther Fürth",
    "Vfl Bochum": "VfL Bochum",
    # Add more as needed
}


def normalize_team_name(team_name: str) -> str:
    """Normalize team name to canonical form."""
    if pd.isna(team_name):
        return team_name
    team_name = str(team_name).strip()
    return TEAM_NAME_ALIASES.get(team_name, team_name)


def normalize_team_column(team_str: str) -> str:
    """Normalize team names, handling 'Multiple Teams (A, B)' format."""
    if pd.isna(team_str):
        return team_str
    team_str = str(team_str)

    if team_str.startswith("Multiple Teams ("):
        # Extract and normalize each team name
        inner = team_str[16:-1]  # Remove "Multiple Teams (" and ")"
        teams = [normalize_team_name(t.strip()) for t in inner.split(", ")]
        return f"Multiple Teams ({', '.join(teams)})"
    else:
        return normalize_team_name(team_str)


# =============================================================================
#                          PLAYER ID & NAME NORMALIZATION
# =============================================================================

def normalize_player_id(pid) -> int | None:
    """
    Normalize player_id to consistent integer form.
    Handles: 'p522' -> 522, '522' -> 522, 522 -> 522, 522.0 -> 522
    """
    if pd.isna(pid):
        return None
    s = str(pid).strip()
    if s.startswith("p"):
        try:
            return int(s[1:])
        except ValueError:
            return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


@st.cache_data
def _build_canonical_player_names() -> dict:
    """
    Build a mapping from normalized player_id to canonical (longest/fullest) name.
    This ensures consistent player names across the dashboard.
    """
    # Try parquet first (preferred for Streamlit Cloud)
    if _DATA_FORMAT == "parquet" and _DATA_DIR:
        # Load all players (including historical) to get complete name mapping
        names = _load_players_parquet(include_historical=True)
        if not names.empty:
            names = names[["player_id", "player_name"]].drop_duplicates()
            names = names[names["player_name"].notna() & (names["player_name"] != "Unknown")]
        else:
            return {}
    elif Path(DB_PATH).exists():
        # Fallback to SQLite
        conn = sqlite3.connect(DB_PATH)
        names = pd.read_sql_query("""
            SELECT player_id, player_name
            FROM players
            WHERE player_name IS NOT NULL AND player_name != 'Unknown'
        """, conn)
        conn.close()
    else:
        return {}

    # Normalize player IDs
    names["pid_norm"] = names["player_id"].apply(normalize_player_id)
    names = names[names["pid_norm"].notna()]

    # For each player_id, pick the longest name (fullest form)
    # e.g., prefer "Thomas Müller" over "T. Müller"
    canonical = {}
    for pid, group in names.groupby("pid_norm"):
        # Get all name variants
        name_list = group["player_name"].dropna().unique().tolist()
        if name_list:
            # Pick longest name (most complete)
            best_name = max(name_list, key=len)
            canonical[pid] = best_name

    return canonical


# Global cache for canonical names (lazy loaded)
_CANONICAL_NAMES_CACHE = None


def get_canonical_player_name(pid, original_name: str = None) -> str:
    """
    Get the canonical (fullest) name for a player.
    Falls back to original_name if not found.
    """
    global _CANONICAL_NAMES_CACHE
    if _CANONICAL_NAMES_CACHE is None:
        _CANONICAL_NAMES_CACHE = _build_canonical_player_names()

    pid_norm = normalize_player_id(pid)
    if pid_norm is not None and pid_norm in _CANONICAL_NAMES_CACHE:
        return _CANONICAL_NAMES_CACHE[pid_norm]
    return original_name if original_name else "Unknown"


# =============================================================================
#                          HOVER TEMPLATE HELPERS
# =============================================================================

def make_player_hover(df: pd.DataFrame, contrib_col: str = "contribution") -> list[str]:
    """
    Create standardized hover text for player scatter plots.

    Format:
    <b>Player Name</b> (contribution)
    Position, Team
    Country, League, Season
    X goals, Y assists
    FTE games played
    """
    hover_texts = []
    for _, row in df.iterrows():
        contrib = row.get(contrib_col, 0)
        contrib_str = f"{contrib:+.1f}" if pd.notna(contrib) else "N/A"

        name = row.get("player_name", "Unknown")
        position = row.get("position_group", row.get("position", ""))
        team = row.get("team", "")
        country = row.get("country", "")
        league = row.get("league", "")
        season = row.get("season", "")
        goals = row.get("goals", 0)
        assists = row.get("assists", 0)
        fte = row.get("FTE_games_played", 0)

        lines = [f"<b>{name}</b> ({contrib_str})"]

        if position or team:
            lines.append(f"{position}, {team}" if position and team else (position or team))

        if country or league or season:
            loc_parts = [p for p in [country, league, str(int(season)) if pd.notna(season) else ""] if p]
            if loc_parts:
                lines.append(", ".join(loc_parts))

        if pd.notna(goals) or pd.notna(assists):
            goals_int = int(goals) if pd.notna(goals) else 0
            assists_int = int(assists) if pd.notna(assists) else 0
            lines.append(f"{goals_int} goals, {assists_int} assists")

        if pd.notna(fte):
            lines.append(f"{fte:.1f} FTE games")

        hover_texts.append("<br>".join(lines))

    return hover_texts


# =============================================================================
#                          CASCADING FILTER HELPERS
# =============================================================================

@st.cache_data
def get_leagues_for_country(country: str = None) -> list:
    """Get leagues available for a specific country."""
    # Try parquet first
    if _DATA_FORMAT == "parquet" and _DATA_DIR:
        df = _load_players_parquet(include_historical=True)
        if not df.empty:
            if country:
                df = df[df["country"] == country]
            return sorted(df["league"].dropna().unique().tolist())

    # Fall back to SQLite
    if not Path(DB_PATH).exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    if country:
        leagues = pd.read_sql_query("""
            SELECT DISTINCT league FROM players WHERE country = ?
        """, conn, params=(country,))
    else:
        leagues = pd.read_sql_query("SELECT DISTINCT league FROM players", conn)
    conn.close()
    return sorted(leagues["league"].dropna().tolist())


@st.cache_data
def get_teams_for_league_season(league: str = None, country: str = None, season: int = None) -> list:
    """
    Get teams available for a specific league/country/season combination.
    Returns individual team names (extracts from 'Multiple Teams' entries).
    """
    teams_df = None

    # Try parquet first
    if _DATA_FORMAT == "parquet" and _DATA_DIR:
        df = _load_players_parquet(include_historical=True)
        if not df.empty:
            # Determine team column name
            team_col = "team(s)" if "team(s)" in df.columns else "team"
            if team_col in df.columns:
                mask = df[team_col].notna()
                if league:
                    mask &= df["league"] == league
                if country:
                    mask &= df["country"] == country
                if season:
                    mask &= df["season"] == season
                teams_df = df.loc[mask, [team_col]].drop_duplicates()
                teams_df = teams_df.rename(columns={team_col: "team"})

    # Fall back to SQLite
    if teams_df is None:
        if not Path(DB_PATH).exists():
            return []

        conn = sqlite3.connect(DB_PATH)

        # Build query with optional filters
        conditions = ["\"team(s)\" IS NOT NULL"]
        params = []

        if league:
            conditions.append("league = ?")
            params.append(league)
        if country:
            conditions.append("country = ?")
            params.append(country)
        if season:
            conditions.append("season = ?")
            params.append(season)

        query = f"""
            SELECT DISTINCT "team(s)" as team
            FROM players
            WHERE {' AND '.join(conditions)}
        """
        teams_df = pd.read_sql_query(query, conn, params=params)
        conn.close()

    # Extract individual team names and normalize
    all_teams = set()
    for team in teams_df["team"].dropna():
        if team.startswith("Multiple Teams ("):
            inner = team[16:-1]
            for t in inner.split(", "):
                all_teams.add(normalize_team_name(t.strip()))
        else:
            all_teams.add(normalize_team_name(team))

    return sorted(all_teams)


@st.cache_data
def get_seasons_for_league(league: str = None, country: str = None) -> list:
    """Get seasons available for a specific league/country."""
    # Try parquet first
    if _DATA_FORMAT == "parquet" and _DATA_DIR:
        df = _load_players_parquet(include_historical=True)
        if not df.empty:
            mask = df["season"].notna()
            if league:
                mask &= df["league"] == league
            if country:
                mask &= df["country"] == country
            return sorted(df.loc[mask, "season"].dropna().astype(int).unique().tolist())

    # Fall back to SQLite
    if not Path(DB_PATH).exists():
        return []

    conn = sqlite3.connect(DB_PATH)
    conditions = ["season IS NOT NULL"]
    params = []

    if league:
        conditions.append("league = ?")
        params.append(league)
    if country:
        conditions.append("country = ?")
        params.append(country)

    query = f"""
        SELECT DISTINCT season FROM players WHERE {' AND '.join(conditions)}
    """
    seasons = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return sorted(seasons["season"].dropna().astype(int).tolist())


# =============================================================================
#                          DATA LOADING
# =============================================================================


@st.cache_data(ttl=600, show_spinner=False)
def load_league_visibility() -> pd.DataFrame:
    """Load league visibility classifications from parquet or database."""
    # Try parquet first
    if _DATA_FORMAT == "parquet" and _DATA_DIR:
        vis_path = _DATA_DIR / "league_visibility.parquet"
        if vis_path.exists():
            return pd.read_parquet(vis_path)

    # Fall back to SQLite
    if Path(DB_PATH).exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT * FROM league_visibility", conn)
            conn.close()
            return df
        except Exception:
            pass

    # In bundled mode without data, return empty DataFrame
    if _BUNDLED_MODE or get_classifier is None:
        return pd.DataFrame(columns=["country", "league", "category", "category_desc", "visibility", "priority"])

    # Fall back to computing from classifier (dev mode only)
    if not Path(DB_PATH).exists():
        return pd.DataFrame(columns=["country", "league", "category", "category_desc", "visibility", "priority"])

    classifier = get_classifier()
    conn = sqlite3.connect(DB_PATH)
    leagues_df = pd.read_sql_query(
        "SELECT DISTINCT country, league FROM players",
        conn
    )
    conn.close()

    if leagues_df.empty:
        return pd.DataFrame(columns=["country", "league", "category", "category_desc", "visibility", "priority"])

    leagues_df["category"] = leagues_df.apply(
        lambda r: classifier.classify(r["country"], r["league"]), axis=1
    )
    leagues_df["category_desc"] = leagues_df["category"].apply(classifier.get_category_description)
    leagues_df["visibility"] = leagues_df.apply(
        lambda r: classifier.classify_visibility(r["country"], r["league"]), axis=1
    )
    leagues_df["priority"] = leagues_df["category"].apply(
        lambda c: classifier.get_category_info(c).get("priority", 99)
    )

    return leagues_df


def get_public_leagues() -> set:
    """Get set of (country, league) tuples for public leagues."""
    vis_df = load_league_visibility()
    public_df = vis_df[vis_df["visibility"] == "public"]
    return set(zip(public_df["country"], public_df["league"]))


def filter_to_public_leagues(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include rows from public leagues."""
    public_leagues = get_public_leagues()
    if not public_leagues:
        return df
    # Vectorized filter using tuple creation (faster than row-wise apply)
    league_tuples = list(zip(df["country"], df["league"]))
    mask = pd.Series([t in public_leagues for t in league_tuples], index=df.index)
    return df[mask]


@st.cache_data(ttl=600, show_spinner=False)  # Reduced TTL (10 min) for fresher data
def load_all_players(min_fte: float = 0.0, include_historical: bool = True) -> pd.DataFrame:
    """Load all player-season records with profile data.

    Args:
        min_fte: Minimum FTE games played filter
        include_historical: If True, load all data. If False, only recent seasons (faster).

    Note:
        When data is pre-enriched (exported with enrichment), heavy transformations are
        skipped for much faster loading. Pre-enrichment is detected via _metadata.json.
    """
    # Check if data is pre-enriched (all transforms done at export time)
    is_pre_enriched = _is_data_pre_enriched()

    # Use parquet if available (faster)
    if _DATA_FORMAT == "parquet" and _HAS_DUCKDB:
        df = _load_players_parquet(include_historical=include_historical)
        if not df.empty:
            # Rename columns for compatibility
            if "team(s)" in df.columns:
                df = df.rename(columns={"team(s)": "team"})
            if CONTRIB_COL in df.columns:
                df = df.rename(columns={CONTRIB_COL: "contribution"})
            if "league_position" in df.columns:
                df = df.rename(columns={"league_position": "team_rank"})

            # Apply FTE filter
            if min_fte > 0:
                df = df[df["FTE_games_played"] >= min_fte]

            # If pre-enriched, data is ready to use - skip all transformations
            if is_pre_enriched:
                return df
    else:
        # Fallback to SQLite
        conn = sqlite3.connect(DB_PATH)
        # Note: contribution_ols excluded from bundled DB to reduce file size
        query = f"""
            SELECT
                player_id,
                player_name,
                position,
                "team(s)" as team,
                country,
                league,
                season,
                {CONTRIB_COL} as contribution,
                FTE_games_played,
                goals,
                assists,
                minutes_played,
                league_position as team_rank
            FROM players
            WHERE FTE_games_played >= ?
            ORDER BY player_name, season
        """
        df = pd.read_sql_query(query, conn, params=(min_fte,))
        conn.close()

    # --- Heavy transformations (only run if data is NOT pre-enriched) ---

    # Normalize player IDs to consistent integer form
    df["player_id"] = df["player_id"].apply(normalize_player_id)

    # Apply canonical player names (longest/fullest form, e.g., "Thomas Müller" over "T. Müller")
    df["player_name"] = df.apply(
        lambda row: get_canonical_player_name(row["player_id"], row["player_name"]),
        axis=1
    )

    # Normalize team names (handles variations like "Bayern Munich" vs "Bayern München")
    df["team"] = df["team"].apply(normalize_team_column)

    # Map position abbreviations to full names
    df["position"] = df["position"].fillna("").map(
        lambda x: POSITION_MAP.get(x.strip(), POSITION_MAP.get(x.strip().upper(), "Unknown"))
    )

    # Fill missing positions from other seasons of the same player
    df = _fill_missing_positions(df)

    # Enrich with player profiles (nationality, age)
    df = _enrich_with_profiles(df)

    return df


def _enrich_with_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add nationality, age, height, weight, BMI, and position fallback from player_profiles database."""
    if df.empty:
        df["nationality"] = "Unknown"
        df["age"] = None
        df["height_cm"] = None
        df["weight_kg"] = None
        df["bmi"] = None
        return df

    try:
        # Load profiles from parquet or SQLite
        if _DATA_FORMAT == "parquet" and _DATA_DIR:
            profiles_path = _DATA_DIR / "profiles.parquet"
            if profiles_path.exists():
                all_profiles = pd.read_parquet(profiles_path)
                # Get most recent profile per player
                profiles = all_profiles.sort_values("season", ascending=False).groupby("player_id").first().reset_index()
                profiles = profiles[profiles["nationality"].notna()]
            else:
                profiles = pd.DataFrame()
        elif Path(PROFILES_DB_PATH).exists():
            conn = sqlite3.connect(PROFILES_DB_PATH)
            # Get the most recent profile for each player (including birth_date for age calculation)
            profiles = pd.read_sql_query("""
                SELECT player_id, nationality, age, birth_date, position, height, weight
                FROM player_profiles
                WHERE nationality IS NOT NULL
                GROUP BY player_id
                HAVING season = MAX(season)
            """, conn)
            conn.close()
        else:
            profiles = pd.DataFrame()

        if profiles.empty:
            df["nationality"] = "Unknown"
            df["age"] = None
            df["height_cm"] = None
            df["weight_kg"] = None
            df["bmi"] = None
            return df

        # Extract birth year from birth_date (format: YYYY-MM-DD)
        profiles["birth_year"] = pd.to_datetime(profiles["birth_date"], errors="coerce").dt.year

        # Sanitize bad age values (API sometimes returns year instead of age)
        profiles.loc[profiles["age"] > 50, "age"] = None

        # Parse height (e.g., "183 cm" -> 183)
        profiles["height_cm"] = profiles["height"].str.extract(r"(\d+)").astype(float)

        # Parse weight (e.g., "77 kg" -> 77)
        profiles["weight_kg"] = profiles["weight"].str.extract(r"(\d+)").astype(float)

        # Calculate BMI: weight / (height_m)^2
        profiles["bmi"] = profiles["weight_kg"] / ((profiles["height_cm"] / 100) ** 2)

        # Map profile position to our standard names
        profiles["profile_position"] = profiles["position"].map(PROFILE_POSITION_MAP)

        # Normalize profile player_ids to match (df.player_id is already normalized at load time)
        profiles["player_id"] = profiles["player_id"].apply(normalize_player_id)

        # Merge with main dataframe using normalized player_id
        df = df.merge(
            profiles[["player_id", "nationality", "birth_year", "age", "height_cm", "weight_kg", "bmi", "profile_position"]],
            on="player_id",
            how="left",
            suffixes=("", "_profile")
        )
        df["nationality"] = df["nationality"].fillna("Unknown")

        # Calculate age per season using birth_year (overrides static age from profile)
        # Age = season - birth_year (approximate, but accurate enough for career trajectories)
        if "birth_year" in df.columns and "season" in df.columns:
            df["age"] = df["season"] - df["birth_year"]
            # Sanitize: age should be between 15 and 45 for professional players
            df.loc[(df["age"] < 15) | (df["age"] > 45), "age"] = None
            # Drop birth_year column (no longer needed)
            df = df.drop(columns=["birth_year"], errors="ignore")

        # Use profile position as fallback for Unknown positions
        unknown_mask = df["position"] == "Unknown"
        if unknown_mask.any() and "profile_position" in df.columns:
            df.loc[unknown_mask, "position"] = df.loc[unknown_mask, "profile_position"].fillna("Unknown")

        # Drop helper column
        df = df.drop(columns=["profile_position"], errors="ignore")

    except Exception as e:
        df["nationality"] = "Unknown"
        df["age"] = None
        df["height_cm"] = None
        df["weight_kg"] = None
        df["bmi"] = None

    return df


def _fill_missing_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing/Unknown positions from other seasons of the same player."""
    if df.empty:
        return df

    df = df.copy()

    # Find players with known positions
    known_positions = df[df["position"] != "Unknown"].groupby("player_id")["position"].first()

    # Fill Unknown positions
    unknown_mask = df["position"] == "Unknown"
    if unknown_mask.any():
        df.loc[unknown_mask, "position"] = df.loc[unknown_mask, "player_id"].map(known_positions)
        # Any still missing become "Unknown"
        df["position"] = df["position"].fillna("Unknown")

    return df


@st.cache_data
def get_unique_values(df: pd.DataFrame):
    """Get unique leagues, seasons, positions, teams, nationalities."""
    return {
        "leagues": sorted(df["league"].dropna().unique()),
        "seasons": sorted(df["season"].dropna().unique()),
        "positions": sorted(df["position"].dropna().unique()),
        "countries": sorted(df["country"].dropna().unique()),
        "teams": sorted(df["team"].dropna().unique()),
        "nationalities": sorted(df["nationality"].dropna().unique()),
    }


@st.cache_data
def load_teams_for_league(league_name: str = None, country: str = None) -> list:
    """Load team names from analysis database (to match actual data used in visualizations)."""
    teams_df = None

    # Try parquet first
    if _DATA_FORMAT == "parquet" and _DATA_DIR:
        try:
            df = _load_players_parquet(include_historical=True)
            if not df.empty:
                team_col = "team(s)" if "team(s)" in df.columns else "team"
                if team_col in df.columns:
                    mask = df[team_col].notna()
                    if league_name and country:
                        mask &= (df["league"] == league_name) & (df["country"] == country)
                    teams_df = df.loc[mask, [team_col]].drop_duplicates()
                    teams_df = teams_df.rename(columns={team_col: "team_name"})
        except Exception:
            teams_df = None

    # Fall back to SQLite
    if teams_df is None:
        try:
            conn = sqlite3.connect(DB_PATH)
            if league_name and country:
                # Get teams for specific league/country from analysis DB
                teams_df = pd.read_sql_query("""
                    SELECT DISTINCT "team(s)" as team_name
                    FROM players
                    WHERE league = ? AND country = ? AND "team(s)" IS NOT NULL
                """, conn, params=(league_name, country))
            else:
                # Get all teams
                teams_df = pd.read_sql_query("""
                    SELECT DISTINCT "team(s)" as team_name
                    FROM players
                    WHERE "team(s)" IS NOT NULL
                """, conn)
            conn.close()
        except Exception:
            return []

    if teams_df is None:
        return []

    # Extract individual team names from "Multiple Teams (A, B)" entries
    # and normalize to canonical forms
    all_teams = set()
    for team in teams_df["team_name"].dropna():
        if team.startswith("Multiple Teams ("):
            # Extract team names from "Multiple Teams (A, B, C)" format
            inner = team[16:-1]  # Remove "Multiple Teams (" and ")"
            for t in inner.split(", "):
                all_teams.add(normalize_team_name(t.strip()))
        else:
            all_teams.add(normalize_team_name(team))

    return sorted(all_teams)


@st.cache_data
def load_player_transfers(player_id: int) -> pd.DataFrame:
    """Load transfer history for a player."""
    transfers = None

    # Try parquet first
    if _DATA_FORMAT == "parquet" and _DATA_DIR:
        transfers_path = _DATA_DIR / "transfers.parquet"
        if transfers_path.exists():
            try:
                all_transfers = pd.read_parquet(transfers_path)
                transfers = all_transfers[
                    (all_transfers["player_id"] == player_id) &
                    (~all_transfers["type"].isin(["checked, no news", "None", "-"]))
                ][["transfer_date", "type", "from_team_name", "to_team_name"]].copy()
                transfers = transfers.sort_values("transfer_date")
            except Exception:
                transfers = None

    # Fall back to SQLite
    if transfers is None and Path(PROFILES_DB_PATH).exists():
        try:
            conn = sqlite3.connect(PROFILES_DB_PATH)
            transfers = pd.read_sql_query("""
                SELECT transfer_date, type, from_team_name, to_team_name
                FROM player_transfers
                WHERE player_id = ?
                AND type NOT IN ('checked, no news', 'None', '-')
                ORDER BY transfer_date
            """, conn, params=(player_id,))
            conn.close()
        except Exception:
            return pd.DataFrame()

    if transfers is None:
        return pd.DataFrame()

    if not transfers.empty:
        # Extract year from transfer_date for positioning
        transfers["year"] = pd.to_datetime(transfers["transfer_date"]).dt.year
        # Create display label
        transfers["label"] = transfers.apply(
            lambda r: f"{r['type']}" if r['type'] not in ['N/A', 'Transfer'] else "", axis=1
        )
    return transfers


@st.cache_data
def get_player_photo(player_id: int) -> str | None:
    """Get player photo URL from profiles database."""
    # Try parquet first
    if _DATA_FORMAT == "parquet" and _DATA_DIR:
        profiles_path = _DATA_DIR / "profiles.parquet"
        if profiles_path.exists():
            try:
                profiles = pd.read_parquet(profiles_path)
                if "photo" in profiles.columns:
                    match = profiles[profiles["player_id"] == player_id]
                    if not match.empty:
                        photo = match.iloc[0].get("photo")
                        return photo if pd.notna(photo) else None
            except Exception:
                pass

    # Fall back to SQLite
    if not Path(PROFILES_DB_PATH).exists():
        return None
    try:
        conn = sqlite3.connect(PROFILES_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT photo FROM player_profiles WHERE player_id = ? LIMIT 1",
            (player_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


@st.cache_data
def get_team_logo(team_name: str, league: str = None) -> str | None:
    """Get team logo URL from teams data in data directory."""
    # In bundled mode, team logos are not available
    if _BUNDLED_MODE or get_absolute_path is None:
        return None

    # Teams are stored in data/{Country}_{League}/{season}/teams.json
    # We'll search for the team name in any teams.json file
    import json
    data_dir = Path(get_data_dir()) if not _BUNDLED_MODE else Path(".")

    # Search for team in any teams.json
    for teams_file in data_dir.glob("*/*/teams.json"):
        try:
            with open(teams_file, "r") as f:
                teams_data = json.load(f)
            for team in teams_data.get("response", []):
                if team.get("team", {}).get("name") == team_name:
                    return team.get("team", {}).get("logo")
        except Exception:
            continue
    return None


# =============================================================================
#                          CAREER & COMPARISON HELPERS
# =============================================================================

def analyze_career_gaps(seasons: list) -> dict:
    """
    Analyze a player's seasons to identify gaps in their career data.

    Returns dict with:
        - first_season: First season with data
        - last_season: Last season with data
        - total_seasons: Number of seasons with data
        - expected_seasons: Number of seasons from first to last
        - missing_seasons: List of missing season years
        - missing_count: Number of missing seasons
        - coverage_pct: Percentage of expected seasons that have data
    """
    if not seasons:
        return {"missing_count": 0, "missing_seasons": [], "coverage_pct": 100}

    seasons_sorted = sorted(seasons)
    first = int(seasons_sorted[0])
    last = int(seasons_sorted[-1])
    expected = list(range(first, last + 1))
    missing = [s for s in expected if s not in seasons_sorted]

    return {
        "first_season": first,
        "last_season": last,
        "total_seasons": len(seasons_sorted),
        "expected_seasons": len(expected),
        "missing_seasons": missing,
        "missing_count": len(missing),
        "coverage_pct": round(100 * len(seasons_sorted) / len(expected), 1) if expected else 100
    }


def find_similar_players(
    df: pd.DataFrame,
    player_id: int,
    position: str = None,
    avg_contribution: float = None,
    age: int = None,
    league: str = None,
    country: str = None,
    contribution_tolerance: float = 0.15,
    age_tolerance: int = 3,
    max_results: int = 10,
    same_league_only: bool = False,
) -> pd.DataFrame:
    """
    Find players similar to a reference player.

    Args:
        df: DataFrame with all player data
        player_id: Reference player ID (will be excluded from results)
        position: Position to match (optional)
        avg_contribution: Average contribution to match within tolerance
        age: Age to match within tolerance
        league: League to prioritize (optional)
        country: Country of league (optional)
        contribution_tolerance: +/- range for contribution matching
        age_tolerance: +/- years for age matching
        max_results: Maximum number of similar players to return
        same_league_only: If True, only return players from the same league

    Returns:
        DataFrame with similar players, sorted by similarity score
    """
    # Compute player averages
    player_stats = df.groupby("player_id").agg({
        "player_name": "first",
        "position": lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
        "contribution": "mean",
        "age": "mean",
        "league": "first",
        "country": "first",
        "season": "count",
        "team": "first"
    }).reset_index()
    player_stats.columns = ["player_id", "player_name", "position", "avg_contribution",
                            "avg_age", "league", "country", "num_seasons", "team"]

    # Exclude the reference player
    candidates = player_stats[player_stats["player_id"] != player_id].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Apply filters
    if same_league_only and league and country:
        candidates = candidates[
            (candidates["league"] == league) &
            (candidates["country"] == country)
        ]

    if position and position != "Unknown":
        # Prefer same position, but include all for scoring
        candidates["position_match"] = (candidates["position"] == position).astype(int) * 3
    else:
        candidates["position_match"] = 0

    # Contribution similarity score (higher = more similar)
    if avg_contribution is not None:
        candidates["contrib_diff"] = abs(candidates["avg_contribution"] - avg_contribution)
        candidates["contrib_match"] = (
            candidates["contrib_diff"] <= contribution_tolerance
        ).astype(int) * 2
        # Bonus for very close matches
        candidates.loc[candidates["contrib_diff"] <= contribution_tolerance / 2, "contrib_match"] += 1
    else:
        candidates["contrib_match"] = 0
        candidates["contrib_diff"] = 0

    # Age similarity score
    if age is not None:
        candidates["age_diff"] = abs(candidates["avg_age"].fillna(99) - age)
        candidates["age_match"] = (candidates["age_diff"] <= age_tolerance).astype(int)
    else:
        candidates["age_match"] = 0
        candidates["age_diff"] = 0

    # League bonus (same league gets bonus)
    if league and country:
        candidates["league_match"] = (
            (candidates["league"] == league) &
            (candidates["country"] == country)
        ).astype(int)
    else:
        candidates["league_match"] = 0

    # Total similarity score
    candidates["similarity_score"] = (
        candidates["position_match"] +
        candidates["contrib_match"] +
        candidates["age_match"] +
        candidates["league_match"]
    )

    # Sort by similarity score, then by contribution difference
    candidates = candidates.sort_values(
        ["similarity_score", "contrib_diff"],
        ascending=[False, True]
    )

    # Add match reason description
    def describe_match(row):
        reasons = []
        if row["position_match"] >= 3:
            reasons.append("same position")
        if row["contrib_match"] >= 2:
            reasons.append(f"similar contribution (±{row['contrib_diff']:.2f})")
        if row["age_match"] >= 1:
            reasons.append(f"similar age (±{int(row['age_diff'])}y)")
        if row["league_match"] >= 1:
            reasons.append("same league")
        return ", ".join(reasons) if reasons else "general match"

    candidates["match_reason"] = candidates.apply(describe_match, axis=1)

    return candidates.head(max_results)[[
        "player_id", "player_name", "position", "avg_contribution",
        "avg_age", "league", "country", "num_seasons", "team",
        "similarity_score", "match_reason"
    ]]


# Preset comparisons for guided distribution analysis
COMPARISON_PRESETS = {
    "custom": {
        "name": "Custom Comparison",
        "description": "Configure your own filters",
        "distributions": None  # User configures manually
    },
    "top5_defenders_latest": {
        "name": "Top 5 Leagues - Defenders (Latest Season)",
        "description": "Compare defender contributions across Europe's top 5 leagues",
        "distributions": [
            {"label": "Premier League", "country": "England", "league": "Premier League", "positions": ["Defender"], "seasons": "latest"},
            {"label": "La Liga", "country": "Spain", "league": "La Liga", "positions": ["Defender"], "seasons": "latest"},
            {"label": "Bundesliga", "country": "Germany", "league": "Bundesliga", "positions": ["Defender"], "seasons": "latest"},
            {"label": "Serie A", "country": "Italy", "league": "Serie A", "positions": ["Defender"], "seasons": "latest"},
            {"label": "Ligue 1", "country": "France", "league": "Ligue 1", "positions": ["Defender"], "seasons": "latest"},
        ]
    },
    "top5_midfielders_latest": {
        "name": "Top 5 Leagues - Midfielders (Latest Season)",
        "description": "Compare midfielder contributions across Europe's top 5 leagues",
        "distributions": [
            {"label": "Premier League", "country": "England", "league": "Premier League", "positions": ["Midfielder"], "seasons": "latest"},
            {"label": "La Liga", "country": "Spain", "league": "La Liga", "positions": ["Midfielder"], "seasons": "latest"},
            {"label": "Bundesliga", "country": "Germany", "league": "Bundesliga", "positions": ["Midfielder"], "seasons": "latest"},
            {"label": "Serie A", "country": "Italy", "league": "Serie A", "positions": ["Midfielder"], "seasons": "latest"},
            {"label": "Ligue 1", "country": "France", "league": "Ligue 1", "positions": ["Midfielder"], "seasons": "latest"},
        ]
    },
    "top5_forwards_latest": {
        "name": "Top 5 Leagues - Forwards (Latest Season)",
        "description": "Compare forward contributions across Europe's top 5 leagues",
        "distributions": [
            {"label": "Premier League", "country": "England", "league": "Premier League", "positions": ["Forward"], "seasons": "latest"},
            {"label": "La Liga", "country": "Spain", "league": "La Liga", "positions": ["Forward"], "seasons": "latest"},
            {"label": "Bundesliga", "country": "Germany", "league": "Bundesliga", "positions": ["Forward"], "seasons": "latest"},
            {"label": "Serie A", "country": "Italy", "league": "Serie A", "positions": ["Forward"], "seasons": "latest"},
            {"label": "Ligue 1", "country": "France", "league": "Ligue 1", "positions": ["Forward"], "seasons": "latest"},
        ]
    },
    "league_evolution": {
        "name": "League Evolution Over Time",
        "description": "Compare a league's contribution distribution across multiple seasons",
        "distributions": None,  # Requires user to select league, then auto-generates season splits
        "type": "league_evolution"
    },
    "club_comparison": {
        "name": "Club vs Club",
        "description": "Compare two clubs within the same league",
        "distributions": None,  # Requires user to select league and two teams
        "type": "club_comparison"
    },
    "age_groups": {
        "name": "Age Group Comparison",
        "description": "Compare contributions by age group within a league",
        "distributions": None,  # Auto-generates age group splits
        "type": "age_groups",
        "age_ranges": [(17, 23, "U23"), (24, 28, "Prime"), (29, 33, "Veteran"), (34, 40, "Late Career")]
    },
    "position_comparison": {
        "name": "Position Comparison",
        "description": "Compare all positions within a selected league",
        "distributions": None,
        "type": "position_comparison"
    }
}


# =============================================================================
#                          STATISTICAL HELPERS
# =============================================================================

def compute_lowess_trend(x: pd.Series, y: pd.Series, frac: float = 0.3, num_points: int = 100):
    """
    Compute a LOWESS-like smoothed trend line using weighted local averaging.
    Returns x_smooth, y_smooth, y_lower, y_upper (for dispersion bands).
    """
    # Remove NaN values
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values

    if len(x_clean) < 10:
        return None, None, None, None

    # Sort by x
    sort_idx = np.argsort(x_clean)
    x_sorted = x_clean[sort_idx]
    y_sorted = y_clean[sort_idx]

    # Create evaluation points
    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), num_points)
    y_smooth = np.zeros(num_points)
    y_std = np.zeros(num_points)

    # Bandwidth based on fraction
    bandwidth = frac * (x_sorted.max() - x_sorted.min())

    for i, x_eval in enumerate(x_smooth):
        # Tricube weights based on distance
        distances = np.abs(x_sorted - x_eval)
        max_dist = max(bandwidth, np.sort(distances)[min(len(distances)-1, int(len(distances) * frac))])
        if max_dist == 0:
            max_dist = 1e-10

        # Tricube kernel
        u = distances / max_dist
        weights = np.where(u < 1, (1 - u**3)**3, 0)

        if weights.sum() > 0:
            y_smooth[i] = np.average(y_sorted, weights=weights)
            # Weighted standard deviation for dispersion
            variance = np.average((y_sorted - y_smooth[i])**2, weights=weights)
            y_std[i] = np.sqrt(variance)
        else:
            y_smooth[i] = np.nan
            y_std[i] = np.nan

    # Standard error bands (approx 1 std)
    y_lower = y_smooth - y_std
    y_upper = y_smooth + y_std

    return x_smooth, y_smooth, y_lower, y_upper


def compute_binned_stats(x: pd.Series, y: pd.Series, bins: int = 10):
    """
    Compute binned means and standard deviations for discrete/categorical-like variables.
    Returns bin_centers, means, lower_bounds, upper_bounds.
    """
    # Remove NaN values
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 10:
        return None, None, None, None

    # For Age, use integer bins
    if x_clean.dtype in [np.int64, np.int32] or (x_clean == x_clean.astype(int)).all():
        # Use actual age values as bins
        bin_edges = np.arange(x_clean.min() - 0.5, x_clean.max() + 1.5, 1)
    else:
        # Equal-width bins for other variables
        bin_edges = np.linspace(x_clean.min(), x_clean.max(), bins + 1)

    bin_centers = []
    means = []
    stds = []

    for i in range(len(bin_edges) - 1):
        mask_bin = (x_clean >= bin_edges[i]) & (x_clean < bin_edges[i+1])
        if mask_bin.sum() >= 3:  # Require at least 3 points per bin
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            means.append(y_clean[mask_bin].mean())
            stds.append(y_clean[mask_bin].std())

    if len(bin_centers) < 2:
        return None, None, None, None

    bin_centers = np.array(bin_centers)
    means = np.array(means)
    stds = np.array(stds)

    return bin_centers, means, means - stds, means + stds


# =============================================================================
#                          UI HELPERS
# =============================================================================

def render_metric_card(label: str, value: str, delta: str = None, help_text: str = None):
    """Render a metric card with optional delta and help text."""
    if help_text:
        st.metric(label=label, value=value, delta=delta, help=help_text)
    else:
        st.metric(label=label, value=value, delta=delta)


def render_summary_cards(df: pd.DataFrame, show_contribution: bool = True):
    """Render summary metric cards for a filtered dataset."""
    if df.empty:
        return

    cols = st.columns(4)
    with cols[0]:
        st.metric("Players", f"{df['player_id'].nunique():,}",
                  help="Unique players in selection")
    with cols[1]:
        st.metric("Records", f"{len(df):,}",
                  help="Player-season records")
    with cols[2]:
        seasons = df["season"].dropna()
        if not seasons.empty:
            st.metric("Seasons", f"{int(seasons.min())}-{int(seasons.max())}",
                      help="Season range covered")
    if show_contribution and "contribution" in df.columns:
        with cols[3]:
            avg_contrib = df["contribution"].mean()
            st.metric("Avg Contribution", f"{avg_contrib:.3f}",
                      help="Mean contribution score")


def render_sparkline(values: list, width: int = 100, height: int = 25, color: str = "#648FFF") -> str:
    """Generate inline SVG sparkline for a list of values."""
    if not values or len(values) < 2:
        return ""

    values = [v for v in values if v is not None and not pd.isna(v)]
    if len(values) < 2:
        return ""

    min_v, max_v = min(values), max(values)
    range_v = max_v - min_v if max_v != min_v else 1

    # Normalize to SVG coordinates
    points = []
    for i, v in enumerate(values):
        x = (i / (len(values) - 1)) * width
        y = height - ((v - min_v) / range_v) * height
        points.append(f"{x:.1f},{y:.1f}")

    path = "M" + " L".join(points)
    svg = f'''<svg width="{width}" height="{height}" style="display:inline-block;vertical-align:middle;">
        <path d="{path}" fill="none" stroke="{color}" stroke-width="1.5"/>
    </svg>'''
    return svg


def paginate_dataframe(df: pd.DataFrame, page_size: int = 25, key: str = "page") -> pd.DataFrame:
    """Add pagination controls and return the current page slice."""
    if len(df) <= page_size:
        return df

    total_pages = (len(df) - 1) // page_size + 1
    page_num = st.session_state.get(f"{key}_num", 0)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Previous", key=f"{key}_prev", disabled=page_num <= 0):
            page_num = max(0, page_num - 1)
            st.session_state[f"{key}_num"] = page_num
    with col2:
        st.caption(f"Page {page_num + 1} of {total_pages} ({len(df):,} total rows)")
    with col3:
        if st.button("Next →", key=f"{key}_next", disabled=page_num >= total_pages - 1):
            page_num = min(total_pages - 1, page_num + 1)
            st.session_state[f"{key}_num"] = page_num

    start_idx = page_num * page_size
    end_idx = min(start_idx + page_size, len(df))
    return df.iloc[start_idx:end_idx]


def export_buttons(df: pd.DataFrame, fig=None, prefix: str = "export"):
    """Render export buttons for CSV and PNG."""
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"{prefix}.csv",
            "text/csv",
            key=f"{prefix}_csv"
        )
    with col2:
        if fig is not None:
            # Plotly figure export
            img_bytes = fig.to_image(format="png", scale=2)
            st.download_button(
                "🖼️ Download PNG",
                img_bytes,
                f"{prefix}.png",
                "image/png",
                key=f"{prefix}_png"
            )


def render_empty_state(message: str = "No data available", suggestion: str = None):
    """Render a friendly empty state message."""
    st.markdown(f"""
    <div style="text-align: center; padding: 40px; color: #666;">
        <div style="font-size: 48px; margin-bottom: 10px;">📊</div>
        <div style="font-size: 18px; font-weight: 500;">{message}</div>
        {f'<div style="font-size: 14px; margin-top: 8px; color: #888;">{suggestion}</div>' if suggestion else ''}
    </div>
    """, unsafe_allow_html=True)


def render_breadcrumbs(items: list):
    """Render breadcrumb navigation."""
    crumbs = " › ".join([f"**{item}**" if i == len(items) - 1 else item
                         for i, item in enumerate(items)])
    st.markdown(crumbs)


def get_smart_defaults(df: pd.DataFrame) -> dict:
    """Get smart default filter values based on data distribution."""
    defaults = {}

    # Default to most recent season
    if "season" in df.columns:
        defaults["season"] = int(df["season"].max())

    # Default to league with most data
    if "league" in df.columns:
        top_league = df["league"].value_counts().idxmax()
        defaults["league"] = top_league

    # Default country based on top league
    if "country" in df.columns and "league" in df.columns:
        if defaults.get("league"):
            country = df[df["league"] == defaults["league"]]["country"].mode()
            defaults["country"] = country.iloc[0] if len(country) > 0 else None

    return defaults


def filter_section(title: str, expanded: bool = True):
    """Create a collapsible filter section (returns expander)."""
    return st.expander(title, expanded=expanded)


# =============================================================================
#                          PAGES
# =============================================================================

def page_scatter_analysis():
    """Scatter analysis page with statistical overlays."""
    st.header("Contribution Analysis")

    with st.spinner("Loading player data..."):
        df = load_all_players(min_fte=0)
    unique = get_unique_values(df)

    # === FILTERS IN SIDEBAR ===
    with st.sidebar:
        st.subheader("Filters")

        # Location filters (collapsible)
        with st.expander("📍 Location", expanded=True):
            selected_country = st.selectbox(
                "Country", ["All"] + unique["countries"],
                key="scatter_country",
                help=f"{len(unique['countries'])} countries available"
            )

            # Cascade: leagues depend on country
            if selected_country != "All":
                available_leagues = get_leagues_for_country(selected_country)
            else:
                available_leagues = unique["leagues"]
            selected_league = st.selectbox(
                "League", ["All"] + available_leagues,
                key="scatter_league",
                help=f"{len(available_leagues)} leagues available"
            )

            # Cascade: seasons depend on country/league
            if selected_country != "All" or selected_league != "All":
                available_seasons = get_seasons_for_league(
                    league=selected_league if selected_league != "All" else None,
                    country=selected_country if selected_country != "All" else None
                )
            else:
                available_seasons = unique["seasons"]
            selected_season = st.selectbox(
                "Season", ["All"] + [str(s) for s in available_seasons],
                key="scatter_season",
                help=f"{len(available_seasons)} seasons available"
            )

            # Cascade: teams depend on country/league/season
            season_filter = int(selected_season) if selected_season != "All" else None
            available_teams = get_teams_for_league_season(
                league=selected_league if selected_league != "All" else None,
                country=selected_country if selected_country != "All" else None,
                season=season_filter
            )
            selected_teams = st.multiselect(
                "Teams", ["All"] + available_teams, default=["All"],
                key="scatter_teams",
                help=f"{len(available_teams)} teams available"
            )

        # Player filters (collapsible)
        with st.expander("👤 Player", expanded=True):
            selected_positions = st.multiselect(
                "Positions",
                ["Goalkeeper", "Defender", "Midfielder", "Forward", "Unknown"],
                default=["Goalkeeper", "Defender", "Midfielder", "Forward"],
                key="scatter_positions"
            )

            nationality_options = ["All"] + [n for n in unique["nationalities"] if n != "Unknown"]
            selected_nationalities = st.multiselect(
                "Nationalities", nationality_options, default=["All"],
                key="scatter_nat",
                help=f"{len(nationality_options)-1} nationalities available"
            )

            age_range = st.slider("Age Range", 15, 45, (15, 45), key="scatter_age")
            min_fte = st.slider(
                "Minimum FTE Games", 0.0, 30.0, 5.0, 0.5,
                key="scatter_fte",
                help="Full-Time Equivalent games: total minutes played ÷ 90. "
                     "E.g., 5 FTE = ~450 minutes played (5 full matches worth)."
            )

        # Chart options (collapsible)
        with st.expander("📊 Chart Options", expanded=False):
            x_axis_options = {
                "FTE Games": "FTE_games_played",
                "Age": "age",
                "Height (cm)": "height_cm",
                "Weight (kg)": "weight_kg",
                "BMI": "bmi"
            }
            x_axis_label = st.selectbox("X-Axis Variable", list(x_axis_options.keys()), key="scatter_xaxis")
            x_axis_col = x_axis_options[x_axis_label]
            dot_size = st.slider("Dot Size", 2, 20, 6, key="scatter_dot")

            st.markdown("**Statistical Overlays**")
            show_trend = st.checkbox(
                "Show Mean Trend", value=False,
                help="LOWESS smoothed mean for continuous variables, binned means for Age",
                key="scatter_trend"
            )
            show_dispersion = st.checkbox(
                "Show Dispersion (±1 SD)", value=False,
                help="Standard deviation bands around mean trend",
                key="scatter_disp"
            )
            show_by_position = st.checkbox(
                "Trend by Position", value=False,
                help="Show separate trend lines for each selected position",
                key="scatter_bypos"
            )

            st.markdown("**Data Transformation**")
            demean_player_league = st.checkbox(
                "Demean by Player-League", value=False,
                help="For each player-league combination, subtract the mean contribution. "
                     "Useful for studying age effects while controlling for player-league baseline performance.",
                key="scatter_demean"
            )

    if show_trend or show_dispersion:
        st.caption("ℹ️ Statistics are computed from all filtered data. Clicking legend entries to hide dots only affects visibility, not the calculations.")

    # === Filter data ===
    filtered = df[df["FTE_games_played"] >= min_fte].copy()

    if selected_country != "All":
        filtered = filtered[filtered["country"] == selected_country]
    if selected_league != "All":
        filtered = filtered[filtered["league"] == selected_league]
    if selected_season != "All":
        filtered = filtered[filtered["season"] == int(selected_season)]
    # Team filter (handles "Multiple Teams (A, B)" format)
    if "All" not in selected_teams and selected_teams:
        team_mask = filtered["team"].apply(
            lambda t: any(team in str(t) for team in selected_teams) if pd.notna(t) else False
        )
        filtered = filtered[team_mask]
    if selected_positions:
        filtered = filtered[filtered["position"].isin(selected_positions)]
    if "All" not in selected_nationalities and selected_nationalities:
        filtered = filtered[filtered["nationality"].isin(selected_nationalities)]
    if age_range != (15, 45):
        filtered = filtered[
            (filtered["age"].notna()) &
            (filtered["age"] >= age_range[0]) &
            (filtered["age"] <= age_range[1])
        ]

    # Filter out rows with missing x-axis values (for non-FTE axes)
    if x_axis_col != "FTE_games_played":
        filtered = filtered[filtered[x_axis_col].notna()]

    # Summary cards
    render_summary_cards(filtered)

    # Scatter plot
    if filtered.empty:
        render_empty_state(
            "No players match your filters",
            "Try adjusting the filters in the sidebar"
        )
        return

    if not filtered.empty:
        # Add standardized hover text
        filtered["hover_text"] = make_player_hover(filtered)

        # Demean by player-league if requested
        y_col = "contribution"
        y_label = "Lasso Contribution (goals/90)"

        if demean_player_league:
            # Compute player-league mean for each player-league combination
            player_league_means = filtered.groupby(["player_id", "league"])["contribution"].transform("mean")
            filtered["contribution_demeaned"] = filtered["contribution"] - player_league_means
            y_col = "contribution_demeaned"
            y_label = "Demeaned Contribution (goals/90, by player-league)"

        # Use fixed color mapping for positions
        fig = px.scatter(
            filtered,
            x=x_axis_col,
            y=y_col,
            color="position",
            color_discrete_map=POSITION_COLORS,
            category_orders={"position": ["Goalkeeper", "Defender", "Midfielder", "Forward", "Unknown"]},
            custom_data=["hover_text"],
            title=f"Player Contributions ({len(filtered)} players)",
            labels={
                "FTE_games_played": "FTE Games Played",
                "age": "Age",
                "height_cm": "Height (cm)",
                "weight_kg": "Weight (kg)",
                "bmi": "BMI",
                "contribution": "Lasso Contribution (goals/90)",
                "contribution_demeaned": y_label,
                "position": "Position"
            }
        )

        # Use standardized hover template
        fig.update_traces(
            marker=dict(size=dot_size),
            hovertemplate="%{customdata[0]}<extra></extra>"
        )

        # === Add statistical overlays ===
        if show_trend or show_dispersion:
            use_binned = x_axis_col == "age"  # Use binned stats for Age

            if show_by_position and len(selected_positions) > 1:
                # Separate trend lines per position
                for position in selected_positions:
                    if position == "Unknown":
                        continue
                    pos_data = filtered[filtered["position"] == position]
                    if len(pos_data) < 10:
                        continue

                    if use_binned:
                        x_trend, y_trend, y_lower, y_upper = compute_binned_stats(
                            pos_data[x_axis_col], pos_data[y_col]
                        )
                    else:
                        x_trend, y_trend, y_lower, y_upper = compute_lowess_trend(
                            pos_data[x_axis_col], pos_data[y_col]
                        )

                    if x_trend is not None:
                        color = POSITION_COLORS.get(position, "#7f7f7f")

                        # Dispersion band (fill)
                        if show_dispersion:
                            fig.add_trace(go.Scatter(
                                x=np.concatenate([x_trend, x_trend[::-1]]),
                                y=np.concatenate([y_upper, y_lower[::-1]]),
                                fill="toself",
                                fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in color else f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                                name=f"{position} ±1 SD"
                            ))

                        # Mean trend line - thicker with markers for visibility
                        if show_trend:
                            fig.add_trace(go.Scatter(
                                x=x_trend,
                                y=y_trend,
                                mode="lines+markers",
                                line=dict(color=color, width=4),
                                marker=dict(size=8, symbol="circle", line=dict(width=1, color="white")),
                                name=f"{position} mean",
                                hovertemplate=f"{position}<br>{x_axis_label}: %{{x:.1f}}<br>Mean: %{{y:.3f}}<extra></extra>"
                            ))
            else:
                # Single overall trend line
                if use_binned:
                    x_trend, y_trend, y_lower, y_upper = compute_binned_stats(
                        filtered[x_axis_col], filtered[y_col]
                    )
                else:
                    x_trend, y_trend, y_lower, y_upper = compute_lowess_trend(
                        filtered[x_axis_col], filtered[y_col]
                    )

                if x_trend is not None:
                    # Dispersion band (fill) - purple/magenta tint
                    if show_dispersion:
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([x_trend, x_trend[::-1]]),
                            y=np.concatenate([y_upper, y_lower[::-1]]),
                            fill="toself",
                            fillcolor="rgba(148, 0, 211, 0.15)",  # Dark violet with transparency
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                            name="±1 SD"
                        ))

                    # Mean trend line - dark magenta/purple for high contrast
                    if show_trend:
                        fig.add_trace(go.Scatter(
                            x=x_trend,
                            y=y_trend,
                            mode="lines+markers",
                            line=dict(color="#8B008B", width=4),  # Dark magenta
                            marker=dict(size=8, symbol="diamond", color="#8B008B",
                                       line=dict(width=1, color="white")),
                            name="Mean trend",
                            hovertemplate=f"{x_axis_label}: %{{x:.1f}}<br>Mean: %{{y:.3f}}<extra></extra>"
                        ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=500, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        # Top/Bottom tables
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Contributors")
            top = filtered.nlargest(10, "contribution")[
                ["player_name", "contribution", "FTE_games_played", "league", "season"]
            ]
            st.dataframe(top, hide_index=True)

        with col2:
            st.subheader("Bottom Contributors")
            bottom = filtered.nsmallest(10, "contribution")[
                ["player_name", "contribution", "FTE_games_played", "league", "season"]
            ]
            st.dataframe(bottom, hide_index=True)

        # Export section
        st.markdown("---")
        with st.expander("📥 Export Data"):
            export_df = filtered[["player_name", "position", "team", "league", "season",
                                  "contribution", "FTE_games_played", "goals", "assists"]].copy()
            csv = export_df.to_csv(index=False)
            st.download_button(
                "Download Filtered Data (CSV)",
                csv,
                "contribution_analysis.csv",
                "text/csv",
                key="scatter_export"
            )


# Alias for backward compatibility - use normalize_player_id instead
_normalize_player_id = normalize_player_id


def page_player_career():
    """Player career exploration with optional comparison."""
    st.header("Player Career")

    with st.spinner("Loading player data..."):
        df = load_all_players(min_fte=0)

    if df.empty:
        render_empty_state("No player data available")
        return

    # Player search with sidebar option
    with st.sidebar:
        st.subheader("🔍 Player Search")
        search = st.text_input("Search player name", "", key="career_search",
                              help="Type a player name to search")

    # Helper function to search and select players
    def search_and_select_player(search_term: str, key_prefix: str, container=st):
        """Search for players and return selected player_id and name."""
        if not search_term:
            return None, None, None

        matches = df[df["player_name"].str.contains(search_term, case=False, na=False)]
        players = matches.groupby("player_id").agg({
            "player_name": "first",
            "season": "count",
            "team": lambda x: ", ".join(sorted(set(x.dropna())))[:50],
            "nationality": "first"
        }).reset_index()
        players.columns = ["player_id", "player_name", "seasons", "teams", "nationality"]
        players = players.sort_values("seasons", ascending=False)

        if players.empty:
            container.warning(f"No players found matching '{search_term}'")
            return None, None, None

        # Build options
        options = []
        for _, row in players.head(15).iterrows():
            teams_short = row["teams"][:25] + "..." if len(str(row["teams"])) > 25 else row["teams"]
            nat = row["nationality"] if row["nationality"] != "Unknown" else ""
            opt = f"{row['player_name']} ({row['seasons']}s) - {teams_short}"
            if nat:
                opt += f" [{nat}]"
            options.append((opt, row["player_id"], row["player_name"]))

        idx = container.selectbox(
            "Select player",
            range(len(options)),
            format_func=lambda i: options[i][0],
            key=f"{key_prefix}_select"
        )

        if idx is not None:
            return options[idx][1], options[idx][2], players
        return None, None, players

    if not search:
        render_empty_state(
            "Search for a player to view their career",
            "Enter a player name in the search box"
        )
        return

    # Primary player selection
    selected_player_id, player_name, player_matches = search_and_select_player(search, "p1")

    if selected_player_id is None:
        return

    player_data = df[df["player_id"] == selected_player_id].sort_values("season")

    if player_data.empty:
        render_empty_state("No data for selected player")
        return

    nationality = player_data["nationality"].iloc[0]
    position = player_data["position"].mode().iloc[0] if not player_data["position"].mode().empty else "Unknown"

    # Comparison mode - add second player search in sidebar
    with st.sidebar:
        st.markdown("---")
        compare_enabled = st.checkbox("⚖️ Compare with another player", key="compare_mode")

        player2_id = None
        player2_name = None

        if compare_enabled:
            search2 = st.text_input("Search second player", key="compare_search2")
            if search2:
                player2_id, player2_name, _ = search_and_select_player(search2, "p2", st.sidebar)

                # Similar player suggestions
                if player2_id is None:
                    with st.expander("💡 Suggest similar players"):
                        p1_position = position
                        p1_avg_contrib = player_data["contribution"].mean()
                        p1_avg_age = player_data["age"].mean() if "age" in player_data.columns else None
                        p1_league = player_data["league"].iloc[0]
                        p1_country = player_data["country"].iloc[0]

                        same_league = st.checkbox("Same league only", value=True, key="suggest_league")
                        contrib_tol = st.slider("Contrib. tolerance", 0.05, 0.30, 0.15, 0.05, key="suggest_tol")

                        similar = find_similar_players(
                            df, player_id=selected_player_id, position=p1_position,
                            avg_contribution=p1_avg_contrib, age=p1_avg_age,
                            league=p1_league, country=p1_country,
                            contribution_tolerance=contrib_tol,
                            same_league_only=same_league, max_results=5
                        )

                        if not similar.empty:
                            for _, row in similar.iterrows():
                                if st.button(f"{row['player_name']} ({row['position']})", key=f"sug_{row['player_id']}"):
                                    st.session_state["compare_search2"] = row["player_name"]
                                    st.rerun()

    # Breadcrumb navigation
    if compare_enabled and player2_id:
        render_breadcrumbs(["Player Career", f"{player_name} vs {player2_name}"])
    else:
        render_breadcrumbs(["Player Career", player_name])

    # =========================================================================
    # COMPARISON MODE: Show both players side by side
    # =========================================================================
    if compare_enabled and player2_id:
        player2_data = df[df["player_id"] == player2_id].sort_values("season")

        # Side-by-side career metrics
        st.markdown("### Career Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{player_name}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Seasons", len(player_data))
            c2.metric("Avg Contrib", f"{player_data['contribution'].mean():.1f}")
            c3.metric("Goals", int(player_data['goals'].sum()))
            c4.metric("Assists", int(player_data['assists'].sum()))

        with col2:
            st.markdown(f"**{player2_name}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Seasons", len(player2_data))
            c2.metric("Avg Contrib", f"{player2_data['contribution'].mean():.1f}")
            c3.metric("Goals", int(player2_data['goals'].sum()))
            c4.metric("Assists", int(player2_data['assists'].sum()))

        # Comparison chart options
        st.markdown("### Career Trajectory Comparison")
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            x_axis_option = st.radio("X-Axis:", ["Season", "Age"], horizontal=True, key="compare_x_axis")
        with opt_col2:
            y_axis_option = st.selectbox("Y-Axis:", ["Contribution", "Goals", "Assists", "Team Rank"], key="compare_y_axis")

        y_col_map = {
            "Contribution": ("contribution", "Contribution (goals/90)"),
            "Goals": ("goals", "Goals"),
            "Assists": ("assists", "Assists"),
            "Team Rank": ("team_rank", "Team Final Position")
        }
        y_col, y_label = y_col_map[y_axis_option]
        reverse_y = y_axis_option == "Team Rank"

        fig = go.Figure()

        def build_hover_data(pdata: pd.DataFrame) -> pd.DataFrame:
            hd = pdata[["team", "league", "country", "position", "season", "goals", "assists", "FTE_games_played", "team_rank", "contribution"]].copy()
            hd["goals"] = hd["goals"].fillna(0).astype(int)
            hd["assists"] = hd["assists"].fillna(0).astype(int)
            hd["FTE_games_played"] = hd["FTE_games_played"].fillna(0)
            hd["team_rank"] = pd.to_numeric(hd["team_rank"], errors="coerce").fillna(0).astype(int)
            return hd

        hover_template = (
            "<b>%{customdata[0]}</b> (%{customdata[9]:+.1f})<br>"
            "%{customdata[3]}, %{customdata[1]}<br>"
            "%{customdata[2]}, Season %{customdata[4]:.0f}<br>"
            "%{customdata[5]} goals, %{customdata[6]} assists<br>"
            "%{customdata[7]:.1f} FTE games, Rank: %{customdata[8]}"
            "<extra></extra>"
        )

        if x_axis_option == "Age" and "age" in player_data.columns and "age" in player2_data.columns:
            p1_ages = player_data[player_data["age"].notna()].copy()
            p2_ages = player2_data[player2_data["age"].notna()].copy()

            if not p1_ages.empty and not p2_ages.empty:
                hd1, hd2 = build_hover_data(p1_ages), build_hover_data(p2_ages)
                fig.add_trace(go.Scatter(x=p1_ages["age"], y=p1_ages[y_col], mode="lines+markers",
                    name=player_name, line=dict(color="#648FFF", width=3), marker=dict(size=10),
                    customdata=hd1.values, hovertemplate=hover_template))
                fig.add_trace(go.Scatter(x=p2_ages["age"], y=p2_ages[y_col], mode="lines+markers",
                    name=player2_name, line=dict(color="#DC267F", width=3), marker=dict(size=10),
                    customdata=hd2.values, hovertemplate=hover_template))
                x_title, title = "Age", f"{y_axis_option} by Age"
            else:
                st.warning("Age data not available. Showing by season.")
                x_axis_option = "Season"

        if x_axis_option == "Season":
            hd1, hd2 = build_hover_data(player_data), build_hover_data(player2_data)
            fig.add_trace(go.Scatter(x=player_data["season"], y=player_data[y_col], mode="lines+markers",
                name=player_name, line=dict(color="#648FFF", width=3), marker=dict(size=10),
                customdata=hd1.values, hovertemplate=hover_template))
            fig.add_trace(go.Scatter(x=player2_data["season"], y=player2_data[y_col], mode="lines+markers",
                name=player2_name, line=dict(color="#DC267F", width=3), marker=dict(size=10),
                customdata=hd2.values, hovertemplate=hover_template))
            x_title, title = "Season", f"{y_axis_option} Over Time"

        if y_axis_option == "Contribution":
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_label, height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        if reverse_y:
            fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, use_container_width=True)

        # Season details side by side
        st.markdown("### Season Details")
        col1, col2 = st.columns(2)

        def format_career_table(pdata: pd.DataFrame) -> pd.DataFrame:
            """Format career table with 1 decimal for contribution/FTE."""
            display = pdata[["season", "age", "team", "league", "contribution", "FTE_games_played", "goals", "assists", "team_rank"]].copy()
            display["contribution"] = display["contribution"].round(1)
            display["FTE_games_played"] = display["FTE_games_played"].round(1)
            display = display.rename(columns={"team_rank": "Rank", "FTE_games_played": "FTE"})
            return display

        with col1:
            st.markdown(f"**{player_name}**")
            st.dataframe(format_career_table(player_data), hide_index=True, height=300)
        with col2:
            st.markdown(f"**{player2_name}**")
            st.dataframe(format_career_table(player2_data), hide_index=True, height=300)

    # =========================================================================
    # SINGLE PLAYER MODE: Detailed career view
    # =========================================================================
    else:
        # Analyze career gaps
        career_gaps = analyze_career_gaps(player_data["season"].tolist())

        # Player header with photo and name prominently displayed
        photo_url = get_player_photo(selected_player_id)

        col_photo, col_info = st.columns([1, 4])
        with col_photo:
            if photo_url:
                st.image(photo_url, width=120)
            else:
                st.markdown("👤")  # Placeholder if no photo

        with col_info:
            st.markdown(f"## {player_name}")
            info_parts = []
            if position and position != "Unknown":
                info_parts.append(position)
            if nationality and nationality != "Unknown":
                info_parts.append(nationality)
            if info_parts:
                st.markdown(" · ".join(info_parts))

        st.markdown("---")

        # Career stats in a styled container
        st.markdown("### Career Overview")

        # Show missing seasons warning if any
        if career_gaps["missing_count"] > 0:
            missing_str = ", ".join(str(s) for s in career_gaps["missing_seasons"][:5])
            if career_gaps["missing_count"] > 5:
                missing_str += f"... (+{career_gaps['missing_count'] - 5} more)"
            st.warning(f"⚠️ Data gaps detected: Missing seasons {missing_str}")

        col1, col2, col3, col4, col5 = st.columns(5)

        seasons_label = f"{career_gaps['total_seasons']}"
        if career_gaps["missing_count"] > 0:
            seasons_help = f"{career_gaps['first_season']}-{career_gaps['last_season']}, {career_gaps['missing_count']} missing ({career_gaps['coverage_pct']}% coverage)"
        else:
            seasons_help = f"{career_gaps['first_season']}-{career_gaps['last_season']} (complete)"
        col1.metric("Seasons", seasons_label, help=seasons_help)
        col2.metric("Avg Contribution", f"{player_data['contribution'].mean():.1f}", help="Average contribution per season")
        col3.metric("Total Goals", int(player_data["goals"].sum()))
        col4.metric("Total Assists", int(player_data["assists"].sum()))
        if nationality != "Unknown":
            col5.metric("Nationality", nationality)
        else:
            col5.metric("Position", position)

        # Chart options in sidebar
        with st.sidebar:
            with st.expander("📊 Chart Options", expanded=True):
                secondary_y_options = ["None", "Goals", "Assists", "Goals + Assists", "Team Rank"]
                secondary_y = st.selectbox("Secondary Y-Axis", secondary_y_options, index=3, key="career_secondary_y")
                show_transfers = st.checkbox("Show Transfers", value=True, key="career_transfers")

        # Load transfers if needed
        transfers = pd.DataFrame()
        if show_transfers:
            transfers = load_player_transfers(selected_player_id)

        # Build the figure with potential secondary Y-axis
        from plotly.subplots import make_subplots
        if secondary_y != "None":
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        else:
            fig = go.Figure()

        # Primary trace: Contribution
        hover_data = player_data[["team", "league", "country", "position", "goals", "assists", "FTE_games_played", "team_rank"]].copy()
        hover_data["goals"] = hover_data["goals"].fillna(0).astype(int)
        hover_data["assists"] = hover_data["assists"].fillna(0).astype(int)
        hover_data["FTE_games_played"] = hover_data["FTE_games_played"].fillna(0)
        hover_data["team_rank"] = pd.to_numeric(hover_data["team_rank"], errors="coerce").fillna(0).astype(int)

        fig.add_trace(
            go.Scatter(
                x=player_data["season"], y=player_data["contribution"],
                mode="lines+markers", name="Contribution",
                line=dict(width=3, color="#648FFF"), marker=dict(size=10),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{y:+.1f})<br>"
                    "%{customdata[3]}, %{customdata[1]}<br>"
                    "%{customdata[2]}, Season %{x}<br>"
                    "%{customdata[4]} goals, %{customdata[5]} assists<br>"
                    "%{customdata[6]:.1f} FTE games, Rank: %{customdata[7]}<extra></extra>"
                ),
                customdata=hover_data.values
            ),
            secondary_y=False if secondary_y != "None" else None
        )

        # Secondary trace if selected
        if secondary_y != "None":
            reverse_secondary_y = False
            if secondary_y == "Goals":
                y_data, trace_name, color = player_data["goals"], "Goals", "#FFB000"
            elif secondary_y == "Assists":
                y_data, trace_name, color = player_data["assists"], "Assists", "#DC267F"
            elif secondary_y == "Team Rank":
                y_data, trace_name, color = player_data["team_rank"], "Team Rank", "#FE6100"
                reverse_secondary_y = True
            else:
                y_data, trace_name, color = player_data["goals"] + player_data["assists"], "G+A", "#785EF0"

            fig.add_trace(
                go.Scatter(x=player_data["season"], y=y_data, mode="lines+markers", name=trace_name,
                    line=dict(width=2, dash="dot", color=color), marker=dict(size=7, symbol="diamond"),
                    hovertemplate=f"{trace_name}: %{{y:.0f}}<extra></extra>"),
                secondary_y=True
            )
            if reverse_secondary_y:
                fig.update_yaxes(autorange="reversed", secondary_y=True)

        # Add transfer markers
        if show_transfers and not transfers.empty:
            min_season, max_season = player_data["season"].min(), player_data["season"].max()
            for _, tr in transfers.iterrows():
                tr_year = tr["year"]
                if min_season <= tr_year <= max_season + 1:
                    fig.add_vline(x=tr_year, line_dash="dash", line_color="rgba(128,128,128,0.5)", line_width=1)
                    label = tr["label"] if tr["label"] else "Transfer"
                    annotation_text = f"→ {tr['to_team_name'][:15]}"
                    if label and label not in ["Transfer", "N/A"]:
                        annotation_text += f"<br><i>{label}</i>"
                    fig.add_annotation(x=tr_year, y=1.02, yref="paper", text=annotation_text,
                        showarrow=False, font=dict(size=18, color="gray"), textangle=-45, xanchor="left")

        # Add horizontal lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=player_data["contribution"].mean(), line_dash="dot", line_color="#FFB000",
            annotation_text=f"avg: {player_data['contribution'].mean():.1f}")

        # Layout
        layout_args = dict(title=f"{player_name} - Career", xaxis_title="Season", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        if secondary_y != "None":
            fig.update_yaxes(title_text="Lasso Contribution (goals/90)", secondary_y=False)
            fig.update_yaxes(title_text=secondary_y, secondary_y=True)
        else:
            layout_args["yaxis_title"] = "Lasso Contribution (goals/90)"

        fig.update_layout(**layout_args)
        st.plotly_chart(fig, use_container_width=True)

        # Transfer history table
        if show_transfers and not transfers.empty:
            with st.expander("Transfer History", expanded=False):
                st.dataframe(transfers[["transfer_date", "from_team_name", "to_team_name", "type"]].rename(columns={
                    "transfer_date": "Date", "from_team_name": "From", "to_team_name": "To", "type": "Type/Fee"
                }), hide_index=True)

        # Season details table
        st.subheader("Season Details")
        display_cols = ["season", "age", "team", "league", "contribution", "FTE_games_played", "goals", "assists", "position"]
        display_df = player_data[display_cols].copy()
        display_df["contribution"] = display_df["contribution"].round(1)
        display_df["FTE_games_played"] = display_df["FTE_games_played"].round(1)
        display_df = display_df.rename(columns={"FTE_games_played": "FTE"})
        st.dataframe(display_df, hide_index=True)

        # Export section
        st.markdown("---")
        with st.expander("📥 Export Data"):
            csv = player_data[display_cols].to_csv(index=False)
            st.download_button(f"Download {player_name}'s Career Data (CSV)", csv,
                f"career_{player_name.replace(' ', '_')}.csv", "text/csv", key="career_export")


# =============================================================================
#                          TEAM ANALYSIS
# =============================================================================

def page_team_analysis():
    """Analyze a team's player contributions compared to the rest of the league."""
    st.header("Team Analysis")

    st.markdown("Compare a team's players against the rest of the league, position by position.")

    with st.spinner("Loading player data..."):
        df = load_all_players(min_fte=3.0)

    if df.empty:
        render_empty_state("No player data available")
        return

    # Build league options with country prefix (public leagues only)
    public_leagues = get_public_leagues()
    country_league_pairs = df[["country", "league"]].drop_duplicates().dropna()
    # Filter to public leagues only
    country_league_pairs = country_league_pairs[
        country_league_pairs.apply(lambda r: (r["country"], r["league"]) in public_leagues, axis=1)
    ]
    league_display_list = sorted([f"{row['country']} - {row['league']}" for _, row in country_league_pairs.iterrows()])
    display_to_country_league = {
        f"{row['country']} - {row['league']}": (row['country'], row['league'])
        for _, row in country_league_pairs.iterrows()
    }

    # Sidebar controls
    with st.sidebar:
        st.subheader("🏆 Select League & Team")

        # League selection
        selected_display = st.selectbox("League", league_display_list, key="team_league")
        if not selected_display:
            return

        country, league = display_to_country_league[selected_display]

        # Filter to selected league
        league_df = df[(df["country"] == country) & (df["league"] == league)]

        # Season selection
        seasons = sorted(league_df["season"].dropna().unique(), reverse=True)
        if not seasons:
            st.warning("No seasons available for this league")
            return

        selected_season = st.selectbox("Season", seasons, key="team_season")
        season_df = league_df[league_df["season"] == selected_season]

        # Team selection
        teams = sorted(season_df["team"].dropna().unique())
        if not teams:
            st.warning("No teams available for this season")
            return

        selected_team = st.selectbox("Team", teams, key="team_select")

        st.markdown("---")
        st.subheader("⚙️ Options")
        min_fte = st.slider("Min FTE Games", 0.0, 20.0, 3.0, 0.5, key="team_min_fte",
                           help="Minimum full-time equivalent games played")
        show_player_names = st.checkbox("Show player names on hover", value=True, key="team_show_names")

    # Filter data
    season_df = season_df[season_df["FTE_games_played"] >= min_fte]
    team_df = season_df[season_df["team"] == selected_team]
    rest_df = season_df[season_df["team"] != selected_team]

    if team_df.empty:
        render_empty_state(f"No players found for {selected_team} with >= {min_fte} FTE games")
        return

    # Breadcrumb
    render_breadcrumbs(["Team Analysis", f"{selected_team} ({selected_season})"])

    # Team header with logo and name prominently displayed
    logo_url = get_team_logo(selected_team, league)

    col_logo, col_info = st.columns([1, 4])
    with col_logo:
        if logo_url:
            st.image(logo_url, width=100)
        else:
            st.markdown("🏟️")  # Placeholder if no logo

    with col_info:
        st.markdown(f"## {selected_team}")
        st.markdown(f"{league} · {selected_season}")

    # Team overview metrics
    st.markdown("### Overview")
    col1, col2, col3, col4, col5 = st.columns(5)

    team_avg = team_df["contribution"].mean()
    league_avg = season_df["contribution"].mean()
    diff = team_avg - league_avg

    col1.metric("Team Avg Contribution", f"{team_avg:.1f}",
               delta=f"{diff:+.1f} vs league", delta_color="normal")
    col2.metric("Players", len(team_df))
    col3.metric("Total Goals", int(team_df["goals"].sum()))
    col4.metric("Total Assists", int(team_df["assists"].sum()))

    # Team rank if available
    team_rank = team_df["team_rank"].iloc[0] if "team_rank" in team_df.columns and not team_df["team_rank"].isna().all() else None
    if team_rank and str(team_rank).isdigit():
        col5.metric("Final Position", f"#{int(team_rank)}")
    else:
        col5.metric("League Avg", f"{league_avg:.1f}")

    # Position mapping - consolidate to main categories
    def map_position(pos):
        if pd.isna(pos):
            return "Unknown"
        pos = str(pos).upper()
        if "GOAL" in pos or pos == "GK" or pos == "G":
            return "Goalkeeper"
        elif "DEF" in pos or pos in ["CB", "LB", "RB", "LWB", "RWB", "D"]:
            return "Defender"
        elif "MID" in pos or pos in ["CM", "DM", "AM", "LM", "RM", "CDM", "CAM", "M"]:
            return "Midfielder"
        elif "ATT" in pos or "FOR" in pos or pos in ["ST", "CF", "LW", "RW", "F", "A"]:
            return "Attacker"
        return "Unknown"

    team_df = team_df.copy()
    rest_df = rest_df.copy()
    team_df["position_group"] = team_df["position"].apply(map_position)
    rest_df["position_group"] = rest_df["position"].apply(map_position)

    # Filter out unknown positions for cleaner visualization
    team_df = team_df[team_df["position_group"] != "Unknown"]
    rest_df = rest_df[rest_df["position_group"] != "Unknown"]

    position_order = ["Goalkeeper", "Defender", "Midfielder", "Attacker"]

    # Create boxplot comparison
    st.markdown("### Contribution by Position")
    st.caption("Compare how each position group performs relative to the rest of the league")

    fig = go.Figure()

    # Color scheme
    team_color = "#648FFF"  # Blue for selected team
    league_color = "#DC267F"  # Magenta for rest of league

    # Track if we've added the first trace for each group (for legend)
    first_league = True
    first_team = True

    for pos in position_order:
        # Rest of league data for this position
        rest_pos = rest_df[rest_df["position_group"] == pos]["contribution"]
        if not rest_pos.empty:
            fig.add_trace(go.Box(
                x=[pos] * len(rest_pos),
                y=rest_pos,
                name="Rest of League",
                legendgroup="league",
                showlegend=first_league,
                marker_color=league_color,
                boxmean=True,
                offsetgroup="league",
                hoverinfo="y+name" if not show_player_names else "all"
            ))
            first_league = False

        # Team data for this position
        team_pos = team_df[team_df["position_group"] == pos]
        if not team_pos.empty:
            hover_text = team_pos["player_name"].tolist() if show_player_names else None
            fig.add_trace(go.Box(
                x=[pos] * len(team_pos),
                y=team_pos["contribution"],
                name=selected_team,
                legendgroup="team",
                showlegend=first_team,
                marker_color=team_color,
                boxmean=True,
                offsetgroup="team",
                text=hover_text,
                hovertemplate="<b>%{text}</b><br>Contribution: %{y:.1f}<extra></extra>" if show_player_names else None
            ))
            first_team = False

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=league_avg, line_dash="dot", line_color="#FFB000",
                 annotation_text=f"League avg: {league_avg:.1f}")

    fig.update_layout(
        title=f"{selected_team} vs Rest of {league} ({selected_season})",
        yaxis_title="Contribution (goals/90)",
        xaxis_title="Position",
        boxmode="group",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(categoryorder="array", categoryarray=position_order)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Position breakdown table
    st.markdown("### Position Summary")

    summary_data = []
    for pos in position_order:
        team_pos = team_df[team_df["position_group"] == pos]
        rest_pos = rest_df[rest_df["position_group"] == pos]

        if not team_pos.empty or not rest_pos.empty:
            team_avg_pos = team_pos["contribution"].mean() if not team_pos.empty else None
            league_avg_pos = rest_pos["contribution"].mean() if not rest_pos.empty else None
            diff_pos = (team_avg_pos - league_avg_pos) if team_avg_pos is not None and league_avg_pos is not None else None

            summary_data.append({
                "Position": pos,
                "Team Players": len(team_pos),
                "Team Avg": f"{team_avg_pos:.1f}" if team_avg_pos is not None else "-",
                "League Avg": f"{league_avg_pos:.1f}" if league_avg_pos is not None else "-",
                "Difference": f"{diff_pos:+.1f}" if diff_pos is not None else "-",
                "Status": "✅ Above" if diff_pos and diff_pos > 0.05 else ("⚠️ Below" if diff_pos and diff_pos < -0.05 else "➖ Similar") if diff_pos is not None else "-"
            })

    if summary_data:
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    # Individual player details
    st.markdown("### Current Squad")

    # Check for new players (not in previous season)
    prev_season = selected_season - 1
    prev_season_players = set()
    prev_team_df = df[(df["team"] == selected_team) & (df["season"] == prev_season)]
    if not prev_team_df.empty:
        prev_season_players = set(prev_team_df["player_id"].dropna().unique())

    # Sort by contribution descending and add age + new player indicator
    team_display = team_df[["player_id", "player_name", "age", "position", "position_group", "contribution", "goals", "assists", "FTE_games_played"]].copy()
    team_display = team_display.sort_values("contribution", ascending=False)

    # Mark new arrivals - append to player name for prominence
    def format_player_name(row):
        is_new = row["player_id"] not in prev_season_players and len(prev_season_players) > 0
        if is_new:
            return f"⭐ {row['player_name']} [NEW]"
        return row["player_name"]

    team_display["player_display"] = team_display.apply(format_player_name, axis=1)

    # Format values
    team_display["contribution"] = team_display["contribution"].round(1)
    team_display["FTE_games_played"] = team_display["FTE_games_played"].round(1)

    # Prepare display
    team_display = team_display[["player_display", "age", "position", "contribution", "goals", "assists", "FTE_games_played"]]
    team_display.columns = ["Player", "Age", "Position", "Contrib", "Goals", "Assists", "FTE"]

    # Count new players for info
    new_count = sum(1 for _, row in team_df.iterrows() if row["player_id"] not in prev_season_players) if prev_season_players else 0
    if new_count > 0:
        st.caption(f"⭐ {new_count} new arrival{'s' if new_count > 1 else ''} this season (marked with [NEW])")

    st.dataframe(team_display, hide_index=True, use_container_width=True)

    # Departed high-contribution players
    st.markdown("### Departed Players (High Contribution)")
    st.caption("Players who left the team after contributing significantly")

    # Find players who were at this team last season but not this season
    if not prev_team_df.empty:
        current_players = set(team_df["player_id"].dropna().unique())
        departed_ids = prev_season_players - current_players

        if departed_ids:
            departed_df = prev_team_df[prev_team_df["player_id"].isin(departed_ids)].copy()
            departed_df = departed_df.sort_values("contribution", ascending=False)

            # Show top departed players with high contribution (contribution > 0)
            high_contrib_departed = departed_df[departed_df["contribution"] > 0]

            if not high_contrib_departed.empty:
                # Check where they went (in current season)
                departed_display = high_contrib_departed[["player_id", "player_name", "age", "position", "contribution", "goals", "assists"]].copy()
                departed_display["contribution"] = departed_display["contribution"].round(1)

                # Find their new team
                def find_new_team(pid):
                    current = df[(df["player_id"] == pid) & (df["season"] == selected_season)]
                    if not current.empty:
                        return current.iloc[0]["team"]
                    return "Unknown"

                departed_display["Now At"] = departed_display["player_id"].apply(find_new_team)
                departed_display = departed_display[["player_name", "age", "position", "contribution", "goals", "assists", "Now At"]]
                departed_display.columns = ["Player", "Age", "Position", "Contrib", "Goals", "Assists", "Now At"]
                st.dataframe(departed_display.head(10), hide_index=True, use_container_width=True)
            else:
                st.caption("No high-contribution players departed")
        else:
            st.caption("No players departed from the team")
    else:
        st.caption("No previous season data available")

    # Export
    st.markdown("---")
    with st.expander("📥 Export Data"):
        csv = team_display.to_csv(index=False)
        st.download_button(
            f"Download {selected_team} Player Data (CSV)",
            csv,
            f"team_{selected_team.replace(' ', '_')}_{selected_season}.csv",
            "text/csv",
            key="team_export"
        )


def page_league_comparison():
    """Compare distributions with flexible filters and boxplot visualization."""
    st.header("Distribution Comparison")

    with st.spinner("Loading player data..."):
        df = load_all_players(min_fte=0)

    # Build league options with country prefix to disambiguate (e.g., "Germany - Bundesliga")
    # Use (country, league) tuples to handle duplicate league names (e.g., Austria/Germany Bundesliga)
    country_league_pairs = df[["country", "league"]].drop_duplicates().dropna()
    league_display_list = sorted([f"{row['country']} - {row['league']}" for _, row in country_league_pairs.iterrows()])
    # Map display name back to (country, league) tuple for filtering
    display_to_country_league = {
        f"{row['country']} - {row['league']}": (row['country'], row['league'])
        for _, row in country_league_pairs.iterrows()
    }

    st.markdown("Configure up to 5 distributions to compare. Use 'Copy from #1' for quick setup.")

    # Global settings in sidebar
    with st.sidebar:
        st.subheader("Global Settings")
        min_fte = st.slider("Minimum FTE Games", 0.0, 30.0, 10.0, 0.5,
                           key="dist_min_fte",
                           help="Full-Time Equivalent: minutes ÷ 90. E.g., 10 FTE = ~900 min.")
        num_distributions = st.number_input("Number of distributions", min_value=1, max_value=5, value=2,
                                           key="dist_num", help="How many groups to compare")

    df_filtered = df[df["FTE_games_played"] >= min_fte]

    # Initialize session state for distribution settings
    if "dist_settings" not in st.session_state:
        st.session_state.dist_settings = [{} for _ in range(5)]

    # Color-blind safe palette
    colors = CB_PALETTE[:5]
    distributions = []

    for i in range(int(num_distributions)):
        with st.expander(f"Distribution {i+1}", expanded=(i < 2)):
            # Copy from Distribution 1 button (for distributions 2-5)
            if i > 0:
                if st.button(f"📋 Copy from Distribution 1", key=f"copy_{i}"):
                    # Copy ALL specs from Distribution 1
                    st.session_state[f"league_{i}"] = st.session_state.get(f"league_0", "All")
                    st.session_state[f"seasons_{i}"] = st.session_state.get(f"seasons_0", ["All"])
                    st.session_state[f"positions_{i}"] = st.session_state.get(f"positions_0", ["All"])
                    st.session_state[f"teams_{i}"] = st.session_state.get(f"teams_0", ["All"])
                    st.session_state[f"nationalities_{i}"] = st.session_state.get(f"nationalities_0", ["All"])
                    st.session_state[f"age_range_{i}"] = st.session_state.get(f"age_range_0", (15, 45))
                    st.rerun()

            col1, col2, col3 = st.columns(3)

            with col1:
                # League filter (single select for cascading)
                league_display_options = ["All"] + league_display_list
                selected_league_display = st.selectbox(
                    "League",
                    league_display_options,
                    key=f"league_{i}"
                )

                # Map back to actual (country, league) tuple for filtering
                if selected_league_display == "All":
                    selected_country_league = None
                    league_subset = df_filtered
                else:
                    selected_country_league = display_to_country_league.get(selected_league_display)
                    if selected_country_league:
                        league_subset = df_filtered[
                            (df_filtered["country"] == selected_country_league[0]) &
                            (df_filtered["league"] == selected_country_league[1])
                        ]
                    else:
                        league_subset = df_filtered

                # Seasons available for selected league
                available_seasons = sorted(league_subset["season"].dropna().unique())
                season_options = ["All"] + [str(s) for s in available_seasons]
                # Only use default if key not already in session state (avoids conflict)
                default_seasons = st.session_state.get(f"seasons_{i}", ["All"])
                selected_seasons = st.multiselect(
                    "Seasons",
                    season_options,
                    default=default_seasons if f"seasons_{i}" not in st.session_state else None,
                    key=f"seasons_{i}"
                )

            with col2:
                # Teams from player_profiles (clean single-team entries)
                if selected_country_league:
                    available_teams = load_teams_for_league(selected_country_league[1], selected_country_league[0])
                else:
                    available_teams = load_teams_for_league()

                team_options = ["All"] + available_teams
                # Only use default if key not already in session state (avoids conflict)
                selected_teams = st.multiselect(
                    "Teams",
                    team_options,
                    default=["All"] if f"teams_{i}" not in st.session_state else None,
                    key=f"teams_{i}"
                )

                # Position filter
                position_options = ["All", "Goalkeeper", "Defender", "Midfielder", "Forward"]
                selected_positions = st.multiselect(
                    "Positions",
                    position_options,
                    default=["All"] if f"positions_{i}" not in st.session_state else None,
                    key=f"positions_{i}"
                )

            with col3:
                # Nationality filter
                available_nationalities = sorted(league_subset["nationality"].dropna().unique())
                nationality_options = ["All"] + [n for n in available_nationalities if n != "Unknown"]
                selected_nationalities = st.multiselect(
                    "Nationalities",
                    nationality_options,
                    default=["All"] if f"nationalities_{i}" not in st.session_state else None,
                    key=f"nationalities_{i}"
                )

                # Age range filter
                # Only use default value if key not already in session state
                default_age = (15, 45) if f"age_range_{i}" not in st.session_state else st.session_state[f"age_range_{i}"]
                age_min, age_max = st.slider(
                    "Age Range",
                    min_value=15,
                    max_value=45,
                    value=default_age,
                    key=f"age_range_{i}"
                )

            # Custom label with smart default
            if selected_country_league:
                default_label = f"{selected_country_league[0]} - {selected_country_league[1]}"
                if "All" not in selected_seasons and len(selected_seasons) == 1:
                    default_label += f" {selected_seasons[0]}"
                if "All" not in selected_teams and len(selected_teams) == 1:
                    default_label = selected_teams[0]
            else:
                default_label = f"Distribution {i+1}"

            label = st.text_input("Label", value=default_label, key=f"label_{i}")

            distributions.append({
                "label": label,
                "country_league": selected_country_league,  # (country, league) tuple or None
                "seasons": selected_seasons,
                "positions": selected_positions,
                "teams": selected_teams,
                "nationalities": selected_nationalities,
                "age_range": (age_min, age_max),
                "color": colors[i]
            })

    # Build filtered datasets and create boxplot
    if st.button("Generate Comparison", type="primary"):
        fig = go.Figure()
        comparison_stats = []
        all_subsets = []

        for dist in distributions:
            # Apply filters
            subset = df_filtered.copy()

            # League filter (using country+league tuple)
            if dist["country_league"]:
                country, league = dist["country_league"]
                subset = subset[(subset["country"] == country) & (subset["league"] == league)]

            # Season filter
            if "All" not in dist["seasons"] and dist["seasons"]:
                subset = subset[subset["season"].astype(str).isin(dist["seasons"])]

            # Position filter
            if "All" not in dist["positions"] and dist["positions"]:
                subset = subset[subset["position"].isin(dist["positions"])]

            # Team filter (handles "Multiple Teams (A, B)" format)
            if "All" not in dist["teams"] and dist["teams"]:
                # Check if any selected team name is contained in the team column
                team_mask = subset["team"].apply(
                    lambda t: any(team in str(t) for team in dist["teams"]) if pd.notna(t) else False
                )
                subset = subset[team_mask]

            # Nationality filter
            if "All" not in dist["nationalities"] and dist["nationalities"]:
                subset = subset[subset["nationality"].isin(dist["nationalities"])]

            # Age range filter
            age_min, age_max = dist["age_range"]
            if age_min > 15 or age_max < 45:
                subset = subset[
                    (subset["age"].notna()) &
                    (subset["age"] >= age_min) &
                    (subset["age"] <= age_max)
                ]

            if subset.empty:
                st.warning(f"No data for '{dist['label']}'")
                continue

            # Add boxplot trace with player names on outliers
            fig.add_trace(go.Box(
                y=subset["contribution"],
                name=dist["label"],
                marker_color=dist["color"],
                boxmean=True,  # Show mean line (no SD - we'll add custom annotation)
                boxpoints="outliers",  # Show outlier points
                text=subset["player_name"],  # Player names for outlier points
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Contribution: %{y:.1f}<br>"
                    "<extra>%{fullData.name}</extra>"
                )
            ))

            all_subsets.append((dist["label"], subset))

            # Collect stats (1 decimal for readability)
            mean_val = subset["contribution"].mean()
            std_val = subset["contribution"].std()
            comparison_stats.append({
                "Distribution": dist["label"],
                "Players": len(subset),
                "Mean": round(mean_val, 1),
                "Median": round(subset["contribution"].median(), 1),
                "Std": round(std_val, 1),
                "Q1": round(subset["contribution"].quantile(0.25), 1),
                "Q3": round(subset["contribution"].quantile(0.75), 1),
                "Min": round(subset["contribution"].min(), 1),
                "Max": round(subset["contribution"].max(), 1),
            })

        if comparison_stats:
            # Add custom annotations for mean ± std for each distribution
            for i, stats in enumerate(comparison_stats):
                fig.add_annotation(
                    x=i,  # Position by trace index
                    y=stats["Mean"],
                    text=f"<b>μ={stats['Mean']:.1f}</b><br>σ={stats['Std']:.1f}",
                    showarrow=False,
                    font=dict(size=10, color="#333"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(100,100,100,0.3)",
                    borderwidth=1,
                    yshift=40,  # Offset above the mean line
                )

            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Contribution Distributions (Boxplot)",
                yaxis_title="Lasso Contribution (goals/90)",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stats table
            st.subheader("Distribution Statistics")
            stats_df = pd.DataFrame(comparison_stats)
            st.dataframe(stats_df, hide_index=True)

            # Top players from each distribution
            st.subheader("Top 5 Players per Distribution")
            cols = st.columns(min(len(all_subsets), 3))
            for idx, (label, subset) in enumerate(all_subsets):
                with cols[idx % 3]:
                    st.markdown(f"**{label}**")
                    top5 = subset.nlargest(5, "contribution")[["player_name", "contribution", "team"]]
                    st.dataframe(top5, hide_index=True, height=200)

            # Export section
            st.markdown("---")
            with st.expander("📥 Export Data"):
                csv = stats_df.to_csv(index=False)
                st.download_button(
                    "Download Statistics (CSV)",
                    csv,
                    "distribution_comparison.csv",
                    "text/csv",
                    key="dist_export"
                )
        else:
            render_empty_state(
                "No data to display",
                "Adjust your filters or click 'Generate Comparison'"
            )


# =============================================================================
#                          NETWORK ANALYSIS
# =============================================================================

@st.cache_data
def _classify_league_type(league_name: str) -> str:
    """Classify league as Men, Women, or Youth based on name patterns."""
    if not league_name:
        return "Men"
    league_lower = league_name.lower()

    # Women's leagues
    women_keywords = ["women", "frauen", "femenina", "femminile", "feminine", "dames", "vrouwen", "feminin"]
    if any(kw in league_lower for kw in women_keywords):
        return "Women"

    # Youth leagues
    youth_keywords = ["u19", "u21", "u23", "u17", "u18", "u20", "youth", "junior", "junioren", "juvenil"]
    if any(kw in league_lower for kw in youth_keywords):
        return "Youth"

    return "Men"


def compute_transfer_network(level: str = "league", min_transfers: int = 3,
                              season_start: int = None, season_end: int = None,
                              filter_cross_country: bool = False,
                              league_type: str = "All"):
    """
    Compute transfer network data from player transitions.

    Args:
        level: "league" or "club"
        min_transfers: Minimum number of transfers to include an edge
        season_start: Filter to include only seasons >= this value
        season_end: Filter to include only seasons <= this value
        filter_cross_country: If True, only include within-country transfers (promotions/relegations)
        league_type: "All", "Men", "Women", or "Youth" - filter leagues by type

    Returns:
        edges_df: DataFrame with source, target, transfers, cumulative_contribution, net_contribution
        nodes_df: DataFrame with node, avg_contribution, n_players
        trans_df: DataFrame with individual transitions for drill-down
    """
    # Load players data (parquet or SQLite)
    if _DATA_FORMAT == "parquet":
        players = _load_players_parquet(include_historical=True)
        if players.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # Rename columns to match expected format
        if "team(s)" in players.columns:
            players = players.rename(columns={"team(s)": "team"})
        elif "team" not in players.columns and "teams" in players.columns:
            players = players.rename(columns={"teams": "team"})
        if CONTRIB_COL in players.columns:
            players = players.rename(columns={CONTRIB_COL: "contribution"})
        # Filter to required columns
        cols = ["player_id", "player_name", "team", "league", "country", "season", "contribution"]
        players = players[[c for c in cols if c in players.columns]]
        players = players[players["contribution"].notna()]
    elif Path(DB_PATH).exists():
        conn = sqlite3.connect(DB_PATH)
        players = pd.read_sql_query("""
            SELECT player_id, player_name, "team(s)" as team, league, country, season,
                   lasso_contribution_alpha_best as contribution
            FROM players
            WHERE lasso_contribution_alpha_best IS NOT NULL
            ORDER BY player_id, season
        """, conn)
        conn.close()
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Apply season filter
    if season_start is not None:
        players = players[players["season"] >= season_start]
    if season_end is not None:
        players = players[players["season"] <= season_end]

    # Apply league type filter
    if league_type and league_type != "All":
        players["_league_type"] = players["league"].apply(_classify_league_type)
        players = players[players["_league_type"] == league_type]
        players = players.drop(columns=["_league_type"])

    # Normalize player_id
    players["pid_norm"] = players["player_id"].apply(_normalize_player_id)

    # Determine node column based on level
    if level == "league":
        players["node"] = players["country"] + " - " + players["league"]
    else:
        players["node"] = players["team"]

    # Find transitions: for each player, track sequential appearances
    transitions = []

    for pid, group in players.groupby("pid_norm"):
        group = group.sort_values("season")
        prev_node = None
        prev_contribution = None
        prev_country = None

        for _, row in group.iterrows():
            current_node = row["node"]
            current_contribution = row["contribution"]
            current_country = row["country"]

            if prev_node is not None and prev_node != current_node:
                # Check cross-country filter
                is_same_country = (prev_country == current_country)

                if not filter_cross_country or is_same_country:
                    transitions.append({
                        "source": prev_node,
                        "target": current_node,
                        "contribution_at_target": current_contribution,
                        "contribution_at_source": prev_contribution,
                        "contribution_delta": current_contribution - prev_contribution,
                        "player_name": row["player_name"],
                        "player_id": pid,
                        "season": row["season"],
                        "same_country": is_same_country
                    })

            prev_node = current_node
            prev_contribution = current_contribution
            prev_country = current_country

    if not transitions:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trans_df = pd.DataFrame(transitions)

    # Aggregate edges - focus on immediate before/after contributions
    edges = trans_df.groupby(["source", "target"]).agg(
        transfers=("player_id", "count"),
        # Outflow: avg contribution immediately before leaving source
        avg_outflow=("contribution_at_source", "mean"),
        # Inflow: avg contribution immediately after arriving at target
        avg_inflow=("contribution_at_target", "mean"),
        # Delta: how players change when moving
        avg_delta=("contribution_delta", "mean"),
        improved_count=("contribution_delta", lambda x: (x > 0).sum()),
        declined_count=("contribution_delta", lambda x: (x < 0).sum()),
    ).reset_index()

    # Filter by minimum transfers
    edges = edges[edges["transfers"] >= min_transfers]

    # Node statistics
    all_nodes = set(edges["source"].unique()) | set(edges["target"].unique())
    node_stats = players[players["node"].isin(all_nodes)].groupby("node").agg(
        avg_contribution=("contribution", "mean"),
        n_players=("pid_norm", "nunique"),
        n_seasons=("season", "count"),
        seasons_list=("season", lambda x: sorted(x.unique().tolist()))
    ).reset_index()

    return edges, node_stats, trans_df


def page_network_analysis():
    """Network analysis page showing transfer flows between leagues/clubs."""
    st.header("🌐 Transfer Network Analysis")

    st.markdown("""
    ### Understanding Player Movement Patterns

    This page visualizes how players move between leagues and clubs, helping you understand:
    - **Transfer corridors**: Which leagues/clubs exchange the most players
    - **Hub leagues**: Leagues that act as central nodes in the transfer network
    - **Player flow patterns**: Where talent tends to move from and to

    ---

    #### 🔍 Visual Guide
    - **Line thickness** = Number of transfers (more players = thicker line)
    - **Node size** = Total transfer activity (in + out)
    - **Hover** for detailed transfer statistics
    """)

    # Controls in sidebar
    with st.sidebar:
        st.subheader("Network Settings")

        with st.expander("🔧 Configuration", expanded=True):
            level = st.selectbox("Network Level", ["League", "Club"], index=0,
                               help="Analyze at league or club level")
            min_transfers = st.slider("Min Transfers", 1, 20, 5,
                                     help="Minimum transfers to show an edge")
            league_type = st.selectbox(
                "League Type",
                ["Men", "Women", "Youth", "All"],
                index=0,
                help="Filter by league type - mixing types is usually not meaningful"
            )

        with st.expander("📅 Season Filter", expanded=True):
            season_start = st.number_input("Season Start", min_value=2010, max_value=2025, value=2016)
            season_end = st.number_input("Season End", min_value=2010, max_value=2025, value=2025)

    # Compute network with spinner
    with st.spinner("Computing transfer network..."):
        edges, nodes, trans_df = compute_transfer_network(
            level=level.lower(),
            min_transfers=min_transfers,
            season_start=season_start,
            season_end=season_end,
            league_type=league_type
        )

    if edges.empty:
        render_empty_state(
            "No transfer data available",
            "Try adjusting the filters in the sidebar"
        )
        return

    # Additional filtering for clubs (too many nodes otherwise)
    if level == "Club":
        # Get top clubs by transfer volume
        top_sources = edges.groupby("source")["transfers"].sum().nlargest(30).index
        top_targets = edges.groupby("target")["transfers"].sum().nlargest(30).index
        top_nodes = set(top_sources) | set(top_targets)
        edges = edges[edges["source"].isin(top_nodes) & edges["target"].isin(top_nodes)]
        nodes = nodes[nodes["node"].isin(top_nodes)]
        st.info(f"Showing top {len(top_nodes)} clubs by transfer volume")

    if edges.empty:
        st.warning("No transfer data to display.")
        return

    # Build Sankey diagram for transfer flows
    # Get top edges by transfer volume for cleaner visualization
    top_edges = edges.nlargest(50, "transfers")

    # Build node list from top edges
    all_nodes_list = list(set(top_edges["source"].unique()) | set(top_edges["target"].unique()))
    node_to_idx = {node: i for i, node in enumerate(all_nodes_list)}

    # Build links
    sources = [node_to_idx[s] for s in top_edges["source"]]
    targets = [node_to_idx[t] for t in top_edges["target"]]
    values = top_edges["transfers"].tolist()

    # Color links by transfer volume
    max_transfers = max(values) if values else 1
    link_colors = [
        f"rgba(100, 143, 255, {0.3 + 0.5 * (v / max_transfers)})"
        for v in values
    ]

    # Node colors based on average contribution
    node_colors = []
    for node in all_nodes_list:
        node_info = nodes[nodes["node"] == node]
        if not node_info.empty:
            contrib = node_info.iloc[0]["avg_contribution"]
            # Map contribution to color (green for high, red for low)
            if contrib > 0.5:
                node_colors.append(f"rgba(46, 139, 87, 0.8)")  # Green
            elif contrib > 0:
                node_colors.append(f"rgba(218, 165, 32, 0.8)")  # Gold
            elif contrib > -0.5:
                node_colors.append(f"rgba(255, 165, 0, 0.8)")  # Orange
            else:
                node_colors.append(f"rgba(205, 92, 92, 0.8)")  # Red
        else:
            node_colors.append("rgba(150, 150, 150, 0.8)")  # Gray

    # Shorten labels for display
    node_labels = [
        n.split(" - ")[-1][:25] if " - " in n else n[:25]
        for n in all_nodes_list
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="white", width=1),
            label=node_labels,
            color=node_colors,
            hovertemplate="%{label}<extra></extra>"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate="%{source.label} → %{target.label}<br>%{value} transfers<extra></extra>"
        )
    )])

    fig.update_layout(
        title=f"{level}-Level Transfer Network ({len(all_nodes_list)} nodes, {len(top_edges)} flows)",
        height=max(500, 25 * len(all_nodes_list)),
        font=dict(size=11),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Flow width = transfer volume | Node color: 🟢 high contribution → 🔴 low contribution")

    # Summary statistics - Fluctuation Rate
    st.subheader("🔄 Transfer Fluctuation")
    st.markdown("""
    **Fluctuation Rate** = (Imports + Exports) / Total Players.
    High fluctuation indicates unstable rosters; low fluctuation suggests squad stability.
    """)

    # Calculate fluctuation per node
    exports_per_node = edges.groupby("source")["transfers"].sum()
    imports_per_node = edges.groupby("target")["transfers"].sum()
    all_nodes_set = set(exports_per_node.index) | set(imports_per_node.index)

    fluctuation_data = []
    for node in all_nodes_set:
        node_info = nodes[nodes["node"] == node]
        if node_info.empty:
            continue
        n_players = node_info.iloc[0]["n_players"]
        exports = exports_per_node.get(node, 0)
        imports = imports_per_node.get(node, 0)
        total_moves = exports + imports
        fluct_rate = total_moves / n_players if n_players > 0 else 0
        fluctuation_data.append({
            "node": node,
            "n_players": n_players,
            "exports": exports,
            "imports": imports,
            "total_moves": total_moves,
            "fluctuation_rate": fluct_rate
        })

    fluct_df = pd.DataFrame(fluctuation_data)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌪️ Highest Fluctuation** (unstable rosters)")
        high_fluct = fluct_df.nlargest(10, "fluctuation_rate")[
            ["node", "n_players", "exports", "imports", "fluctuation_rate"]
        ].copy()
        high_fluct.columns = ["Node", "Players", "Out", "In", "Rate"]
        st.dataframe(high_fluct.round(2), hide_index=True)

    with col2:
        st.markdown("**🏠 Lowest Fluctuation** (stable rosters)")
        low_fluct = fluct_df[fluct_df["total_moves"] > 5].nsmallest(10, "fluctuation_rate")[
            ["node", "n_players", "exports", "imports", "fluctuation_rate"]
        ].copy()
        low_fluct.columns = ["Node", "Players", "Out", "In", "Rate"]
        st.dataframe(low_fluct.round(2), hide_index=True)

    # Top transfer corridors
    st.subheader("🛤️ Top Transfer Corridors")
    st.markdown("The most active transfer routes between leagues/clubs.")

    top_corridors = edges.nlargest(15, "transfers")[
        ["source", "target", "transfers"]
    ].copy()
    top_corridors.columns = ["From", "To", "Transfers"]
    st.dataframe(top_corridors, hide_index=True)

    # Node Detail Panel
    st.subheader("Node Detail")
    st.markdown("Select a node to see its transfer network. Edge width and color intensity = transfer volume.")

    # Filter node options to public leagues only (or clubs from public leagues)
    public_leagues = get_public_leagues()
    all_node_list = nodes["node"].tolist()

    def is_public_node(node_name: str) -> bool:
        """Check if a node is from a public league."""
        # For league-level nodes (format: "Country - League")
        if " - " in node_name:
            parts = node_name.split(" - ", 1)
            if len(parts) == 2:
                country, league = parts
                return (country, league) in public_leagues
        # For club-level or unrecognized format, allow (clubs use all leagues in transfer network)
        return True

    public_node_list = [n for n in all_node_list if is_public_node(n)]
    node_options = ["Select a node..."] + sorted(public_node_list)
    selected_node = st.selectbox("Choose League/Club", node_options, key="node_detail_select")

    if selected_node and selected_node != "Select a node...":
        # Get node stats
        node_info = nodes[nodes["node"] == selected_node].iloc[0] if len(nodes[nodes["node"] == selected_node]) > 0 else None

        if node_info is not None:
            # Incoming and outgoing edges for this node
            incoming_edges = edges[edges["target"] == selected_node]
            outgoing_edges = edges[edges["source"] == selected_node]

            total_incoming = incoming_edges["transfers"].sum() if not incoming_edges.empty else 0
            total_outgoing = outgoing_edges["transfers"].sum() if not outgoing_edges.empty else 0

            # Summary metrics (simplified)
            col1, col2, col3 = st.columns(3)
            col1.metric("Incoming Transfers", int(total_incoming))
            col2.metric("Outgoing Transfers", int(total_outgoing))
            col3.metric("Net Flow", int(total_incoming - total_outgoing))

            # Create focused graph visualization
            st.markdown("### Transfer Map")

            # Build mini-network for this node
            import networkx as nx
            mini_G = nx.DiGraph()

            # Add central node
            mini_G.add_node(selected_node, is_central=True,
                           avg_contribution=node_info["avg_contribution"],
                           n_players=node_info["n_players"])

            # Add predecessors (sources that feed into this node)
            for _, row in incoming_edges.iterrows():
                source_info = nodes[nodes["node"] == row["source"]]
                if not source_info.empty:
                    mini_G.add_node(row["source"], is_central=False,
                                   avg_contribution=source_info.iloc[0]["avg_contribution"],
                                   n_players=source_info.iloc[0]["n_players"])
                    mini_G.add_edge(row["source"], selected_node,
                                   transfers=row["transfers"])

            # Add successors (targets that this node feeds into)
            for _, row in outgoing_edges.iterrows():
                target_info = nodes[nodes["node"] == row["target"]]
                if not target_info.empty:
                    mini_G.add_node(row["target"], is_central=False,
                                   avg_contribution=target_info.iloc[0]["avg_contribution"],
                                   n_players=target_info.iloc[0]["n_players"])
                    mini_G.add_edge(selected_node, row["target"],
                                   transfers=row["transfers"])

            if len(mini_G.nodes()) > 1:
                # Get top feeders and customers by transfer volume
                top_incoming = incoming_edges.nlargest(15, "transfers")
                top_outgoing = outgoing_edges.nlargest(15, "transfers")

                # Build Sankey diagram - much better for flow visualization
                # Node indices: 0 = feeders, then central, then customers
                feeders = list(top_incoming["source"].unique())
                customers = list(top_outgoing["target"].unique())

                # Create node list: feeders + [central] + customers
                all_nodes = feeders + [selected_node] + customers
                node_to_idx = {node: i for i, node in enumerate(all_nodes)}
                central_idx = node_to_idx[selected_node]

                # Build links
                sources = []
                targets = []
                values = []
                link_colors = []

                # Incoming links (feeders -> central) - green
                for _, row in top_incoming.iterrows():
                    if row["source"] in node_to_idx:
                        sources.append(node_to_idx[row["source"]])
                        targets.append(central_idx)
                        values.append(row["transfers"])
                        link_colors.append("rgba(46, 139, 87, 0.5)")  # SeaGreen

                # Outgoing links (central -> customers) - red/coral
                for _, row in top_outgoing.iterrows():
                    if row["target"] in node_to_idx:
                        sources.append(central_idx)
                        targets.append(node_to_idx[row["target"]])
                        values.append(row["transfers"])
                        link_colors.append("rgba(205, 92, 92, 0.5)")  # IndianRed

                # Node colors: green for feeders, gold for central, red for customers
                node_colors = (
                    ["#2E8B57"] * len(feeders) +
                    ["gold"] +
                    ["#CD5C5C"] * len(customers)
                )

                # Shorten labels for display
                node_labels = [
                    n.split(" - ")[-1][:25] if " - " in n else n[:25]
                    for n in all_nodes
                ]

                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=20,
                        thickness=25,
                        line=dict(color="white", width=1),
                        label=node_labels,
                        color=node_colors,
                        hovertemplate="%{label}<extra></extra>"
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                        hovertemplate="%{source.label} → %{target.label}<br>%{value} transfers<extra></extra>"
                    )
                )])

                fig_sankey.update_layout(
                    height=max(400, 35 * len(all_nodes)),
                    margin=dict(l=10, r=10, t=10, b=10),
                    font=dict(size=11)
                )
                st.plotly_chart(fig_sankey, use_container_width=True)

                st.caption("🟢 **Feeders** (left): send players | 🟡 **Selected** (center) | 🔴 **Customers** (right): receive players | Flow width = transfer volume")
            else:
                st.info("No connected nodes found for this selection.")

            # Player-level details
            st.markdown("### Player Transitions")
            incoming_trans = trans_df[trans_df["target"] == selected_node].copy() if not trans_df.empty else pd.DataFrame()
            outgoing_trans = trans_df[trans_df["source"] == selected_node].copy() if not trans_df.empty else pd.DataFrame()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Recent Arrivals** (players who joined)")
                if not incoming_trans.empty:
                    top_arrivals = incoming_trans.nlargest(8, "season")[
                        ["player_name", "source", "season"]
                    ].copy()
                    top_arrivals.columns = ["Player", "From", "Season"]
                    st.dataframe(top_arrivals, hide_index=True)
                else:
                    st.caption("No incoming transitions")

            with col2:
                st.markdown("**Recent Departures** (players who left)")
                if not outgoing_trans.empty:
                    top_departures = outgoing_trans.nlargest(8, "season")[
                        ["player_name", "target", "season"]
                    ].copy()
                    top_departures.columns = ["Player", "To", "Season"]
                    st.dataframe(top_departures, hide_index=True)
                else:
                    st.caption("No outgoing transitions")

    # =========================================================================
    # CAREER PATH MINING
    # =========================================================================
    st.markdown("---")
    st.subheader("🛤️ Career Path Mining")
    st.markdown("""
    Discover common **multi-step career trajectories** in the data.
    A career path is a sequence of leagues/clubs a player moved through.
    """)

    if not trans_df.empty and "season" in trans_df.columns:
        # Build career paths for each player
        career_paths = []

        # Group transitions by player
        for pid, group in trans_df.groupby("player_id"):
            group = group.sort_values("season")
            path = []
            for _, row in group.iterrows():
                if not path or path[-1] != row["source"]:
                    path.append(row["source"])
                path.append(row["target"])

            if len(path) >= 2:
                career_paths.append({
                    "player_id": pid,
                    "path": " → ".join(path),
                    "path_tuple": tuple(path),
                    "path_length": len(path),
                    "player_name": group.iloc[-1]["player_name"]
                })

        if career_paths:
            paths_df = pd.DataFrame(career_paths)

            # Find most common 2-step and 3-step patterns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Most Common 2-Step Paths**")
                # Extract 2-step subpaths
                two_step_paths = []
                for path in paths_df["path_tuple"]:
                    for i in range(len(path) - 1):
                        two_step_paths.append(f"{path[i]} → {path[i+1]}")

                if two_step_paths:
                    from collections import Counter
                    path_counts = Counter(two_step_paths)
                    top_2step = pd.DataFrame(path_counts.most_common(15),
                                            columns=["Path", "Count"])
                    st.dataframe(top_2step, hide_index=True)

            with col2:
                st.markdown("**Most Common 3-Step Paths**")
                # Extract 3-step subpaths
                three_step_paths = []
                for path in paths_df["path_tuple"]:
                    for i in range(len(path) - 2):
                        three_step_paths.append(f"{path[i]} → {path[i+1]} → {path[i+2]}")

                if three_step_paths:
                    path_counts = Counter(three_step_paths)
                    top_3step = pd.DataFrame(path_counts.most_common(15),
                                            columns=["Path", "Count"])
                    st.dataframe(top_3step, hide_index=True)

            # Path length distribution
            with st.expander("Career Path Length Distribution"):
                length_counts = paths_df["path_length"].value_counts().sort_index()
                fig_lengths = px.bar(x=length_counts.index, y=length_counts.values,
                                    labels={"x": "Career Path Length (leagues)", "y": "Number of Players"},
                                    title="Distribution of Career Path Lengths")
                fig_lengths.update_layout(height=300)
                st.plotly_chart(fig_lengths, use_container_width=True)

            # Search for specific paths
            with st.expander("Search Career Paths"):
                search_term = st.text_input("Search for league/club in path",
                                           placeholder="e.g., Premier League")
                if search_term:
                    matching_paths = paths_df[paths_df["path"].str.contains(search_term, case=False, na=False)]
                    if not matching_paths.empty:
                        st.caption(f"Found {len(matching_paths)} players with '{search_term}' in their path")
                        display_paths = matching_paths.nlargest(20, "path_length")[
                            ["player_name", "path", "path_length"]
                        ].copy()
                        display_paths.columns = ["Player", "Career Path", "Steps"]
                        st.dataframe(display_paths, hide_index=True)
                    else:
                        st.caption("No matching paths found")
        else:
            st.caption("Not enough career path data available")
    else:
        st.caption("No transition data available for career path mining")


# =============================================================================
#                          PERSISTENCE ANALYSIS
# =============================================================================

@st.cache_data
def compute_season_transitions(min_fte: float = 3.0):
    """
    Compute season-to-season transitions for each player.
    Returns DataFrame with prev_season, curr_season contributions and metadata.
    Uses vectorized pandas operations for performance.
    """
    if not Path(DB_PATH).exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    players = pd.read_sql_query(f"""
        SELECT player_id, player_name, "team(s)" as team, league, country, season,
               {CONTRIB_COL} as contribution, FTE_games_played
        FROM players
        WHERE {CONTRIB_COL} IS NOT NULL AND FTE_games_played >= ?
        ORDER BY player_id, season
    """, conn, params=(min_fte,))
    conn.close()

    if players.empty:
        return pd.DataFrame()

    # Normalize player_id in-place (avoid creating duplicate column names)
    players["player_id"] = players["player_id"].apply(_normalize_player_id)

    # Flag multi-team seasons
    players["is_multi_team"] = players["team"].str.startswith("Multiple Teams", na=False)

    # Sort by player and season for correct shift operations
    players = players.sort_values(["player_id", "season"]).reset_index(drop=True)

    # Use vectorized shift operations within each player group
    # Shift creates the "previous" row's values
    players["prev_season"] = players.groupby("player_id")["season"].shift(1)
    players["prev_contribution"] = players.groupby("player_id")["contribution"].shift(1)
    players["prev_team"] = players.groupby("player_id")["team"].shift(1)
    players["prev_league"] = players.groupby("player_id")["league"].shift(1)
    players["prev_country"] = players.groupby("player_id")["country"].shift(1)
    players["prev_is_multi"] = players.groupby("player_id")["is_multi_team"].shift(1)

    # Drop rows without a previous season (first season for each player)
    transitions = players.dropna(subset=["prev_season"]).copy()

    if transitions.empty:
        return pd.DataFrame()

    # Compute derived columns
    transitions["season_gap"] = transitions["season"] - transitions["prev_season"]
    transitions["switched_league"] = (
        (transitions["prev_league"] != transitions["league"]) |
        (transitions["prev_country"] != transitions["country"])
    )
    transitions["switched_team"] = transitions["prev_team"] != transitions["team"]

    # Rename columns to match expected output format
    transitions = transitions.rename(columns={
        "season": "curr_season",
        "contribution": "curr_contribution",
        "team": "curr_team",
        "league": "curr_league",
        "country": "curr_country",
        "is_multi_team": "curr_is_multi",
    })

    # Select and reorder columns
    result_cols = [
        "player_id", "player_name", "prev_season", "curr_season", "season_gap",
        "prev_contribution", "curr_contribution",
        "prev_team", "curr_team", "prev_league", "curr_league",
        "prev_country", "curr_country", "prev_is_multi", "curr_is_multi",
        "switched_league", "switched_team"
    ]
    transitions = transitions[result_cols]

    # Convert prev_season to int (was float due to shift)
    transitions["prev_season"] = transitions["prev_season"].astype(int)

    return transitions


def page_markov_analysis():
    """Persistence analysis - season-to-season contribution evolution."""
    st.header("Persistence Analysis")

    st.markdown("""
    How well does a player's contribution predict their next season performance?
    Use the sidebar filters to customize the analysis.
    """)

    # Load base data with spinner
    with st.spinner("Loading transition data..."):
        transitions_all = compute_season_transitions(min_fte=0)

    if transitions_all.empty:
        render_empty_state(
            "No transition data available",
            "Ensure you have player data across multiple seasons"
        )
        return

    # === FILTERS IN SIDEBAR ===
    with st.sidebar:
        st.subheader("Filters")

        # Player filters
        with st.expander("👤 Player Filters", expanded=True):
            min_fte = st.slider("Minimum FTE Games", 0.0, 20.0, 5.0, 0.5,
                               key="markov_fte",
                               help="Full-Time Equivalent: minutes ÷ 90. E.g., 5 FTE = ~450 min.")

            gap_options = ["1 (consecutive)", "Any gap"]
            season_gap_filter = st.selectbox("Season Gap", gap_options, index=0,
                                            key="markov_gap")

            exclude_multi = st.checkbox("Exclude Multi-Team Seasons", value=True,
                                       key="markov_multi",
                                       help="Exclude players who played for multiple teams")

            exclude_league_switch = st.checkbox("Exclude League Switchers", value=True,
                                               key="markov_league_switch",
                                               help="Exclude players who changed leagues")

            only_team_switchers = st.checkbox("Only Team Switchers", value=True,
                                             key="markov_team_switch",
                                             help="Only include players who switched teams")

        # Location filters (cascading)
        with st.expander("📍 Location", expanded=True):
            all_countries = sorted(set(transitions_all["prev_country"].unique()) |
                                  set(transitions_all["curr_country"].unique()))
            selected_countries = st.multiselect(
                "Country", ["All"] + all_countries, default=["All"],
                key="markov_countries",
                help=f"{len(all_countries)} countries available"
            )

            # League filter (cascades from country)
            if "All" not in selected_countries and selected_countries:
                available_leagues = []
                for country in selected_countries:
                    available_leagues.extend(get_leagues_for_country(country))
                available_leagues = sorted(set(available_leagues))
            else:
                available_leagues = sorted(set(transitions_all["prev_league"].unique()) |
                                    set(transitions_all["curr_league"].unique()))
            selected_leagues = st.multiselect(
                "League", ["All"] + available_leagues, default=["All"],
                key="markov_leagues",
                help=f"{len(available_leagues)} leagues available"
            )

            # Season range
            all_seasons = sorted(transitions_all["curr_season"].unique())
            if len(all_seasons) >= 2:
                season_range = st.slider("Season Range", min(all_seasons), max(all_seasons),
                                        (min(all_seasons), max(all_seasons)), key="markov_seasons")
            else:
                season_range = (min(all_seasons), max(all_seasons))

            # Team filter (cascades from country/league)
            if "All" not in selected_leagues and selected_leagues:
                available_teams = []
                for league in selected_leagues:
                    country = None
                    if "All" not in selected_countries and selected_countries:
                        country = selected_countries[0] if len(selected_countries) == 1 else None
                    available_teams.extend(get_teams_for_league_season(league=league, country=country))
                available_teams = sorted(set(available_teams))
            elif "All" not in selected_countries and selected_countries:
                available_teams = []
                for country in selected_countries:
                    for league in get_leagues_for_country(country):
                        available_teams.extend(get_teams_for_league_season(league=league, country=country))
                available_teams = sorted(set(available_teams))
            else:
                available_teams = []
            if available_teams:
                selected_teams = st.multiselect(
                    "Team", ["All"] + available_teams, default=["All"],
                    key="markov_teams",
                    help=f"{len(available_teams)} teams available"
                )
            else:
                selected_teams = ["All"]
                st.caption("Select a country or league to filter by team")

    # Apply filters
    df = transitions_all.copy()

    # Reload with FTE filter
    if min_fte > 0:
        df = compute_season_transitions(min_fte=min_fte)

    # Season gap filter
    if season_gap_filter == "1 (consecutive)":
        df = df[df["season_gap"] == 1]

    # Multi-team filter
    if exclude_multi:
        df = df[~df["prev_is_multi"] & ~df["curr_is_multi"]]

    # League switch filter
    if exclude_league_switch:
        df = df[~df["switched_league"]]

    # Team switchers only filter
    if only_team_switchers:
        df = df[df["switched_team"]]

    # League filter
    if "All" not in selected_leagues and selected_leagues:
        df = df[df["prev_league"].isin(selected_leagues) | df["curr_league"].isin(selected_leagues)]

    # Country filter
    if "All" not in selected_countries and selected_countries:
        df = df[df["prev_country"].isin(selected_countries) | df["curr_country"].isin(selected_countries)]

    # Team filter (filter if team appears in either prev or curr)
    if "All" not in selected_teams and selected_teams:
        team_mask = df.apply(
            lambda row: (
                any(team in str(row["prev_team"]) for team in selected_teams) or
                any(team in str(row["curr_team"]) for team in selected_teams)
            ),
            axis=1
        )
        df = df[team_mask]

    # Season range filter
    df = df[(df["curr_season"] >= season_range[0]) & (df["curr_season"] <= season_range[1])]

    if df.empty:
        st.warning("No data matches the current filters. Try: lowering FTE threshold, "
                   "allowing all season gaps, or unchecking 'Only Team Switchers'.")
        return

    st.caption(f"Showing {len(df):,} player-season transitions")

    # === KEY STATISTICS (always visible) ===
    correlation = df["prev_contribution"].corr(df["curr_contribution"])
    mean_change = (df["curr_contribution"] - df["prev_contribution"]).mean()
    improved_pct = (df["curr_contribution"] > df["prev_contribution"]).mean() * 100
    pos_players = df[df["prev_contribution"] > 0]
    regression_to_mean = pos_players["curr_contribution"].mean() - pos_players["prev_contribution"].mean() if len(pos_players) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Correlation (r)", f"{correlation:.3f}", help="How predictive is prev season?")
    col2.metric("Mean Change", f"{mean_change:+.3f}")
    col3.metric("% Improved", f"{improved_pct:.1f}%")
    col4.metric("Regression", f"{regression_to_mean:+.3f}", help="Avg change for positive contributors")

    # === MAIN SCATTER PLOT ===
    st.subheader("Contribution: Previous → Current Season")

    col1, col2 = st.columns([3, 1])

    with col2:
        color_by = st.selectbox("Color by", ["None", "Switched League", "Switched Team", "Current League"])
        dot_size = st.slider("Dot Size", 2, 15, 5, key="markov_dot_size")
        show_identity = st.checkbox("Show y=x line", value=True, help="Players above this line improved")
        show_regression = st.checkbox("Show regression line", value=True)

    with col1:
        # Build scatter
        if color_by == "Switched League":
            df["_color"] = df["switched_league"].map({True: "Switched", False: "Same League"})
            fig = px.scatter(df, x="prev_contribution", y="curr_contribution",
                            color="_color", hover_data=["player_name", "prev_season", "curr_season",
                                                        "prev_team", "curr_team"])
        elif color_by == "Switched Team":
            df["_color"] = df["switched_team"].map({True: "Switched", False: "Same Team"})
            fig = px.scatter(df, x="prev_contribution", y="curr_contribution",
                            color="_color", hover_data=["player_name", "prev_season", "curr_season",
                                                        "prev_team", "curr_team"])
        elif color_by == "Current League":
            fig = px.scatter(df, x="prev_contribution", y="curr_contribution",
                            color="curr_league", hover_data=["player_name", "prev_season", "curr_season",
                                                             "prev_team", "curr_team"])
        else:
            fig = px.scatter(df, x="prev_contribution", y="curr_contribution",
                            hover_data=["player_name", "prev_season", "curr_season",
                                        "prev_team", "curr_team"])

        fig.update_traces(marker=dict(size=dot_size, opacity=0.6))

        # Add y=x identity line
        if show_identity:
            min_val = min(df["prev_contribution"].min(), df["curr_contribution"].min())
            max_val = max(df["prev_contribution"].max(), df["curr_contribution"].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines", line=dict(dash="dash", color="gray", width=2),
                name="y = x (no change)", showlegend=True
            ))

        # Add regression line
        if show_regression and len(df) > 10:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df["prev_contribution"], df["curr_contribution"]
            )
            x_line = np.array([df["prev_contribution"].min(), df["prev_contribution"].max()])
            y_line = slope * x_line + intercept
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines", line=dict(color="red", width=2),
                name=f"Regression (r={r_value:.3f})", showlegend=True
            ))

        fig.update_layout(
            xaxis_title="Previous Season Contribution",
            yaxis_title="Current Season Contribution",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # === NOTABLE TRANSITIONS (always visible) ===
    st.subheader("Notable Transitions")
    df["change"] = df["curr_contribution"] - df["prev_contribution"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Improvers**")
        improvers = df.nlargest(10, "change")[
            ["player_name", "prev_season", "curr_season", "prev_contribution", "curr_contribution", "change"]
        ].copy()
        improvers.columns = ["Player", "From", "To", "Prev", "Curr", "Δ"]
        st.dataframe(improvers.round(3), hide_index=True, height=350)

    with col2:
        st.markdown("**Top Decliners**")
        decliners = df.nsmallest(10, "change")[
            ["player_name", "prev_season", "curr_season", "prev_contribution", "curr_contribution", "change"]
        ].copy()
        decliners.columns = ["Player", "From", "To", "Prev", "Curr", "Δ"]
        st.dataframe(decliners.round(3), hide_index=True, height=350)

    # === ADVANCED SECTIONS (in expanders) ===
    st.markdown("---")

    # --- Within-League Correlation ---
    with st.expander("📊 Within-League Correlation", expanded=False):
        st.markdown("How well does contribution predict next season **within the same league**?")

        league_stats = []
        for (country, league), group in df.groupby(["curr_country", "curr_league"]):
            if len(group) >= 20:
                corr = group["prev_contribution"].corr(group["curr_contribution"])
                league_stats.append({
                    "League": f"{country} - {league}",
                    "Transitions": len(group),
                    "Correlation": corr,
                    "Mean Change": group["curr_contribution"].mean() - group["prev_contribution"].mean()
                })

        if league_stats:
            league_df = pd.DataFrame(league_stats).sort_values("Correlation", ascending=False)
            st.dataframe(league_df.round(3), hide_index=True, height=300)
        else:
            st.info("Not enough data for league-level analysis (need ≥20 transitions per league).")

    # --- Transfer Corridors ---
    with st.expander("🔄 Transfer Corridors", expanded=False):
        st.markdown("Correlation between contribution in **source** and **destination** league.")

        transfers_df = compute_season_transitions(min_fte=min_fte)
        if season_gap_filter == "1 (consecutive)":
            transfers_df = transfers_df[transfers_df["season_gap"] == 1]
        if exclude_multi:
            transfers_df = transfers_df[~transfers_df["prev_is_multi"] & ~transfers_df["curr_is_multi"]]
        transfers_df = transfers_df[transfers_df["switched_league"]]

        if not transfers_df.empty:
            corridor_stats = []
            for (prev_lg, curr_lg), grp in transfers_df.groupby(["prev_league", "curr_league"]):
                if len(grp) >= 5:
                    corr = grp["prev_contribution"].corr(grp["curr_contribution"])
                    corridor_stats.append({
                        "From": prev_lg, "To": curr_lg, "N": len(grp),
                        "Corr": corr if not pd.isna(corr) else 0,
                        "Δ": grp["curr_contribution"].mean() - grp["prev_contribution"].mean()
                    })
            if corridor_stats:
                corridor_df = pd.DataFrame(corridor_stats).sort_values("N", ascending=False)
                st.dataframe(corridor_df.head(20).round(3), hide_index=True, height=400)
            else:
                st.info("Need ≥5 transfers per corridor.")
        else:
            st.info("No league-switching transfers found.")

    # --- Regression Prediction ---
    with st.expander("🎯 Regression-to-Mean Prediction", expanded=False):
        st.markdown("**Predict next-season contribution** using regression-to-mean.")
        st.markdown("Players with extreme contributions tend to regress toward the average.")

        all_players = load_all_players(min_fte=min_fte)
        if not all_players.empty:
            latest_season = all_players["season"].max()
            current_players = all_players[all_players["season"] == latest_season].copy()

            if not current_players.empty:
                overall_mean = all_players["contribution"].mean()
                r_value = correlation  # Use the correlation we already computed

                current_players["predicted_next"] = overall_mean + r_value * (current_players["contribution"] - overall_mean)
                current_players["expected_change"] = current_players["predicted_next"] - current_players["contribution"]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Expected to Decline** (high performers)")
                    decline = current_players[current_players["expected_change"] < -0.05].nlargest(10, "contribution")
                    if not decline.empty:
                        st.dataframe(decline[["player_name", "team", "contribution", "predicted_next"]].round(3), hide_index=True)
                    else:
                        st.caption("No significant expected declines")

                with col2:
                    st.markdown("**Expected to Improve** (underperformers)")
                    improve = current_players[current_players["expected_change"] > 0.05].nsmallest(10, "contribution")
                    if not improve.empty:
                        st.dataframe(improve[["player_name", "team", "contribution", "predicted_next"]].round(3), hide_index=True)
                    else:
                        st.caption("No significant expected improvements")


# =============================================================================
#                          FREE AGENTS (DISAPPEARED TALENT)
# =============================================================================

@st.cache_data
def compute_free_agents(min_fte: float = 3.0, max_age: int | None = None):
    """
    Find players who appeared in season N but NOT in season N+1 anywhere in the database.
    These are potential 'free agents' - talent that disappeared from tracked leagues.

    IMPORTANT: Only considers players as "disappeared" if BOTH:
    1. Their league has data for the following season
    2. Their team/club exists in the database for the following season
       (otherwise the team may have been relegated to an untracked league)

    Returns DataFrame with player info from their last tracked season.
    """
    if not Path(DB_PATH).exists():
        return pd.DataFrame()

    df = load_all_players(min_fte=min_fte)
    if df.empty:
        return pd.DataFrame()

    # Build set of (league, season) pairs that exist in the database
    league_seasons = set(zip(df["league"], df["season"]))

    # Build set of (team, league, season) pairs for club existence check
    # Extract individual teams from "Multiple Teams (A, B)" format
    team_league_seasons = set()
    for _, row in df.iterrows():
        team_str = str(row["team"]) if pd.notna(row["team"]) else ""
        league = row["league"]
        season = row["season"]

        if team_str.startswith("Multiple Teams ("):
            # Extract individual team names
            inner = team_str[16:-1]  # Remove "Multiple Teams (" and ")"
            teams = [t.strip() for t in inner.split(", ")]
            for t in teams:
                team_league_seasons.add((t, league, season))
        else:
            team_league_seasons.add((team_str, league, season))

    # For each player, find their last season per league
    # (a player might play in multiple leagues)
    player_league_last = df.groupby(["player_id", "league"])["season"].max().reset_index()
    player_league_last.columns = ["player_id", "league", "last_season"]

    # Only keep entries where the NEXT season exists for that league
    # (otherwise we can't know if the player disappeared or if we just don't have data)
    player_league_last["next_season_exists"] = player_league_last.apply(
        lambda row: (row["league"], row["last_season"] + 1) in league_seasons, axis=1
    )
    potential_free_agents = player_league_last[player_league_last["next_season_exists"]].copy()

    if potential_free_agents.empty:
        return pd.DataFrame()

    # Get the player's info from their last season in that league
    result = df.merge(
        potential_free_agents[["player_id", "league", "last_season"]],
        on=["player_id", "league"]
    )
    result = result[result["season"] == result["last_season"]]

    # Check if player's TEAM exists in the database for the next season
    # If the team doesn't exist, we can't conclude the player "disappeared"
    def team_exists_next_season(row):
        team_str = str(row["team"]) if pd.notna(row["team"]) else ""
        league = row["league"]
        next_season = row["last_season"] + 1

        if team_str.startswith("Multiple Teams ("):
            # Check if ANY of the teams exists next season
            inner = team_str[16:-1]
            teams = [t.strip() for t in inner.split(", ")]
            return any((t, league, next_season) in team_league_seasons for t in teams)
        else:
            return (team_str, league, next_season) in team_league_seasons

    result["team_exists_next"] = result.apply(team_exists_next_season, axis=1)

    # Only keep players whose team still exists next season
    result = result[result["team_exists_next"]].copy()

    if result.empty:
        return pd.DataFrame()

    # Check if player appeared ANYWHERE in the database in the next season
    # (not just the same league - they might have transferred)
    next_season_players = df[["player_id", "season"]].copy()
    next_season_players["check_season"] = next_season_players["season"] - 1
    next_season_players = next_season_players[["player_id", "check_season"]].drop_duplicates()

    result = result.merge(
        next_season_players,
        left_on=["player_id", "last_season"],
        right_on=["player_id", "check_season"],
        how="left",
        indicator=True
    )

    # Keep only those who did NOT appear in next season anywhere
    result = result[result["_merge"] == "left_only"].drop(columns=["_merge", "check_season"])

    # Apply age filter if specified (exclude unknown ages when filtering)
    if max_age is not None and "age" in result.columns:
        result = result[result["age"].notna() & (result["age"] <= max_age)]

    # Drop helper columns
    result = result.drop(columns=["next_season_exists", "team_exists_next"], errors="ignore")

    return result


def page_free_agents():
    """Top Free Agents - players who disappeared from the database."""
    st.header("🔍 Top Free Agents")

    st.markdown("""
    Identify top contributors who **did not appear in the following season** anywhere in our database.
    These could be players who retired, moved to untracked leagues, or represent untapped scouting opportunities.
    """)

    # Load base data for filter options with spinner
    with st.spinner("Loading player data..."):
        df_all = load_all_players(min_fte=0)

    if df_all.empty:
        render_empty_state(
            "No player data available",
            "Ensure the database has player records"
        )
        return

    # === FILTERS IN SIDEBAR ===
    with st.sidebar:
        st.subheader("Filters")

        # Age/FTE filters
        with st.expander("👤 Player Filters", expanded=True):
            age_filter = st.selectbox(
                "Max Age",
                ["No limit", "≤19 (Youth)", "≤21 (U21)", "≤23 (Young)", "≤25", "≤30"],
                index=0,
                key="fa_age",
                help="Filter by maximum player age in their last season"
            )
            max_age = None
            if age_filter == "≤19 (Youth)":
                max_age = 19
            elif age_filter == "≤21 (U21)":
                max_age = 21
            elif age_filter == "≤23 (Young)":
                max_age = 23
            elif age_filter == "≤25":
                max_age = 25
            elif age_filter == "≤30":
                max_age = 30

            min_fte = st.slider("Minimum FTE Games", 0.0, 20.0, 3.0, 0.5,
                               key="fa_fte",
                               help="Full-Time Equivalent: minutes ÷ 90. E.g., 3 FTE = ~270 min.")

            # Position filter
            position_options = ["All Positions", "Goalkeeper", "Defender", "Midfielder", "Attacker"]
            selected_position = st.selectbox(
                "Position",
                position_options,
                index=0,
                key="fa_position",
                help="Filter by player position"
            )

            min_contribution = st.slider("Min Contribution", -1.0, 2.0, 0.0, 0.05,
                                        key="fa_contrib",
                                        help="Regression coefficient: positive = improves team scoring, "
                                             "negative = worsens it. Range typically -0.5 to +1.0.")

            top_n = st.selectbox("Show Top N", [25, 50, 100, 200, 500], index=1,
                                key="fa_top_n",
                                help="Number of top players to display")

        # Location filters (cascading)
        with st.expander("📍 Location", expanded=True):
            all_countries = sorted(df_all["country"].unique())
            selected_countries = st.multiselect(
                "Country", ["All"] + all_countries, default=["All"],
                key="fa_countries",
                help=f"{len(all_countries)} countries available"
            )

            # Cascade: leagues depend on country
            if "All" not in selected_countries and selected_countries:
                available_leagues = []
                for country in selected_countries:
                    available_leagues.extend(get_leagues_for_country(country))
                available_leagues = sorted(set(available_leagues))
            else:
                available_leagues = sorted(df_all["league"].unique())
            selected_leagues = st.multiselect(
                "League", ["All"] + available_leagues, default=["All"],
                key="fa_leagues",
                help=f"{len(available_leagues)} leagues available"
            )

            # Cascade: seasons depend on country/league
            if "All" not in selected_countries and selected_countries or "All" not in selected_leagues and selected_leagues:
                available_seasons = get_seasons_for_league(
                    league=selected_leagues[0] if "All" not in selected_leagues and len(selected_leagues) == 1 else None,
                    country=selected_countries[0] if "All" not in selected_countries and len(selected_countries) == 1 else None
                )
            else:
                available_seasons = sorted(df_all["season"].unique())
            if len(available_seasons) >= 2:
                season_range = st.slider(
                    "Last Seen Season Range",
                    min(available_seasons), max(available_seasons) - 1,
                    (min(available_seasons), max(available_seasons) - 1),
                    key="fa_seasons",
                    help="Filter by the season when players were last seen"
                )
            else:
                season_range = (min(available_seasons), max(available_seasons) if available_seasons else 2024)

            # Team filter (cascades from country/league)
            if "All" not in selected_leagues and selected_leagues:
                available_teams = []
                for league in selected_leagues:
                    country = None
                    if "All" not in selected_countries and len(selected_countries) == 1:
                        country = selected_countries[0]
                    available_teams.extend(get_teams_for_league_season(league=league, country=country))
                available_teams = sorted(set(available_teams))
                selected_teams = st.multiselect(
                    "Team", ["All"] + available_teams, default=["All"],
                    key="fa_teams",
                    help=f"{len(available_teams)} teams available"
                )
            elif "All" not in selected_countries and selected_countries:
                available_teams = []
                for country in selected_countries:
                    for league in get_leagues_for_country(country):
                        available_teams.extend(get_teams_for_league_season(league=league, country=country))
                available_teams = sorted(set(available_teams))
                selected_teams = st.multiselect(
                    "Team", ["All"] + available_teams, default=["All"],
                    key="fa_teams",
                    help=f"{len(available_teams)} teams available"
                )
            else:
                selected_teams = ["All"]
                st.caption("Select a country or league to filter by team")

    # Compute free agents
    with st.spinner("Finding disappeared talent..."):
        free_agents = compute_free_agents(min_fte=min_fte, max_age=max_age)

    if free_agents.empty:
        st.info("No free agents found with current filters. Try adjusting the minimum FTE or age filter.")
        return

    # Apply additional filters
    df = free_agents.copy()

    # Contribution filter
    df = df[df["contribution"] >= min_contribution]

    # League filter
    if "All" not in selected_leagues and selected_leagues:
        df = df[df["league"].isin(selected_leagues)]

    # Country filter
    if "All" not in selected_countries and selected_countries:
        df = df[df["country"].isin(selected_countries)]

    # Team filter (handles "Multiple Teams (A, B)" format)
    if "All" not in selected_teams and selected_teams:
        team_mask = df["team"].apply(
            lambda t: any(team in str(t) for team in selected_teams) if pd.notna(t) else False
        )
        df = df[team_mask]

    # Season range filter
    df = df[(df["last_season"] >= season_range[0]) & (df["last_season"] <= season_range[1])]

    # Position filter
    if selected_position != "All Positions" and "position" in df.columns:
        df = df[df["position"] == selected_position]

    if df.empty:
        st.warning("No players match the current filters. Try: removing age limit, "
                   "lowering minimum contribution, expanding the season range, or selecting a different position.")
        return

    # Sort by contribution and take top N
    df = df.nlargest(top_n, "contribution")

    st.caption(f"Showing {len(df):,} players who disappeared after their last season")

    # === DISPLAY TABLE ===
    st.subheader("Top Disappeared Contributors")

    # Prepare display columns
    display_cols = ["player_name", "last_season", "country", "league", "team", "contribution", "position"]
    if "age" in df.columns:
        display_cols.insert(5, "age")
    if "nationality" in df.columns:
        display_cols.append("nationality")
    display_cols.extend(["FTE_games_played", "goals", "assists"])

    # Filter to available columns
    display_cols = [c for c in display_cols if c in df.columns]

    display_df = df[display_cols].copy()

    # Rename columns with custom mapping for clarity
    column_rename = {
        "player_name": "Player Name",
        "last_season": "Last Season",
        "country": "Country",
        "league": "League",
        "team": "Team",
        "contribution": "Contribution",
        "position": "Position",
        "age": "Age (at season)",  # Clarify this is age at time of season
        "nationality": "Nationality",
        "FTE_games_played": "FTE Games",
        "goals": "Goals",
        "assists": "Assists"
    }
    display_df.columns = [column_rename.get(c, c.replace("_", " ").title()) for c in display_cols]

    st.dataframe(display_df.round(3), hide_index=True, height=600)

    # === BREAKDOWN BY LEAGUE ===
    st.subheader("Breakdown by League")

    league_stats = df.groupby("league").agg({
        "player_id": "count",
        "contribution": ["mean", "max", "sum"]
    }).round(3)
    league_stats.columns = ["Count", "Avg Contribution", "Max Contribution", "Total Contribution"]
    league_stats = league_stats.sort_values("Count", ascending=False)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.dataframe(league_stats.reset_index(), hide_index=True, height=400)

    with col2:
        # Bar chart of free agents by league
        fig = px.bar(
            league_stats.reset_index().head(15),
            x="league",
            y="Count",
            color="Avg Contribution",
            title="Free Agents by League (Top 15)",
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # === AGE DISTRIBUTION ===
    if "age" in df.columns and df["age"].notna().sum() > 0:
        st.subheader("Age Distribution of Disappeared Players")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df[df["age"].notna()],
                x="age",
                nbins=20,
                title="Age Distribution",
                color_discrete_sequence=["steelblue"]
            )
            fig.update_layout(xaxis_title="Age", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Age vs Contribution scatter
            fig = px.scatter(
                df[df["age"].notna()],
                x="age",
                y="contribution",
                hover_data=["player_name", "league", "team", "last_season"],
                title="Age vs Contribution",
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
#                          LEAGUE QUALITY (SYNTHETIC CONTROL)
# =============================================================================


@st.cache_data
def compute_synthetic_control_matches(min_fte: float = 5.0, n_matches: int = 5):
    """
    For players who changed leagues, find similar peers who stayed.

    Returns DataFrame with:
    - Switcher info (player who changed leagues)
    - Average performance of matched peers who stayed
    - Inferred league quality difference
    """
    if not Path(DB_PATH).exists():
        return pd.DataFrame()

    # Load all players
    conn = sqlite3.connect(DB_PATH)
    players = pd.read_sql_query(f"""
        SELECT player_id, player_name, "team(s)" as team, league, country, season,
               {CONTRIB_COL} as contribution, FTE_games_played, position
        FROM players
        WHERE {CONTRIB_COL} IS NOT NULL AND FTE_games_played >= ?
        ORDER BY player_id, season
    """, conn, params=(min_fte,))
    conn.close()

    if players.empty:
        return pd.DataFrame()

    # Ensure season is numeric (in case stored as string)
    players["season"] = pd.to_numeric(players["season"], errors="coerce")
    players = players[players["season"].notna()]

    # Ensure contribution is numeric
    players["contribution"] = pd.to_numeric(players["contribution"], errors="coerce")

    # Normalize player_id
    players["pid_norm"] = players["player_id"].apply(_normalize_player_id)

    # Try to get profile data for matching (age, height, weight)
    profile_data = {}
    if Path(PROFILES_DB_PATH).exists():
        try:
            conn = sqlite3.connect(PROFILES_DB_PATH)
            profiles = pd.read_sql_query("""
                SELECT player_id, height, weight
                FROM player_profiles
            """, conn)
            conn.close()
            profiles["pid_norm"] = profiles["player_id"].apply(_normalize_player_id)
            for _, row in profiles.iterrows():
                profile_data[row["pid_norm"]] = {
                    "height": row.get("height"),
                    "weight": row.get("weight")
                }
        except Exception:
            pass

    # Find players who changed leagues (consecutive seasons)
    switchers = []
    stayers = []

    for pid, group in players.groupby("pid_norm"):
        group = group.sort_values("season")
        rows = group.to_dict("records")

        for i in range(1, len(rows)):
            prev = rows[i-1]
            curr = rows[i]

            # Only consider consecutive seasons
            if curr["season"] - prev["season"] != 1:
                continue

            league_changed = prev["league"] != curr["league"] or prev["country"] != curr["country"]

            entry = {
                "player_id": pid,
                "player_name": curr["player_name"],
                "season": curr["season"],
                "prev_season": prev["season"],
                "prev_league": f"{prev['country']} - {prev['league']}",
                "curr_league": f"{curr['country']} - {curr['league']}",
                "prev_contribution": prev["contribution"],
                "curr_contribution": curr["contribution"],
                "contribution_change": curr["contribution"] - prev["contribution"],
                "position": prev.get("position", "Unknown"),
                "height": profile_data.get(pid, {}).get("height"),
                "weight": profile_data.get(pid, {}).get("weight"),
            }

            if league_changed:
                switchers.append(entry)
            else:
                stayers.append(entry)

    if not switchers or not stayers:
        return pd.DataFrame()

    switchers_df = pd.DataFrame(switchers)
    stayers_df = pd.DataFrame(stayers)

    # For each switcher, find n_matches similar stayers from the SAME source league
    results = []

    for _, switcher in switchers_df.iterrows():
        # Get stayers from the same source league and season
        pool = stayers_df[
            (stayers_df["prev_league"] == switcher["prev_league"]) &
            (stayers_df["season"] == switcher["season"])
        ].copy()

        if pool.empty:
            continue

        # Calculate similarity score based on available features
        # Features: prev_contribution, position (if available)
        pool["contrib_diff"] = abs(pool["prev_contribution"] - switcher["prev_contribution"])

        # Position match (bonus for same position)
        pool["pos_match"] = (pool["position"] == switcher["position"]).astype(float) * 0.1

        # Height/weight similarity (if available)
        pool["height_diff"] = 0.0
        pool["weight_diff"] = 0.0
        switcher_height = pd.to_numeric(switcher["height"], errors="coerce")
        switcher_weight = pd.to_numeric(switcher["weight"], errors="coerce")
        if pd.notna(switcher_height):
            pool_height = pd.to_numeric(pool["height"], errors="coerce").fillna(0)
            pool["height_diff"] = abs(pool_height - switcher_height) / 100
        if pd.notna(switcher_weight):
            pool_weight = pd.to_numeric(pool["weight"], errors="coerce").fillna(0)
            pool["weight_diff"] = abs(pool_weight - switcher_weight) / 50

        # Combined similarity (lower is better)
        pool["similarity"] = pool["contrib_diff"] - pool["pos_match"] + pool["height_diff"] + pool["weight_diff"]

        # Get top n matches
        matches = pool.nsmallest(n_matches, "similarity")

        if len(matches) < 2:  # Need at least 2 matches for meaningful comparison
            continue

        # Calculate synthetic control outcome
        avg_stayer_change = matches["contribution_change"].mean()
        std_stayer_change = matches["contribution_change"].std()

        # League quality effect = switcher's change - what stayers experienced
        league_effect = switcher["contribution_change"] - avg_stayer_change

        results.append({
            "player_name": switcher["player_name"],
            "player_id": switcher["player_id"],
            "season": switcher["season"],
            "from_league": switcher["prev_league"],
            "to_league": switcher["curr_league"],
            "prev_contribution": switcher["prev_contribution"],
            "curr_contribution": switcher["curr_contribution"],
            "switcher_change": switcher["contribution_change"],
            "avg_stayer_change": avg_stayer_change,
            "std_stayer_change": std_stayer_change,
            "league_effect": league_effect,
            "n_matches": len(matches),
            "match_names": ", ".join(matches["player_name"].tolist()[:3])
        })

    return pd.DataFrame(results)


@st.cache_data(ttl=300)
def _load_league_quality_parquet() -> pd.DataFrame:
    """Load league quality parquet file (contains rankings, pairwise, transfers)."""
    if _DATA_FORMAT == "parquet" and _DATA_DIR:
        lq_path = _DATA_DIR / "league_quality.parquet"
        if lq_path.exists():
            return pd.read_parquet(lq_path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_precomputed_league_rankings(season: int = None) -> pd.DataFrame:
    """Load pre-computed league rankings from parquet or database."""
    # Try parquet first
    lq_df = _load_league_quality_parquet()
    if not lq_df.empty and "_table" in lq_df.columns:
        rankings = lq_df[lq_df["_table"] == "lq_league_rankings"].copy()
        if not rankings.empty:
            cols = ["league", "quality_index", "std_error", "reference_league", "rank", "created_at"]
            cols = [c for c in cols if c in rankings.columns]
            if season is None:
                # Cross-season pooled rankings (where season is null)
                df = rankings[rankings["season"].isna()][cols].sort_values("rank")
            else:
                df = rankings[rankings["season"] == season][cols].sort_values("rank")
            return df.reset_index(drop=True)

    # Fall back to SQLite
    if not Path(DB_PATH).exists():
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_PATH)

        if season is None:
            # Cross-season pooled rankings
            query = """
                SELECT league, quality_index, std_error, reference_league, rank, created_at
                FROM lq_league_rankings
                WHERE season IS NULL
                ORDER BY rank
            """
            df = pd.read_sql_query(query, conn)
        else:
            query = """
                SELECT league, quality_index, std_error, reference_league, rank, created_at
                FROM lq_league_rankings
                WHERE season = ?
                ORDER BY rank
            """
            df = pd.read_sql_query(query, conn, params=(season,))

        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_precomputed_pairwise_estimates(season: int = None) -> pd.DataFrame:
    """Load pre-computed pairwise estimates from parquet or league_quality.db."""
    # Try parquet first
    lq_df = _load_league_quality_parquet()
    if not lq_df.empty and "_table" in lq_df.columns:
        pairwise = lq_df[lq_df["_table"] == "lq_pairwise_estimates"].copy()
        if not pairwise.empty:
            cols = ["from_league", "to_league", "season", "estimate", "std_error", "n_transfers",
                    "n_ab", "n_ba", "t_statistic", "p_value", "ci_lower", "ci_upper", "significant", "created_at"]
            cols = [c for c in cols if c in pairwise.columns]
            if season is None:
                df = pairwise[cols].sort_values(["season", "n_transfers"], ascending=[False, False])
            else:
                df = pairwise[pairwise["season"] == season][cols].sort_values("n_transfers", ascending=False)
            return df.reset_index(drop=True)

    # Fall back to SQLite
    lq_db = Path("league_quality.db")
    if not lq_db.exists():
        # Fallback to old location
        lq_db = Path(DB_PATH)
    if not lq_db.exists():
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(lq_db)

        if season is None:
            # Load all season-specific estimates
            query = """
                SELECT from_league, to_league, season, estimate, std_error, n_transfers,
                       n_ab, n_ba, t_statistic, p_value, ci_lower, ci_upper, significant, created_at
                FROM lq_pairwise_estimates
                ORDER BY season DESC, n_transfers DESC
            """
            df = pd.read_sql_query(query, conn)
        else:
            query = """
                SELECT from_league, to_league, season, estimate, std_error, n_transfers,
                       n_ab, n_ba, t_statistic, p_value, ci_lower, ci_upper, significant, created_at
                FROM lq_pairwise_estimates
                WHERE season = ?
                ORDER BY n_transfers DESC
            """
            df = pd.read_sql_query(query, conn, params=(season,))

        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# =============================================================================
# Synthetic Results (from compare_leagues.py) - parquet or SQLite
# =============================================================================

SYNTHETIC_RESULTS_DB = Path(__file__).parent / "synthetic_results.db"


@st.cache_data(ttl=300)
def _load_synthetic_comparisons_parquet() -> pd.DataFrame:
    """Load synthetic comparisons from parquet file."""
    parquet_path = _DATA_DIR / "synthetic_comparisons.parquet" if _DATA_DIR else None
    if parquet_path and parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_synthetic_league_comparisons(pooled: bool = True, season: int = None) -> pd.DataFrame:
    """
    Load league comparisons from parquet (preferred) or SQLite fallback.

    Args:
        pooled: If True, load pooled (all-seasons) results. If False, load season-specific.
        season: If pooled=False, filter to this specific season.

    Returns:
        DataFrame with columns: source_country, source_league, dest_country, dest_league,
                               delta, delta_std, delta_se, t_statistic, p_value, n_transfers, n_controls
    """
    # Try parquet first (preferred for Streamlit Cloud)
    df = _load_synthetic_comparisons_parquet()

    if df.empty and SYNTHETIC_RESULTS_DB.exists():
        # Fall back to SQLite for local development
        try:
            conn = sqlite3.connect(SYNTHETIC_RESULTS_DB)
            df = pd.read_sql_query("SELECT * FROM league_comparisons", conn)
            conn.close()
        except Exception as e:
            print(f"Error loading synthetic results from SQLite: {e}")
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Filter by pooled/season
    if pooled:
        df = df[df["pooled"] == 1].sort_values("n_transfers", ascending=False)
    else:
        df = df[df["pooled"] == 0]
        if season is not None:
            df = df[df["season"] == season]
        df = df.sort_values(["season", "n_transfers"], ascending=[False, False])

    # Add combined league names for display
    if not df.empty:
        df["from_league"] = df["source_country"] + " - " + df["source_league"]
        df["to_league"] = df["dest_country"] + " - " + df["dest_league"]

    return df


@st.cache_data(ttl=300)
def _get_synthetic_comparison_seasons() -> list:
    """Get list of seasons with synthetic comparison data."""
    df = _load_synthetic_comparisons_parquet()

    if df.empty and SYNTHETIC_RESULTS_DB.exists():
        try:
            conn = sqlite3.connect(SYNTHETIC_RESULTS_DB)
            query = "SELECT DISTINCT season FROM league_comparisons WHERE pooled = 0 AND season IS NOT NULL ORDER BY season DESC"
            df = pd.read_sql_query(query, conn)
            conn.close()
        except Exception:
            return []

    if df.empty:
        return []

    # Filter to non-pooled seasons
    if "pooled" in df.columns:
        df = df[df["pooled"] == 0]

    if "season" not in df.columns:
        return []

    return sorted(df["season"].dropna().unique().tolist(), reverse=True)


@st.cache_data(ttl=300)
def _get_synthetic_leagues() -> list:
    """Get list of leagues that have synthetic comparison data."""
    df = _load_synthetic_comparisons_parquet()

    if df.empty and SYNTHETIC_RESULTS_DB.exists():
        try:
            conn = sqlite3.connect(SYNTHETIC_RESULTS_DB)
            query = """
                SELECT DISTINCT league FROM (
                    SELECT source_country || ' - ' || source_league as league FROM league_comparisons
                    UNION
                    SELECT dest_country || ' - ' || dest_league as league FROM league_comparisons
                )
                ORDER BY league
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df["league"].tolist()
        except Exception:
            return []

    if df.empty:
        return []

    # Build league list from parquet
    leagues = set()
    if "source_country" in df.columns and "source_league" in df.columns:
        leagues.update((df["source_country"] + " - " + df["source_league"]).unique())
    if "dest_country" in df.columns and "dest_league" in df.columns:
        leagues.update((df["dest_country"] + " - " + df["dest_league"]).unique())

    return sorted(leagues)


@st.cache_data(ttl=300)
def _get_available_comparison_seasons() -> list:
    """Get list of seasons with pairwise comparison data."""
    # First try synthetic_results.db
    synthetic_seasons = _get_synthetic_comparison_seasons()
    if synthetic_seasons:
        return synthetic_seasons

    # Try parquet
    lq_df = _load_league_quality_parquet()
    if not lq_df.empty and "_table" in lq_df.columns:
        pairwise = lq_df[lq_df["_table"] == "lq_pairwise_estimates"]
        if not pairwise.empty and "season" in pairwise.columns:
            seasons = pairwise["season"].dropna().unique().tolist()
            return sorted([int(s) for s in seasons], reverse=True)

    # Fallback to league_quality.db
    lq_db = Path("league_quality.db")
    if not lq_db.exists():
        lq_db = Path(DB_PATH)
    if not lq_db.exists():
        return []

    try:
        conn = sqlite3.connect(lq_db)
        query = "SELECT DISTINCT season FROM lq_pairwise_estimates WHERE season IS NOT NULL ORDER BY season DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df["season"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=300)
def _get_leagues_with_comparisons() -> list:
    """Get list of leagues that have comparison data."""
    # First try synthetic_results.db
    synthetic_leagues = _get_synthetic_leagues()
    if synthetic_leagues:
        return synthetic_leagues

    # Try parquet
    lq_df = _load_league_quality_parquet()
    if not lq_df.empty and "_table" in lq_df.columns:
        pairwise = lq_df[lq_df["_table"] == "lq_pairwise_estimates"]
        if not pairwise.empty:
            leagues = set()
            if "from_league" in pairwise.columns:
                leagues.update(pairwise["from_league"].dropna().unique())
            if "to_league" in pairwise.columns:
                leagues.update(pairwise["to_league"].dropna().unique())
            if leagues:
                return sorted(leagues)

    # Fallback to league_quality.db
    lq_db = Path("league_quality.db")
    if not lq_db.exists():
        lq_db = Path(DB_PATH)
    if not lq_db.exists():
        return []

    try:
        conn = sqlite3.connect(lq_db)
        query = """
            SELECT DISTINCT league FROM (
                SELECT from_league as league FROM lq_pairwise_estimates
                UNION
                SELECT to_league as league FROM lq_pairwise_estimates
            ) ORDER BY league
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df["league"].tolist()
    except Exception:
        return []


def _normalize_league_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize league pairs so harder league is always League A (left).

    Convention: negative estimate = destination is harder.
    After normalization: estimate is always positive, League A is always harder.
    """
    normalized = df.copy()

    # For each row where estimate < 0, swap leagues and flip sign
    needs_swap = normalized["estimate"] < 0

    # Swap from_league and to_league
    temp_from = normalized.loc[needs_swap, "from_league"].copy()
    normalized.loc[needs_swap, "from_league"] = normalized.loc[needs_swap, "to_league"]
    normalized.loc[needs_swap, "to_league"] = temp_from

    # Swap n_ab and n_ba
    if "n_ab" in normalized.columns and "n_ba" in normalized.columns:
        temp_n_ab = normalized.loc[needs_swap, "n_ab"].copy()
        normalized.loc[needs_swap, "n_ab"] = normalized.loc[needs_swap, "n_ba"]
        normalized.loc[needs_swap, "n_ba"] = temp_n_ab

    # Flip estimate sign
    normalized.loc[needs_swap, "estimate"] = -normalized.loc[needs_swap, "estimate"]

    return normalized


def _display_precomputed_league_quality(rankings: pd.DataFrame, pairwise: pd.DataFrame):
    """Display pre-computed bilateral league comparisons."""

    # Load full pairwise data with directional info
    all_pairwise = _load_precomputed_pairwise_estimates()

    if all_pairwise.empty:
        render_empty_state("No comparisons computed", "Run league_quality.py --incremental to generate comparisons")
        return

    # Normalize pairs: harder league is always League A (left), estimate is always positive
    all_pairwise = _normalize_league_pairs(all_pairwise)

    # Get available seasons and leagues for filters
    available_seasons = _get_available_comparison_seasons()
    available_leagues = _get_leagues_with_comparisons()

    # Sidebar filters
    with st.sidebar:
        st.subheader("Comparison Filters")

        season_options = ["All Seasons (Pooled)"] + [str(s) for s in available_seasons]
        selected_season = st.selectbox("Season", season_options, index=0, key="lq_season")

        league_options = ["All Leagues"] + available_leagues
        selected_league = st.selectbox("Focus on League", league_options, index=0,
                                       help="Show comparisons involving this league", key="lq_league")

        show_significant_only = st.checkbox("Significant Only", value=True, key="lq_sig",
                                            help="Show only statistically significant comparisons (p < 0.05)")

        min_transfers = st.slider("Min Transfers", 3, 30, 5, key="lq_min_transfers",
                                  help="Minimum player transfers to include a comparison")

    # Load and filter data
    if selected_season == "All Seasons (Pooled)":
        # For pooled, get estimates where season is NULL (cross-season)
        filtered = all_pairwise[all_pairwise["season"].isna()].copy()
        if filtered.empty:
            # Fallback to aggregating all seasons
            filtered = all_pairwise.copy()
        season_label = "All Seasons"
    else:
        filtered = all_pairwise[all_pairwise["season"] == int(selected_season)].copy()
        season_label = f"Season {selected_season}"

    # Apply filters
    if selected_league != "All Leagues":
        filtered = filtered[
            (filtered["from_league"] == selected_league) |
            (filtered["to_league"] == selected_league)
        ]

    if show_significant_only:
        filtered = filtered[filtered["significant"] == 1]

    filtered = filtered[filtered["n_transfers"] >= min_transfers]

    if filtered.empty:
        st.warning("No comparisons match the selected filters. Try adjusting the filters.")
        return

    # Summary metrics
    st.subheader(f"Bilateral Comparisons ({season_label})")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("League Pairs", len(filtered))
    col2.metric("Significant", f"{(filtered['significant'] == 1).sum()}")
    total_transfers = filtered['n_transfers'].sum()
    col3.metric("Total Transfers", f"{total_transfers:,}")
    unique_leagues = set(filtered['from_league'].tolist() + filtered['to_league'].tolist())
    col4.metric("Leagues Involved", len(unique_leagues))

    st.markdown("""
    Each row shows a **bilateral comparison** between two leagues based on player transfers.
    - **League A** (Harder) → **League B** (Easier): Pairs are normalized so A is always the harder league
    - **Estimate**: Quality difference (always positive; larger = bigger quality gap)
    - **A→B / B→A**: Player transfers in each direction
    - **✓**: Statistically significant (p < 0.05)
    """)

    # Prepare display table with directional info
    display_df = filtered.copy()

    # Handle missing n_ab/n_ba columns
    if 'n_ab' not in display_df.columns:
        display_df['n_ab'] = display_df['n_transfers'] // 2
        display_df['n_ba'] = display_df['n_transfers'] - display_df['n_ab']

    # Format for display (A = harder league, B = easier league after normalization)
    display_df["Harder (A)"] = display_df["from_league"].str.replace("_", " - ")
    display_df["Easier (B)"] = display_df["to_league"].str.replace("_", " - ")
    display_df["A→B"] = display_df["n_ab"].fillna(0).astype(int)
    display_df["B→A"] = display_df["n_ba"].fillna(0).astype(int)
    display_df["Total"] = display_df["n_transfers"]
    display_df["Estimate"] = display_df["estimate"].round(3)
    display_df["SE"] = display_df["std_error"].round(3)
    display_df["p-value"] = display_df["p_value"].round(4)
    display_df["Sig"] = display_df["significant"].apply(lambda x: "✓" if x else "")

    # Sort by total transfers (most data first)
    display_df = display_df.sort_values("n_transfers", ascending=False)

    # Select columns for display
    show_cols = ["Harder (A)", "Easier (B)", "Estimate", "SE", "A→B", "B→A", "Total", "p-value", "Sig"]
    if "season" in display_df.columns and display_df["season"].notna().any():
        display_df["Season"] = display_df["season"].astype(int)
        show_cols = ["Season"] + show_cols

    st.dataframe(display_df[show_cols], hide_index=True, height=400)

    # Network Visualization (primary view)
    st.markdown("---")
    st.subheader("Comparison Network")

    st.markdown("""
    **Directed graph** of significant league comparisons.
    - **Arrow direction**: Points from easier league → harder league
    - **Arrow thickness**: Proportional to number of transfers
    - **Node size**: Number of connections (more = more comparisons)
    """)

    # Build network
    import networkx as nx

    net_data = filtered[filtered["significant"] == 1] if show_significant_only else filtered

    if len(net_data) >= 2:
        G = nx.DiGraph()

        for _, row in net_data.iterrows():
            harder_lg = row["from_league"].replace("_", " - ")  # After normalization, from_league = harder
            easier_lg = row["to_league"].replace("_", " - ")    # to_league = easier
            # Arrow points from easier → harder league
            G.add_edge(
                easier_lg, harder_lg,
                weight=abs(row["estimate"]),
                estimate=row["estimate"],
                n_transfers=row["n_transfers"],
                n_ab=row.get("n_ab", 0),
                n_ba=row.get("n_ba", 0)
            )

        # Layout
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except Exception:
            pos = nx.circular_layout(G)

        # Create plotly figure
        edge_annotations = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]  # Easier league (source)
            x1, y1 = pos[edge[1]]  # Harder league (target)

            estimate = edge[2].get("estimate", 0)
            n_transfers = edge[2].get("n_transfers", 0)
            # Orange/amber - contrasts with blue nodes and white background
            color = "rgba(230, 120, 20, 0.9)"

            edge_annotations.append(dict(
                ax=x0, ay=y0,
                x=x1, y=y1,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=3,  # Larger arrowhead style
                arrowsize=2.0,  # Bigger arrowhead (was 1.5)
                arrowwidth=max(2, min(6, n_transfers / 8)),  # Thicker lines (was 1-4, now 2-6)
                arrowcolor=color,
                hovertext=f"{edge[0]} → {edge[1]}: +{estimate:.3f} quality gap ({n_transfers} transfers)"
            ))

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = list(G.nodes())
        node_size = [12 + G.degree(n) * 4 for n in G.nodes()]

        # Node hover info
        node_hover = []
        for n in G.nodes():
            in_deg = G.in_degree(n)
            out_deg = G.out_degree(n)
            node_hover.append(f"{n}<br>Incoming: {in_deg}<br>Outgoing: {out_deg}")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(size=node_size, color="#1f77b4", line=dict(width=2, color="white")),
            text=node_text,
            textposition="top center",
            textfont=dict(size=9),
            hovertext=node_hover,
            hoverinfo="text",
            name="Leagues"
        ))

        fig.update_layout(
            annotations=edge_annotations,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            title=f"League Comparison Network ({len(G.nodes())} leagues, {len(G.edges())} comparisons)"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("Arrows point from easier leagues to harder leagues. "
                   "Arrow thickness indicates number of transfers supporting the comparison.")
    else:
        st.info("Not enough comparisons for network visualization.")

    # Focused league view
    if selected_league != "All Leagues" and len(filtered) >= 1:
        st.markdown("---")
        st.subheader(f"Focus: {selected_league.replace('_', ' - ')}")

        # Split comparisons
        as_source = filtered[filtered["from_league"] == selected_league].copy()
        as_dest = filtered[filtered["to_league"] == selected_league].copy()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Players leaving to other leagues** ({len(as_source)} comparisons)")
            if not as_source.empty:
                chart_data = []
                for _, row in as_source.iterrows():
                    dest = row["to_league"].replace("_", " - ")
                    chart_data.append({
                        "destination": dest,
                        "estimate": row["estimate"],
                        "se": row["std_error"],
                        "transfers": row["n_transfers"],
                        "sig": "✓" if row["significant"] else ""
                    })
                chart_df = pd.DataFrame(chart_data).sort_values("estimate")

                fig = px.bar(
                    chart_df, x="estimate", y="destination", orientation="h",
                    error_x="se",
                    color="estimate",
                    color_continuous_scale="RdYlGn",
                    hover_data=["transfers", "sig"],
                    title="Effect when leaving"
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=max(250, len(chart_df) * 25), showlegend=False,
                                  yaxis_title="", xaxis_title="Quality Difference")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No data")

        with col2:
            st.markdown(f"**Players arriving from other leagues** ({len(as_dest)} comparisons)")
            if not as_dest.empty:
                chart_data = []
                for _, row in as_dest.iterrows():
                    source = row["from_league"].replace("_", " - ")
                    chart_data.append({
                        "source": source,
                        "estimate": -row["estimate"],  # Flip for interpretation
                        "se": row["std_error"],
                        "transfers": row["n_transfers"],
                        "sig": "✓" if row["significant"] else ""
                    })
                chart_df = pd.DataFrame(chart_data).sort_values("estimate")

                fig = px.bar(
                    chart_df, x="estimate", y="source", orientation="h",
                    error_x="se",
                    color="estimate",
                    color_continuous_scale="RdYlGn",
                    hover_data=["transfers", "sig"],
                    title="Effect when arriving"
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=max(250, len(chart_df) * 25), showlegend=False,
                                  yaxis_title="", xaxis_title="Quality Difference (inverted)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No data")

    # Pair Time Series Analysis
    st.markdown("---")
    st.subheader("📈 Pair Time Series")

    st.markdown("""
    Select a league pair to see how the quality difference has evolved over seasons.
    The pair ordering is fixed (harder league = A) to ensure coefficients are comparable over time.
    """)

    # Get unique pairs from the original (unnormalized) data to show all available
    # Then normalize them for consistent display
    all_raw = _load_precomputed_pairwise_estimates()
    if not all_raw.empty:
        all_normalized = _normalize_league_pairs(all_raw)

        # Create unique pair identifiers (always A-B with A being harder)
        all_normalized["pair"] = all_normalized.apply(
            lambda r: f"{r['from_league'].replace('_', ' - ')} vs {r['to_league'].replace('_', ' - ')}", axis=1
        )

        # Get pairs that have multiple seasons of data
        pair_season_counts = all_normalized.groupby("pair")["season"].nunique()
        multi_season_pairs = pair_season_counts[pair_season_counts > 1].index.tolist()

        if multi_season_pairs:
            # Sort pairs alphabetically
            multi_season_pairs.sort()

            selected_pair = st.selectbox(
                "Select League Pair",
                options=["Select a pair..."] + multi_season_pairs,
                key="lq_pair_select",
                help="Choose a pair with multiple seasons of data to see coefficient trends"
            )

            if selected_pair != "Select a pair...":
                # Filter to this pair
                pair_data = all_normalized[all_normalized["pair"] == selected_pair].copy()
                pair_data = pair_data[pair_data["season"].notna()].sort_values("season")

                if len(pair_data) >= 2:
                    # Extract league names for display
                    harder_league = pair_data["from_league"].iloc[0].replace("_", " - ")
                    easier_league = pair_data["to_league"].iloc[0].replace("_", " - ")

                    # Create time series chart
                    fig = go.Figure()

                    # Add estimate line with confidence bands if available
                    pair_data["season_int"] = pair_data["season"].astype(int)

                    # Main estimate line
                    fig.add_trace(go.Scatter(
                        x=pair_data["season_int"],
                        y=pair_data["estimate"],
                        mode="lines+markers",
                        name="Quality Gap",
                        line=dict(color="steelblue", width=2),
                        marker=dict(size=10),
                        hovertemplate="Season %{x}<br>Gap: +%{y:.3f}<extra></extra>"
                    ))

                    # Add standard error bands
                    if "std_error" in pair_data.columns:
                        upper = pair_data["estimate"] + 1.96 * pair_data["std_error"]
                        lower = pair_data["estimate"] - 1.96 * pair_data["std_error"]

                        fig.add_trace(go.Scatter(
                            x=pair_data["season_int"].tolist() + pair_data["season_int"].tolist()[::-1],
                            y=upper.tolist() + lower.tolist()[::-1],
                            fill="toself",
                            fillcolor="rgba(70, 130, 180, 0.2)",
                            line=dict(color="rgba(255,255,255,0)"),
                            name="95% CI",
                            hoverinfo="skip"
                        ))

                    # Mark significant seasons
                    sig_data = pair_data[pair_data["significant"] == 1]
                    if not sig_data.empty:
                        fig.add_trace(go.Scatter(
                            x=sig_data["season_int"],
                            y=sig_data["estimate"],
                            mode="markers",
                            name="Significant",
                            marker=dict(size=14, color="green", symbol="circle-open", line=dict(width=2)),
                            hovertemplate="Season %{x}<br>Gap: +%{y:.3f} (significant)<extra></extra>"
                        ))

                    fig.update_layout(
                        title=f"Quality Gap Over Time: {harder_league} (harder) vs {easier_league} (easier)",
                        xaxis_title="Season",
                        yaxis_title="Quality Difference (positive = A is harder)",
                        height=400,
                        hovermode="x unified",
                        xaxis=dict(tickmode="array", tickvals=pair_data["season_int"].tolist()),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    # Add zero reference line
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

                    st.plotly_chart(fig, use_container_width=True)

                    # Show data table for this pair
                    with st.expander("View data"):
                        pair_display = pair_data[["season_int", "estimate", "std_error", "n_transfers", "n_ab", "n_ba", "significant"]].copy()
                        pair_display.columns = ["Season", "Estimate", "SE", "Total Transfers", "A→B", "B→A", "Significant"]
                        pair_display["Significant"] = pair_display["Significant"].apply(lambda x: "✓" if x else "")
                        pair_display["Estimate"] = pair_display["Estimate"].round(3)
                        pair_display["SE"] = pair_display["SE"].round(3)
                        st.dataframe(pair_display, hide_index=True)
                else:
                    st.info("Not enough seasonal data for this pair to show a time series.")
        else:
            st.info("No league pairs with multiple seasons of data available for time series analysis.")
    else:
        st.info("No pairwise data available.")

    # Export
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Export Comparisons (CSV)",
            data=filtered.to_csv(index=False),
            file_name="league_comparisons.csv",
            mime="text/csv"
        )


def page_league_quality():
    """Synthetic control analysis for league quality comparisons."""
    st.header("🔬 League Comparison")

    st.markdown("""
    Estimate **league quality differences** using synthetic control analysis of player transfers:

    1. Find players who **transferred between leagues**
    2. Match each with **similar players who stayed** (synthetic control)
    3. Compare contribution changes: **Δ = transferred - synthetic control**
    4. **Negative Δ** = destination is harder · **Positive Δ** = destination is easier

    *Data from `compare_leagues.py` using synthetic control methodology (Abadie 2021).*
    """)

    # Load synthetic results
    pooled_comparisons = _load_synthetic_league_comparisons(pooled=True)
    has_data = not pooled_comparisons.empty

    if not has_data:
        render_empty_state(
            "No comparison data available",
            "Run `python compare_leagues.py` to generate synthetic control comparisons"
        )
        return

    # Sidebar controls
    with st.sidebar:
        st.subheader("Analysis Settings")

        available_seasons = _get_synthetic_comparison_seasons()
        season_options = ["All Seasons (Pooled)"] + [str(s) for s in available_seasons]
        selected_season = st.selectbox("Season", season_options, index=0, key="lq_season_select")

        # League filter - get all unique leagues from pooled comparisons
        all_leagues = sorted(set(pooled_comparisons["from_league"].unique()) |
                            set(pooled_comparisons["to_league"].unique()))
        league_options = ["All Leagues"] + all_leagues
        selected_league = st.selectbox(
            "Focus on League",
            league_options,
            index=0,
            key="lq_league_filter",
            help="Filter to show comparisons involving this league"
        )

        min_transfers = st.slider("Min Transfers", 5, 100, 20,
                                 help="Minimum transfers to include a comparison")

        significance_level = st.select_slider(
            "Significance Level",
            options=[0.01, 0.05, 0.10, 0.20, 1.0],
            value=0.10,
            format_func=lambda x: f"p < {x}" if x < 1 else "All"
        )

    # Load data based on selection
    if selected_season == "All Seasons (Pooled)":
        comparisons = pooled_comparisons.copy()
    else:
        comparisons = _load_synthetic_league_comparisons(pooled=False, season=int(selected_season))

    if comparisons.empty:
        render_empty_state("No data for selection", "Try a different season or pooled view")
        return

    # Filter by selected league
    if selected_league != "All Leagues":
        comparisons = comparisons[
            (comparisons["from_league"] == selected_league) |
            (comparisons["to_league"] == selected_league)
        ].copy()

    # Filter by minimum transfers
    comparisons = comparisons[comparisons["n_transfers"] >= min_transfers].copy()

    if comparisons.empty:
        msg = f"No comparisons with ≥{min_transfers} transfers"
        if selected_league != "All Leagues":
            msg += f" involving {selected_league}"
        msg += ". Try lowering the threshold."
        st.warning(msg)
        return

    # Get last update time
    if "created_at" in comparisons.columns:
        last_update = comparisons["created_at"].max()
        st.caption(f"Last updated: {last_update}")

    # Summary metrics
    st.subheader("Overview")

    # Count unique league pairs (not direction-specific)
    unique_pairs = len(comparisons) // 2  # Data is symmetric
    total_transfers = comparisons["n_transfers"].sum() // 2
    significant = comparisons[comparisons["p_value"] < significance_level]
    n_significant = len(significant) // 2

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("League Pairs", f"{unique_pairs:,}")
    col2.metric("Total Transfers", f"{total_transfers:,}")
    col3.metric("Significant", f"{n_significant:,}", help=f"p < {significance_level}")
    col4.metric("Avg |Δ|", f"{comparisons['delta'].abs().mean():.4f}",
                help="Average absolute contribution difference")

    # Pairwise comparisons (the actual data we have)
    st.markdown("---")
    st.subheader("Pairwise League Comparisons")

    st.info(
        "**Note:** These are pairwise comparisons only. A global ranking would require "
        "transitivity (A > B and B > C → A > C), which isn't guaranteed in this data."
    )

    # Filter significant only for display
    if significance_level < 1.0:
        display_comparisons = comparisons[comparisons["p_value"] < significance_level].copy()
    else:
        display_comparisons = comparisons.copy()

    # Keep only one direction (positive delta = destination easier)
    display_comparisons = display_comparisons[display_comparisons["delta"] >= 0].copy()

    if not display_comparisons.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Hardest Transitions** *(biggest performance drop)*")
            # Actually, delta >= 0 means dest is easier, so we want delta < 0 for hard
            # Let me fix this - use the other direction
            hard_transitions = comparisons[comparisons["delta"] < -0.01].nsmallest(15, "delta")
            if not hard_transitions.empty:
                display_df = hard_transitions[["from_league", "to_league", "delta", "t_statistic", "p_value", "n_transfers"]].copy()
                display_df.columns = ["From", "To", "Δ", "t-stat", "p-value", "N"]
                st.dataframe(display_df.round(4), hide_index=True)
            else:
                st.caption("No significant hard transitions found")

        with col2:
            st.markdown("**Easiest Transitions** *(biggest performance gain)*")
            easy_transitions = comparisons[comparisons["delta"] > 0.01].nlargest(15, "delta")
            if not easy_transitions.empty:
                display_df = easy_transitions[["from_league", "to_league", "delta", "t_statistic", "p_value", "n_transfers"]].copy()
                display_df.columns = ["From", "To", "Δ", "t-stat", "p-value", "N"]
                st.dataframe(display_df.round(4), hide_index=True)
            else:
                st.caption("No significant easy transitions found")

    # Visualization: Scatter of delta vs n_transfers
    st.markdown("---")
    st.subheader("Comparison Quality")

    # Use one direction only for plotting
    plot_data = comparisons[comparisons["delta"] >= 0].copy()
    if len(plot_data) >= 5:
        plot_data["significant"] = plot_data["p_value"] < significance_level
        plot_data["label"] = plot_data["from_league"] + " → " + plot_data["to_league"]

        fig = px.scatter(
            plot_data,
            x="n_transfers",
            y="delta",
            color="significant",
            hover_data=["label", "t_statistic", "p_value"],
            title="League Quality Difference vs Sample Size",
            labels={"n_transfers": "Number of Transfers", "delta": "Δ (contribution difference)"},
            color_discrete_map={True: "#2196F3", False: "#BDBDBD"}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=400, showlegend=True)
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Blue = significant at p < {significance_level} · Gray = not significant")

    # Full data table
    st.markdown("---")
    st.subheader("All Comparisons")

    with st.expander("View Full Comparison Data", expanded=False):
        export_data = comparisons[comparisons["delta"] >= 0].copy()  # One direction only
        display_cols = ["from_league", "to_league", "delta", "delta_se", "t_statistic", "p_value", "n_transfers", "n_controls"]
        available_cols = [c for c in display_cols if c in export_data.columns]
        display_data = export_data[available_cols].copy()
        display_data.columns = ["From", "To", "Δ", "SE", "t-stat", "p-value", "Transfers", "Controls"][:len(available_cols)]
        st.dataframe(display_data.round(4), hide_index=True, height=400)

    # Export
    st.download_button(
        "📥 Export Comparisons",
        data=comparisons.to_csv(index=False),
        file_name="league_comparisons.csv",
        mime="text/csv"
    )


# =============================================================================
#                          HOME PAGE
# =============================================================================


def page_home():
    """Home page with narrative and visualizations explaining the contribution metric."""

    # Load data once for all visualizations
    df = load_all_players(min_fte=0)

    # Add position_group column for visualization
    # Note: position is already mapped to full names in load_all_players(), so just copy it
    if not df.empty and "position" in df.columns:
        df["position_group"] = df["position"]
    elif not df.empty:
        df["position_group"] = "Unknown"

    # ==========================================================================
    # TITLE AND DEFINITION
    # ==========================================================================
    st.title("⚽ Off-Balance-Sheet Contribution")

    st.markdown("""
    **Off-balance-sheet contributions** measure how much a player changes their team's goal difference
    when they are on the pitch — both scoring and defending — beyond goals and assists.

    These contributions are difficult to see with the naked eye, and traditional stats often miss or
    misrepresent them. Metrics like distance covered or passing success rate can be inflated and misleading.

    Examples of off-balance-sheet impact:
    - The **pressing of a forward** that forces turnovers in dangerous areas
    - The **tempo-setting of a midfielder** who dictates the rhythm of play
    - The **pre-assist of a fullback** — the pass before the pass
    - The **quick distribution of a goalkeeper** to initiate counterattacks
    - Even intangibles like **motivation or reassuring presence** on the pitch

    This framework identifies these players by asking: **when this player is on the pitch,
    does the team's goal difference improve?**
    """)

    st.markdown("---")

    # ==========================================================================
    # INTUITION: HOW IT WORKS
    # ==========================================================================
    st.markdown("### How does it work?")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **The Intuition:**

        When 22 players are on the pitch, goals happen in **segments** — periods where the
        same players are on the field. We ask: **who was on the pitch when goals were scored
        or conceded?**

        Over a season, we observe thousands of segments with different player combinations.
        Using regularized regression, we attribute goal differences across many segments to
        estimate each player's contribution.

        A player with **+1.0** contribution means the team tends to score ~1 more goal per game
        (net) when they're on the pitch, controlling for teammates and opponents.
        """)

    with col2:
        st.markdown("""
        **Interpretation:**

        - **Positive contribution**: The team tends to score more (net) when the player is
          on the pitch, controlling for teammates and opponents
        - **Negative contribution**: The opposite
        - **Near zero**: Either neutral impact or insufficient signal (few minutes played)

        **Important**: Contribution is a relative, context-dependent measure. Comparisons
        should be made within leagues, roles, and sample sizes.
        """)

    # ==========================================================================
    # EXAMPLE INSIGHT
    # ==========================================================================
    st.info("""
    **Example insight:** Two midfielders may have similar goal and assist numbers, but one
    consistently improves team goal difference when on the pitch, while the other does not.
    Off-balance-sheet contributions help identify this difference.
    """)

    st.markdown("---")

    # ==========================================================================
    # METHODOLOGY DETAILS (EXPANDABLE)
    # ==========================================================================
    with st.expander("📘 Methodology Details"):
        st.markdown("### Match Segments")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            We divide each match into **segments** — periods where no substitutions
            or red cards occur. In each segment:

            1. **Record** the goal difference (from home team's view)
            2. **Track** all 22 players on the pitch
            3. **Scale** to per-90-minutes for comparability

            Over a season, we observe **thousands of segments** with different
            player combinations. Using regression, we estimate each player's
            **contribution** to goal difference.
            """)

        with col2:
            st.markdown("""
            **Example segment:**

            ```
            Minute 23-45 (22 mins)
            Home scores 1, Away scores 0
            Scaled: +1 × (22/90) = +0.24
            ```

            All 22 players get credit/blame for this +0.24.
            Over many segments, each player's true impact emerges.

            | Contribution | Meaning |
            |--------------|---------|
            | **+1.0** | Team scores ~1 more goal/game |
            | **0.0** | Average player |
            | **-1.0** | Team concedes ~1 more goal/game |
            """)

        st.markdown("### Why Regularization (Lasso)?")

        st.markdown("""
        This framework uses **Lasso regression** rather than standard OLS regression to estimate
        contributions. When players rarely rotate, OLS cannot separate individual effects from
        team performance — leading to extreme, implausible estimates.

        Lasso regularization introduces a small bias to dramatically reduce variance, producing
        stable and interpretable results while retaining most explanatory power (R² ~0.53).
        """)

        # Get German Bundesliga 2023 data for comparison
        buli_ols = df[(df["country"] == "Germany") & (df["league"] == "Bundesliga") &
                      (df["season"] == 2023) & (df["FTE_games_played"] >= 5)].copy()

        # OLS comparison only available in full database (not bundled mode)
        has_ols = not buli_ols.empty and "contribution_ols" in buli_ols.columns

        if has_ols:
            buli_ols["hover_ols"] = make_player_hover(buli_ols, contrib_col="contribution_ols")
            buli_ols["hover_lasso"] = make_player_hover(buli_ols, contrib_col="contribution")

            col1, col2 = st.columns(2)

            with col1:
                # OLS scatter plot
                fig_ols = px.scatter(
                    buli_ols,
                    x="FTE_games_played",
                    y="contribution_ols",
                    color="position_group",
                    color_discrete_map=POSITION_COLORS,
                    custom_data=["hover_ols"],
                    title="OLS Contribution (extreme outliers)",
                    labels={"FTE_games_played": "Full Games Played", "contribution_ols": "OLS Contribution"}
                )
                fig_ols.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
                fig_ols.update_layout(height=350, legend_title_text="")
                fig_ols.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(fig_ols, use_container_width=True)
                st.caption("*Bundesliga 2023/24, players with ≥5 FTE games*")

            with col2:
                # Lasso scatter plot
                fig_lasso = px.scatter(
                    buli_ols,
                    x="FTE_games_played",
                    y="contribution",
                    color="position_group",
                    color_discrete_map=POSITION_COLORS,
                    custom_data=["hover_lasso"],
                    title="Lasso Contribution (regularized)",
                    labels={"FTE_games_played": "Full Games Played", "contribution": "Lasso Contribution"}
                )
                fig_lasso.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
                fig_lasso.update_layout(height=350, legend_title_text="")
                fig_lasso.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(fig_lasso, use_container_width=True)
                st.caption("*Same data with Lasso — extreme values shrink toward zero*")

            # Stats comparison
            ols_range = buli_ols["contribution_ols"].agg(["min", "max"])
            lasso_range = buli_ols["contribution"].agg(["min", "max"])

            st.markdown(f"""
            | Metric | OLS | Lasso |
            |--------|-----|-------|
            | **Min** | {ols_range['min']:.1f} | {lasso_range['min']:.1f} |
            | **Max** | {ols_range['max']:.1f} | {lasso_range['max']:.1f} |
            | **Range** | {ols_range['max'] - ols_range['min']:.1f} | {lasso_range['max'] - lasso_range['min']:.1f} |
            | **R²** | 0.54 | 0.53 |

            Lasso retains most explanatory power (R² 0.53 vs 0.54) while shrinking the range from ~{ols_range['max'] - ols_range['min']:.0f} to ~{lasso_range['max'] - lasso_range['min']:.0f} goals.
            """)
        else:
            # Simplified explanation without OLS comparison (bundled mode)
            st.info("""
            **Technical note:** OLS regression produces extreme contribution estimates when players rarely rotate,
            as individual effects cannot be separated from team performance. Lasso regularization introduces a
            small bias to dramatically reduce variance, producing stable and interpretable results while
            retaining most explanatory power (R² ~0.53 vs 0.54 for OLS).
            """)

    st.markdown("---")

    # ==========================================================================
    # SLIDE 5: Example - Bundesliga 2023/24
    # ==========================================================================
    st.markdown("## Case Study: Bundesliga 2023/24")

    buli_2023 = df[(df["country"] == "Germany") & (df["league"] == "Bundesliga") & (df["season"] == 2023) & (df["FTE_games_played"] >= 10)]

    if not buli_2023.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Top Contributors")
            top_10 = buli_2023.nlargest(10, "contribution")[
                ["player_name", "team", "position_group", "FTE_games_played", "goals", "contribution"]
            ].reset_index(drop=True)
            top_10.index = top_10.index + 1
            top_10.columns = ["Player", "Team", "Position", "FTE Games", "Goals", "Contribution"]
            st.dataframe(top_10.round(2), height=380)

        with col2:
            st.markdown("### Key Observations")
            st.markdown("""
            - **Kane's 36 goals** don't make him a top contributor — Bayern's
              quality is spread across many players

            - **Midfielders** and **Defenders** often rank higher than forwards —
              they control the game's tempo

            - **Weaker teams'** stars often show higher values — the team
              depends more on them individually
            """)

        # Scatter: FTE vs Contribution by position
        buli_2023_plot = buli_2023.copy()
        buli_2023_plot["hover_text"] = make_player_hover(buli_2023_plot)
        fig = px.scatter(
            buli_2023_plot,
            x="FTE_games_played",
            y="contribution",
            color="position_group",
            color_discrete_map=POSITION_COLORS,
            custom_data=["hover_text"],
            title="Playing Time vs Contribution (Bundesliga 23/24)",
            labels={"FTE_games_played": "Full Games Played", "contribution": "Contribution"}
        )
        fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
        fig.update_layout(height=400, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("*Players with at least 10 FTE games played*")

    st.markdown("---")

    # ==========================================================================
    # SLIDE 7: Database Overview
    # ==========================================================================
    st.markdown("## Database Coverage")

    n_players = len(df)
    n_unique = df["player_id"].nunique()
    n_leagues = df["league"].nunique()
    n_countries = df["country"].nunique()
    season_range = f"{df['season'].min()}-{df['season'].max()}"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Player-Seasons", f"{n_players:,}")
    col2.metric("Unique Players", f"{n_unique:,}")
    col3.metric("Leagues", n_leagues)
    col4.metric("Countries", n_countries)
    col5.metric("Seasons", season_range)

    # Data completeness note
    st.info(
        "**Note on data completeness:** Some seasons may have incomplete lineup data, "
        "particularly for lower divisions and youth leagues. The 2019-2020 seasons are "
        "notably affected due to the COVID-19 pandemic, which disrupted data collection "
        "for many leagues (e.g., Germany U19 Bundesliga 2020 has only 22% lineup coverage "
        "vs. 100% for surrounding seasons). Analysis is only performed on fixtures with "
        "complete lineup information."
    )

    # Coverage table by country and league
    coverage = df.groupby(["country", "league"]).agg(
        From=("season", "min"),
        To=("season", "max")
    ).reset_index()

    # Use league visibility data for categorization (from database)
    league_vis = load_league_visibility()

    # Color scheme for league categories (semi-transparent backgrounds)
    CATEGORY_COLORS = {
        "top_5": "rgba(76, 175, 80, 0.25)",       # Green - Top 5 European
        "major_european": "rgba(56, 142, 60, 0.2)",  # Dark green - Major European
        "secondary_european": "rgba(139, 195, 74, 0.2)",  # Light green - Secondary
        "lower_tier": "rgba(205, 220, 57, 0.2)",  # Lime - Lower tier
        "regional": "rgba(255, 235, 59, 0.2)",    # Yellow - Regional
        "womens": "rgba(233, 30, 99, 0.2)",       # Pink - Women's
        "youth": "rgba(33, 150, 243, 0.2)",       # Blue - Youth
        "cup_competitions": "rgba(156, 39, 176, 0.2)",  # Purple - Cups
        "south_american": "rgba(255, 152, 0, 0.2)",  # Orange - South America
        "north_american": "rgba(255, 87, 34, 0.2)",  # Deep orange - North America
        "asian": "rgba(0, 188, 212, 0.2)",        # Cyan - Asian
        "african": "rgba(121, 85, 72, 0.2)",      # Brown - African
        "other_european": "rgba(158, 158, 158, 0.15)",  # Gray - Other European
        "other": "rgba(224, 224, 224, 0.1)",      # Light gray - Other
    }

    # Merge with league visibility data to get category and priority
    if not league_vis.empty:
        coverage = coverage.merge(
            league_vis[["country", "league", "category", "priority"]],
            on=["country", "league"],
            how="left"
        )
        coverage["_category"] = coverage["category"].fillna("other")
        coverage["_tier"] = coverage["priority"].fillna(99).astype(int)
    else:
        coverage["_category"] = "other"
        coverage["_tier"] = 99

    coverage = coverage.sort_values(["country", "_tier", "league"])

    # Format for display with merged country cells (blank duplicates)
    coverage_display = coverage.copy()
    coverage_display["From"] = coverage_display["From"].astype(int)
    coverage_display["To"] = coverage_display["To"].astype(int)

    # Blank out repeated country values to simulate merged cells
    prev_country = None
    for idx in coverage_display.index:
        if coverage_display.loc[idx, "country"] == prev_country:
            coverage_display.loc[idx, "country"] = ""
        else:
            prev_country = coverage_display.loc[idx, "country"]

    # Store category for styling, then drop from display
    categories = coverage_display["_category"].tolist()

    # Prepare display dataframe (without _category)
    display_df = coverage_display[["country", "league", "From", "To"]].copy()
    display_df.columns = ["Country", "League", "From", "To"]

    # Style rows by category using index-based lookup
    def style_league_rows(row):
        idx = row.name
        # Find position in the list
        try:
            pos = list(display_df.index).index(idx)
            category = categories[pos]
        except (ValueError, IndexError):
            category = "other"

        color = CATEGORY_COLORS.get(category, "")
        if color:
            return [f"background-color: {color}"] * len(row)
        return [""] * len(row)

    styled_df = display_df.style.apply(style_league_rows, axis=1)
    styled_df = styled_df.format({"From": "{:.0f}", "To": "{:.0f}"})

    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        # Records by season chart
        season_counts = df.groupby("season").size().reset_index(name="count")
        fig = px.bar(
            season_counts,
            x="season",
            y="count",
            title="Player-Season Records by Year",
            labels={"season": "Season", "count": "Records"}
        )
        # Add COVID-19 marker for 2019-2020 seasons (2019 = 2019-20 season, disrupted Mar 2020)
        max_count = season_counts["count"].max()
        fig.add_vrect(
            x0=2018.5, x1=2020.5,
            fillcolor="rgba(255, 0, 0, 0.1)",
            layer="below",
            line_width=0,
        )
        fig.add_annotation(
            x=2019.5, y=max_count * 0.95,
            text="COVID-19",
            showarrow=False,
            font=dict(size=10, color="red"),
        )
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("**League Coverage**")
        st.dataframe(
            styled_df,
            hide_index=True,
            height=320,
            use_container_width=True
        )
        st.caption("🟩 Top 5 · 🟢 European · 🟦 Youth · 🩷 Women's · 🟠 Americas · 🔵 Asian")

    st.markdown("---")

    # ==========================================================================
    # SLIDE 8: Navigation Guide
    # ==========================================================================
    st.markdown("## Explore the Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **👤 Player Career**

        Track any player across seasons.
        See contribution evolve over time.
        Compare to league averages.
        """)

        st.markdown("""
        **🏟️ Team Analysis**

        Analyze team composition and contributions.
        See how players contribute to team success.
        """)

    with col2:
        st.markdown("""
        **🔗 Transfer Networks**

        Visualize talent flow between clubs/leagues.
        Which corridors produce consistent value?
        """)

        st.markdown("""
        **🔬 League Comparison**

        Compare league quality using transfer data.
        See how players perform across leagues.
        """)

    with col3:
        st.markdown("""
        *Use the sidebar to navigate between different analysis pages.*
        """)

    # --- ARCHIVED PAGE DESCRIPTIONS (code preserved but hidden) ---
    # **🔍 Top Free Agents**
    # Find high contributors who disappeared.
    # Potential scouting opportunities.
    # Filter by position and age.
    # st.markdown("""
    # **📊 Contribution Analysis**
    # Filter by league, team, position. Compare goals vs contribution. Find hidden gems.
    # """)
    # st.markdown("""
    # **🔄 Persistence**
    # Does contribution persist year-to-year? Test predictive power of the metric.
    # """)
    # st.markdown("""
    # **📈 Distribution Comparison**
    # Compare leagues head-to-head. Which have more variance in quality?
    # """)
    # --- END ARCHIVED ---

    st.markdown("---")

    st.caption("""
    *Data: API-Football match events. Method: Lasso regression on match segments.
    Use the sidebar to navigate between pages.*
    """)


# NOTE: page_compare_players() has been merged into page_player_career()
# The comparison functionality is now accessible via the "Compare with another player" checkbox


# =============================================================================
#                          MAIN
# =============================================================================

def main():
    st.set_page_config(
        page_title="Soccer Analytics",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Check database exists
    if not Path(DB_PATH).exists():
        st.error(f"Database not found: {DB_PATH}")
        st.info("Run analysis first to generate data.")
        return

    # Navigation with icons
    st.sidebar.title("⚽ Soccer Analytics")

    pages = {
        "💡 Motivation": page_home,
        # --- ARCHIVED PAGES (code preserved but hidden) ---
        # "📊 Contribution Analysis": page_scatter_analysis,
        # "📈 Distribution Comparison": page_league_comparison,
        # "🔄 Persistence": page_markov_analysis,
        # "🔍 Top Free Agents": page_free_agents,  # Removed from public dashboard
        # --- END ARCHIVED ---
        "👤 Player Career": page_player_career,
        "🏟️ Team Analysis": page_team_analysis,
        "🔗 Transfer Networks": page_network_analysis,
        "🔬 League Comparison": page_league_quality,
    }

    page = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")

    # Run selected page
    pages[page]()

    # Data info and refresh at bottom of sidebar
    st.sidebar.markdown("---")
    with st.sidebar:
        # Data summary
        df = load_all_players(min_fte=0)
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"📊 {len(df):,} records")
        with col2:
            st.caption(f"👥 {df['player_id'].nunique():,} players")

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Footer
    st.sidebar.markdown("---")
    if _USING_DASHBOARD_DB:
        st.sidebar.caption("📦 Using dashboard.db (standalone)")
    else:
        st.sidebar.caption("🗃️ Using full databases")
    st.sidebar.caption("Off-Balance-Sheet Contributions · v2.0")


if __name__ == "__main__":
    main()
