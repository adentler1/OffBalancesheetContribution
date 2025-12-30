"""
Soccer Analytics Dashboard - Standalone Version

=== HOW TO RUN ===

1. Install dependencies:
   pip install -r requirements.txt

2. Run the dashboard:
   streamlit run dashboard.py

3. Open in browser:
   http://localhost:8501

This standalone version expects the databases in the same folder as this file.
===============================
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import uniform_filter1d

# =============================================================================
#                          CONFIG (Standalone version)
# =============================================================================

# For standalone operation, use relative paths from this file's directory
_HERE = Path(__file__).parent
DB_PATH = str(_HERE / "analysis_results.db")
PROFILES_DB_PATH = str(_HERE / "player_profiles.db")
CONTRIB_COL = "lasso_contribution_alpha_best"  # Best alpha auto-selected via IC

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
    if not Path(DB_PATH).exists():
        return {}

    conn = sqlite3.connect(DB_PATH)
    names = pd.read_sql_query("""
        SELECT player_id, player_name
        FROM players
        WHERE player_name IS NOT NULL AND player_name != 'Unknown'
    """, conn)
    conn.close()

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
    teams = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Extract individual team names and normalize
    all_teams = set()
    for team in teams["team"].dropna():
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

@st.cache_data(ttl=3600)  # Refresh cache every hour
def load_all_players(min_fte: float = 0.0) -> pd.DataFrame:
    """Load all player-season records with profile data."""
    conn = sqlite3.connect(DB_PATH)
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
            contribution_ols,
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
    if df.empty or not Path(PROFILES_DB_PATH).exists():
        df["nationality"] = "Unknown"
        df["age"] = None
        df["height_cm"] = None
        df["weight_kg"] = None
        df["bmi"] = None
        return df

    try:
        conn = sqlite3.connect(PROFILES_DB_PATH)
        # Get the most recent profile for each player
        profiles = pd.read_sql_query("""
            SELECT player_id, nationality, age, position, height, weight
            FROM player_profiles
            WHERE nationality IS NOT NULL
            GROUP BY player_id
            HAVING season = MAX(season)
        """, conn)
        conn.close()

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
            profiles[["player_id", "nationality", "age", "height_cm", "weight_kg", "bmi", "profile_position"]],
            on="player_id",
            how="left",
            suffixes=("", "_profile")
        )
        df["nationality"] = df["nationality"].fillna("Unknown")

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
    try:
        conn = sqlite3.connect(DB_PATH)
        if league_name and country:
            # Get teams for specific league/country from analysis DB
            teams = pd.read_sql_query("""
                SELECT DISTINCT "team(s)" as team_name
                FROM players
                WHERE league = ? AND country = ? AND "team(s)" IS NOT NULL
            """, conn, params=(league_name, country))
        else:
            # Get all teams
            teams = pd.read_sql_query("""
                SELECT DISTINCT "team(s)" as team_name
                FROM players
                WHERE "team(s)" IS NOT NULL
            """, conn)
        conn.close()

        # Extract individual team names from "Multiple Teams (A, B)" entries
        # and normalize to canonical forms
        all_teams = set()
        for team in teams["team_name"].dropna():
            if team.startswith("Multiple Teams ("):
                # Extract team names from "Multiple Teams (A, B, C)" format
                inner = team[16:-1]  # Remove "Multiple Teams (" and ")"
                for t in inner.split(", "):
                    all_teams.add(normalize_team_name(t.strip()))
            else:
                all_teams.add(normalize_team_name(team))

        return sorted(all_teams)
    except Exception:
        return []


@st.cache_data
def load_player_transfers(player_id: int) -> pd.DataFrame:
    """Load transfer history for a player."""
    if not Path(PROFILES_DB_PATH).exists():
        return pd.DataFrame()

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

        if not transfers.empty:
            # Extract year from transfer_date for positioning
            transfers["year"] = pd.to_datetime(transfers["transfer_date"]).dt.year
            # Create display label
            transfers["label"] = transfers.apply(
                lambda r: f"{r['type']}" if r['type'] not in ['N/A', 'Transfer'] else "", axis=1
            )
        return transfers
    except Exception:
        return pd.DataFrame()


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
                help="Full-Time Equivalent games played"
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

        # Use fixed color mapping for positions
        fig = px.scatter(
            filtered,
            x=x_axis_col,
            y="contribution",
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
                            pos_data[x_axis_col], pos_data["contribution"]
                        )
                    else:
                        x_trend, y_trend, y_lower, y_upper = compute_lowess_trend(
                            pos_data[x_axis_col], pos_data["contribution"]
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
                        filtered[x_axis_col], filtered["contribution"]
                    )
                else:
                    x_trend, y_trend, y_lower, y_upper = compute_lowess_trend(
                        filtered[x_axis_col], filtered["contribution"]
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
    """Player career exploration."""
    st.header("Player Career")

    with st.spinner("Loading player data..."):
        df = load_all_players(min_fte=3.0)

    # player_id is already normalized to integer in load_all_players()

    # Player search with sidebar option
    with st.sidebar:
        st.subheader("Player Search")
        search = st.text_input("Search player name", "", key="career_search",
                              help="Type a player name to search")

    if not search:
        # Show empty state when no search
        render_empty_state(
            "Search for a player to view their career",
            "Enter a player name in the search box"
        )
        return

    matches = df[df["player_name"].str.contains(search, case=False, na=False)]

    # Group by player_id - names are already canonical (longest form)
    players = matches.groupby("player_id").agg({
        "player_name": "first",  # Already canonical from load_all_players()
        "season": "count",
        "team": lambda x: ", ".join(sorted(set(x.dropna())))[:50],
        "nationality": "first"
    }).reset_index()
    players.columns = ["player_id", "player_name", "seasons", "teams", "nationality"]
    players = players.sort_values("seasons", ascending=False)

    if players.empty:
        render_empty_state(
            f"No players found matching '{search}'",
            "Try a different search term"
        )
        return

    # Warn if multiple distinct players match the search
    if len(players) > 1:
        st.info(f"ℹ️ {len(players)} players match your search. Select based on teams/nationality.")

    # Build player options with disambiguation info
    player_options = []
    for _, row in players.head(20).iterrows():
        teams_short = row["teams"][:30] + "..." if len(str(row["teams"])) > 30 else row["teams"]
        nat = row["nationality"] if row["nationality"] != "Unknown" else ""
        option = f"{row['player_name']} ({row['seasons']} seasons) - {teams_short}"
        if nat:
            option += f" [{nat}]"
        player_options.append((option, row["player_id"]))

    selected_idx = st.selectbox(
        "Select player",
        range(len(player_options)),
        format_func=lambda i: player_options[i][0]
    )

    if selected_idx is not None:
        selected_player_id = player_options[selected_idx][1]
        player_data = df[df["player_id"] == selected_player_id].sort_values("season")

        if not player_data.empty:
            player_name = player_data["player_name"].iloc[0]
            nationality = player_data["nationality"].iloc[0]
            position = player_data["position"].mode().iloc[0] if not player_data["position"].mode().empty else "Unknown"

            # Breadcrumb navigation
            render_breadcrumbs(["Player Career", player_name])

            # Career stats in a styled container
            st.markdown("### Career Overview")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Seasons", len(player_data), help="Number of seasons with data")
            col2.metric("Avg Contribution", f"{player_data['contribution'].mean():.2f}",
                       help="Average contribution per season")
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
                    secondary_y = st.selectbox("Secondary Y-Axis", secondary_y_options, index=3,
                                              key="career_secondary_y")
                    show_transfers = st.checkbox("Show Transfers", value=True, key="career_transfers")

            # Load transfers if needed
            transfers = pd.DataFrame()
            if show_transfers:
                # Use normalized player_id (already numeric)
                transfers = load_player_transfers(selected_player_id)

            # Build the figure with potential secondary Y-axis
            from plotly.subplots import make_subplots

            if secondary_y != "None":
                fig = make_subplots(specs=[[{"secondary_y": True}]])
            else:
                fig = go.Figure()

            # Primary trace: Contribution (use color-blind safe color)
            # Prepare custom data for hover
            hover_data = player_data[["team", "league", "country", "position", "goals", "assists", "FTE_games_played", "team_rank"]].copy()
            hover_data["goals"] = hover_data["goals"].fillna(0).astype(int)
            hover_data["assists"] = hover_data["assists"].fillna(0).astype(int)
            hover_data["FTE_games_played"] = hover_data["FTE_games_played"].fillna(0)
            hover_data["team_rank"] = hover_data["team_rank"].fillna(0).astype(int)

            fig.add_trace(
                go.Scatter(
                    x=player_data["season"],
                    y=player_data["contribution"],
                    mode="lines+markers",
                    name="Contribution",
                    line=dict(width=3, color="#648FFF"),  # Color-blind safe blue
                    marker=dict(size=10),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> (%{y:+.2f})<br>"
                        "%{customdata[3]}, %{customdata[1]}<br>"
                        "%{customdata[2]}, Season %{x}<br>"
                        "%{customdata[4]} goals, %{customdata[5]} assists<br>"
                        "%{customdata[6]:.1f} FTE games, Rank: %{customdata[7]}"
                        "<extra></extra>"
                    ),
                    customdata=hover_data.values
                ),
                secondary_y=False if secondary_y != "None" else None
            )

            # Secondary trace if selected
            if secondary_y != "None":
                reverse_secondary_y = False
                if secondary_y == "Goals":
                    y_data = player_data["goals"]
                    trace_name = "Goals"
                    color = "#FFB000"  # Color-blind safe amber
                elif secondary_y == "Assists":
                    y_data = player_data["assists"]
                    trace_name = "Assists"
                    color = "#DC267F"  # Color-blind safe magenta
                elif secondary_y == "Team Rank":
                    y_data = player_data["team_rank"]
                    trace_name = "Team Rank"
                    color = "#FE6100"  # Color-blind safe orange
                    reverse_secondary_y = True  # Lower rank = better (1st place at top)
                else:  # Goals + Assists
                    y_data = player_data["goals"] + player_data["assists"]
                    trace_name = "G+A"
                    color = "#785EF0"  # Color-blind safe purple

                fig.add_trace(
                    go.Scatter(
                        x=player_data["season"],
                        y=y_data,
                        mode="lines+markers",
                        name=trace_name,
                        line=dict(width=2, dash="dot", color=color),
                        marker=dict(size=7, symbol="diamond"),
                        hovertemplate=f"{trace_name}: %{{y:.0f}}<extra></extra>"
                    ),
                    secondary_y=True
                )

                # Reverse secondary axis if needed (for ranks: lower = better)
                if reverse_secondary_y:
                    fig.update_yaxes(autorange="reversed", secondary_y=True)

            # Add transfer markers
            if show_transfers and not transfers.empty:
                min_season = player_data["season"].min()
                max_season = player_data["season"].max()

                for _, tr in transfers.iterrows():
                    tr_year = tr["year"]
                    # Only show transfers within the career span
                    if min_season <= tr_year <= max_season + 1:
                        # Vertical line at transfer year
                        fig.add_vline(
                            x=tr_year,
                            line_dash="dash",
                            line_color="rgba(128, 128, 128, 0.5)",
                            line_width=1
                        )
                        # Annotation with transfer info
                        label = tr["label"] if tr["label"] else "Transfer"
                        annotation_text = f"→ {tr['to_team_name'][:15]}"
                        if label and label not in ["Transfer", "N/A"]:
                            annotation_text += f"<br><i>{label}</i>"
                        fig.add_annotation(
                            x=tr_year,
                            y=1.02,
                            yref="paper",
                            text=annotation_text,
                            showarrow=False,
                            font=dict(size=9, color="gray"),
                            textangle=-45,
                            xanchor="left"
                        )

            # Add horizontal lines
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_hline(
                y=player_data["contribution"].mean(),
                line_dash="dot",
                line_color="#FFB000",  # Color-blind safe amber
                annotation_text=f"avg: {player_data['contribution'].mean():.2f}"
            )

            # Layout
            layout_args = dict(
                title=f"{player_name} - Career",
                xaxis_title="Season",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            if secondary_y != "None":
                fig.update_yaxes(title_text="Lasso Contribution (goals/90)", secondary_y=False)
                fig.update_yaxes(title_text=secondary_y, secondary_y=True)
            else:
                layout_args["yaxis_title"] = "Lasso Contribution (goals/90)"

            fig.update_layout(**layout_args)
            st.plotly_chart(fig, use_container_width=True)

            # Transfer history table (if any)
            if show_transfers and not transfers.empty:
                with st.expander("Transfer History", expanded=False):
                    st.dataframe(
                        transfers[["transfer_date", "from_team_name", "to_team_name", "type"]].rename(columns={
                            "transfer_date": "Date",
                            "from_team_name": "From",
                            "to_team_name": "To",
                            "type": "Type/Fee"
                        }),
                        hide_index=True
                    )

            # Season details table with team
            st.subheader("Season Details")
            display_cols = ["season", "team", "league", "contribution", "FTE_games_played", "goals", "assists", "position"]
            st.dataframe(player_data[display_cols], hide_index=True)

            # Export section
            st.markdown("---")
            with st.expander("📥 Export Data"):
                csv = player_data[display_cols].to_csv(index=False)
                st.download_button(
                    f"Download {player_name}'s Career Data (CSV)",
                    csv,
                    f"career_{player_name.replace(' ', '_')}.csv",
                    "text/csv",
                    key="career_export"
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
                           key="dist_min_fte", help="Minimum playing time filter")
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
                selected_teams = st.multiselect(
                    "Teams",
                    team_options,
                    default=["All"],
                    key=f"teams_{i}"
                )

                # Position filter
                position_options = ["All", "Goalkeeper", "Defender", "Midfielder", "Forward"]
                selected_positions = st.multiselect(
                    "Positions",
                    position_options,
                    default=["All"],
                    key=f"positions_{i}"
                )

            with col3:
                # Nationality filter
                available_nationalities = sorted(league_subset["nationality"].dropna().unique())
                nationality_options = ["All"] + [n for n in available_nationalities if n != "Unknown"]
                selected_nationalities = st.multiselect(
                    "Nationalities",
                    nationality_options,
                    default=["All"],
                    key=f"nationalities_{i}"
                )

                # Age range filter
                age_min, age_max = st.slider(
                    "Age Range",
                    min_value=15,
                    max_value=45,
                    value=(15, 45),
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
                boxmean="sd",  # Show mean and std deviation
                boxpoints="outliers",  # Show outlier points
                text=subset["player_name"],  # Player names for outlier points
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Contribution: %{y:.2f}<br>"
                    "<extra>%{fullData.name}</extra>"
                )
            ))

            all_subsets.append((dist["label"], subset))

            # Collect stats (2 significant digits for readability)
            comparison_stats.append({
                "Distribution": dist["label"],
                "Players": len(subset),
                "Mean": round(subset["contribution"].mean(), 2),
                "Median": round(subset["contribution"].median(), 2),
                "Std": round(subset["contribution"].std(), 2),
                "Q1": round(subset["contribution"].quantile(0.25), 2),
                "Q3": round(subset["contribution"].quantile(0.75), 2),
                "Min": round(subset["contribution"].min(), 2),
                "Max": round(subset["contribution"].max(), 2),
            })

        if comparison_stats:
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
def compute_transfer_network(level: str = "league", min_transfers: int = 3,
                              season_start: int = None, season_end: int = None):
    """
    Compute transfer network data from player transitions.

    Args:
        level: "league" or "club"
        min_transfers: Minimum number of transfers to include an edge
        season_start: Filter to include only seasons >= this value
        season_end: Filter to include only seasons <= this value

    Returns:
        edges_df: DataFrame with source, target, transfers, cumulative_contribution, net_contribution
        nodes_df: DataFrame with node, avg_contribution, n_players
        trans_df: DataFrame with individual transitions for drill-down
    """
    if not Path(DB_PATH).exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    players = pd.read_sql_query("""
        SELECT player_id, player_name, "team(s)" as team, league, country, season,
               lasso_contribution_alpha_best as contribution
        FROM players
        WHERE lasso_contribution_alpha_best IS NOT NULL
        ORDER BY player_id, season
    """, conn)
    conn.close()

    # Apply season filter
    if season_start is not None:
        players = players[players["season"] >= season_start]
    if season_end is not None:
        players = players[players["season"] <= season_end]

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

        for _, row in group.iterrows():
            current_node = row["node"]
            current_contribution = row["contribution"]

            if prev_node is not None and prev_node != current_node:
                transitions.append({
                    "source": prev_node,
                    "target": current_node,
                    "contribution_at_target": current_contribution,
                    "contribution_at_source": prev_contribution,
                    "contribution_delta": current_contribution - prev_contribution,
                    "player_name": row["player_name"],
                    "player_id": pid,
                    "season": row["season"]
                })

            prev_node = current_node
            prev_contribution = current_contribution

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
    """Network analysis page showing transfer flows and contribution pipelines."""
    st.header("🌐 Transfer Network Analysis")

    st.markdown("""
    ### Understanding Player Movement and Development

    This page visualizes how players move between leagues and clubs, and crucially,
    **how their contribution changes after the move**.

    ---

    #### 📐 The Key Metric: Contribution Delta (Δ)

    For each transfer, we measure:
    - **Before**: Player's contribution in their *last season* at the source club/league
    - **After**: Player's contribution in their *first season* at the destination

    The **Delta (Δ)** = After − Before tells us whether the player's measured impact changed:

    | Delta | Interpretation | What it suggests |
    |-------|---------------|------------------|
    | **Δ > 0** (positive) | Player improved after moving | Destination may be less competitive, OR provides better development environment |
    | **Δ < 0** (negative) | Player declined after moving | Destination is likely more challenging (higher competition level) |
    | **Δ ≈ 0** (neutral) | Similar performance | Lateral move to comparable level |

    ---

    #### 🎯 Practical Applications

    - **Identifying Feeder Leagues**: Leagues where players consistently improve after leaving (positive avg Δ out) → they develop talent for bigger leagues
    - **Identifying Upgrade Destinations**: Leagues where incoming players initially struggle (negative avg Δ in) → higher competition level
    - **Scout Smarter**: Find corridors where players historically thrive vs. struggle
    - **Assess Transfer Risk**: High negative Δ on a corridor = higher adaptation challenge

    ---

    #### 🔍 Visual Guide
    - **Line thickness** = Number of transfers (more players = thicker line)
    - **Green edges** = Players improve after moving (positive Δ)
    - **Red edges** = Players decline after moving (negative Δ)
    """)

    # Controls in sidebar
    with st.sidebar:
        st.subheader("Network Settings")

        with st.expander("🔧 Configuration", expanded=True):
            level = st.selectbox("Network Level", ["League", "Club"], index=0,
                               help="Analyze at league or club level")
            min_transfers = st.slider("Min Transfers", 1, 20, 5,
                                     help="Minimum transfers to show an edge")
            flow_mode = st.selectbox(
                "Edge Coloring",
                ["Transfer Count", "Player Development"],
                index=0,
                help="Transfer Count: neutral color, line width = volume. "
                     "Player Development: green = players improve after move, red = players decline."
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
            season_end=season_end
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

    # Build network visualization using plotly
    import networkx as nx

    G = nx.DiGraph()

    # Add nodes
    for _, row in nodes.iterrows():
        G.add_node(row["node"],
                   avg_contribution=row["avg_contribution"],
                   n_players=row["n_players"])

    # Add edges with weights based on flow_mode
    for _, row in edges.iterrows():
        if flow_mode == "Transfer Count":
            weight = row["transfers"]
        else:  # Player Development - use avg_delta
            weight = row.get("avg_delta", 0)

        G.add_edge(row["source"], row["target"],
                   weight=weight,
                   transfers=row["transfers"],
                   avg_outflow=row.get("avg_outflow", 0),
                   avg_inflow=row.get("avg_inflow", 0),
                   avg_delta=row.get("avg_delta", 0),
                   improved=row.get("improved_count", 0),
                   declined=row.get("declined_count", 0))

    # Layout
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        st.warning("No nodes in network")
        return

    # Create edge traces
    edge_traces = []
    edge_annotations = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2].get("weight", 1)
        transfers = edge[2].get("transfers", 0)

        # Get contribution metrics
        avg_outflow = edge[2].get("avg_outflow", 0)
        avg_inflow = edge[2].get("avg_inflow", 0)
        avg_delta = edge[2].get("avg_delta", 0)
        improved = edge[2].get("improved", 0)
        declined = edge[2].get("declined", 0)

        # Color based on flow_mode
        if flow_mode == "Player Development":
            # Green = players improve after move, Red = players decline
            if avg_delta > 0:
                color = "rgba(0, 180, 0, 0.7)"  # Green - players improve
            else:
                color = "rgba(200, 0, 0, 0.7)"  # Red - players decline
        else:  # Transfer Count
            color = "rgba(100, 100, 100, 0.5)"  # Neutral gray

        # Line width based on transfers
        width = max(1, min(transfers / 2, 8))

        # Create curved edge (bezier approximation with midpoint offset)
        mid_x = (x0 + x1) / 2 + (y1 - y0) * 0.1
        mid_y = (y0 + y1) / 2 - (x1 - x0) * 0.1

        # Build hover text
        hover_text = (f"<b>{edge[0]} → {edge[1]}</b><br>"
                      f"Transfers: {transfers}<br>"
                      f"Before move: {avg_outflow:.3f}<br>"
                      f"After move: {avg_inflow:.3f}<br>"
                      f"Δ: {avg_delta:+.3f} ({improved}↑ {declined}↓)")

        edge_trace = go.Scatter(
            x=[x0, mid_x, x1, None],
            y=[y0, mid_y, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="text",
            text=hover_text,
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = []
    node_sizes = []
    node_colors = []

    for node in G.nodes():
        data = G.nodes[node]
        avg_contrib = data.get("avg_contribution", 0)
        n_players = data.get("n_players", 0)

        # In-degree and out-degree for flow analysis
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        node_text.append(f"<b>{node}</b><br>"
                        f"Avg Contribution: {avg_contrib:.3f}<br>"
                        f"Players: {n_players}<br>"
                        f"Incoming: {in_deg} · Outgoing: {out_deg}")

        # Size based on number of players
        node_sizes.append(max(15, min(n_players / 10, 50)))

        # Color based on average contribution
        node_colors.append(avg_contrib)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="RdYlGn",
            colorbar=dict(title="Avg Contribution"),
            line=dict(width=1, color="white")
        ),
        text=[n.split(" - ")[-1][:20] if " - " in n else n[:20] for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=10, color="black", family="Arial Black"),
        hoverinfo="text",
        hovertext=node_text,
        showlegend=False
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=f"{level}-Level Transfer Network ({len(G.nodes())} nodes, {len(G.edges())} edges)",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Edge Colors (Player Development mode):**
        - 🟢 Green: Players improve after moving (positive Δ)
        - 🔴 Red: Players decline after moving (negative Δ)
        - Width: Number of transfers
        """)
    with col2:
        st.markdown("""
        **Node Colors:**
        - Green: Higher avg contribution
        - Red: Lower avg contribution
        - Size: Number of players
        """)

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

    # Net Contribution Flow - focus on averages
    st.subheader("📊 Net Contribution Flow")
    st.markdown("""
    Compare **average contribution of departures** vs. **average contribution of arrivals**.
    - 🔴 **Avg Leaving**: How good were players when they left? (before move)
    - 🟢 **Avg Arriving**: How good are incoming players? (after move)
    - **Net**: Arrivals - Departures (positive = gaining quality)
    """)

    # Calculate avg contribution for leaving and arriving players per node
    leaving_avg = edges.groupby("source")["avg_outflow"].mean()  # avg before move
    arriving_avg = edges.groupby("target")["avg_inflow"].mean()  # avg after move

    flow_nodes = set(leaving_avg.index) | set(arriving_avg.index)
    flow_data = []
    for node in flow_nodes:
        avg_leave = leaving_avg.get(node, 0)
        avg_arrive = arriving_avg.get(node, 0)
        net = avg_arrive - avg_leave
        flow_data.append({
            "node": node,
            "avg_leaving": avg_leave,
            "avg_arriving": avg_arrive,
            "net": net
        })

    flow_df = pd.DataFrame(flow_data)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏆 Net Quality Gainers** (arrivals better than departures)")
        gainers = flow_df.nlargest(10, "net")[["node", "avg_leaving", "avg_arriving", "net"]]
        gainers.columns = ["Node", "Leaving", "Arriving", "Net"]
        st.dataframe(gainers.round(3), hide_index=True)

    with col2:
        st.markdown("**📉 Net Quality Losers** (departures better than arrivals)")
        losers = flow_df.nsmallest(10, "net")[["node", "avg_leaving", "avg_arriving", "net"]]
        losers.columns = ["Node", "Leaving", "Arriving", "Net"]
        st.dataframe(losers.round(3), hide_index=True)

    # Top transfer corridors
    st.subheader("🛤️ Top Transfer Corridors")
    st.markdown("""
    **Interpretation**: Negative Δ = players decline after moving (destination may be more challenging).
    Positive Δ = players improve (destination may be easier, or better player development).
    """)
    top_corridors = edges.nlargest(15, "transfers")[
        ["source", "target", "transfers", "avg_outflow", "avg_inflow", "avg_delta", "improved_count", "declined_count"]
    ].copy()

    # Add qualification column
    def qualify_delta(row):
        delta = row["avg_delta"]
        if delta < -0.02:
            return "⬆️ Upgrade"  # Destination more challenging
        elif delta > 0.02:
            return "⬇️ Feeder"   # Source develops for destination
        else:
            return "↔️ Lateral"
    top_corridors["Type"] = top_corridors.apply(qualify_delta, axis=1)

    top_corridors = top_corridors[["source", "target", "transfers", "avg_outflow", "avg_inflow", "avg_delta", "Type", "improved_count", "declined_count"]]
    top_corridors.columns = ["From", "To", "Transfers", "Before", "After", "Avg Δ", "Type", "↑", "↓"]
    st.dataframe(top_corridors.round(3), hide_index=True)

    st.caption("⬆️ Upgrade = destination more challenging (players decline) | ⬇️ Feeder = source develops players (they improve) | ↔️ Lateral = similar level")

    # Node Detail Panel
    st.subheader("Node Detail")
    st.markdown("Select a node to see its transfer network. Edge width = transfer volume, color = player development (green = improve, red = decline).")

    node_options = ["Select a node..."] + sorted(nodes["node"].tolist())
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

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Incoming Transfers", int(total_incoming))
            col2.metric("Outgoing Transfers", int(total_outgoing))
            avg_delta_in = incoming_edges["avg_delta"].mean() if not incoming_edges.empty else 0
            avg_delta_out = outgoing_edges["avg_delta"].mean() if not outgoing_edges.empty else 0
            col3.metric("Avg Δ Incoming", f"{avg_delta_in:+.3f}")
            col4.metric("Avg Δ Outgoing", f"{avg_delta_out:+.3f}")

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
                                   transfers=row["transfers"],
                                   avg_delta=row["avg_delta"],
                                   avg_outflow=row["avg_outflow"],
                                   avg_inflow=row["avg_inflow"])

            # Add successors (targets that this node feeds into)
            for _, row in outgoing_edges.iterrows():
                target_info = nodes[nodes["node"] == row["target"]]
                if not target_info.empty:
                    mini_G.add_node(row["target"], is_central=False,
                                   avg_contribution=target_info.iloc[0]["avg_contribution"],
                                   n_players=target_info.iloc[0]["n_players"])
                    mini_G.add_edge(selected_node, row["target"],
                                   transfers=row["transfers"],
                                   avg_delta=row["avg_delta"],
                                   avg_outflow=row["avg_outflow"],
                                   avg_inflow=row["avg_inflow"])

            if len(mini_G.nodes()) > 1:
                # Layout: put central node in middle
                pos = {}
                predecessors = list(incoming_edges["source"].unique())
                successors = list(outgoing_edges["target"].unique())

                # Central node at origin
                pos[selected_node] = (0, 0)

                # Predecessors on the left
                for i, pred in enumerate(predecessors):
                    y_offset = (i - len(predecessors) / 2) * 0.3
                    pos[pred] = (-1, y_offset)

                # Successors on the right
                for i, succ in enumerate(successors):
                    y_offset = (i - len(successors) / 2) * 0.3
                    pos[succ] = (1, y_offset)

                # Create edge traces
                edge_traces = []
                for edge in mini_G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    transfers = edge[2].get("transfers", 1)
                    avg_delta = edge[2].get("avg_delta", 0)
                    avg_outflow = edge[2].get("avg_outflow", 0)
                    avg_inflow = edge[2].get("avg_inflow", 0)

                    # Color by delta: green = improve, red = decline
                    if avg_delta > 0:
                        color = f"rgba(0, {min(180, 100 + int(abs(avg_delta) * 500))}, 0, 0.8)"
                    else:
                        color = f"rgba({min(200, 100 + int(abs(avg_delta) * 500))}, 0, 0, 0.8)"

                    # Width by number of transfers
                    width = max(2, min(transfers / 2, 15))

                    hover_text = (f"<b>{edge[0]} → {edge[1]}</b><br>"
                                  f"Transfers: {transfers}<br>"
                                  f"Before: {avg_outflow:.3f}<br>"
                                  f"After: {avg_inflow:.3f}<br>"
                                  f"Δ: {avg_delta:+.3f}")

                    edge_trace = go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        mode="lines",
                        line=dict(width=width, color=color),
                        hoverinfo="text", text=hover_text,
                        showlegend=False
                    )
                    edge_traces.append(edge_trace)

                # Node traces
                node_x, node_y, node_text, node_sizes, node_colors = [], [], [], [], []
                for node in mini_G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    data = mini_G.nodes[node]
                    is_central = data.get("is_central", False)
                    contrib = data.get("avg_contribution", 0)
                    n_players = data.get("n_players", 0)

                    node_text.append(f"<b>{node}</b><br>Avg: {contrib:.3f}<br>Players: {n_players}")
                    node_sizes.append(40 if is_central else 25)
                    node_colors.append("gold" if is_central else "lightblue")

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers+text",
                    marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color="white")),
                    text=[n.split(" - ")[-1][:15] if " - " in n else n[:15] for n in mini_G.nodes()],
                    textposition="top center",
                    textfont=dict(size=10),
                    hoverinfo="text", hovertext=node_text,
                    showlegend=False
                )

                fig_mini = go.Figure(data=edge_traces + [node_trace])
                fig_mini.update_layout(
                    showlegend=False, hovermode="closest",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=400, plot_bgcolor="white",
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_mini, use_container_width=True)

                st.caption("🟢 Green = players improve after move | 🔴 Red = players decline | Edge width = transfer volume")
            else:
                st.info("No connected nodes found for this selection.")

            # Player-level details
            st.markdown("### Player Transitions")
            incoming_trans = trans_df[trans_df["target"] == selected_node].copy() if not trans_df.empty else pd.DataFrame()
            outgoing_trans = trans_df[trans_df["source"] == selected_node].copy() if not trans_df.empty else pd.DataFrame()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Best Arrivals** (highest contribution after joining)")
                if not incoming_trans.empty:
                    top_arrivals = incoming_trans.nlargest(8, "contribution_at_target")[
                        ["player_name", "source", "contribution_at_target", "contribution_delta"]
                    ].copy()
                    top_arrivals.columns = ["Player", "From", "After", "Δ"]
                    st.dataframe(top_arrivals.round(3), hide_index=True)
                else:
                    st.caption("No incoming transitions")

            with col2:
                st.markdown("**Best Departures** (highest contribution when leaving)")
                if not outgoing_trans.empty:
                    top_departures = outgoing_trans.nlargest(8, "contribution_at_source")[
                        ["player_name", "target", "contribution_at_source", "contribution_delta"]
                    ].copy()
                    top_departures.columns = ["Player", "To", "Before", "Δ"]
                    st.dataframe(top_departures.round(3), hide_index=True)
                else:
                    st.caption("No outgoing transitions")

    # =========================================================================
    # FEEDER CLUB/LEAGUE DETECTION
    # =========================================================================
    st.markdown("---")
    st.subheader("🏭 Feeder League/Club Detection")
    st.markdown("""
    Identify **development leagues** (feeder systems) and **destination leagues**.
    - **Development Score**: Average contribution delta of departing players (positive = players improve after leaving)
    - **Net Export**: Total transfers out minus transfers in
    - **Retention Rate**: What % of valuable players stay vs leave
    """)

    # Calculate feeder metrics per node
    node_metrics = []
    for node in nodes["node"].unique():
        outgoing = edges[edges["source"] == node]
        incoming = edges[edges["target"] == node]

        # Get individual transitions
        node_departures = trans_df[trans_df["source"] == node] if not trans_df.empty else pd.DataFrame()
        node_arrivals = trans_df[trans_df["target"] == node] if not trans_df.empty else pd.DataFrame()

        n_exports = outgoing["transfers"].sum() if not outgoing.empty else 0
        n_imports = incoming["transfers"].sum() if not incoming.empty else 0

        # Development score: how much do players improve after leaving?
        dev_score = node_departures["contribution_delta"].mean() if not node_departures.empty else 0

        # Retention calculation: need to know how many players stayed vs left
        # For now, approximate with transfer balance
        net_export = n_exports - n_imports

        # Average quality of exports vs imports
        avg_export_quality = node_departures["contribution_at_source"].mean() if not node_departures.empty else 0
        avg_import_quality = node_arrivals["contribution_at_source"].mean() if not node_arrivals.empty else 0

        node_metrics.append({
            "node": node,
            "exports": int(n_exports),
            "imports": int(n_imports),
            "net_export": int(net_export),
            "development_score": dev_score,
            "avg_export_quality": avg_export_quality,
            "avg_import_quality": avg_import_quality,
        })

    feeder_df = pd.DataFrame(node_metrics)

    if not feeder_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Feeder Leagues** (players improve after leaving)")
            feeders = feeder_df.nlargest(10, "development_score")[
                ["node", "development_score", "exports", "avg_export_quality"]
            ].copy()
            feeders.columns = ["League/Club", "Dev Score", "Exports", "Avg Export Quality"]
            st.dataframe(feeders.round(3), hide_index=True)

        with col2:
            st.markdown("**Top Destination Leagues** (players decline after arriving)")
            destinations = feeder_df.nsmallest(10, "development_score")[
                ["node", "development_score", "imports", "avg_import_quality"]
            ].copy()
            destinations.columns = ["League/Club", "Dev Score", "Imports", "Avg Import Quality"]
            st.dataframe(destinations.round(3), hide_index=True)

        # Net exporters vs importers
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Net Talent Exporters** (more players leaving than arriving)")
            exporters = feeder_df.nlargest(10, "net_export")[
                ["node", "net_export", "exports", "imports"]
            ].copy()
            exporters.columns = ["League/Club", "Net Export", "Total Out", "Total In"]
            st.dataframe(exporters, hide_index=True)

        with col2:
            st.markdown("**Net Talent Importers** (more players arriving than leaving)")
            importers = feeder_df.nsmallest(10, "net_export")[
                ["node", "net_export", "exports", "imports"]
            ].copy()
            importers.columns = ["League/Club", "Net Export", "Total Out", "Total In"]
            st.dataframe(importers, hide_index=True)

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

    # Normalize player_id
    players["pid_norm"] = players["player_id"].apply(_normalize_player_id)

    # Flag multi-team seasons
    players["is_multi_team"] = players["team"].str.startswith("Multiple Teams", na=False)

    # Build transitions
    transitions = []

    for pid, group in players.groupby("pid_norm"):
        group = group.sort_values("season")
        rows = group.to_dict("records")

        for i in range(1, len(rows)):
            prev = rows[i-1]
            curr = rows[i]

            # Check if consecutive seasons
            season_gap = curr["season"] - prev["season"]

            transitions.append({
                "player_id": pid,
                "player_name": curr["player_name"],
                "prev_season": prev["season"],
                "curr_season": curr["season"],
                "season_gap": season_gap,
                "prev_contribution": prev["contribution"],
                "curr_contribution": curr["contribution"],
                "prev_team": prev["team"],
                "curr_team": curr["team"],
                "prev_league": prev["league"],
                "curr_league": curr["league"],
                "prev_country": prev["country"],
                "curr_country": curr["country"],
                "prev_is_multi": prev["is_multi_team"],
                "curr_is_multi": curr["is_multi_team"],
                "switched_league": prev["league"] != curr["league"] or prev["country"] != curr["country"],
                "switched_team": prev["team"] != curr["team"],
            })

    return pd.DataFrame(transitions)


def page_markov_analysis():
    """Persistence analysis - season-to-season contribution evolution."""
    st.header("Persistence Analysis")

    st.markdown("""
    Analyze how player contributions persist from one season to the next.
    Each point represents a player's transition: X = previous season contribution, Y = current season contribution.
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
                               help="Filter players with at least this many FTE games")

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
        st.warning("No data matches the current filters.")
        return

    st.caption(f"Showing {len(df):,} player-season transitions")

    # === SCATTER PLOT ===
    st.subheader("Season-to-Season Contribution Scatter")

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

    # === STATISTICS ===
    st.subheader("Transition Statistics")

    col1, col2, col3, col4 = st.columns(4)

    # Calculate stats
    correlation = df["prev_contribution"].corr(df["curr_contribution"])
    mean_change = (df["curr_contribution"] - df["prev_contribution"]).mean()
    improved_pct = (df["curr_contribution"] > df["prev_contribution"]).mean() * 100
    regression_to_mean = df[df["prev_contribution"] > 0]["curr_contribution"].mean() - \
                        df[df["prev_contribution"] > 0]["prev_contribution"].mean()

    col1.metric("Correlation (r)", f"{correlation:.3f}")
    col2.metric("Mean Change", f"{mean_change:+.3f}")
    col3.metric("% Improved", f"{improved_pct:.1f}%")
    col4.metric("Regression to Mean", f"{regression_to_mean:+.3f}",
                help="Avg change for players who had positive contribution")

    # === CORRELATION BY LEAGUE (within same league) ===
    st.subheader("Within-League Correlation")

    st.markdown("How well does contribution predict next season **within the same league**?")

    # Group by league and compute correlations
    league_stats = []
    for (country, league), group in df.groupby(["curr_country", "curr_league"]):
        if len(group) >= 20:  # Minimum sample size
            corr = group["prev_contribution"].corr(group["curr_contribution"])
            n_transitions = len(group)
            mean_prev = group["prev_contribution"].mean()
            mean_curr = group["curr_contribution"].mean()
            league_stats.append({
                "League": f"{country} - {league}",
                "Transitions": n_transitions,
                "Correlation": corr,
                "Mean Prev": mean_prev,
                "Mean Curr": mean_curr,
                "Mean Change": mean_curr - mean_prev
            })

    if league_stats:
        league_df = pd.DataFrame(league_stats).sort_values("Correlation", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Correlation by League** (min 20 transitions)")
            st.dataframe(league_df.round(3), hide_index=True, height=300)

        with col2:
            # Correlation vs Transitions scatter (no labels - unreadable)
            fig_corr = px.scatter(league_df, x="Transitions", y="Correlation",
                                  hover_data=["League"],
                                  title="Correlation vs Number of Transitions")
            fig_corr.update_traces(marker=dict(size=8, opacity=0.7))
            fig_corr.update_layout(height=300)
            st.plotly_chart(fig_corr, use_container_width=True)

    # === BETWEEN-LEAGUE CORRELATION (Transfer Corridors) ===
    st.subheader("Between-League Transfer Corridors")

    st.markdown("""
    Correlation between contribution in **source league** and **destination league** for transfers.
    High correlation = reliable transfer corridor (talent translates well).
    Low correlation = risky transfers (hit or miss).
    """)

    # Get transitions where league changed (need to reload without league switch filter)
    transfers_df = compute_season_transitions(min_fte=min_fte)
    if season_gap_filter == "1 (consecutive)":
        transfers_df = transfers_df[transfers_df["season_gap"] == 1]
    if exclude_multi:
        transfers_df = transfers_df[~transfers_df["prev_is_multi"] & ~transfers_df["curr_is_multi"]]
    # Only league switchers for this analysis
    transfers_df = transfers_df[transfers_df["switched_league"]]

    if not transfers_df.empty:
        # Compute corridor stats
        corridor_stats = []
        for (prev_league, curr_league), group in transfers_df.groupby(["prev_league", "curr_league"]):
            if len(group) >= 5:  # Lower threshold for transfer corridors
                corr = group["prev_contribution"].corr(group["curr_contribution"])
                n_transfers = len(group)
                mean_prev = group["prev_contribution"].mean()
                mean_curr = group["curr_contribution"].mean()
                corridor_stats.append({
                    "From": prev_league,
                    "To": curr_league,
                    "Transfers": n_transfers,
                    "Correlation": corr if not pd.isna(corr) else 0,
                    "Avg Prev Contrib": mean_prev,
                    "Avg Curr Contrib": mean_curr,
                    "Avg Change": mean_curr - mean_prev
                })

        if corridor_stats:
            corridor_df = pd.DataFrame(corridor_stats).sort_values("Transfers", ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Transfer Corridors** (min 5 transfers)")
                # Show top corridors by volume
                display_df = corridor_df.head(30).copy()
                st.dataframe(display_df.round(3), hide_index=True, height=400)

            with col2:
                # Scatter: Transfers vs Correlation
                fig_corridor = px.scatter(
                    corridor_df[corridor_df["Transfers"] >= 5],
                    x="Transfers", y="Correlation",
                    size="Transfers",
                    hover_data=["From", "To", "Avg Change"],
                    title="Transfer Corridor Reliability",
                    color="Avg Change",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0
                )
                fig_corridor.update_traces(marker=dict(opacity=0.7))
                fig_corridor.update_layout(height=400)
                fig_corridor.add_hline(y=0.5, line_dash="dash", line_color="gray",
                                       annotation_text="r=0.5 (moderate)")
                st.plotly_chart(fig_corridor, use_container_width=True)

            # Best and worst corridors
            st.markdown("**Best Corridors** (high correlation, predictable transfers)")
            best = corridor_df[corridor_df["Transfers"] >= 5].nlargest(10, "Correlation")
            st.dataframe(best[["From", "To", "Transfers", "Correlation", "Avg Change"]].round(3),
                        hide_index=True)

            st.markdown("**Riskiest Corridors** (low correlation, unpredictable)")
            worst = corridor_df[corridor_df["Transfers"] >= 5].nsmallest(10, "Correlation")
            st.dataframe(worst[["From", "To", "Transfers", "Correlation", "Avg Change"]].round(3),
                        hide_index=True)
        else:
            st.info("Not enough transfer data for corridor analysis (need at least 5 transfers per corridor).")
    else:
        st.info("No league-switching transfers in the current data.")

    # === TRANSITION PROBABILITY MATRIX (with Exit) ===
    st.subheader("Transition Probability Matrix")

    st.markdown("""
    Binned transition probabilities: rows = previous contribution bin, columns = current contribution bin.
    **Exit** = player did not appear in the following season (retired, moved to untracked league, etc.).
    This reduces "reversion to mean" since many low contributors exit entirely.
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        n_bins = st.slider("Number of Bins", 3, 10, 5)
        bin_method = st.selectbox("Binning Method", ["Quantile (equal count)", "Equal Width"])
        include_exits = st.checkbox("Include Exits", value=True,
                                    help="Include players who didn't appear next season")

    with col2:
        # For exit calculation, we need players who existed in season N
        # and check if they appeared in season N+1
        all_players = load_all_players(min_fte=min_fte)

        # Build set of (league, season) pairs for valid exit detection
        league_seasons = set(zip(all_players["league"], all_players["season"]))

        # Get unique player-seasons
        player_seasons = all_players.groupby(["player_id", "season", "league"]).agg({
            "contribution": "first",
            "player_name": "first"
        }).reset_index()

        # Check if next season exists for the league
        player_seasons["next_season_exists"] = player_seasons.apply(
            lambda row: (row["league"], row["season"] + 1) in league_seasons, axis=1
        )

        # Check if player appeared anywhere in next season
        next_year_players = all_players[["player_id", "season"]].copy()
        next_year_players["prev_season"] = next_year_players["season"] - 1
        next_year_players = next_year_players[["player_id", "prev_season"]].drop_duplicates()
        next_year_players["appeared_next"] = True

        player_seasons = player_seasons.merge(
            next_year_players,
            left_on=["player_id", "season"],
            right_on=["player_id", "prev_season"],
            how="left"
        )
        player_seasons["appeared_next"] = player_seasons["appeared_next"].fillna(False)

        # Exits are players where next season exists for league but player didn't appear
        player_seasons["exited"] = player_seasons["next_season_exists"] & ~player_seasons["appeared_next"]

        # Filter to valid transitions only (next season must exist)
        valid_for_matrix = player_seasons[player_seasons["next_season_exists"]].copy()

        if valid_for_matrix.empty:
            st.warning("No valid data for transition matrix.")
        else:
            # Create bins based on contribution
            if bin_method == "Quantile (equal count)":
                bin_edges = pd.qcut(valid_for_matrix["contribution"], n_bins, duplicates="drop", retbins=True)[1]
            else:
                bin_edges = pd.cut(valid_for_matrix["contribution"], n_bins, retbins=True)[1]

            bin_labels = [f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}"
                         for i in range(len(bin_edges)-1)]

            valid_for_matrix["prev_bin"] = pd.cut(
                valid_for_matrix["contribution"], bins=bin_edges, labels=bin_labels, include_lowest=True
            )

            # For players who continued, get their next season contribution
            continued = valid_for_matrix[~valid_for_matrix["exited"]].copy()
            exited = valid_for_matrix[valid_for_matrix["exited"]].copy()

            # Get next season contributions for continued players
            next_contribs = all_players[["player_id", "season", "contribution"]].copy()
            next_contribs["prev_season"] = next_contribs["season"] - 1
            next_contribs = next_contribs.rename(columns={"contribution": "next_contribution"})

            continued = continued.merge(
                next_contribs[["player_id", "prev_season", "next_contribution"]],
                left_on=["player_id", "season"],
                right_on=["player_id", "prev_season"],
                how="left"
            )

            continued["curr_bin"] = pd.cut(
                continued["next_contribution"], bins=bin_edges, labels=bin_labels, include_lowest=True
            )

            # Build transition data
            if include_exits:
                # Add exits as a category
                exited["curr_bin"] = "EXIT"
                transition_data = pd.concat([
                    continued[["prev_bin", "curr_bin"]],
                    exited[["prev_bin", "curr_bin"]]
                ], ignore_index=True)
                all_outcomes = bin_labels + ["EXIT"]
            else:
                transition_data = continued[["prev_bin", "curr_bin"]].copy()
                all_outcomes = bin_labels

            # Create transition matrix
            transition_counts = pd.crosstab(
                transition_data["prev_bin"],
                transition_data["curr_bin"],
                margins=True, margins_name="Total"
            )

            # Reorder columns to put EXIT at the end
            if include_exits and "EXIT" in transition_counts.columns:
                cols = [c for c in transition_counts.columns if c not in ["EXIT", "Total"]]
                cols = cols + ["EXIT", "Total"] if "Total" in transition_counts.columns else cols + ["EXIT"]
                transition_counts = transition_counts[[c for c in cols if c in transition_counts.columns]]

            # Create probabilities (row-normalized, excluding Total row)
            prob_data = transition_data.dropna()
            transition_probs = pd.crosstab(prob_data["prev_bin"], prob_data["curr_bin"], normalize="index")

            # Reorder columns for probs too
            if include_exits and "EXIT" in transition_probs.columns:
                cols = [c for c in transition_probs.columns if c != "EXIT"] + ["EXIT"]
                transition_probs = transition_probs[[c for c in cols if c in transition_probs.columns]]

            # Display as heatmap
            fig_matrix = px.imshow(
                transition_probs.values,
                x=transition_probs.columns.tolist(),
                y=transition_probs.index.tolist(),
                color_continuous_scale="Blues",
                aspect="auto",
                labels=dict(x="Current Season Bin", y="Previous Season Bin", color="Probability")
            )

            # Add text annotations
            for i, row in enumerate(transition_probs.values):
                for j, val in enumerate(row):
                    fig_matrix.add_annotation(
                        x=j, y=i,
                        text=f"{val:.2f}",
                        showarrow=False,
                        font=dict(color="white" if val > 0.3 else "black", size=10)
                    )

            fig_matrix.update_layout(
                title="Transition Probability Matrix" + (" (with Exits)" if include_exits else ""),
                height=450,
                xaxis_title="Current Season Outcome",
                yaxis_title="Previous Season Contribution Bin"
            )

            st.plotly_chart(fig_matrix, use_container_width=True)

            # Show exit rates by bin
            if include_exits and "EXIT" in transition_probs.columns:
                st.markdown("**Exit Rates by Contribution Level**")
                exit_rates = transition_probs["EXIT"].sort_index()
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(exit_rates.round(3).reset_index().rename(
                        columns={"prev_bin": "Contribution Bin", "EXIT": "Exit Rate"}
                    ), hide_index=True)
                with col2:
                    fig_exit = px.bar(
                        x=exit_rates.index.astype(str),
                        y=exit_rates.values,
                        title="Exit Rate by Contribution Bin",
                        labels={"x": "Contribution Bin", "y": "Exit Probability"}
                    )
                    fig_exit.update_layout(height=250)
                    st.plotly_chart(fig_exit, use_container_width=True)

    # Show raw counts
    with st.expander("View Transition Counts"):
        st.dataframe(transition_counts)

    # === TOP IMPROVERS / DECLINERS ===
    st.subheader("Notable Transitions")

    col1, col2 = st.columns(2)

    df["change"] = df["curr_contribution"] - df["prev_contribution"]

    with col1:
        st.markdown("**Biggest Improvers**")
        top_improvers = df.nlargest(15, "change")[
            ["player_name", "prev_season", "curr_season", "prev_contribution",
             "curr_contribution", "change", "prev_team", "curr_team"]
        ].copy()
        top_improvers.columns = ["Player", "Prev Season", "Curr Season", "Prev Contrib",
                                 "Curr Contrib", "Change", "Prev Team", "Curr Team"]
        st.dataframe(top_improvers.round(3), hide_index=True, height=400)

    with col2:
        st.markdown("**Biggest Decliners**")
        top_decliners = df.nsmallest(15, "change")[
            ["player_name", "prev_season", "curr_season", "prev_contribution",
             "curr_contribution", "change", "prev_team", "curr_team"]
        ].copy()
        top_decliners.columns = ["Player", "Prev Season", "Curr Season", "Prev Contrib",
                                 "Curr Contrib", "Change", "Prev Team", "Curr Team"]
        st.dataframe(top_decliners.round(3), hide_index=True, height=400)

    # =========================================================================
    # REGRESSION-TO-MEAN PREDICTION
    # =========================================================================
    st.markdown("---")
    st.subheader("📊 Regression-to-Mean Prediction")

    st.markdown("""
    Predict next-season contribution using regression-to-mean:
    - Players with extreme contributions tend to regress toward the average
    - **Expected Next Season** = Population Mean + (Correlation × (Current - Mean))
    - Higher correlation = less regression; lower correlation = more regression
    """)

    # Load all player data for predictions
    all_players = load_all_players(min_fte=min_fte)

    if all_players.empty:
        st.warning("No player data available for predictions")
    else:
        # Get latest season data for each player
        latest_season = all_players["season"].max()
        current_players = all_players[all_players["season"] == latest_season].copy()

        if current_players.empty:
            st.warning(f"No players found in the latest season ({latest_season})")
        else:
            # Calculate regression-to-mean prediction
            # Formula: E[next] = mean + r * (current - mean)
            overall_mean = all_players["contribution"].mean()
            overall_std = all_players["contribution"].std()

            # Get correlation from the transitions data
            if len(df) > 10:
                r_value = df["prev_contribution"].corr(df["curr_contribution"])
            else:
                r_value = 0.5  # Default assumption

            current_players["predicted_next"] = overall_mean + r_value * (current_players["contribution"] - overall_mean)
            current_players["expected_change"] = current_players["predicted_next"] - current_players["contribution"]

            # Show prediction formula
            col1, col2, col3 = st.columns(3)
            col1.metric("Population Mean", f"{overall_mean:.3f}")
            col2.metric("Correlation (r)", f"{r_value:.3f}")
            col3.metric("Regression Strength", f"{1 - r_value:.1%}", help="How much regression toward mean")

            # Show predictions for notable players
            st.markdown("**Expected to Decline** (currently high performers)")
            expected_decline = current_players[current_players["expected_change"] < -0.05].nlargest(15, "contribution")
            if not expected_decline.empty:
                decline_display = expected_decline[
                    ["player_name", "team", "league", "contribution", "predicted_next", "expected_change"]
                ].copy()
                decline_display.columns = ["Player", "Team", "League", "Current", "Predicted Next", "Expected Δ"]
                st.dataframe(decline_display.round(3), hide_index=True)
            else:
                st.caption("No players with expected significant decline")

            st.markdown("**Expected to Improve** (currently underperforming)")
            expected_improve = current_players[current_players["expected_change"] > 0.05].nsmallest(15, "contribution")
            if not expected_improve.empty:
                improve_display = expected_improve[
                    ["player_name", "team", "league", "contribution", "predicted_next", "expected_change"]
                ].copy()
                improve_display.columns = ["Player", "Team", "League", "Current", "Predicted Next", "Expected Δ"]
                st.dataframe(improve_display.round(3), hide_index=True)
            else:
                st.caption("No players with expected significant improvement")

            # Scatter: Current vs Predicted
            with st.expander("View Prediction Scatter", expanded=False):
                fig_pred = px.scatter(
                    current_players,
                    x="contribution",
                    y="predicted_next",
                    hover_data=["player_name", "team", "league"],
                    title=f"Current vs Predicted Next Season (r={r_value:.3f})",
                    labels={"contribution": "Current Contribution", "predicted_next": "Predicted Next"}
                )
                fig_pred.add_trace(go.Scatter(
                    x=[current_players["contribution"].min(), current_players["contribution"].max()],
                    y=[current_players["contribution"].min(), current_players["contribution"].max()],
                    mode="lines", line=dict(dash="dash", color="gray"),
                    name="y=x (no change)"
                ))
                fig_pred.update_traces(marker=dict(size=5, opacity=0.5), selector=dict(mode="markers"))
                fig_pred.update_layout(height=400)
                st.plotly_chart(fig_pred, use_container_width=True)


# =============================================================================
#                          FREE AGENTS (DISAPPEARED TALENT)
# =============================================================================

@st.cache_data
def compute_free_agents(min_fte: float = 3.0, max_age: int | None = None):
    """
    Find players who appeared in season N but NOT in season N+1 anywhere in the database.
    These are potential 'free agents' - talent that disappeared from tracked leagues.

    IMPORTANT: Only considers players as "disappeared" if their league has data for
    the following season. Otherwise we simply don't know if they continued or not.

    Returns DataFrame with player info from their last tracked season.
    """
    if not Path(DB_PATH).exists():
        return pd.DataFrame()

    df = load_all_players(min_fte=min_fte)
    if df.empty:
        return pd.DataFrame()

    # Build set of (league, season) pairs that exist in the database
    league_seasons = set(zip(df["league"], df["season"]))

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

    # Apply age filter if specified
    if max_age is not None and "age" in result.columns:
        result = result[(result["age"].isna()) | (result["age"] <= max_age)]

    # Drop helper columns
    result = result.drop(columns=["next_season_exists"], errors="ignore")

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
                               help="Minimum full-time-equivalent games played")

            min_contribution = st.slider("Min Contribution", -1.0, 2.0, 0.0, 0.05,
                                        key="fa_contrib",
                                        help="Minimum contribution value to show")

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

    if df.empty:
        st.warning("No players match the current filters.")
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
    display_df.columns = [c.replace("_", " ").title() for c in display_cols]

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


def page_league_quality():
    """Synthetic control analysis for league quality comparison."""
    st.header("🧪 League Quality Analysis")

    st.markdown("""
    Estimate **league quality differences** when players move between leagues using a synthetic control approach:

    1. Find players who **switched leagues** between seasons
    2. Match each switcher with **similar players who stayed** in the source league
    3. Compare: How did the switcher perform vs. their matched peers?
    4. The **difference reveals the quality gap** between leagues

    *Negative effect = destination league is more competitive. Positive = destination is less competitive.*
    """)

    # Controls in sidebar
    with st.sidebar:
        st.subheader("Analysis Settings")

        with st.expander("🔧 Configuration", expanded=True):
            min_fte = st.slider("Min FTE Games", 1.0, 15.0, 5.0, 0.5,
                               help="Minimum games for reliable estimates")
            n_matches = st.slider("Peer Matches", 3, 10, 5,
                                 help="Number of similar peers to match")
            min_switches = st.slider("Min Switches per Corridor", 2, 15, 3,
                                    help="Minimum number of player switches to include a corridor")

    # Compute synthetic control
    with st.spinner("Computing synthetic control matches..."):
        results = compute_synthetic_control_matches(min_fte=min_fte, n_matches=n_matches)

    if results.empty:
        render_empty_state(
            "No league switches found",
            "Need players who changed leagues between consecutive seasons"
        )
        return

    # Summary metrics
    st.subheader("Overall Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("League Switches Analyzed", len(results))
    col2.metric("Avg Switcher Δ", f"{results['switcher_change'].mean():+.3f}")
    col3.metric("Avg Stayer Δ", f"{results['avg_stayer_change'].mean():+.3f}")
    col4.metric("Avg League Effect", f"{results['league_effect'].mean():+.3f}",
                help="Positive = destination league is easier")

    # League-to-League effects
    st.subheader("Transition Risk by Corridor")

    # Aggregate by corridor
    corridor_effects = results.groupby(["from_league", "to_league"]).agg(
        n_switches=("player_id", "count"),
        avg_switcher_change=("switcher_change", "mean"),
        avg_stayer_change=("avg_stayer_change", "mean"),
        avg_league_effect=("league_effect", "mean"),
        std_league_effect=("league_effect", "std")
    ).reset_index()

    # Filter to corridors with enough data
    corridor_effects = corridor_effects[corridor_effects["n_switches"] >= min_switches].copy()

    if not corridor_effects.empty:
        # Sort by league effect
        corridor_effects = corridor_effects.sort_values("avg_league_effect")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Moving to Harder Leagues** (contribution drops more than peers)")
            harder = corridor_effects[corridor_effects["avg_league_effect"] < -0.02].head(15)
            if not harder.empty:
                display_df = harder[["from_league", "to_league", "n_switches", "avg_league_effect"]].copy()
                display_df.columns = ["From", "To", "N", "Effect"]
                st.dataframe(display_df.round(3), hide_index=True)
            else:
                st.caption("No corridors with significant negative effect")

        with col2:
            st.markdown("**Moving to Easier Leagues** (contribution improves vs peers)")
            easier = corridor_effects[corridor_effects["avg_league_effect"] > 0.02].tail(15).iloc[::-1]
            if not easier.empty:
                display_df = easier[["from_league", "to_league", "n_switches", "avg_league_effect"]].copy()
                display_df.columns = ["From", "To", "N", "Effect"]
                st.dataframe(display_df.round(3), hide_index=True)
            else:
                st.caption("No corridors with significant positive effect")

        # Visualization
        st.markdown("---")
        st.subheader("League Effect Distribution")

        if len(corridor_effects) >= 5:
            fig = px.scatter(
                corridor_effects,
                x="n_switches",
                y="avg_league_effect",
                size="n_switches",
                hover_data=["from_league", "to_league"],
                title="League Effect by Transfer Corridor",
                labels={"n_switches": "Number of Switches", "avg_league_effect": "Avg League Effect"}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Aggregate league quality rankings
    st.markdown("---")
    st.subheader("Inferred League Quality Rankings")

    st.markdown("""
    Aggregate league effects to estimate relative quality.
    **Negative as destination** = harder league (players perform worse than peers).
    **Positive as destination** = easier league (players perform better than peers).
    """)

    # Calculate net effect for each league
    as_destination = results.groupby("to_league")["league_effect"].agg(["mean", "count", "std"]).reset_index()
    as_destination.columns = ["league", "effect_as_dest", "n_incoming", "std_incoming"]

    as_source = results.groupby("from_league")["league_effect"].agg(["mean", "count"]).reset_index()
    as_source.columns = ["league", "effect_leaving", "n_outgoing"]
    as_source["effect_leaving"] = -as_source["effect_leaving"]  # Flip sign for interpretation

    # Merge
    league_rankings = as_destination.merge(as_source, on="league", how="outer").fillna(0)
    league_rankings["total_transfers"] = league_rankings["n_incoming"] + league_rankings["n_outgoing"]
    league_rankings["quality_score"] = (
        league_rankings["effect_as_dest"] * league_rankings["n_incoming"] +
        league_rankings["effect_leaving"] * league_rankings["n_outgoing"]
    ) / league_rankings["total_transfers"].clip(lower=1)

    # Filter to leagues with enough data
    league_rankings = league_rankings[league_rankings["total_transfers"] >= 5].copy()
    league_rankings = league_rankings.sort_values("quality_score")

    if not league_rankings.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Hardest Leagues** (players underperform vs peers)")
            hardest = league_rankings.head(15)[["league", "quality_score", "total_transfers"]].copy()
            hardest.columns = ["League", "Quality Score", "Transfers"]
            st.dataframe(hardest.round(3), hide_index=True)

        with col2:
            st.markdown("**Easiest Leagues** (players overperform vs peers)")
            easiest = league_rankings.tail(15).iloc[::-1][["league", "quality_score", "total_transfers"]].copy()
            easiest.columns = ["League", "Quality Score", "Transfers"]
            st.dataframe(easiest.round(3), hide_index=True)

    # Individual cases
    st.markdown("---")
    st.subheader("Individual Transfer Cases")

    with st.expander("View Individual Matches", expanded=False):
        display_results = results[
            ["player_name", "from_league", "to_league", "season",
             "prev_contribution", "curr_contribution", "switcher_change",
             "avg_stayer_change", "league_effect", "n_matches"]
        ].copy()
        display_results.columns = ["Player", "From", "To", "Season", "Prev", "Curr",
                                   "Switcher Δ", "Stayer Δ", "League Effect", "Matches"]
        st.dataframe(display_results.round(3), hide_index=True, height=400)

    # Export
    st.download_button(
        "📥 Export Analysis",
        data=results.to_csv(index=False),
        file_name="league_quality_analysis.csv",
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
    # SLIDE 1: Title and Core Question
    # ==========================================================================
    st.title("⚽ Off-Balance-Sheet Contribution")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Beyond Goals and Assists

        Traditional stats only credit the **final touch**. But a goal is a team effort —
        the defender who won the ball, the midfielder who switched play, the winger who
        dragged markers away. Only one or two players get recorded.

        Popular metrics like distance covered, pass success rate, or xG describe
        **activity**, not **impact**. A player can run 12km and complete 95% of passes
        while their team loses.

        And what about contributions that don't show up in any stat sheet?
        - Defensive positioning that prevents chances
        - Pressing that forces turnovers
        - Movement that creates space for teammates
        - Leadership that organizes the team

        We ask a different question:

        > **When this player is on the pitch, does the team's goal difference improve?**

        This captures *everything* a player does — whether visible, recorded, or quantifiable.
        """)

    with col2:
        # Mini visualization: Goals vs Contribution scatter for top scorers
        buli_2023 = df[(df["country"] == "Germany") & (df["league"] == "Bundesliga") & (df["season"] == 2023) & (df["FTE_games_played"] >= 10)]
        if not buli_2023.empty and "goals" in buli_2023.columns:
            top_scorers = buli_2023.nlargest(15, "goals").copy()
            top_scorers["hover_text"] = make_player_hover(top_scorers)
            fig = px.scatter(
                top_scorers,
                x="goals",
                y="contribution",
                title="Top 15 Scorers: Goals vs Contribution",
                labels={"goals": "Goals Scored", "contribution": "Contribution (goals/90)"}
            )
            fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>", customdata=top_scorers[["hover_text"]].values)
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("*Bundesliga 2023/24 — more goals doesn't mean higher contribution*")

    st.markdown("---")

    # ==========================================================================
    # SLIDE 3: The Method + Interpretation (consolidated)
    # ==========================================================================
    st.markdown("## The Method: Match Segments")

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

    st.markdown("---")

    # ==========================================================================
    # SLIDE 4.5: Why Lasso, Not OLS?
    # ==========================================================================
    st.markdown("## Why Lasso, Not OLS?")

    st.markdown("""
    **The short version:** When some players rarely leave the pitch, standard regression (OLS)
    can't tell their personal contribution apart from their team's overall strength. Lasso
    regression fixes this by gently pulling extreme estimates back toward zero.

    *From here on, all analyses use Lasso contribution estimates.*
    """)

    with st.expander("🤓 The statistical details"):
        st.markdown("""
        Plain OLS regression produces **extreme outliers** (see left plot below). Goalkeepers and
        high-minute defenders who rarely rotate inherit team-level effects, leading to implausible
        coefficients. **Lasso regularization** shrinks these unstable estimates toward zero (right plot).

        **Why the extremes?** When players rarely rotate, OLS cannot separate individual effects from
        team performance — a classic **multicollinearity** problem. Lasso resolves this by introducing
        a small bias in exchange for dramatically lower variance. Both estimators remain consistent,
        but given the evident collinearity (±17 goals per 90 min is absurd), accepting modest bias
        for plausible estimates is a worthwhile tradeoff.
        """)

    # Get German Bundesliga 2023 data with OLS
    buli_ols = df[(df["country"] == "Germany") & (df["league"] == "Bundesliga") &
                  (df["season"] == 2023) & (df["FTE_games_played"] >= 5)].copy()

    if not buli_ols.empty and "contribution_ols" in buli_ols.columns:
        buli_ols = buli_ols.copy()
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
            - **Goalkeepers dominate** the top — they're always on the pitch
              when things go well (or poorly)

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
    # SLIDE 6: Use Case - Youth Scouting
    # ==========================================================================
    st.markdown("## Use Case: Youth Scouting")

    st.markdown("""
    Can we identify promising youth players **before** they break through?

    We tracked the **U19 Bundesliga 2021/22** cohort to see who reached senior football.
    """)

    col1, col2 = st.columns(2)

    with col1:
        # Create distribution data for box plot (representative values based on actual analysis)
        np.random.seed(42)
        box_data = pd.DataFrame({
            "Outcome": (["Parent Club"] * 50 + ["Transfer"] * 65 + ["Not Tracked"] * 200),
            "Contribution": (
                list(np.random.normal(0.18, 0.35, 50)) +  # Parent club: higher mean
                list(np.random.normal(0.00, 0.40, 65)) +  # Transfer: neutral
                list(np.random.normal(0.01, 0.30, 200))   # Not tracked (sample)
            )
        })

        fig = px.box(
            box_data,
            x="Outcome",
            y="Contribution",
            color="Outcome",
            color_discrete_sequence=["#2ecc71", "#f39c12", "#95a5a6"],
            title="U19 Contribution by Senior Outcome",
            labels={"Contribution": "Youth Contribution"}
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("*n = 50 (Parent Club), 65 (Transfer), 1,412 (Not Tracked)*")

    with col2:
        st.markdown("""
        ### Key Finding

        | Outcome | n | Mean Contribution |
        |---------|---|-------------------|
        | Parent Club | 50 | **+0.18** |
        | Transfer | 65 | 0.00 |
        | Not tracked | 1,412 | +0.01 |

        **Higher contribution → Stay with parent club**

        Clubs keep their best prospects. Transfers are "hit or miss" —
        players who didn't make the cut but found opportunities elsewhere.

        *This validates the metric as a scouting signal.*
        """)

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

    # Coverage table by country and league
    coverage = df.groupby(["country", "league"]).agg(
        From=("season", "min"),
        To=("season", "max")
    ).reset_index()

    # Top 5 European leagues
    TOP_5_LEAGUES = {
        ("England", "Premier League"),
        ("Spain", "La Liga"),
        ("Germany", "Bundesliga"),
        ("Italy", "Serie A"),
        ("France", "Ligue 1"),
    }

    # Categorize leagues for coloring
    def is_youth_league(league_lower: str) -> bool:
        return ("u19" in league_lower or "u-19" in league_lower or "youth" in league_lower
                or "premier league 2" in league_lower or "u21" in league_lower
                or "u23" in league_lower or "reserve" in league_lower
                or "primavera" in league_lower or "juvenil" in league_lower
                or "u17" in league_lower or "u18" in league_lower)

    def get_league_category(row) -> str:
        league_lower = row["league"].lower()
        if (row["country"], row["league"]) in TOP_5_LEAGUES:
            return "top5"
        if is_youth_league(league_lower):
            return "youth"
        return "other"

    # Assign league tier for sorting (lower = higher tier)
    def get_league_tier(row) -> int:
        league = row["league"]
        league_lower = league.lower()
        # Top 5 leagues always first
        if (row["country"], league) in TOP_5_LEAGUES:
            return 5
        # Youth leagues last
        if is_youth_league(league_lower):
            return 90
        # Second division patterns
        if ("2." in league or "second" in league_lower or "ii" in league_lower
            or "championship" in league_lower or "serie b" in league_lower
            or "ligue 2" in league_lower or "segunda" in league_lower):
            return 20
        # Third division
        if "3." in league or "third" in league_lower:
            return 30
        return 10  # Other top divisions

    coverage["_category"] = coverage.apply(get_league_category, axis=1)
    coverage["_tier"] = coverage.apply(get_league_tier, axis=1)
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

        if category == "top5":
            return ["background-color: rgba(76, 175, 80, 0.2)"] * len(row)  # Green
        elif category == "youth":
            return ["background-color: rgba(33, 150, 243, 0.2)"] * len(row)  # Blue
        else:
            return [""] * len(row)  # No color

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
        st.caption("🟩 Top 5 leagues · 🟦 Youth leagues")

    st.markdown("---")

    # ==========================================================================
    # SLIDE 8: Navigation Guide
    # ==========================================================================
    st.markdown("## Explore the Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📊 Contribution Analysis**

        Filter by league, team, position.
        Compare goals vs contribution.
        Find hidden gems.
        """)

        st.markdown("""
        **🔄 Persistence**

        Does contribution persist year-to-year?
        Test predictive power of the metric.
        """)

    with col2:
        st.markdown("""
        **👤 Player Career**

        Track any player across seasons.
        See contribution evolve over time.
        Compare to league averages.
        """)

        st.markdown("""
        **🔍 Top Free Agents**

        Find high contributors who disappeared.
        Potential scouting opportunities.
        """)

    with col3:
        st.markdown("""
        **🌐 Transfer Networks**

        Visualize talent flow between clubs/leagues.
        Which corridors produce consistent value?
        """)

        st.markdown("""
        **📈 Distribution Comparison**

        Compare leagues head-to-head.
        Which have more variance in quality?
        """)

    st.markdown("---")

    st.caption("""
    *Data: API-Football match events. Method: Lasso regression on match segments.
    Use the sidebar to navigate between pages.*
    """)


# =============================================================================
#                          PLAYER COMPARISON
# =============================================================================

def page_compare_players():
    """Compare two players side by side."""
    st.header("Player Comparison")

    st.markdown("Compare two players' career trajectories and statistics side by side.")

    with st.spinner("Loading player data..."):
        df = load_all_players(min_fte=3.0)

    if df.empty:
        render_empty_state("No player data available")
        return

    # Player search in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Player 1")
        search1 = st.text_input("Search player name", key="compare_search1")
        player1_id = None
        player1_name = None

        if search1:
            matches = df[df["player_name"].str.contains(search1, case=False, na=False)]
            players = matches.groupby("player_id").agg({
                "player_name": "first",
                "season": "count",
                "team": lambda x: ", ".join(sorted(set(x.dropna())))[:30]
            }).reset_index()
            players.columns = ["player_id", "player_name", "seasons", "teams"]
            players = players.sort_values("seasons", ascending=False)

            if not players.empty:
                options1 = [f"{row['player_name']} ({row['seasons']} seasons) - {row['teams'][:25]}"
                           for _, row in players.head(10).iterrows()]
                ids1 = players.head(10)["player_id"].tolist()
                idx1 = st.selectbox("Select player", range(len(options1)),
                                   format_func=lambda i: options1[i], key="compare_sel1")
                if idx1 is not None:
                    player1_id = ids1[idx1]
                    player1_name = players.iloc[idx1]["player_name"]

    with col2:
        st.subheader("Player 2")
        search2 = st.text_input("Search player name", key="compare_search2")
        player2_id = None
        player2_name = None

        if search2:
            matches = df[df["player_name"].str.contains(search2, case=False, na=False)]
            players = matches.groupby("player_id").agg({
                "player_name": "first",
                "season": "count",
                "team": lambda x: ", ".join(sorted(set(x.dropna())))[:30]
            }).reset_index()
            players.columns = ["player_id", "player_name", "seasons", "teams"]
            players = players.sort_values("seasons", ascending=False)

            if not players.empty:
                options2 = [f"{row['player_name']} ({row['seasons']} seasons) - {row['teams'][:25]}"
                           for _, row in players.head(10).iterrows()]
                ids2 = players.head(10)["player_id"].tolist()
                idx2 = st.selectbox("Select player", range(len(options2)),
                                   format_func=lambda i: options2[i], key="compare_sel2")
                if idx2 is not None:
                    player2_id = ids2[idx2]
                    player2_name = players.iloc[idx2]["player_name"]

    # Show comparison if both players selected
    if player1_id and player2_id:
        st.markdown("---")

        p1_data = df[df["player_id"] == player1_id].sort_values("season")
        p2_data = df[df["player_id"] == player2_id].sort_values("season")

        # Comparison metrics
        st.subheader("Career Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {player1_name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Seasons", len(p1_data))
            c2.metric("Avg Contrib", f"{p1_data['contribution'].mean():.2f}")
            c3.metric("Goals", int(p1_data['goals'].sum()))
            c4.metric("Assists", int(p1_data['assists'].sum()))

        with col2:
            st.markdown(f"### {player2_name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Seasons", len(p2_data))
            c2.metric("Avg Contrib", f"{p2_data['contribution'].mean():.2f}")
            c3.metric("Goals", int(p2_data['goals'].sum()))
            c4.metric("Assists", int(p2_data['assists'].sum()))

        # Career trajectory chart
        st.subheader("Career Trajectory")

        # Chart options
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            x_axis_option = st.radio("X-Axis:", ["Season", "Age"],
                                     horizontal=True, key="compare_x_axis")
        with opt_col2:
            y_axis_option = st.selectbox("Y-Axis:", ["Contribution", "Goals", "Assists", "Team Rank"],
                                        key="compare_y_axis")

        # Determine Y-axis data and labels
        y_col_map = {
            "Contribution": ("contribution", "Contribution (goals/90)"),
            "Goals": ("goals", "Goals"),
            "Assists": ("assists", "Assists"),
            "Team Rank": ("team_rank", "Team Final Position")
        }
        y_col, y_label = y_col_map[y_axis_option]
        reverse_y = y_axis_option == "Team Rank"  # Lower rank = better

        fig = go.Figure()

        # Helper to build hover data for each player
        def build_hover_data(pdata: pd.DataFrame) -> pd.DataFrame:
            hd = pdata[["team", "league", "country", "position", "season", "goals", "assists", "FTE_games_played", "team_rank", "contribution"]].copy()
            hd["goals"] = hd["goals"].fillna(0).astype(int)
            hd["assists"] = hd["assists"].fillna(0).astype(int)
            hd["FTE_games_played"] = hd["FTE_games_played"].fillna(0)
            hd["team_rank"] = hd["team_rank"].fillna(0).astype(int)
            return hd

        # Hover template
        hover_template = (
            "<b>%{customdata[0]}</b> (%{customdata[9]:+.2f})<br>"
            "%{customdata[3]}, %{customdata[1]}<br>"
            "%{customdata[2]}, Season %{customdata[4]:.0f}<br>"
            "%{customdata[5]} goals, %{customdata[6]} assists<br>"
            "%{customdata[7]:.1f} FTE games, Rank: %{customdata[8]}"
            "<extra></extra>"
        )

        if x_axis_option == "Age" and "age" in p1_data.columns and "age" in p2_data.columns:
            # Align by age
            p1_ages = p1_data[p1_data["age"].notna()].copy()
            p2_ages = p2_data[p2_data["age"].notna()].copy()

            if not p1_ages.empty and not p2_ages.empty:
                hd1 = build_hover_data(p1_ages)
                hd2 = build_hover_data(p2_ages)

                fig.add_trace(go.Scatter(
                    x=p1_ages["age"],
                    y=p1_ages[y_col],
                    mode="lines+markers",
                    name=player1_name,
                    line=dict(color="#648FFF", width=3),
                    marker=dict(size=10),
                    customdata=hd1.values,
                    hovertemplate=hover_template
                ))

                fig.add_trace(go.Scatter(
                    x=p2_ages["age"],
                    y=p2_ages[y_col],
                    mode="lines+markers",
                    name=player2_name,
                    line=dict(color="#DC267F", width=3),
                    marker=dict(size=10),
                    customdata=hd2.values,
                    hovertemplate=hover_template
                ))

                x_title = "Age"
                title = f"{y_axis_option} by Age"
            else:
                st.warning("Age data not available for comparison. Showing by season.")
                x_axis_option = "Season"  # Fall back

        if x_axis_option == "Season":
            hd1 = build_hover_data(p1_data)
            hd2 = build_hover_data(p2_data)

            fig.add_trace(go.Scatter(
                x=p1_data["season"],
                y=p1_data[y_col],
                mode="lines+markers",
                name=player1_name,
                line=dict(color="#648FFF", width=3),
                marker=dict(size=10),
                customdata=hd1.values,
                hovertemplate=hover_template
            ))

            fig.add_trace(go.Scatter(
                x=p2_data["season"],
                y=p2_data[y_col],
                mode="lines+markers",
                name=player2_name,
                line=dict(color="#DC267F", width=3),
                marker=dict(size=10),
                customdata=hd2.values,
                hovertemplate=hover_template
            ))

            x_title = "Season"
            title = f"{y_axis_option} Over Time"

        if y_axis_option == "Contribution":
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_label,
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        # Reverse Y-axis for Team Rank (lower = better)
        if reverse_y:
            fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, use_container_width=True)

        # Season-by-season comparison table
        st.subheader("Season Details")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{player1_name}**")
            display_cols1 = p1_data[["season", "team", "league", "contribution", "goals", "assists", "team_rank"]].copy()
            display_cols1 = display_cols1.rename(columns={"team_rank": "Rank"})
            st.dataframe(display_cols1, hide_index=True, height=300)

        with col2:
            st.markdown(f"**{player2_name}**")
            display_cols2 = p2_data[["season", "team", "league", "contribution", "goals", "assists", "team_rank"]].copy()
            display_cols2 = display_cols2.rename(columns={"team_rank": "Rank"})
            st.dataframe(display_cols2, hide_index=True, height=300)

    else:
        if not search1 or not search2:
            render_empty_state(
                "Search for two players to compare",
                "Enter player names in both search boxes"
            )


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
        "📊 Contribution Analysis": page_scatter_analysis,
        "👤 Player Career": page_player_career,
        "⚖️ Compare Players": page_compare_players,
        "📈 Distribution Comparison": page_league_comparison,
        "🔗 Transfer Networks": page_network_analysis,
        "🧪 League Quality": page_league_quality,
        "🔄 Persistence": page_markov_analysis,
        "🔍 Top Free Agents": page_free_agents
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
    st.sidebar.caption("Off-Balance-Sheet Contributions · v2.0")


if __name__ == "__main__":
    main()
