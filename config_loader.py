"""
Configuration loader for Soccer Analytics Pipeline.

Loads settings from config.yaml and provides typed access to configuration values.
Falls back to sensible defaults if config.yaml is missing or incomplete.

Usage:
    from config_loader import config

    api_key = config.api.key
    verbose = config.downloader.verbose
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import yaml, fall back gracefully
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class APISettings:
    """API configuration settings."""
    key: str = ""
    requests_per_minute: float = 300.0
    max_player_pages: int = 3
    use_squads_fallback: bool = True
    request_timeout: int = 45

    @property
    def rate_limit_sec(self) -> float:
        """Derived seconds between requests."""
        return 60.0 / self.requests_per_minute if self.requests_per_minute > 0 else 0.2


@dataclass
class DownloaderSettings:
    """Downloader behavior settings."""
    verbose: bool = True
    debug: bool = False
    max_api_calls_per_run: int = 1000000000
    fetch_transfers: bool = True
    transfers_only_most_recent: bool = True
    refresh_transfers: bool = False
    auto_analyze: bool = True


@dataclass
class AnalysisSettings:
    """Analysis/regression settings."""
    lasso_alphas: List[float] = field(default_factory=lambda: [0.001])
    ridge_alphas: List[float] = field(default_factory=lambda: [10.0])
    auto_select_lasso_alpha: bool = True
    select_alpha_by: str = "ic"
    info_criterion: str = "AIC"
    ic_alpha_bounds: List[float] = field(default_factory=lambda: [1e-6, 10.0])
    ic_tol: float = 0.01
    ic_max_iter: int = 60
    keep_player_after_red: bool = False
    save_intermediate: bool = False


@dataclass
class VisualizationSettings:
    """Visualization settings."""
    strategy: str = "single_season"
    country: str = "Germany"
    league: str = "Bundesliga"
    season: int = 2023
    min_fte: float = 5.0
    position_filter: Optional[str] = None
    output_dir: str = "output"
    output_prefix: str = "visualization"
    dpi: int = 150
    preferred_contrib_col: str = "contribution_ols"


@dataclass
class PathSettings:
    """File path settings."""
    data_dir: str = "data"  # Directory containing country_league folders
    download_queue: str = "download_queue.csv"
    profiles_db: str = "player_profiles.db"
    analysis_db: str = "analysis_results.db"
    log_dir: str = "logs"
    log_file: str = "football_fetcher.log"


@dataclass
class LoggingSettings:
    """Logging settings."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(message)s"


@dataclass
class Config:
    """Main configuration container."""
    api: APISettings = field(default_factory=APISettings)
    downloader: DownloaderSettings = field(default_factory=DownloaderSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        """Create Config from dictionary (parsed YAML)."""
        return cls(
            api=APISettings(**data.get("api", {})),
            downloader=DownloaderSettings(**data.get("downloader", {})),
            analysis=AnalysisSettings(**data.get("analysis", {})),
            visualization=VisualizationSettings(**data.get("visualization", {})),
            paths=PathSettings(**data.get("paths", {})),
            logging=LoggingSettings(**data.get("logging", {})),
        )


# =============================================================================
# Config Loading
# =============================================================================

def find_config_file() -> Optional[Path]:
    """
    Find config.yaml in the Soccer directory.

    Searches:
    1. Current working directory
    2. Directory containing this file
    3. Parent directories up to 3 levels
    """
    candidates = [
        Path.cwd() / "config.yaml",
        Path(__file__).parent / "config.yaml",
    ]

    # Also check parent directories
    current = Path.cwd()
    for _ in range(3):
        candidates.append(current / "config.yaml")
        candidates.append(current / "Soccer" / "config.yaml")
        current = current.parent

    for path in candidates:
        if path.exists():
            return path

    return None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Optional explicit path to config.yaml

    Returns:
        Config object with loaded or default settings
    """
    # Find config file
    if config_path:
        path = Path(config_path)
    else:
        path = find_config_file()

    # If no yaml available or no config file, return defaults
    if not YAML_AVAILABLE:
        print("Warning: PyYAML not installed. Using default configuration.")
        print("  Install with: pip install pyyaml")
        return Config()

    if path is None or not path.exists():
        print(f"Warning: config.yaml not found. Using default configuration.")
        return Config()

    # Load YAML
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return Config.from_dict(data)
    except Exception as e:
        print(f"Warning: Error loading config.yaml: {e}")
        print("  Using default configuration.")
        return Config()


def reload_config() -> Config:
    """Reload configuration from disk."""
    global config
    config = load_config()
    return config


# =============================================================================
# Global Config Instance
# =============================================================================

# Load config on import
config = load_config()

# Track the directory where config.yaml was found (for resolving relative paths)
_config_file = find_config_file()
CONFIG_DIR: Path = _config_file.parent if _config_file else Path(__file__).parent


def get_absolute_path(relative_path: str) -> Path:
    """
    Resolve a path relative to the config directory (Soccer/).

    This ensures paths work regardless of the current working directory.

    Args:
        relative_path: Path relative to Soccer/ directory

    Returns:
        Absolute Path object
    """
    return (CONFIG_DIR / relative_path).resolve()


# =============================================================================
# Convenience Functions for Migration
# =============================================================================

def get_api_key() -> str:
    """Get API key from config."""
    return config.api.key


def get_rate_limit_sec() -> float:
    """Get rate limit in seconds."""
    return config.api.rate_limit_sec


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return config.downloader.verbose


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return config.downloader.debug
