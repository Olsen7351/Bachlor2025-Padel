from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

# Constants for tracked players (only near-side players)
TRACKED_PLAYERS = ["player_1", "player_2"]

@dataclass
class PlayerStats:
    """
    Stats for a single player from ML analysis.
    Only populated for player_1 and player_2 (near-side players).
    """
    player_identifier: str  # "player_1" or "player_2"
    total_hits: int = 0
    overhead_hits: int = 0
    lob: int = 0
    serve: int = 0
    groundstrokes: int = 0


@dataclass
class RallyData:
    """
    Individual rally from ML analysis.
    Rallies are match-level, not player-specific.
    """
    rally_id: int
    duration_seconds: float


@dataclass 
class HeatmapData:
    """
    Heatmap output from ML analysis.
    Only generated for player_1 and player_2.
    """
    player_identifier: str
    image_path: Path  # Temp file path - read as bytes for storage


@dataclass
class MLAnalysisResult:
    """Complete result from ML pipeline - only includes tracked players"""
    player_stats: Dict[str, PlayerStats]  # keyed by player_identifier (player_1, player_2)
    rallies: List[RallyData]
    heatmaps: Dict[str, HeatmapData]  # keyed by player_identifier
    total_rallies: int
    output_video_path: Optional[Path] = None