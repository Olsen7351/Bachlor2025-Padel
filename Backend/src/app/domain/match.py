from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class Match:
    """Domain model for Match entity"""
    id: Optional[int]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class MatchPlayer:
    """Domain model for MatchPlayer entity"""
    id: Optional[int]
    match_id: int
    player_identifier: str  # "player_0", "player_1", etc.
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class SummaryMetrics:
    """Domain model for SummaryMetrics entity"""
    id: Optional[int]
    match_player_id: int
    total_hits: int
    total_rallies: int
    hits_id: Optional[int] = None  # Reference to detailed hits
    rally_id: Optional[int] = None  # Reference to rally data
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class Hits:
    """Domain model for Hits entity"""
    id: Optional[int]
    summary_metrics_id: int
    hit_errors: int
    overhead_hits: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class Rally:
    """Domain model for Rally entity"""
    id: Optional[int]
    summary_metrics_id: int
    hits: int
    length_in_time: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class Heatmap:
    """Domain model for Heatmap entity"""
    id: Optional[int]
    summary_metrics_id: int
    offensive_zone_time: float
    defensive_zone_time: float
    transition_zone_time: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class HeatmapCoord:
    """Domain model for HeatmapCoord entity"""
    id: Optional[int]
    heatmap_id: int
    x_coord: float
    y_coord: float
    intensity: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None