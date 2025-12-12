from dataclasses import dataclass
from datetime import datetime
from typing import Optional

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
    player_identifier: str  # "player_1", "player_2", etc.
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class SummaryMetrics:
    """
    Domain model for SummaryMetrics entity - aggregated stats per player.
    Contains FKs to related entities for back-mapping.
    """
    id: Optional[int]
    match_player_id: int
    total_hits: int
    total_rallies: int
    hits_id: Optional[int] = None  # FK to Hits
    heatmap_id: Optional[int] = None  # FK to Heatmap
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class Hits:
    """
    Domain model for Hits entity - breakdown of hit types per player.
    
    Note: Only populated for player_1 and player_2 (the players on the 
    analyzed side of the court). player_3 and player_4 won't have hit data.
    """
    id: Optional[int]
    overhead_hits: int = 0
    lob: int = 0
    serve: int = 0
    backhand: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class Rally:
    """
    Domain model for Rally entity - individual rally information.
    
    Note: Rallies are detected for the entire match (both sides of court).
    """
    id: Optional[int]
    summary_metrics_id: int
    duration: float  # Duration in seconds
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class Heatmap:
    """
    Domain model for Heatmap entity - player position heatmap as binary image.
    
    Note: Only generated for player_1 and player_2.
    Stores the heatmap as a PNG binary blob.
    """
    id: Optional[int]
    heatmap: bytes  # PNG binary data
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
