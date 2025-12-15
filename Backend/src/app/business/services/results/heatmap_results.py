from dataclasses import dataclass
from typing import Optional, List

@dataclass
class PlayerHeatmapResult:
    """Result type for single player heatmap"""
    match_id: int
    player_identifier: str
    heatmap_2d: Optional[str]
    content_type: str = "image/png"


@dataclass
class HeatmapDataResult:
    """Result type for heatmap comparison data"""
    player_identifier: str
    heatmap_base64: str
    content_type: str = "image/png"


@dataclass
class HeatmapComparisonResult:
    """Result type for heatmap comparison"""
    match_id: int
    heatmaps: List[HeatmapDataResult]