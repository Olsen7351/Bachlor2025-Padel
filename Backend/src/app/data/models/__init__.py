from .base import Base

# Import all models to register them with SQLAlchemy
from .player_model import PlayerModel
from .video_model import VideoModel  
from .analysis_model import AnalysisModel
from .match_model import (
    MatchModel,
    MatchPlayerModel, 
    SummaryMetricsModel,
    HitsModel,
    RallyModel,
    HeatmapModel
)

# Export for easy importing
__all__ = [
    "Base",
    "PlayerModel",
    "VideoModel", 
    "AnalysisModel",
    "MatchModel",
    "MatchPlayerModel",
    "SummaryMetricsModel", 
    "HitsModel",
    "RallyModel",
    "HeatmapModel"
]