from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class HeatmapErrorResponse(BaseModel):
    """Error response model for heatmap endpoints"""
    error: str
    detail: Optional[str] = None


class HeatmapUnavailableResponse(BaseModel):
    """UC-02 F1/F2: Response when heatmap data is not available"""
    status: str = Field(default="heatmap_unavailable")
    message: str
    reason: str
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "heatmap_unavailable",
                "message": "Heatmap not available for player 'player_3'",
                "reason": "The AI could not track this player sufficiently"
            }
        }
    )