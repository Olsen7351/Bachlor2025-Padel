from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class HeatmapDataDto(BaseModel):
    """DTO for individual player heatmap data"""
    player_identifier: str = Field(..., description="Player identifier (e.g., 'player_1')")
    heatmap_base64: str = Field(..., description="Base64 encoded PNG image")
    content_type: str = Field(default="image/png", description="MIME type of the image")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "player_identifier": "player_1",
                "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "content_type": "image/png"
            }
        }
    )


class PlayerHeatmapDto(BaseModel):
    """
    UC-02 S1: Response for single player heatmap visualization.
    
    Frontend can use heatmap_2d directly in img tag:
    <img src={`data:${content_type};base64,${heatmap_2d}`} />
    """
    match_id: int = Field(..., description="Match ID")
    player_identifier: str = Field(..., description="Player identifier")
    heatmap_2d: Optional[str] = Field(None, description="Base64 encoded 2D heatmap PNG")
    # heatmap_3d: Optional[str] = Field(None, description="Base64 encoded 3D visualization - TODO")
    content_type: str = Field(default="image/png", description="MIME type")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "match_id": 1,
                "player_identifier": "player_1",
                "heatmap_2d": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "content_type": "image/png"
            }
        }
    )


class HeatmapComparisonDto(BaseModel):
    """
    UC-02 S2: Response for comparing multiple player heatmaps.
    """
    match_id: int = Field(..., description="Match ID")
    heatmaps: List[HeatmapDataDto] = Field(..., description="List of player heatmaps")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "match_id": 1,
                "heatmaps": [
                    {
                        "player_identifier": "player_1",
                        "heatmap_base64": "base64data...",
                        "content_type": "image/png"
                    },
                    {
                        "player_identifier": "player_2",
                        "heatmap_base64": "base64data...",
                        "content_type": "image/png"
                    }
                ]
            }
        }
    )


class AvailableHeatmapsDto(BaseModel):
    """Response for listing which players have heatmap data"""
    match_id: int
    available_players: List[str] = Field(..., description="Players with heatmap data available")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "match_id": 1,
                "available_players": ["player_1", "player_2"]
            }
        }
    )


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