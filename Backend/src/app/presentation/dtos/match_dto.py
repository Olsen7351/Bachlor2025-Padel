from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional, List


# ============================================================================
# UC-04: Summary Metrics DTOs
# ============================================================================

class PlayerHitCountDto(BaseModel):
    """
    DTO for player hit count display
    Implements UC-04 Success Scenario S1
    """
    player_identifier: str = Field(
        ..., 
        description="Player identifier from ML model (e.g., 'player_1')"
    )
    total_hits: int = Field(..., ge=0, description="Total number of hits by this player")
    
    model_config = ConfigDict(from_attributes=True)


class MatchSummaryDto(BaseModel):
    """
    DTO for match overview with player statistics
    Implements UC-04 Success Scenarios S1, S2, S3
    
    Note: analysis_id is Optional because:
    - Analysis may not exist yet (entity creation in progress)
    - Analysis may have failed to create
    - Supports backward compatibility
    """
    match_id: int
    analysis_id: Optional[int] = Field(
        None,
        description="ID of the associated analysis (None if analysis not yet created)"
    )
    player_statistics: List[PlayerHitCountDto] = Field(
        ..., 
        description="List of players with their hit counts"
    )
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class SetFilterDto(BaseModel):
    """
    DTO for filtering statistics by set
    Implements UC-04 Success Scenario S2
    """
    set_number: Optional[int] = Field(
        None, 
        ge=1, 
        description="Filter by specific set number (1, 2, 3, etc.)"
    )


class HitComparisonChartDto(BaseModel):
    """
    DTO for visual hit comparison chart data
    Implements UC-04 Success Scenario S3
    """
    chart_type: str = Field(default="bar", description="Type of chart (bar, pie, etc.)")
    players: List[PlayerHitCountDto]
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Error Response DTOs
# ============================================================================

class MatchErrorResponse(BaseModel):
    """Error response for match-related operations"""
    error: str
    details: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class DataUnavailableResponse(BaseModel):
    """
    Response when data is not available
    Implements UC-04 Failure Scenario F1
    """
    status: str = "data_unavailable"
    message: str
    reason: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)