from pydantic import BaseModel, Field, ConfigDict
from typing import List


class RallyDto(BaseModel):
    """Individual rally data"""
    rally_id: int = Field(..., description="Unique rally identifier")
    duration: float = Field(..., description="Rally duration in seconds")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rally_id": 1,
                "duration": 12.5
            }
        }
    )


class RallyAnalysisDto(BaseModel):
    """
    UC-08 S1: Rally analysis overview response
    
    Contains aggregate statistics about rallies in a match.
    """
    match_id: int = Field(..., description="Match identifier")
    total_rallies: int = Field(..., description="Total number of rallies detected")
    average_duration: float = Field(..., description="Average rally duration in seconds")
    min_duration: float = Field(..., description="Shortest rally duration in seconds")
    max_duration: float = Field(..., description="Longest rally duration in seconds")
    rallies: List[RallyDto] = Field(default_factory=list, description="Individual rally details")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "match_id": 1,
                "total_rallies": 45,
                "average_duration": 8.3,
                "min_duration": 2.1,
                "max_duration": 32.5,
                "rallies": [
                    {"rally_id": 1, "duration": 6.47},
                    {"rally_id": 2, "duration": 17.13}
                ]
            }
        }
    )


class RallyDistributionBucketDto(BaseModel):
    """Single bucket in rally duration distribution"""
    bucket: str = Field(..., description="Bucket identifier (short, medium, long, very_long)")
    label: str = Field(..., description="Human-readable label (e.g., '< 5s')")
    count: int = Field(..., description="Number of rallies in this bucket")
    percentage: float = Field(..., description="Percentage of total rallies")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bucket": "medium",
                "label": "5-15s",
                "count": 18,
                "percentage": 40.0
            }
        }
    )


class RallyDistributionDto(BaseModel):
    """
    Rally duration distribution for visualization.
    
    Groups rallies into buckets for histogram/chart display.
    """
    match_id: int = Field(..., description="Match identifier")
    total_rallies: int = Field(..., description="Total number of rallies")
    distribution: List[RallyDistributionBucketDto] = Field(
        ..., 
        description="Rally distribution across duration buckets"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "match_id": 1,
                "total_rallies": 45,
                "distribution": [
                    {"bucket": "short", "label": "< 5s", "count": 8, "percentage": 17.8},
                    {"bucket": "medium", "label": "5-15s", "count": 25, "percentage": 55.6},
                    {"bucket": "long", "label": "15-30s", "count": 10, "percentage": 22.2},
                    {"bucket": "very_long", "label": "> 30s", "count": 2, "percentage": 4.4}
                ]
            }
        }
    )


class RallyErrorResponse(BaseModel):
    """Error response for rally endpoints"""
    error: str = Field(..., description="Error message")
    type: str = Field(default="rally_error", description="Error type identifier")


class RallyDataUnavailableResponse(BaseModel):
    """
    UC-08 F1: Rally data unavailable response
    
    Returned when the AI couldn't distinguish individual rallies.
    """
    status: str = Field(default="data_unavailable", description="Status indicator")
    message: str = Field(..., description="Human-readable error message")
    reason: str = Field(..., description="Technical reason for data unavailability")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "data_unavailable",
                "message": "Rally data is not available for this match",
                "reason": "The AI could not distinguish individual rallies from each other"
            }
        }
    )