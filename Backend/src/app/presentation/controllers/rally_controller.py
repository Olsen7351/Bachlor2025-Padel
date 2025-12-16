from fastapi import APIRouter, Depends, HTTPException, status

# Business layer
from ...business.services.interfaces import IRallyService
from ...business.exceptions import (
    MatchNotFoundException,
    DataUnavailableException,
    RallyDataUnavailableException
)
from ..dtos.rally_dto import (
    RallyAnalysisDto,
    RallyDto,
    RallyDistributionDto,
    RallyDistributionBucketDto,
    RallyErrorResponse,
    RallyDataUnavailableResponse
)

from ...domain.player import Player

# DI container
from ...dependencies import get_rally_service

from ...auth.dependencies import get_current_user



router = APIRouter(prefix="/rallies", tags=["rallies"])


@router.get(
    "/{match_id}/analysis",
    response_model=RallyAnalysisDto,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": RallyErrorResponse, "description": "Match not found"},
        503: {"model": RallyDataUnavailableResponse, "description": "Rally data not available"}
    },
    summary="Get rally analysis overview",
    description="""
    Get rally length analysis for a match.
    Implements UC-08 Success Scenario S1.
    
    **Success Scenario S1:**
    - Display rally length overview
    - Show average rally duration
    - Show total number of rallies
    - List individual rally durations
    
    **Failure Scenario F1:**
    - Returns 503 if AI couldn't distinguish individual rallies
    - Analysis may have failed to detect rally boundaries
    """
)
async def get_rally_analysis(
    match_id: int,
    current_user: Player = Depends(get_current_user),
    rally_service: IRallyService = Depends(get_rally_service)
) -> RallyAnalysisDto:
    """
    UC-08 S1: Get rally analysis for a match
    
    Returns:
    - Total rallies detected
    - Average, min, max duration
    - List of all rally durations
    """
    try:
        analysis = await rally_service.get_rally_analysis(match_id)
        
        # Map to DTO
        rallies = [
            RallyDto(
                rally_id=rally["rally_id"],
                duration=rally["duration"]
            )
            for rally in analysis["rallies"]
        ]
        
        return RallyAnalysisDto(
            match_id=analysis["match_id"],
            total_rallies=analysis["total_rallies"],
            average_duration=analysis["average_duration"],
            min_duration=analysis["min_duration"],
            max_duration=analysis["max_duration"],
            rallies=rallies
        )
        
    except MatchNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": str(e), "type": "match_not_found"}
        )
    
    except RallyDataUnavailableException as e:
        # UC-08 F1: Rally data not available
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "data_unavailable",
                "message": str(e),
                "reason": "The AI could not distinguish individual rallies from each other"
            }
        )
    
    except DataUnavailableException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "data_unavailable",
                "message": str(e),
                "reason": "Analysis data is not available for this match"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred", "details": str(e)}
        )


@router.get(
    "/{match_id}/distribution",
    response_model=RallyDistributionDto,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": RallyErrorResponse, "description": "Match not found"},
        503: {"model": RallyDataUnavailableResponse, "description": "Rally data not available"}
    },
    summary="Get rally duration distribution",
    description="""
    Get rally duration distribution for chart visualization.
    
    Groups rallies into duration buckets:
    - Short: < 5 seconds
    - Medium: 5-15 seconds
    - Long: 15-30 seconds
    - Very Long: > 30 seconds
    
    Useful for histogram or pie chart displays.
    """
)
async def get_rally_distribution(
    match_id: int,
    current_user: Player = Depends(get_current_user),
    rally_service: IRallyService = Depends(get_rally_service)
) -> RallyDistributionDto:
    """
    Get rally duration distribution for visualization
    
    Returns rally counts and percentages grouped by duration buckets.
    """
    try:
        distribution = await rally_service.get_rally_duration_distribution(match_id)
        
        # Map to DTO
        buckets = [
            RallyDistributionBucketDto(
                bucket=bucket["bucket"],
                label=bucket["label"],
                count=bucket["count"],
                percentage=bucket["percentage"]
            )
            for bucket in distribution["distribution"]
        ]
        
        return RallyDistributionDto(
            match_id=distribution["match_id"],
            total_rallies=distribution["total_rallies"],
            distribution=buckets
        )
        
    except MatchNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": str(e), "type": "match_not_found"}
        )
    
    except RallyDataUnavailableException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "data_unavailable",
                "message": str(e),
                "reason": "The AI could not distinguish individual rallies from each other"
            }
        )
    
    except DataUnavailableException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "data_unavailable",
                "message": str(e),
                "reason": "Analysis data is not available for this match"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred", "details": str(e)}
        )