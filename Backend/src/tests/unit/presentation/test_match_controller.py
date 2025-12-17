import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import HTTPException
from datetime import datetime

from app.domain.player import Player
from app.business.exceptions import (
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException
)


class TestMatchController:
    """
    Test cases for Match Controller endpoints (Presentation Layer)
    Tests UC-04: Opgørelse af samlet antal slag
    """
    
    @pytest.fixture
    def mock_player(self):
        """Mock authenticated player"""
        return Player(
            id="test-player-123",
            name="Test Player",
            email="test@example.com",
            role="player"
        )
    
    @pytest.fixture
    def mock_match_service(self):
        """Mock match service"""
        service = Mock()
        service.get_match_overview = AsyncMock()
        service.get_player_hit_count = AsyncMock()
        return service
    
    @pytest.fixture
    def mock_match_overview_data(self):
        """Mock data returned by service for match overview"""
        return {
            "match_id": 1,
            "analysis_id": 123,
            "player_statistics": [
                {
                    "player_identifier": "player_1", 
                    "total_hits": 10,
                    "overhead_hits": 2,
                    "lob": 1,
                    "serve": 3,
                    "groundstrokes": 4
                },
                {
                    "player_identifier": "player_2", 
                    "total_hits": 5,
                    "overhead_hits": 0,
                    "lob": 0,
                    "serve": 2,
                    "groundstrokes": 3
                }
            ],
            "created_at": datetime(2024, 10, 29, 15, 0, 0)
        }
    
    # ===================================================================
    # UC-04 Success Scenario S1: Display match overview with details
    # ===================================================================
    
    @pytest.mark.asyncio
    async def test_get_match_overview_success(
        self,
        mock_player,
        mock_match_service,
        mock_match_overview_data
    ):
        """
        UC-04 S1: Get match overview with player statistics and details
        """
        # Arrange
        match_id = 1
        mock_match_service.get_match_overview.return_value = mock_match_overview_data
        
        from app.presentation.controllers.match_controller import get_match_overview
        
        # Act
        response = await get_match_overview(
            match_id=match_id,
            current_user=mock_player,
            match_service=mock_match_service
        )
        
        # Assert - Verify DTO mapping
        assert response.match_id == 1
        assert response.analysis_id == 123
        assert len(response.player_statistics) == 2
        
        # Verify details for player 1
        p1 = response.player_statistics[0]
        assert p1.player_identifier == "player_1"
        assert p1.total_hits == 10
        assert p1.overhead_hits == 2
        assert p1.groundstrokes == 4
        
        # Verify service was called correctly
        mock_match_service.get_match_overview.assert_called_once_with(match_id)
    
    @pytest.mark.asyncio
    async def test_get_match_overview_analysis_id_optional(
        self,
        mock_player,
        mock_match_service
    ):
        """
        Test that analysis_id can be None
        """
        # Arrange
        mock_match_service.get_match_overview.return_value = {
            "match_id": 1,
            "analysis_id": None,
            "player_statistics": [],
            "created_at": datetime.now()
        }
        
        from app.presentation.controllers.match_controller import get_match_overview
        
        # Act
        response = await get_match_overview(
            match_id=1,
            current_user=mock_player,
            match_service=mock_match_service
        )
        
        # Assert
        assert response.analysis_id is None
        assert response.match_id == 1
    
    @pytest.mark.asyncio
    async def test_get_match_overview_match_not_found(
        self,
        mock_player,
        mock_match_service
    ):
        """Test error handling when match doesn't exist"""
        mock_match_service.get_match_overview.side_effect = MatchNotFoundException("Not found")
        
        from app.presentation.controllers.match_controller import get_match_overview
        
        with pytest.raises(HTTPException) as exc_info:
            await get_match_overview(999, mock_player, mock_match_service)
        
        assert exc_info.value.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_match_overview_data_unavailable(
        self,
        mock_player,
        mock_match_service
    ):
        """Test UC-04 F1: Hit data not available"""
        mock_match_service.get_match_overview.side_effect = DataUnavailableException("Failed")
        
        from app.presentation.controllers.match_controller import get_match_overview
        
        with pytest.raises(HTTPException) as exc_info:
            await get_match_overview(1, mock_player, mock_match_service)
        
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["status"] == "data_unavailable"
