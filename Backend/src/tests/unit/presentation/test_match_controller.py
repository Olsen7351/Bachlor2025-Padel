import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import HTTPException
from datetime import datetime

from app.domain.player import Player
from app.business.exceptions import (
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException,
    InvalidSetNumberException
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
        service.get_match_statistics_by_set = AsyncMock()
        service.get_hit_comparison_chart_data = AsyncMock()
        service.get_player_hit_count = AsyncMock()
        return service
    
    @pytest.fixture
    def mock_match_overview_data(self):
        """Mock data returned by service for match overview"""
        return {
            "match_id": 1,
            "analysis_id": 123,
            "player_statistics": [
                {"player_identifier": "player_1", "total_hits": 245},
                {"player_identifier": "player_2", "total_hits": 238},
                {"player_identifier": "player_3", "total_hits": 221},
                {"player_identifier": "player_4", "total_hits": 215}
            ],
            "created_at": datetime(2024, 10, 29, 15, 0, 0)
        }
    
    @pytest.fixture
    def mock_chart_data(self):
        """Mock data for hit comparison chart"""
        return {
            "chart_type": "bar",
            "players": [
                {"player_identifier": "player_1", "total_hits": 245},
                {"player_identifier": "player_2", "total_hits": 238},
                {"player_identifier": "player_3", "total_hits": 221},
                {"player_identifier": "player_4", "total_hits": 215}
            ]
        }
    
    # ===================================================================
    # UC-04 Success Scenario S1: Display match overview with hit counts
    # ===================================================================
    
    @pytest.mark.asyncio
    async def test_get_match_overview_success(
        self,
        mock_player,
        mock_match_service,
        mock_match_overview_data
    ):
        """
        UC-04 S1: Get match overview with player statistics
        GIVEN a match with completed analysis
        WHEN user requests match overview
        THEN system returns match data with analysis_id and player hit counts
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
        assert response.analysis_id == 123  # ✅ Critical: analysis_id is present!
        assert len(response.player_statistics) == 4
        assert response.player_statistics[0].player_identifier == "player_1"
        assert response.player_statistics[0].total_hits == 245
        assert response.created_at == datetime(2024, 10, 29, 15, 0, 0)
        
        # Verify service was called correctly
        mock_match_service.get_match_overview.assert_called_once_with(match_id)
    
    @pytest.mark.asyncio
    async def test_get_match_overview_analysis_id_optional(
        self,
        mock_player,
        mock_match_service
    ):
        """
        Test that analysis_id can be None (Optional handling)
        GIVEN a match where analysis doesn't exist yet
        WHEN user requests match overview
        THEN system returns data with analysis_id as None
        """
        # Arrange
        mock_match_service.get_match_overview.return_value = {
            "match_id": 1,
            "analysis_id": None,  # Can be None
            "player_statistics": [
                {"player_identifier": "player_1", "total_hits": 245}
            ],
            "created_at": datetime(2024, 10, 29, 15, 0, 0)
        }
        
        from app.presentation.controllers.match_controller import get_match_overview
        
        # Act
        response = await get_match_overview(
            match_id=1,
            current_user=mock_player,
            match_service=mock_match_service
        )
        
        # Assert
        assert response.analysis_id is None  # Should handle None gracefully
        assert response.match_id == 1
    
    @pytest.mark.asyncio
    async def test_get_match_overview_match_not_found(
        self,
        mock_player,
        mock_match_service
    ):
        """
        Test error handling when match doesn't exist
        GIVEN a non-existent match_id
        WHEN user requests match overview
        THEN system returns 404 Not Found
        """
        # Arrange
        mock_match_service.get_match_overview.side_effect = MatchNotFoundException(
            "Match with ID 999 not found"
        )
        
        from app.presentation.controllers.match_controller import get_match_overview
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_match_overview(
                match_id=999,
                current_user=mock_player,
                match_service=mock_match_service
            )
        
        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail).lower()
    
    @pytest.mark.asyncio
    async def test_get_match_overview_data_unavailable(
        self,
        mock_player,
        mock_match_service
    ):
        """
        UC-04 F1: Hit data not available
        GIVEN a match where analysis failed
        WHEN user requests match overview
        THEN system returns 503 Service Unavailable
        """
        # Arrange
        mock_match_service.get_match_overview.side_effect = DataUnavailableException(
            "Hit data is not available for this match. Analysis may have failed."
        )
        
        from app.presentation.controllers.match_controller import get_match_overview
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_match_overview(
                match_id=1,
                current_user=mock_player,
                match_service=mock_match_service
            )
        
        assert exc_info.value.status_code == 503
        detail = exc_info.value.detail
        assert detail["status"] == "data_unavailable"
        assert "not available" in detail["message"].lower()
    
    # ===================================================================
    # UC-04 Success Scenario S2: Filter statistics by set
    # ===================================================================
    
    @pytest.mark.asyncio
    async def test_get_match_statistics_by_set(
        self,
        mock_player,
        mock_match_service,
        mock_match_overview_data
    ):
        """
        UC-04 S2: Filter match statistics by set number
        GIVEN a match with multiple sets
        WHEN user filters by set_number
        THEN system returns statistics for that set only
        """
        # Arrange
        mock_match_service.get_match_statistics_by_set.return_value = {
            "match_id": 1,
            "analysis_id": 123,
            "set_number": 2,
            "player_statistics": [
                {"player_identifier": "player_1", "total_hits": 120},
                {"player_identifier": "player_2", "total_hits": 115}
            ],
            "created_at": datetime(2024, 10, 29, 15, 0, 0)
        }
        
        from app.presentation.controllers.match_controller import get_match_statistics_by_set
        
        # Act
        response = await get_match_statistics_by_set(
            match_id=1,
            set_number=2,
            current_user=mock_player,
            match_service=mock_match_service
        )
        
        # Assert
        assert response.match_id == 1
        assert len(response.player_statistics) == 2
        
        # Verify service called with set_number
        mock_match_service.get_match_statistics_by_set.assert_called_once_with(1, 2)
    
    @pytest.mark.asyncio
    async def test_get_match_statistics_without_set_filter(
        self,
        mock_player,
        mock_match_service,
        mock_match_overview_data
    ):
        """
        Test that no set_number returns overall statistics
        GIVEN a match request without set_number
        WHEN user requests statistics
        THEN system returns overall match statistics
        """
        # Arrange
        mock_match_service.get_match_overview.return_value = mock_match_overview_data
        
        from app.presentation.controllers.match_controller import get_match_statistics_by_set
        
        # Act
        response = await get_match_statistics_by_set(
            match_id=1,
            set_number=None,  # No filter
            current_user=mock_player,
            match_service=mock_match_service
        )
        
        # Assert
        assert response.match_id == 1
        # Should call overview, not set-filtered method
        mock_match_service.get_match_overview.assert_called_once_with(1)
        mock_match_service.get_match_statistics_by_set.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_match_statistics_invalid_set_number(
        self,
        mock_player,
        mock_match_service
    ):
        """
        Test validation of set_number
        GIVEN an invalid set_number (< 1)
        WHEN user requests statistics
        THEN system returns 400 Bad Request
        """
        # Arrange
        mock_match_service.get_match_statistics_by_set.side_effect = InvalidSetNumberException(
            "Set number must be >= 1, got 0"
        )
        
        from app.presentation.controllers.match_controller import get_match_statistics_by_set
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_match_statistics_by_set(
                match_id=1,
                set_number=0,  # Invalid
                current_user=mock_player,
                match_service=mock_match_service
            )
        
        assert exc_info.value.status_code == 400
    
    # ===================================================================
    # UC-04 Success Scenario S3: Visual hit comparison chart
    # ===================================================================
    
    @pytest.mark.asyncio
    async def test_get_hit_comparison_chart(
        self,
        mock_player,
        mock_match_service,
        mock_chart_data
    ):
        """
        UC-04 S3: Get data for hit comparison chart
        GIVEN a match with player statistics
        WHEN user requests chart data
        THEN system returns formatted chart data
        """
        # Arrange
        mock_match_service.get_hit_comparison_chart_data.return_value = mock_chart_data
        
        from app.presentation.controllers.match_controller import get_hit_comparison_chart
        
        # Act
        response = await get_hit_comparison_chart(
            match_id=1,
            current_user=mock_player,
            match_service=mock_match_service
        )
        
        # Assert
        assert response.chart_type == "bar"
        assert len(response.players) == 4
        assert response.players[0].player_identifier == "player_1"
        assert response.players[0].total_hits == 245
        
        # Verify players are sorted by hit count (highest first)
        assert response.players[0].total_hits >= response.players[1].total_hits
        assert response.players[1].total_hits >= response.players[2].total_hits
    
    @pytest.mark.asyncio
    async def test_get_hit_comparison_chart_no_data(
        self,
        mock_player,
        mock_match_service
    ):
        """
        Test chart endpoint when data is unavailable
        GIVEN a match where analysis failed
        WHEN user requests chart data
        THEN system returns 503 Service Unavailable
        """
        # Arrange
        mock_match_service.get_hit_comparison_chart_data.side_effect = DataUnavailableException(
            "Cannot generate chart without hit data"
        )
        
        from app.presentation.controllers.match_controller import get_hit_comparison_chart
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_hit_comparison_chart(
                match_id=1,
                current_user=mock_player,
                match_service=mock_match_service
            )
        
        assert exc_info.value.status_code == 503
    
    # ===================================================================
    # Specific Player Hit Count
    # ===================================================================
    
    @pytest.mark.asyncio
    async def test_get_player_hit_count_success(
        self,
        mock_player,
        mock_match_service
    ):
        """
        Test getting hit count for specific player
        GIVEN a match and player_identifier
        WHEN user requests hit count
        THEN system returns integer hit count
        """
        # Arrange
        mock_match_service.get_player_hit_count.return_value = 245
        
        from app.presentation.controllers.match_controller import get_player_hit_count
        
        # Act
        response = await get_player_hit_count(
            match_id=1,
            player_identifier="player_1",
            current_user=mock_player,
            match_service=mock_match_service
        )
        
        # Assert
        assert response == 245
        assert isinstance(response, int)
        
        # Verify service was called correctly
        mock_match_service.get_player_hit_count.assert_called_once_with(1, "player_1")
    
    @pytest.mark.asyncio
    async def test_get_player_hit_count_player_not_found(
        self,
        mock_player,
        mock_match_service
    ):
        """
        Test error when player doesn't exist in match
        GIVEN a non-existent player_identifier
        WHEN user requests hit count
        THEN system returns 404 Not Found
        """
        # Arrange
        mock_match_service.get_player_hit_count.side_effect = PlayerInMatchNotFoundException(
            "Player 'player_99' not found in match 1"
        )
        
        from app.presentation.controllers.match_controller import get_player_hit_count
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_player_hit_count(
                match_id=1,
                player_identifier="player_99",
                current_user=mock_player,
                match_service=mock_match_service
            )
        
        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail).lower()
    
    @pytest.mark.asyncio
    async def test_get_player_hit_count_data_unavailable(
        self,
        mock_player,
        mock_match_service
    ):
        """
        Test when player's hit data is not available
        GIVEN a player whose analysis data is missing
        WHEN user requests hit count
        THEN system returns 503 Service Unavailable
        """
        # Arrange
        mock_match_service.get_player_hit_count.side_effect = DataUnavailableException(
            "Hit data is not available for player 'player_1'"
        )
        
        from app.presentation.controllers.match_controller import get_player_hit_count
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_player_hit_count(
                match_id=1,
                player_identifier="player_1",
                current_user=mock_player,
                match_service=mock_match_service
            )
        
        assert exc_info.value.status_code == 503
