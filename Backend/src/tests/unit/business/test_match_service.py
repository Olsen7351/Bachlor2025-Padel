import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from app.business.services.match_service import MatchService
from app.business.exceptions import (
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException,
    InvalidSetNumberException
)
from app.domain.match import Match, MatchPlayer, SummaryMetrics
from app.domain.analysis import Analysis


# ==================== FIXTURES ====================

@pytest.fixture
def mock_match_repository():
    """Mock for IMatchRepository"""
    return AsyncMock()


@pytest.fixture
def mock_match_player_repository():
    """Mock for IMatchPlayerRepository"""
    return AsyncMock()


@pytest.fixture
def mock_summary_metrics_repository():
    """Mock for ISummaryMetricsRepository"""
    return AsyncMock()


@pytest.fixture
def mock_analysis_repository():
    """Mock for IAnalysisRepository"""
    return AsyncMock()


@pytest.fixture
def match_service(
    mock_match_repository,
    mock_match_player_repository,
    mock_summary_metrics_repository,
    mock_analysis_repository
):
    """Create MatchService with all mocked dependencies"""
    return MatchService(
        match_repository=mock_match_repository,
        match_player_repository=mock_match_player_repository,
        summary_metrics_repository=mock_summary_metrics_repository,
        analysis_repository=mock_analysis_repository
    )


@pytest.fixture
def sample_match():
    """Sample Match entity"""
    return Match(
        id=1,
        created_at=datetime(2025, 1, 15, 10, 0, 0),
        updated_at=datetime(2025, 1, 15, 10, 0, 0)
    )


@pytest.fixture
def sample_analysis():
    """Sample Analysis entity"""
    return Analysis(
        id=100,
        player_id="firebase-uid-123",  # FK to Player
        video_id=50,
        match_id=1,
        analysis_timestamp=datetime(2025, 1, 15, 10, 30, 0),
        created_at=datetime(2025, 1, 15, 10, 0, 0),
        updated_at=datetime(2025, 1, 15, 10, 0, 0)
    )


@pytest.fixture
def sample_match_players():
    """Sample MatchPlayer entities"""
    return [
        MatchPlayer(
            id=1,
            match_id=1,
            player_identifier="player_1",
            created_at=datetime(2025, 1, 15, 10, 0, 0),
            updated_at=datetime(2025, 1, 15, 10, 0, 0)
        ),
        MatchPlayer(
            id=2,
            match_id=1,
            player_identifier="player_2",
            created_at=datetime(2025, 1, 15, 10, 0, 0),
            updated_at=datetime(2025, 1, 15, 10, 0, 0)
        ),
        MatchPlayer(
            id=3,
            match_id=1,
            player_identifier="player_3",
            created_at=datetime(2025, 1, 15, 10, 0, 0),
            updated_at=datetime(2025, 1, 15, 10, 0, 0)
        ),
        MatchPlayer(
            id=4,
            match_id=1,
            player_identifier="player_4",
            created_at=datetime(2025, 1, 15, 10, 0, 0),
            updated_at=datetime(2025, 1, 15, 10, 0, 0)
        )
    ]


@pytest.fixture
def sample_summary_metrics():
    """Sample SummaryMetrics entities"""
    return [
        SummaryMetrics(
            id=1,
            match_player_id=1,
            total_hits=150,
            total_rallies=25,
            created_at=datetime(2025, 1, 15, 10, 30, 0),
            updated_at=datetime(2025, 1, 15, 10, 30, 0)
        ),
        SummaryMetrics(
            id=2,
            match_player_id=2,
            total_hits=145,
            total_rallies=25,
            created_at=datetime(2025, 1, 15, 10, 30, 0),
            updated_at=datetime(2025, 1, 15, 10, 30, 0)
        ),
        SummaryMetrics(
            id=3,
            match_player_id=3,
            total_hits=138,
            total_rallies=25,
            created_at=datetime(2025, 1, 15, 10, 30, 0),
            updated_at=datetime(2025, 1, 15, 10, 30, 0)
        ),
        SummaryMetrics(
            id=4,
            match_player_id=4,
            total_hits=142,
            total_rallies=25,
            created_at=datetime(2025, 1, 15, 10, 30, 0),
            updated_at=datetime(2025, 1, 15, 10, 30, 0)
        )
    ]


# ==================== TEST CLASSES ====================

class TestGetMatchOverview:
    """
    Test suite for MatchService.get_match_overview method
    Implements UC-04 Success Scenario S1
    """
    
    # ==================== SUCCESS SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_get_match_overview_success(
        self,
        match_service,
        mock_match_repository,
        mock_match_player_repository,
        mock_summary_metrics_repository,
        mock_analysis_repository,
        sample_match,
        sample_analysis,
        sample_match_players,
        sample_summary_metrics
    ):
        """
        UC-04 S1: Display total hit counts
        GIVEN a successfully analyzed match
        WHEN viewing match overview
        THEN system displays list of players with total hit counts
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = sample_match
        mock_analysis_repository.get_by_match_id.return_value = sample_analysis
        mock_summary_metrics_repository.get_all_by_match_id.return_value = sample_summary_metrics
        
        # Mock match player lookups
        mock_match_player_repository.get_by_id.side_effect = sample_match_players
        
        # Act
        result = await match_service.get_match_overview(match_id=1)
        
        # Assert
        assert result["match_id"] == 1
        assert result["analysis_id"] == 100
        assert len(result["player_statistics"]) == 4
        assert result["created_at"] == datetime(2025, 1, 15, 10, 0, 0)
        
        # Verify player statistics
        player_stats = result["player_statistics"]
        assert player_stats[0]["player_identifier"] == "player_1"
        assert player_stats[0]["total_hits"] == 150
        assert player_stats[1]["player_identifier"] == "player_2"
        assert player_stats[1]["total_hits"] == 145
        
        # Verify repository interactions
        mock_match_repository.get_by_id.assert_called_once_with(1)
        mock_analysis_repository.get_by_match_id.assert_called_once_with(1)
        mock_summary_metrics_repository.get_all_by_match_id.assert_called_once_with(1)
        assert mock_match_player_repository.get_by_id.call_count == 4
    
    @pytest.mark.asyncio
    async def test_get_match_overview_no_analysis(
        self,
        match_service,
        mock_match_repository,
        mock_match_player_repository,
        mock_summary_metrics_repository,
        mock_analysis_repository,
        sample_match,
        sample_match_players,
        sample_summary_metrics
    ):
        """
        GIVEN a match without analysis (analysis_id is None)
        WHEN getting match overview
        THEN should return overview with analysis_id as None
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = sample_match
        mock_analysis_repository.get_by_match_id.return_value = None
        mock_summary_metrics_repository.get_all_by_match_id.return_value = sample_summary_metrics
        mock_match_player_repository.get_by_id.side_effect = sample_match_players
        
        # Act
        result = await match_service.get_match_overview(match_id=1)
        
        # Assert
        assert result["analysis_id"] is None
        assert result["match_id"] == 1
    
    # ==================== FAILURE SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_get_match_overview_match_not_found(
        self,
        match_service,
        mock_match_repository
    ):
        """
        GIVEN match does not exist
        WHEN getting match overview
        THEN should raise MatchNotFoundException
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(MatchNotFoundException) as exc_info:
            await match_service.get_match_overview(match_id=999)
        
        assert "999" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_get_match_overview_data_unavailable(
        self,
        match_service,
        mock_match_repository,
        mock_analysis_repository,
        mock_summary_metrics_repository,
        sample_match,
        sample_analysis
    ):
        """
        UC-04 F1: Hit data is not available
        GIVEN hit identification failed during analysis
        WHEN viewing match overview
        THEN should raise DataUnavailableException
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = sample_match
        mock_analysis_repository.get_by_match_id.return_value = sample_analysis
        mock_summary_metrics_repository.get_all_by_match_id.return_value = []
        
        # Act & Assert
        with pytest.raises(DataUnavailableException) as exc_info:
            await match_service.get_match_overview(match_id=1)
        
        assert "not available" in str(exc_info.value).lower()
        assert "analysis may not have completed" in str(exc_info.value).lower()


class TestGetMatchStatisticsBySet:
    """
    Test suite for MatchService.get_match_statistics_by_set method
    Implements UC-04 Success Scenario S2
    """
    
    # ==================== SUCCESS SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_get_match_statistics_by_set_success(
        self,
        match_service,
        mock_match_repository,
        mock_match_player_repository,
        mock_summary_metrics_repository,
        mock_analysis_repository,
        sample_match,
        sample_analysis,
        sample_match_players,
        sample_summary_metrics
    ):
        """
        UC-04 S2: Filter by set
        GIVEN match overview page with multiple sets
        WHEN filtering to show only "2nd set"
        THEN hit count updates to reflect only second set
        """
        # Arrange
        set_number = 2
        mock_match_repository.get_by_id.return_value = sample_match
        mock_analysis_repository.get_by_match_id.return_value = sample_analysis
        mock_summary_metrics_repository.get_by_match_and_set.return_value = sample_summary_metrics[:2]
        mock_match_player_repository.get_by_id.side_effect = sample_match_players[:2]
        
        # Act
        result = await match_service.get_match_statistics_by_set(match_id=1, set_number=set_number)
        
        # Assert
        assert result["match_id"] == 1
        assert result["analysis_id"] == 100
        assert result["set_number"] == 2
        assert len(result["player_statistics"]) == 2
        
        # Verify repository interactions
        mock_match_repository.get_by_id.assert_called_once_with(1)
        mock_summary_metrics_repository.get_by_match_and_set.assert_called_once_with(1, 2)
    
    @pytest.mark.asyncio
    async def test_get_match_statistics_by_set_first_set(
        self,
        match_service,
        mock_match_repository,
        mock_match_player_repository,
        mock_summary_metrics_repository,
        mock_analysis_repository,
        sample_match,
        sample_analysis,
        sample_match_players,
        sample_summary_metrics
    ):
        """
        GIVEN a request for first set statistics
        WHEN set_number is 1
        THEN should return statistics for first set
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = sample_match
        mock_analysis_repository.get_by_match_id.return_value = sample_analysis
        mock_summary_metrics_repository.get_by_match_and_set.return_value = sample_summary_metrics
        mock_match_player_repository.get_by_id.side_effect = sample_match_players
        
        # Act
        result = await match_service.get_match_statistics_by_set(match_id=1, set_number=1)
        
        # Assert
        assert result["set_number"] == 1
        mock_summary_metrics_repository.get_by_match_and_set.assert_called_once_with(1, 1)
    
    # ==================== FAILURE SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_get_match_statistics_by_set_invalid_set_number_zero(
        self,
        match_service
    ):
        """
        GIVEN set number is 0
        WHEN getting match statistics by set
        THEN should raise InvalidSetNumberException
        """
        # Act & Assert
        with pytest.raises(InvalidSetNumberException) as exc_info:
            await match_service.get_match_statistics_by_set(match_id=1, set_number=0)
        
        assert "must be >= 1" in str(exc_info.value)
        assert "0" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_match_statistics_by_set_invalid_set_number_negative(
        self,
        match_service
    ):
        """
        GIVEN set number is negative
        WHEN getting match statistics by set
        THEN should raise InvalidSetNumberException
        """
        # Act & Assert
        with pytest.raises(InvalidSetNumberException) as exc_info:
            await match_service.get_match_statistics_by_set(match_id=1, set_number=-1)
        
        assert "must be >= 1" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_match_statistics_by_set_match_not_found(
        self,
        match_service,
        mock_match_repository
    ):
        """
        GIVEN match does not exist
        WHEN getting statistics by set
        THEN should raise MatchNotFoundException
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(MatchNotFoundException):
            await match_service.get_match_statistics_by_set(match_id=999, set_number=1)
    
    @pytest.mark.asyncio
    async def test_get_match_statistics_by_set_data_unavailable(
        self,
        match_service,
        mock_match_repository,
        mock_analysis_repository,
        mock_summary_metrics_repository,
        sample_match,
        sample_analysis
    ):
        """
        UC-04 F1: Hit data not available for set
        GIVEN hit data is not available for the specified set
        WHEN viewing statistics for that set
        THEN should raise DataUnavailableException
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = sample_match
        mock_analysis_repository.get_by_match_id.return_value = sample_analysis
        mock_summary_metrics_repository.get_by_match_and_set.return_value = []
        
        # Act & Assert
        with pytest.raises(DataUnavailableException) as exc_info:
            await match_service.get_match_statistics_by_set(match_id=1, set_number=2)
        
        assert "set 2" in str(exc_info.value).lower()
        assert "not available" in str(exc_info.value).lower()


class TestGetHitComparisonChartData:
    """
    Test suite for MatchService.get_hit_comparison_chart_data method
    Implements UC-04 Success Scenario S3
    """
    
    # ==================== SUCCESS SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_get_hit_comparison_chart_data_success(
        self,
        match_service,
        mock_match_repository,
        mock_match_player_repository,
        mock_summary_metrics_repository,
        sample_match,
        sample_match_players,
        sample_summary_metrics
    ):
        """
        UC-04 S3: Visual comparison
        GIVEN match overview page
        WHEN viewing player comparisons
        THEN system displays bar chart comparing hit counts for all players
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = sample_match
        mock_summary_metrics_repository.get_all_by_match_id.return_value = sample_summary_metrics
        mock_match_player_repository.get_by_id.side_effect = sample_match_players
        
        # Act
        result = await match_service.get_hit_comparison_chart_data(match_id=1)
        
        # Assert
        assert result["chart_type"] == "bar"
        assert len(result["players"]) == 4
        
        # Verify data is sorted by hit count (descending)
        hit_counts = [p["total_hits"] for p in result["players"]]
        assert hit_counts == sorted(hit_counts, reverse=True)
        
        # Verify first player has highest hit count
        assert result["players"][0]["player_identifier"] == "player_1"
        assert result["players"][0]["total_hits"] == 150
    
    @pytest.mark.asyncio
    async def test_get_hit_comparison_chart_data_sorting(
        self,
        match_service,
        mock_match_repository,
        mock_match_player_repository,
        mock_summary_metrics_repository,
        sample_match
    ):
        """
        GIVEN players with different hit counts
        WHEN generating chart data
        THEN players should be sorted by hit count descending
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = sample_match
        
        # Create metrics in non-sorted order
        unsorted_metrics = [
            SummaryMetrics(id=1, match_player_id=1, total_hits=100, total_rallies=20),
            SummaryMetrics(id=2, match_player_id=2, total_hits=200, total_rallies=20),
            SummaryMetrics(id=3, match_player_id=3, total_hits=150, total_rallies=20),
        ]
        mock_summary_metrics_repository.get_all_by_match_id.return_value = unsorted_metrics
        
        match_players = [
            MatchPlayer(id=1, match_id=1, player_identifier="player_1"),
            MatchPlayer(id=2, match_id=1, player_identifier="player_2"),
            MatchPlayer(id=3, match_id=1, player_identifier="player_3"),
        ]
        mock_match_player_repository.get_by_id.side_effect = match_players
        
        # Act
        result = await match_service.get_hit_comparison_chart_data(match_id=1)
        
        # Assert - verify descending order
        assert result["players"][0]["total_hits"] == 200  # player_2
        assert result["players"][1]["total_hits"] == 150  # player_3
        assert result["players"][2]["total_hits"] == 100  # player_1
    
    # ==================== FAILURE SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_get_hit_comparison_chart_data_match_not_found(
        self,
        match_service,
        mock_match_repository
    ):
        """
        GIVEN match does not exist
        WHEN generating chart data
        THEN should raise MatchNotFoundException
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(MatchNotFoundException):
            await match_service.get_hit_comparison_chart_data(match_id=999)
    
    @pytest.mark.asyncio
    async def test_get_hit_comparison_chart_data_unavailable(
        self,
        match_service,
        mock_match_repository,
        mock_summary_metrics_repository,
        sample_match
    ):
        """
        UC-04 F1: Cannot generate chart without data
        GIVEN hit data is not available
        WHEN attempting to generate comparison chart
        THEN should raise DataUnavailableException
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = sample_match
        mock_summary_metrics_repository.get_all_by_match_id.return_value = []
        
        # Act & Assert
        with pytest.raises(DataUnavailableException) as exc_info:
            await match_service.get_hit_comparison_chart_data(match_id=1)
        
        assert "cannot generate comparison chart" in str(exc_info.value).lower()


class TestGetPlayerHitCount:
    """Test suite for MatchService.get_player_hit_count method"""
    
    # ==================== SUCCESS SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_get_player_hit_count_success(
        self,
        match_service,
        mock_match_repository,
        mock_match_player_repository,
        mock_summary_metrics_repository,
        sample_match
    ):
        """
        GIVEN a player exists in a match with hit data
        WHEN getting player hit count
        THEN should return the total hit count
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = sample_match
        match_player = MatchPlayer(id=1, match_id=1, player_identifier="player_1")
        mock_match_player_repository.get_by_identifier.return_value = match_player
        metrics = SummaryMetrics(id=1, match_player_id=1, total_hits=150, total_rallies=25)
        mock_summary_metrics_repository.get_by_match_player_id.return_value = metrics
        
        # Act
        result = await match_service.get_player_hit_count(match_id=1, player_identifier="player_1")
        
        # Assert
        assert result == 150
        mock_match_player_repository.get_by_identifier.assert_called_once_with(1, "player_1")
        mock_summary_metrics_repository.get_by_match_player_id.assert_called_once_with(1)
    
    # ==================== FAILURE SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_get_player_hit_count_match_not_found(
        self,
        match_service,
        mock_match_repository
    ):
        """
        GIVEN match does not exist
        WHEN getting player hit count
        THEN should raise MatchNotFoundException
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(MatchNotFoundException):
            await match_service.get_player_hit_count(match_id=999, player_identifier="player_1")
    
    @pytest.mark.asyncio
    async def test_get_player_hit_count_player_not_found(
        self,
        match_service,
        mock_match_repository,
        mock_match_player_repository,
        sample_match
    ):
        """
        GIVEN player does not exist in match
        WHEN getting player hit count
        THEN should raise PlayerInMatchNotFoundException
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = sample_match
        mock_match_player_repository.get_by_identifier.return_value = None
        
        # Act & Assert
        with pytest.raises(PlayerInMatchNotFoundException) as exc_info:
            await match_service.get_player_hit_count(match_id=1, player_identifier="unknown_player")
        
        assert "unknown_player" in str(exc_info.value)
        assert "not found in match" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_get_player_hit_count_data_unavailable(
        self,
        match_service,
        mock_match_repository,
        mock_match_player_repository,
        mock_summary_metrics_repository,
        sample_match
    ):
        """
        GIVEN hit data is not available for player
        WHEN getting player hit count
        THEN should raise DataUnavailableException
        """
        # Arrange
        mock_match_repository.get_by_id.return_value = sample_match
        match_player = MatchPlayer(id=1, match_id=1, player_identifier="player_1")
        mock_match_player_repository.get_by_identifier.return_value = match_player
        mock_summary_metrics_repository.get_by_match_player_id.return_value = None
        
        # Act & Assert
        with pytest.raises(DataUnavailableException) as exc_info:
            await match_service.get_player_hit_count(match_id=1, player_identifier="player_1")
        
        assert "player_1" in str(exc_info.value)
        assert "not available" in str(exc_info.value).lower()