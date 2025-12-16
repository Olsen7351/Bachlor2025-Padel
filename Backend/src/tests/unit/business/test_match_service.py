import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from dataclasses import dataclass

from app.business.services.match_service import MatchService
from app.business.exceptions import (
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException
)
from app.domain.match import Match, MatchPlayer, SummaryMetrics, Hits
from app.domain.analysis import Analysis


@pytest.fixture
def mock_repos():
    """Return all mock repositories"""
    return {
        "match": AsyncMock(),
        "player": AsyncMock(),
        "metrics": AsyncMock(),
        "analysis": AsyncMock(),
        "hits": AsyncMock() 
    }

@pytest.fixture
def match_service(mock_repos):
    return MatchService(
        match_repository=mock_repos["match"],
        match_player_repository=mock_repos["player"],
        summary_metrics_repository=mock_repos["metrics"],
        analysis_repository=mock_repos["analysis"],
        hits_repository=mock_repos["hits"]
    )

@pytest.fixture
def sample_data():
    """Create sample domain entities"""
    return {
        "match": Match(id=1, created_at=datetime.now(), updated_at=datetime.now()),
        "analysis": Analysis(
            id=100, player_id="uid", video_id=50, match_id=1, 
            analysis_timestamp=datetime.now(), created_at=datetime.now(), updated_at=datetime.now()
        ),
        "match_players": [
            MatchPlayer(id=1, match_id=1, player_identifier="player_1"),
            MatchPlayer(id=2, match_id=1, player_identifier="player_2")
        ],
        "metrics": [
            SummaryMetrics(id=1, match_player_id=1, total_hits=10, hits_id=500, total_rallies=5),
            SummaryMetrics(id=2, match_player_id=2, total_hits=5, hits_id=501, total_rallies=5)
        ],
        "hits": [
            Hits(id=500, overhead_hits=2, lob=1, serve=3, groundstrokes=4),
            Hits(id=501, overhead_hits=0, lob=0, serve=2, groundstrokes=3)
        ]
    }

class TestGetMatchOverview:
    
    @pytest.mark.asyncio
    async def test_get_match_overview_success(
        self,
        match_service,
        mock_repos,
        sample_data
    ):
        """
        UC-04 S1: Display total hit counts AND detailed breakdown
        """
        # Arrange
        mock_repos["match"].get_by_id.return_value = sample_data["match"]
        mock_repos["analysis"].get_by_match_id.return_value = sample_data["analysis"]
        mock_repos["metrics"].get_all_by_match_id.return_value = sample_data["metrics"]
        mock_repos["player"].get_by_id.side_effect = sample_data["match_players"]
        
        # Mock hits lookup (first call returns hits[0], second returns hits[1])
        mock_repos["hits"].get_by_id.side_effect = sample_data["hits"]
        
        # Act
        result = await match_service.get_match_overview(match_id=1)
        
        # Assert
        assert result["match_id"] == 1
        assert len(result["player_statistics"]) == 2
        
        # Verify Player 1 stats
        p1 = result["player_statistics"][0]
        assert p1["player_identifier"] == "player_1"
        assert p1["total_hits"] == 10
        assert p1["overhead_hits"] == 2
        assert p1["groundstrokes"] == 4
        
        # Verify repos called
        mock_repos["hits"].get_by_id.assert_any_call(500)
        mock_repos["hits"].get_by_id.assert_any_call(501)

    @pytest.mark.asyncio
    async def test_get_match_overview_no_hits_data(
        self,
        match_service,
        mock_repos,
        sample_data
    ):
        """
        Test when Hits object is missing (should default to 0)
        """
        # Arrange
        mock_repos["match"].get_by_id.return_value = sample_data["match"]
        mock_repos["metrics"].get_all_by_match_id.return_value = sample_data["metrics"]
        mock_repos["player"].get_by_id.side_effect = sample_data["match_players"]
        
        # Hits repo returns None
        mock_repos["hits"].get_by_id.return_value = None
        
        # Act
        result = await match_service.get_match_overview(match_id=1)
        
        # Assert
        p1 = result["player_statistics"][0]
        assert p1["total_hits"] == 10  # From metrics
        assert p1["groundstrokes"] == 0  # Defaulted
        assert p1["overhead_hits"] == 0  # Defaulted

    @pytest.mark.asyncio
    async def test_get_match_overview_match_not_found(
        self,
        match_service,
        mock_repos
    ):
        mock_repos["match"].get_by_id.return_value = None
        with pytest.raises(MatchNotFoundException):
            await match_service.get_match_overview(match_id=999)

    @pytest.mark.asyncio
    async def test_get_match_overview_data_unavailable(
        self,
        match_service,
        mock_repos,
        sample_data
    ):
        mock_repos["match"].get_by_id.return_value = sample_data["match"]
        mock_repos["metrics"].get_all_by_match_id.return_value = [] # Empty list
        
        with pytest.raises(DataUnavailableException):
            await match_service.get_match_overview(match_id=1)