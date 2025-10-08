"""
Shared test fixtures and configuration
pytest will automatically load this file
"""
import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime

from app.domain.player import Player


@pytest.fixture
def sample_player():
    """Create a sample Player domain entity for testing"""
    return Player(
        id="firebase-uid-123",
        name="John Doe",
        email="john@example.com",
        role="player",
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        updated_at=datetime(2024, 1, 1, 12, 0, 0)
    )


@pytest.fixture
def sample_player_2():
    """Create a second sample Player for testing"""
    return Player(
        id="firebase-uid-456",
        name="Jane Smith",
        email="jane@example.com",
        role="player",
        created_at=datetime(2024, 1, 2, 12, 0, 0),
        updated_at=datetime(2024, 1, 2, 12, 0, 0)
    )


@pytest.fixture
def mock_player_repository():
    """Create a mock PlayerRepository"""
    mock_repo = AsyncMock()
    
    # Set default return values
    mock_repo.get_by_id.return_value = None
    mock_repo.get_by_email.return_value = None
    mock_repo.get_all.return_value = []
    
    return mock_repo