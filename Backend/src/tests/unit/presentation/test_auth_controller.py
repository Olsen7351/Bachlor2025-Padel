"""
Unit tests for auth controller endpoints

Tests UC-00: Player Login (with auto-registration)
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from datetime import datetime

from app.presentation.controllers.auth_controller import login, get_current_user_info
from app.auth.dependencies import AuthenticatedUser
from app.business.exceptions import PlayerNotFoundException, ValidationException
from app.domain.player import Player


class TestLoginEndpoint:
    """Test suite for /auth/login endpoint (UC-00 with auto-registration)"""
    
    @pytest.fixture
    def authenticated_user(self):
        """Mock authenticated user from Firebase token"""
        return AuthenticatedUser(
            uid="firebase-uid-123",
            email="john@example.com",
            firebase_data={
                'uid': 'firebase-uid-123',
                'email': 'john@example.com',
                'email_verified': True,
                'name': 'John Doe'
            }
        )
    
    @pytest.fixture
    def authenticated_user_no_name(self):
        """Mock authenticated user without display name"""
        return AuthenticatedUser(
            uid="firebase-uid-456",
            email="noname@example.com",
            firebase_data={
                'uid': 'firebase-uid-456',
                'email': 'noname@example.com',
                'email_verified': True,
                'name': ''
            }
        )
    
    @pytest.fixture
    def mock_player_service(self):
        """Mock player service"""
        return AsyncMock()
    
    # ==================== SUCCESS SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_login_existing_user_success(self, authenticated_user, mock_player_service, sample_player):
        """
        UC-00 S1: Successful login with existing user
        GIVEN valid Firebase token and user exists in database
        WHEN calling login endpoint
        THEN should return user data
        """
        # Arrange
        mock_player_service.get_player_by_id.return_value = sample_player
        
        # Act
        result = await login(
            firebase_user=authenticated_user,
            player_service=mock_player_service
        )
        
        # Assert
        assert result.message == "Login successful"
        assert result.user.id == "firebase-uid-123"
        assert result.user.name == "John Doe"
        assert result.user.email == "john@example.com"
        mock_player_service.get_player_by_id.assert_called_once_with("firebase-uid-123")
        mock_player_service.create_player.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_login_new_user_auto_registration_success(self, authenticated_user, mock_player_service):
        """
        UC-00 with auto-registration: New user is automatically created
        GIVEN valid Firebase token but user NOT in database
        WHEN calling login endpoint
        THEN should create user and return user data
        """
        # Arrange
        mock_player_service.get_player_by_id.side_effect = PlayerNotFoundException("Not found")
        
        created_player = Player(
            id="firebase-uid-123",
            name="John Doe",
            email="john@example.com",
            role="player",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        mock_player_service.create_player.return_value = created_player
        
        # Act
        result = await login(
            firebase_user=authenticated_user,
            player_service=mock_player_service
        )
        
        # Assert
        assert result.message == "Login successful"
        assert result.user.id == "firebase-uid-123"
        assert result.user.name == "John Doe"
        assert result.user.email == "john@example.com"
        
        mock_player_service.get_player_by_id.assert_called_once_with("firebase-uid-123")
        mock_player_service.create_player.assert_called_once_with(
            id="firebase-uid-123",
            name="John Doe",
            email="john@example.com",
            role="player"
        )
    
    # ==================== FAILURE SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_login_auto_registration_fails_no_display_name(
        self, authenticated_user_no_name, mock_player_service
    ):
        """
        Auto-registration failure: Firebase display_name not set
        GIVEN valid Firebase token but no display_name
        WHEN calling login endpoint and user doesn't exist
        THEN should return 400 error
        """
        # Arrange
        mock_player_service.get_player_by_id.side_effect = PlayerNotFoundException("Not found")
        mock_player_service.create_player.side_effect = ValidationException(
            "Name cannot be empty or whitespace"
        )
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await login(
                firebase_user=authenticated_user_no_name,
                player_service=mock_player_service
            )
        
        assert exc_info.value.status_code == 400
        assert "display name" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_login_service_error_on_get(self, authenticated_user, mock_player_service):
        """
        GIVEN service throws unexpected error during get
        WHEN calling login endpoint
        THEN should return 500 error
        """
        # Arrange
        mock_player_service.get_player_by_id.side_effect = Exception("Database connection failed")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await login(
                firebase_user=authenticated_user,
                player_service=mock_player_service
            )
        
        assert exc_info.value.status_code == 500
        assert "failed" in exc_info.value.detail.lower()


class TestGetCurrentUserEndpoint:
    """Test suite for /auth/me endpoint"""
    
    @pytest.fixture
    def authenticated_user(self):
        """Mock authenticated user"""
        return AuthenticatedUser(
            uid="firebase-uid-123",
            email="john@example.com",
            firebase_data={'uid': 'firebase-uid-123', 'email': 'john@example.com'}
        )
    
    @pytest.fixture
    def mock_player_service(self):
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_get_me_success(self, authenticated_user, mock_player_service, sample_player):
        """
        GIVEN valid authenticated user
        WHEN calling /me endpoint
        THEN should return user profile
        """
        # Arrange
        mock_player_service.get_player_by_id.return_value = sample_player
        
        # Act
        result = await get_current_user_info(
            firebase_user=authenticated_user,
            player_service=mock_player_service
        )
        
        # Assert
        assert result.id == "firebase-uid-123"
        assert result.name == "John Doe"
        assert result.email == "john@example.com"
        mock_player_service.get_player_by_id.assert_called_once_with("firebase-uid-123")
    
    @pytest.mark.asyncio
    async def test_get_me_user_not_found(self, authenticated_user, mock_player_service):
        """
        GIVEN valid token but user not in database
        WHEN calling /me endpoint
        THEN should return 404 error
        """
        # Arrange
        mock_player_service.get_player_by_id.side_effect = PlayerNotFoundException("Not found")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_info(
                firebase_user=authenticated_user,
                player_service=mock_player_service
            )
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_get_me_service_error(self, authenticated_user, mock_player_service):
        """
        GIVEN service error
        WHEN calling /me endpoint
        THEN should return 500 error
        """
        # Arrange
        mock_player_service.get_player_by_id.side_effect = Exception("Internal error")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_info(
                firebase_user=authenticated_user,
                player_service=mock_player_service
            )
        
        assert exc_info.value.status_code == 500