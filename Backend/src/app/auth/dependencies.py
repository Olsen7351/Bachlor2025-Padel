from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from .firebase_service import FirebaseService
from ..business.exceptions import AuthenticationException, PlayerNotFoundException
from ..data.connection import get_db_session
from ..data.repositories.player_repository import PlayerRepository
from ..domain.player import Player


security = HTTPBearer()
firebase_service = FirebaseService()


class AuthenticatedUser:
    """Represents an authenticated user from Firebase"""
    
    def __init__(self, uid: str, email: str, firebase_data: dict):
        self.uid = uid
        self.email = email
        self.firebase_data = firebase_data
        self.email_verified = firebase_data.get('email_verified', False)
        self.name = firebase_data.get('name', '')


async def verify_firebase_token(credentials: HTTPAuthorizationCredentials) -> AuthenticatedUser:
    """
    Internal helper to verify Firebase token and return AuthenticatedUser
    """
    try:
        # Verify the Firebase ID token
        decoded_token = await firebase_service.verify_token(credentials.credentials)
        
        # Create authenticated user object
        user = AuthenticatedUser(
            uid=decoded_token['uid'],
            email=decoded_token.get('email', ''),
            firebase_data=decoded_token
        )
        
        return user
        
    except AuthenticationException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_db_session)
) -> Player:
    """
    FastAPI dependency to get current authenticated user as a Player object
    
    Process:
    1. Verify Firebase token
    2. Get firebase_uid from token
    3. Look up Player in database
    4. Return Player object
    
    Usage: 
        def protected_endpoint(current_user: Player = Depends(get_current_user)):
            # current_user.id contains the firebase_uid
            # current_user has all Player fields (name, email, role, etc.)
    """
    # Step 1 & 2: Verify Firebase token and get user info
    firebase_user = await verify_firebase_token(credentials)
    
    # Step 3: Look up Player in database using firebase_uid
    player_repository = PlayerRepository(session)
    
    try:
        player = await player_repository.get_by_id(firebase_user.uid)
        
        if not player:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Player not found for Firebase UID: {firebase_user.uid}. Please register first.",
            )
        
        return player
        
    except PlayerNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player not found for Firebase UID: {firebase_user.uid}. Please register first.",
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching player: {str(e)}",
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    session: AsyncSession = Depends(get_db_session)
) -> Optional[Player]:
    """
    Optional authentication - returns None if no token provided
    Returns Player object if authenticated, None otherwise
    
    Usage: 
        def endpoint(user: Optional[Player] = Depends(get_optional_user)):
            if user:
                # User is authenticated
                player_id = user.id
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, session)
    except HTTPException:
        return None


async def get_firebase_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> AuthenticatedUser:
    """
    Get Firebase user info without database lookup
    Use this if you only need Firebase authentication without Player data
    
    Usage:
        def endpoint(firebase_user: AuthenticatedUser = Depends(get_firebase_user)):
            # firebase_user.uid, firebase_user.email, etc.
    """
    return await verify_firebase_token(credentials)


def require_role(required_role: str):
    """
    Dependency factory for role-based access control
    
    Usage:
        @router.get("/admin")
        async def admin_endpoint(
            current_user: Player = Depends(require_role("admin"))
        ):
            # Only users with role="admin" can access this
    
    Args:
        required_role: Required role (e.g., "admin", "player")
        
    Returns:
        Dependency function that checks role
    """
    async def role_checker(
        current_user: Player = Depends(get_current_user)
    ) -> Player:
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}",
            )
        return current_user
    
    return role_checker