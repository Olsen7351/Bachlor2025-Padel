from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from .firebase_service import FirebaseService
from ..business.exceptions import AuthenticationException

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

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> AuthenticatedUser:
    """
    FastAPI dependency to get current authenticated user from Firebase token
    Usage: def protected_endpoint(current_user: AuthenticatedUser = Depends(get_current_user)):
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

async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[AuthenticatedUser]:
    """
    Optional authentication - returns None if no token provided
    Usage: def endpoint(user: Optional[AuthenticatedUser] = Depends(get_optional_user)):
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
