from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import requests

from ...data.connection import get_db_session
from ...data.repositories.interfaces import IPlayerRepository
from ...data.repositories.player_repository import PlayerRepository
from ...business.services.interfaces import IPlayerService
from ...business.services.player_service import PlayerService
from ...business.exceptions import PlayerAlreadyExistsException, ValidationException, PlayerNotFoundException
from ...auth.dependencies import get_current_user, get_firebase_user, AuthenticatedUser
from ..dtos.auth_dto import RegisterRequest, LoginResponse
from ..dtos.player_dto import PlayerResponse
from firebase_admin import auth as firebase_auth
from app.config import get_settings

router = APIRouter(prefix="/auth", tags=["authentication"])

async def get_player_service(
    session: AsyncSession = Depends(get_db_session)
) -> IPlayerService:  # Return interface type
    """
    Dependency injection - returns IPlayerService interface
    Controller depends on abstraction, not concrete class
    """
    player_repository: IPlayerRepository = PlayerRepository(session)
    return PlayerService(player_repository)

@router.post("/register", response_model=PlayerResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    firebase_user: AuthenticatedUser = Depends(get_firebase_user),  # ← CHANGED: Use get_firebase_user
    player_service: IPlayerService = Depends(get_player_service)
):
    """
    Complete registration after Firebase user creation
    
    Implements UC-09: Player Registration

    Registration Flow:
    1. Frontend: Create user in Firebase (email + password)
    2. Frontend: Get Firebase ID token
    3. Frontend: Call this endpoint with token + name
    4. Backend: Verify token (extracts email + UID) - NO DATABASE LOOKUP
    5. Backend: Create user in database with Firebase data + provided name
    
    Validation (UC-09 Failure Scenarios):
    - F4: Name cannot be empty or whitespace only (validated in DTO)
    - F5: Name max 100 characters (validated in DTO)
    - F6: Player cannot already exist in database
    - F7: Token must be valid
    """
    try:
        # Email and UID come from the verified Firebase token
        # We trust Firebase more than frontend input for these
        player = await player_service.create_player(
            id=firebase_user.uid,        # From Firebase token
            name=request.name,           # From request body
            email=firebase_user.email,   # From Firebase token
            role="player"                # Default role
        )
        
        return PlayerResponse(
            id=player.id,
            name=player.name,
            email=player.email,
            role=player.role,
            created_at=player.created_at,
            updated_at=player.updated_at
        )
        
    except PlayerAlreadyExistsException as e:
        # F6: Player already exists in database
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Profile already exists. Please use login instead."
        )
    except ValidationException as e:
        # F4, F5: Validation errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again later."
        )
    
@router.post("/login", response_model=LoginResponse)
async def login(
    firebase_user: AuthenticatedUser = Depends(get_firebase_user),  # ← CHANGED: Use get_firebase_user
    player_service: IPlayerService = Depends(get_player_service)
):
    """Login endpoint - verifies Firebase token and returns user info
    
    Implements UC-00: Player Login

    Login Flow:
    1. Frontend: User enters email + password
    2. Frontend: Authenticate with Firebase (signInWithEmailAndPassword)
    3. Frontend: Get Firebase ID token
    4. Frontend: Call this endpoint with token
    5. Backend: Verify token (handled by get_firebase_user dependency)
    6. Backend: Return user data from database

    Failure Scenarios (UC-00):
    - F1: Wrong password - handled by Firebase
    - F2: User not in Firebase - handled by Firebase
    - F3: User in Firebase but not in the backend DB - handled here (404)
    - F4: Invalid/expired token - handled by get_firebase_user (401)
    """
    try:
        # Get user from database using Firebase UID
        player = await player_service.get_player_by_id(firebase_user.uid)

        return LoginResponse(
            message="Login successful",
            user=PlayerResponse(
                id=player.id,
                name=player.name,
                email=player.email,
                role=player.role,
                created_at=player.created_at,
                updated_at=player.updated_at
            )
        )
    
    except PlayerNotFoundException:
        # UC-00 F3: User exists in Firebase but not in backend DB
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found. Please complete registration first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed. Please try again later."
        )
    
@router.get("/me", response_model=PlayerResponse)
async def get_current_user_info(
    firebase_user: AuthenticatedUser = Depends(get_firebase_user),  # ← CHANGED: Use get_firebase_user
    player_service: IPlayerService = Depends(get_player_service)
):
    """
    Get current authenticated user's profile

    Used for:
    - Checking auth state after page refresh
    - Getting current user info anywhere in the app
    - Verifying token is still valid

    Returns:
    - 200: User profile data
    - 401: Invalid/expired token (handled by get_firebase_user)
    - 404: User not found in database
    """
    try:
        player = await player_service.get_player_by_id(firebase_user.uid)

        return PlayerResponse(
            id=player.id,
            name=player.name,
            email=player.email,
            role=player.role,
            created_at=player.created_at,
            updated_at=player.updated_at
        )
    
    except PlayerNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found. Please complete registration first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile."
        )

#region Helper endpoints (Firebase) for testing player registration

@router.post("/test/create-user-and-token") 
async def create_test_user_and_token(
    test_email: str = "test@example.com",
    test_password: str = "testpassword123",
    test_name: str = "Test User"
):
    """DEVELOPMENT ONLY: Create a Firebase user and return auth token"""
    if get_settings().environment != "development":
        raise HTTPException(status_code=403, detail="Test endpoints only available in development")
    
    try:
        user_record = firebase_auth.create_user(
            email=test_email,
            password=test_password,
            display_name=test_name
        )
        
        custom_token = firebase_auth.create_custom_token(user_record.uid)
        
        return {
            "message": "Test user created successfully",
            "firebase_uid": user_record.uid,
            "email": test_email,
            "custom_token": custom_token.decode('utf-8'),
        }
        
    except Exception as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=400, detail=f"User {test_email} already exists. Try a different email or delete the user first.")
        raise HTTPException(status_code=500, detail=f"Failed to create test user: {str(e)}")

@router.post("/test/login-existing-user")
async def login_existing_user(
    email: str = "test@example.com",
    password: str = "testpassword123"
):
    """DEVELOPMENT ONLY: Login with existing Firebase user and get ID token"""
    if get_settings().environment != "development":
        raise HTTPException(status_code=403, detail="Test endpoints only available in development")
    
    try:
        settings = get_settings()
        firebase_web_api_key = settings.firebase_web_api_key
        
        if not firebase_web_api_key:
            raise HTTPException(
                status_code=500, 
                detail="Firebase Web API Key not configured. Add FIREBASE_WEB_API_KEY to .env"
            )
        
        login_response = requests.post(
            f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={firebase_web_api_key}",
            json={
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
        )
        
        if login_response.status_code != 200:
            error_data = login_response.json()
            error_message = error_data.get('error', {}).get('message', 'Unknown error')
            raise HTTPException(status_code=400, detail=f"Firebase login failed: {error_message}")
        
        login_data = login_response.json()
        firebase_uid = login_data['localId']
        firebase_user = firebase_auth.get_user(firebase_uid)
        
        return {
            "message": "Login successful! Copy the id_token below",
            "user_info": {
                "firebase_uid": firebase_uid,
                "email": firebase_user.email,
                "display_name": firebase_user.display_name,
            },
            "id_token": login_data['idToken'],
            "instructions": "Copy id_token and use it in Authorize button as: Bearer <id_token>"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")
    
@router.get("/test/list-firebase-users")
async def list_firebase_users(max_results: int = 10):
    """
    DEVELOPMENT ONLY: List Firebase users for testing
    """
    if get_settings().environment != "development":
        raise HTTPException(status_code=403, detail="Test endpoints only available in development")
    
    try:
        # List users from Firebase
        page = firebase_auth.list_users(max_results=max_results)
        
        users = []
        for user in page.users:
            users.append({
                "uid": user.uid,
                "email": user.email,
                "display_name": user.display_name,
                "email_verified": user.email_verified,
                "disabled": user.disabled,
                "creation_time": user.user_metadata.creation_timestamp,
                "last_sign_in": user.user_metadata.last_sign_in_timestamp
            })
        
        return {
            "message": f"Found {len(users)} Firebase users",
            "users": users,
            "note": "Use email/password from these users with /test/login-existing-user endpoint"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list users: {str(e)}")

@router.delete("/test/delete-user/{uid}")
async def delete_test_user(uid: str):
    """DEVELOPMENT ONLY: Delete a Firebase user"""
    if get_settings().environment != "development":
        raise HTTPException(status_code=403, detail="Test endpoints only available in development")
    
    try:
        firebase_auth.delete_user(uid)
        return {"message": f"User {uid} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")
    
#endregion