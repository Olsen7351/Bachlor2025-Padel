from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import requests

from ...data.connection import get_db_session
from ...data.repositories.player_repository import PlayerRepository
from ...business.services.player_service import PlayerService
from ...business.exceptions import PlayerAlreadyExistsException, ValidationException
from ...auth.dependencies import get_current_user, AuthenticatedUser
from ...auth.firebase_service import FirebaseService
from ..dtos.auth_dto import RegisterRequest, LoginResponse, AuthUserInfo
from ..dtos.player_dto import PlayerResponse
from firebase_admin import auth as firebase_auth
from app.config import get_settings

router = APIRouter(prefix="/auth", tags=["authentication"])

async def get_player_service(session: AsyncSession = Depends(get_db_session)) -> PlayerService:
    """Dependency injection for PlayerService"""
    player_repository = PlayerRepository(session)
    return PlayerService(player_repository)

@router.post("/register", response_model=PlayerResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    player_service: PlayerService = Depends(get_player_service)
):
    """
    Complete registration after Firebase user creation
    
    Registration Flow:
    1. Frontend: Create user in Firebase (email + password)
    2. Frontend: Get Firebase ID token
    3. Frontend: Call this endpoint with token + name
    4. Backend: Verify token (extracts email + UID)
    5. Backend: Create user in database with Firebase data + provided name
    
    Why only name is needed in request:
    - Email: Extracted from verified Firebase token (trusted source)
    - UID: Extracted from verified Firebase token (unique identifier)
    - Password: Handled by Firebase (not stored in our DB)
    - Role: Defaults to "player"
    """
    try:
        # Email and UID come from the verified Firebase token
        # We trust Firebase more than frontend input for these
        player = await player_service.create_player(
            id=current_user.uid,        # From Firebase token
            name=request.name,          # From request body
            email=current_user.email,   # From Firebase token
            role="player"               # Default role
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValidationException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Registration failed: {str(e)}")

@router.post("/login", response_model=LoginResponse)
async def login(
    current_user: AuthenticatedUser = Depends(get_current_user),
    player_service: PlayerService = Depends(get_player_service)
):
    """
    Login endpoint - verifies Firebase token and returns user info
    
    Login Flow:
    1. Frontend: User enters email + password
    2. Frontend: Authenticate with Firebase
    3. Frontend: Get Firebase ID token
    4. Frontend: Call this endpoint with token
    5. Backend: Verify token and return user data from database
    """
    try:
        # Get user from our database using Firebase UID
        player = await player_service.get_player_by_id(current_user.uid)
        
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
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User not found in database. Please complete registration first."
        )

@router.get("/me", response_model=PlayerResponse)
async def get_current_user_info(
    current_user: AuthenticatedUser = Depends(get_current_user),
    player_service: PlayerService = Depends(get_player_service)
):
    """Get current user information from database"""
    try:
        player = await player_service.get_player_by_id(current_user.uid)
        
        return PlayerResponse(
            id=player.id,
            name=player.name,
            email=player.email,
            role=player.role,
            created_at=player.created_at,
            updated_at=player.updated_at
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found"
        )

@router.get("/firebase-info", response_model=AuthUserInfo)
async def get_firebase_user_info(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Get Firebase user information (useful for debugging/testing)
    This shows what data we get from the Firebase token
    """
    return AuthUserInfo(
        uid=current_user.uid,
        email=current_user.email,
        email_verified=current_user.email_verified,
        name=current_user.name
    )

@router.post("/test/create-user-and-token") 
async def create_test_user_and_token(
    test_email: str = "test@example.com",
    test_password: str = "testpassword123",
    test_name: str = "Test User"
):
    """
    DEVELOPMENT ONLY: Create a Firebase user and return auth token
    This simulates what your frontend would do
    """
    if get_settings().environment != "development":
        raise HTTPException(status_code=403, detail="Test endpoints only available in development")
    
    try:
        # Create user in Firebase
        user_record = firebase_auth.create_user(
            email=test_email,
            password=test_password,
            display_name=test_name
        )
        
        # Create custom token for immediate login
        custom_token = firebase_auth.create_custom_token(user_record.uid)
        
        return {
            "message": "Test user created successfully",
            "firebase_uid": user_record.uid,
            "email": test_email,
            "custom_token": custom_token.decode('utf-8'),
            "instructions": [
                "1. Use the custom_token to get an ID token (see next_step_url)",
                "2. Or use the email/password with Firebase Auth REST API",
                "3. Then call /api/auth/register with the ID token to complete registration"
            ],
            "signin_url": f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={get_settings().firebase_project_id}",
            "custom_token_url": f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key={get_settings().firebase_project_id}"
        }
        
    except Exception as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=400, detail=f"User {test_email} already exists. Try a different email or delete the user first.")
        raise HTTPException(status_code=500, detail=f"Failed to create test user: {str(e)}")

@router.delete("/test/delete-user/{uid}")
async def delete_test_user(uid: str):
    """
    DEVELOPMENT ONLY: Delete a Firebase user for testing
    """
    if get_settings().environment != "development":
        raise HTTPException(status_code=403, detail="Test endpoints only available in development")
    
    try:
        firebase_auth.delete_user(uid)
        return {"message": f"User {uid} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")
    
@router.post("/test/login-existing-user")
async def login_existing_user(
    email: str = "test@example.com",
    password: str = "testpassword123"
):
    """
    DEVELOPMENT ONLY: Login with existing Firebase user and get ready-to-use ID token
    This is for testing with users that already exist in Firebase
    """
    if get_settings().environment != "development":
        raise HTTPException(status_code=403, detail="Test endpoints only available in development")
    
    try:
        settings = get_settings()
        
        # You need to add your Firebase Web API Key here or to config
        firebase_web_api_key = settings.firebase_web_api_key
        
        if not firebase_web_api_key:
            return {
                "error": "Firebase Web API Key not configured",
                "instructions": [
                    "1. Go to Firebase Console > Project Settings > General",
                    "2. Copy the Web API Key",
                    "3. Add FIREBASE_WEB_API_KEY to your .env file",
                    "4. Restart the server"
                ]
            }
        
        # Call Firebase Auth REST API to sign in with email/password
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
            
            if 'EMAIL_NOT_FOUND' in error_message:
                return {
                    "error": "User not found",
                    "suggestion": "Create user first with /test/create-user-and-token endpoint"
                }
            elif 'INVALID_PASSWORD' in error_message:
                return {
                    "error": "Invalid password",
                    "suggestion": "Check the password or create a new user"
                }
            else:
                return {
                    "error": f"Firebase login failed: {error_message}",
                    "firebase_response": error_data
                }
        
        login_data = login_response.json()
        
        # Get user info from Firebase
        firebase_uid = login_data['localId']
        
        try:
            firebase_user = firebase_auth.get_user(firebase_uid)
        except Exception as e:
            return {"error": f"Failed to get Firebase user info: {str(e)}"}
        
        return {
            "message": "🎉 Login successful! ID token ready to use",
            "user_info": {
                "firebase_uid": firebase_uid,
                "email": firebase_user.email,
                "display_name": firebase_user.display_name,
                "email_verified": firebase_user.email_verified
            },
            "id_token": login_data['idToken'],
            "refresh_token": login_data['refreshToken'],
            "expires_in_seconds": login_data.get('expiresIn', '3600'),
            "instructions": [
                "✅ Copy the id_token above",
                "✅ Use this token in Authorization header: Bearer <id_token>",
                "✅ Test /api/auth/me to verify it works",
                "✅ If user not in database yet, call /api/auth/register first"
            ]
        }
        
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

@router.delete("/test/clean-firebase-users")
async def clean_firebase_test_users():
    """
    DEVELOPMENT ONLY: Delete all test Firebase users (emails containing 'test')
    """
    if get_settings().environment != "development":
        raise HTTPException(status_code=403, detail="Test endpoints only available in development")
    
    try:
        page = firebase_auth.list_users(max_results=100)
        deleted_users = []
        
        for user in page.users:
            if user.email and 'test' in user.email.lower():
                firebase_auth.delete_user(user.uid)
                deleted_users.append({"uid": user.uid, "email": user.email})
        
        return {
            "message": f"Deleted {len(deleted_users)} test users",
            "deleted_users": deleted_users
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clean users: {str(e)}")