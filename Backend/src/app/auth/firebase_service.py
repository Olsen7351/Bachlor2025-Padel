import firebase_admin
from firebase_admin import credentials, auth
from typing import Optional, Dict, Any
import os
from ..business.exceptions import AuthenticationException
from ..config import get_settings, get_firebase_config

class FirebaseService:
    """Service for Firebase authentication operations"""
    
    def __init__(self):
        if not firebase_admin._apps:
            self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK using config settings"""
        try:
            settings = get_settings()
            
            # Check if Firebase is configured
            if not settings.validate_firebase_config():
                raise ValueError(
                    "Firebase configuration not found. Please set Firebase environment variables:\n"
                    "- FIREBASE_PROJECT_ID\n"
                    "- FIREBASE_PRIVATE_KEY_ID\n" 
                    "- FIREBASE_PRIVATE_KEY\n"
                    "- FIREBASE_CLIENT_EMAIL\n"
                    "- FIREBASE_CLIENT_ID\n"
                    "Or provide FIREBASE_SERVICE_ACCOUNT_PATH"
                )
            
            # Get Firebase config
            firebase_config = get_firebase_config()
            
            # Remove None values
            service_account_info = {k: v for k, v in firebase_config.items() if v is not None}
            
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
            print(f"✅ Firebase initialized with environment variables for project: {settings.firebase_project_id}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Firebase: {str(e)}")
            raise Exception(f"Failed to initialize Firebase: {str(e)}")
    
    async def verify_token(self, id_token: str) -> Dict[str, Any]:
        """Verify Firebase ID token and return decoded token"""
        try:
            # Verify the ID token while checking if the token is revoked
            decoded_token = auth.verify_id_token(id_token, check_revoked=True)
            return decoded_token
        except auth.InvalidIdTokenError:
            raise AuthenticationException("Invalid authentication token")
        except auth.ExpiredIdTokenError:
            raise AuthenticationException("Authentication token has expired")
        except auth.RevokedIdTokenError:
            raise AuthenticationException("Authentication token has been revoked")
        except Exception as e:
            raise AuthenticationException(f"Authentication failed: {str(e)}")
    
    async def get_user_by_uid(self, uid: str) -> Dict[str, Any]:
        """Get Firebase user by UID"""
        try:
            user_record = auth.get_user(uid)
            return {
                'uid': user_record.uid,
                'email': user_record.email,
                'email_verified': user_record.email_verified,
                'display_name': user_record.display_name,
                'disabled': user_record.disabled,
                'creation_time': user_record.user_metadata.creation_timestamp,
                'last_sign_in': user_record.user_metadata.last_sign_in_timestamp,
            }
        except auth.UserNotFoundError:
            raise AuthenticationException("User not found")
        except Exception as e:
            raise AuthenticationException(f"Failed to get user: {str(e)}")
    
    async def create_custom_token(self, uid: str, additional_claims: Optional[Dict] = None) -> str:
        """Create a custom token for a user (useful for testing)"""
        try:
            return auth.create_custom_token(uid, additional_claims)
        except Exception as e:
            raise AuthenticationException(f"Failed to create custom token: {str(e)}")