class PlayerAlreadyExistsException(Exception):
    """Raised when trying to create a player that already exists"""
    pass

class PlayerNotFoundException(Exception):
    """Raised when a player is not found"""
    pass

class ValidationException(Exception):
    """Raised when validation fails"""
    pass

class AnalysisNotFoundException(Exception):
    """Raised when an analysis is not found"""
    pass

class VideoNotFoundException(Exception):
    """Raised when a video is not found"""
    pass

class AuthenticationException(Exception):
    """Raised when authentication fails"""
    pass