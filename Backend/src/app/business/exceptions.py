# Player exceptions
class PlayerAlreadyExistsException(Exception):
    """Raised when trying to create a player that already exists"""
    pass


class PlayerNotFoundException(Exception):
    """Raised when a player is not found"""
    pass


# Video exceptions
class VideoNotFoundException(Exception):
    """Raised when a video is not found"""
    pass


class InvalidFileFormatException(Exception):
    """Raised when uploaded file format is not supported"""
    pass


class FileTooLargeException(Exception):
    """Raised when uploaded file exceeds size limit"""
    pass


class StorageException(Exception):
    """Raised when file storage operation fails"""
    pass


class AnalysisException(Exception):
    """Raised when video analysis fails"""
    pass


# Analysis exceptions
class AnalysisNotFoundException(Exception):
    """Raised when an analysis is not found"""
    pass


# Authentication exceptions
class AuthenticationException(Exception):
    """Raised when authentication fails"""
    pass


class UnauthorizedAccessException(Exception):
    """Raised when user tries to access resource they don't own"""
    pass


# Validation exceptions
class ValidationException(Exception):
    """Raised when validation fails"""
    pass


# Match exceptions
class MatchNotFoundException(Exception):
    """Raised when a match is not found"""
    pass


class PlayerInMatchNotFoundException(Exception):
    """Raised when a player identifier is not found in a specific match"""
    pass


class DataUnavailableException(Exception):
    """Raised when required data is not available
    Implements UC-04 F1
    """
    pass


class InvalidSetNumberException(Exception):
    """Raised when an invalid set number is provided"""
    pass


class AnalysisNotCompleteException(Exception):
    """Raised when trying to access data from incomplete analysis"""
    pass
