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

# Authentication exceptions
class AuthenticationException(Exception):
    """Raised when authentication fails"""
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

class AnalysisNotCompleteException(Exception):
    """Raised when trying to access data from incomplete analysis"""
    pass

# Heatmap exceptions
class HeatmapNotFoundException(Exception):
    """Raised when heatmap data is not found - maps to HTTP 404"""
    pass

class InsufficientPositionDataException(Exception):
    """
    UC-02 F2: Raised when AI couldn't track a player sufficiently.
    The position data is too sparse to generate a meaningful heatmap.
    """
    pass

# Rally exceptions
class RallyDataUnavailableException(Exception):
    """
    UC-08 F1: Raised when rally data could not be generated.
    The AI could not distinguish individual rallies from each other.
    Maps to HTTP 503 Service Unavailable.
    """
    pass