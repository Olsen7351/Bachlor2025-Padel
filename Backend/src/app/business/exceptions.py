class BusinessLogicException(Exception):
    """Base exception for business logic errors"""
    pass

class PlayerAlreadyExistsException(BusinessLogicException):
    """Raised when trying to create a player with existing email"""
    pass

class PlayerNotFoundException(BusinessLogicException):
    """Raised when player is not found"""
    pass

class ValidationException(BusinessLogicException):
    """Raised when domain validation fails"""
    pass
