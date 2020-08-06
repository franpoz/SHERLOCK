class ObjectProcessingError(Exception):
    """Raised when object processing fails and wants the other objects to still be processed."""
    pass