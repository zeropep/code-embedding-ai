from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ErrorCode(str, Enum):
    """Standardized error codes for client handling"""

    # Client errors (4xx)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"

    # Specific errors
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"


class ErrorDetail(BaseModel):
    """Detailed error information"""
    field: Optional[str] = Field(None, description="Field name if error is field-specific")
    message: str = Field(..., description="Human-readable error message")
    code: Optional[str] = Field(None, description="Specific error code for this detail")


class ErrorResponse(BaseModel):
    """
    Standardized error response format

    Provides clear, actionable information to help users understand and resolve errors.
    """
    success: bool = Field(False, description="Always false for error responses")
    error_code: ErrorCode = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Additional error details")
    suggestion: Optional[str] = Field(None, description="Actionable suggestion for resolving the error")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    documentation_url: Optional[str] = Field(None, description="Link to relevant documentation")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "EMBEDDING_ERROR",
                "message": "Failed to generate embeddings for the provided content",
                "details": [
                    {
                        "field": "content",
                        "message": "Content exceeds maximum length of 8000 characters",
                        "code": "CONTENT_TOO_LONG"
                    }
                ],
                "suggestion": "Try splitting the content into smaller chunks or reducing the input size",
                "timestamp": "2025-12-07T10:30:00.000Z",
                "request_id": "req_123abc",
                "documentation_url": "https://docs.example.com/errors/embedding-error"
            }
        }


class ErrorResponseBuilder:
    """Helper class for building standardized error responses"""

    @staticmethod
    def validation_error(
        message: str,
        field: Optional[str] = None,
        suggestion: Optional[str] = None
    ) -> ErrorResponse:
        """Build a validation error response"""
        details = None
        if field:
            details = [ErrorDetail(field=field, message=message)]

        return ErrorResponse(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message if not field else f"Validation error: {field}",
            details=details,
            suggestion=suggestion or "Please check the request parameters and try again"
        )

    @staticmethod
    def not_found(
        resource_type: str,
        resource_id: Optional[str] = None
    ) -> ErrorResponse:
        """Build a resource not found error"""
        message = f"{resource_type} not found"
        if resource_id:
            message += f": {resource_id}"

        return ErrorResponse(
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            message=message,
            suggestion=f"Verify that the {resource_type.lower()} exists and try again"
        )

    @staticmethod
    def internal_error(
        message: str = "An internal error occurred",
        details: Optional[str] = None
    ) -> ErrorResponse:
        """Build an internal error response"""
        error_details = None
        if details:
            error_details = [ErrorDetail(message=details)]

        return ErrorResponse(
            error_code=ErrorCode.INTERNAL_ERROR,
            message=message,
            details=error_details,
            suggestion="Please try again later. If the problem persists, contact support"
        )

    @staticmethod
    def service_unavailable(
        service_name: str,
        retry_after_seconds: Optional[int] = None
    ) -> ErrorResponse:
        """Build a service unavailable error"""
        message = f"{service_name} is temporarily unavailable"
        suggestion = "Please try again in a few moments"

        if retry_after_seconds:
            suggestion = f"Please try again after {retry_after_seconds} seconds"

        return ErrorResponse(
            error_code=ErrorCode.SERVICE_UNAVAILABLE,
            message=message,
            suggestion=suggestion
        )

    @staticmethod
    def timeout_error(
        operation: str,
        timeout_seconds: Optional[float] = None
    ) -> ErrorResponse:
        """Build a timeout error response"""
        message = f"Operation timed out: {operation}"
        if timeout_seconds:
            message += f" (limit: {timeout_seconds}s)"

        return ErrorResponse(
            error_code=ErrorCode.TIMEOUT_ERROR,
            message=message,
            suggestion="The operation took too long to complete. Try reducing the request size or complexity"
        )

    @staticmethod
    def embedding_error(
        reason: str,
        recoverable: bool = False
    ) -> ErrorResponse:
        """Build an embedding generation error"""
        suggestion = "Please check the input content and try again"
        if recoverable:
            suggestion = "This is a temporary issue. Please try again in a moment"

        return ErrorResponse(
            error_code=ErrorCode.EMBEDDING_ERROR,
            message=f"Failed to generate embeddings: {reason}",
            suggestion=suggestion,
            details=[ErrorDetail(message=reason)]
        )

    @staticmethod
    def database_error(
        operation: str,
        recoverable: bool = True
    ) -> ErrorResponse:
        """Build a database error response"""
        suggestion = "This is a temporary database issue. Please try again"
        if not recoverable:
            suggestion = "A database error occurred. Please contact support if this persists"

        return ErrorResponse(
            error_code=ErrorCode.DATABASE_ERROR,
            message=f"Database operation failed: {operation}",
            suggestion=suggestion
        )

    @staticmethod
    def model_load_error(
        model_name: str,
        reason: str
    ) -> ErrorResponse:
        """Build a model loading error"""
        return ErrorResponse(
            error_code=ErrorCode.MODEL_LOAD_ERROR,
            message=f"Failed to load model '{model_name}'",
            details=[ErrorDetail(message=reason)],
            suggestion="The embedding model could not be loaded. Please check the system configuration or contact support"
        )

    @staticmethod
    def rate_limit_error(
        limit: int,
        window_seconds: int,
        retry_after_seconds: int
    ) -> ErrorResponse:
        """Build a rate limit error"""
        return ErrorResponse(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message=f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            suggestion=f"Please wait {retry_after_seconds} seconds before making more requests"
        )

    @staticmethod
    def from_exception(
        exception: Exception,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        include_traceback: bool = False
    ) -> ErrorResponse:
        """Build an error response from an exception"""
        message = str(exception) or type(exception).__name__

        details = None
        if include_traceback:
            import traceback
            tb = traceback.format_exc()
            details = [ErrorDetail(message=tb, code="STACK_TRACE")]

        # Map common exception types to error codes
        if isinstance(exception, ValueError):
            error_code = ErrorCode.VALIDATION_ERROR
        elif isinstance(exception, KeyError):
            error_code = ErrorCode.RESOURCE_NOT_FOUND
        elif isinstance(exception, TimeoutError):
            error_code = ErrorCode.TIMEOUT_ERROR
        elif isinstance(exception, ConnectionError):
            error_code = ErrorCode.SERVICE_UNAVAILABLE

        return ErrorResponse(
            error_code=error_code,
            message=message,
            details=details,
            suggestion="An unexpected error occurred. Please try again"
        )


# HTTP status code mapping for error codes
ERROR_CODE_TO_HTTP_STATUS: Dict[ErrorCode, int] = {
    # Client errors (4xx)
    ErrorCode.VALIDATION_ERROR: 400,
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.RESOURCE_NOT_FOUND: 404,
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,

    # Server errors (5xx)
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.SERVICE_UNAVAILABLE: 503,
    ErrorCode.TIMEOUT_ERROR: 504,
    ErrorCode.DATABASE_ERROR: 500,
    ErrorCode.EMBEDDING_ERROR: 500,
    ErrorCode.EXTERNAL_API_ERROR: 502,
    ErrorCode.MODEL_LOAD_ERROR: 500,
    ErrorCode.VECTOR_STORE_ERROR: 500,
    ErrorCode.CACHE_ERROR: 500,
}


def get_http_status_for_error(error_code: ErrorCode) -> int:
    """Get HTTP status code for an error code"""
    return ERROR_CODE_TO_HTTP_STATUS.get(error_code, 500)
