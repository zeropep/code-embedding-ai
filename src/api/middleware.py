import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import structlog

from .error_models import (
    ErrorResponse, ErrorResponseBuilder, ErrorCode,
    get_http_status_for_error
)
from ..utils.error_tracker import get_error_tracker, ErrorCategory, ErrorSeverity


logger = structlog.get_logger(__name__)


async def error_handling_middleware(request: Request, call_next: Callable) -> Response:
    """
    Global error handling middleware

    Catches all unhandled exceptions and converts them to standardized error responses.
    Also tracks errors for monitoring.
    """
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()

    try:
        # Process the request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Log successful request
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=f"{duration_ms:.2f}",
            request_id=request_id
        )

        return response

    except Exception as e:
        # Calculate request duration
        duration_ms = (time.time() - start_time) * 1000

        # Determine error severity based on exception type
        severity = ErrorSeverity.HIGH
        if isinstance(e, (ValueError, KeyError, TypeError)):
            severity = ErrorSeverity.MEDIUM
        elif isinstance(e, TimeoutError):
            severity = ErrorSeverity.HIGH
        elif isinstance(e, (ConnectionError, OSError)):
            severity = ErrorSeverity.CRITICAL

        # Track the error
        error_tracker = get_error_tracker()
        error_tracker.record_error(
            e,
            category=_categorize_error(e),
            severity=severity,
            context={
                "method": request.method,
                "path": str(request.url.path),
                "request_id": request_id,
                "duration_ms": duration_ms
            }
        )

        # Log the error
        logger.error(
            "Request failed with unhandled exception",
            method=request.method,
            path=request.url.path,
            error_type=type(e).__name__,
            error=str(e),
            duration_ms=f"{duration_ms:.2f}",
            request_id=request_id
        )

        # Build error response
        error_response = _build_error_response(e, request_id)

        # Return JSON error response
        return JSONResponse(
            status_code=get_http_status_for_error(error_response.error_code),
            content=error_response.model_dump(),
            headers={"X-Request-ID": request_id}
        )


def _categorize_error(exception: Exception) -> ErrorCategory:
    """Categorize exception into error category"""
    error_type = type(exception).__name__

    if isinstance(exception, (ConnectionError, TimeoutError)):
        return ErrorCategory.NETWORK
    elif "database" in str(exception).lower() or "chroma" in str(exception).lower():
        return ErrorCategory.DATABASE
    elif "embedding" in str(exception).lower() or "model" in str(exception).lower():
        return ErrorCategory.EMBEDDING
    elif isinstance(exception, ValueError):
        return ErrorCategory.VALIDATION
    elif isinstance(exception, TimeoutError):
        return ErrorCategory.TIMEOUT
    else:
        return ErrorCategory.INTERNAL


def _build_error_response(exception: Exception, request_id: str) -> ErrorResponse:
    """Build error response from exception"""
    # Map exception to appropriate error response
    if isinstance(exception, ValueError):
        return ErrorResponseBuilder.validation_error(
            message=str(exception)
        )
    elif isinstance(exception, KeyError):
        return ErrorResponseBuilder.not_found(
            resource_type="Resource",
            resource_id=str(exception)
        )
    elif isinstance(exception, TimeoutError):
        return ErrorResponseBuilder.timeout_error(
            operation="request processing"
        )
    elif isinstance(exception, ConnectionError):
        return ErrorResponseBuilder.service_unavailable(
            service_name="backend service"
        )
    else:
        # Generic internal error
        error_response = ErrorResponseBuilder.internal_error(
            message="An unexpected error occurred",
            details=str(exception)
        )
        error_response.request_id = request_id
        return error_response


async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    """
    Request logging middleware

    Logs all incoming requests for debugging and monitoring.
    """
    # Log incoming request
    logger.info(
        "Incoming request",
        method=request.method,
        path=request.url.path,
        query_params=dict(request.query_params) if request.query_params else None,
        client_host=request.client.host if request.client else None
    )

    # Process request
    response = await call_next(request)

    return response


async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
    """
    Basic rate limiting middleware (placeholder)

    This is a simple implementation. For production, use a proper rate limiting library
    like slowapi or fastapi-limiter.
    """
    # TODO: Implement proper rate limiting
    # For now, just pass through
    response = await call_next(request)
    return response
