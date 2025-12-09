from .models import ErrorResponse
from .dependencies import startup_event, shutdown_event
from .routes import all_routers
from .error_models import ErrorResponseBuilder, ErrorCode, get_http_status_for_error
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import uuid
import structlog
from dotenv import load_dotenv
from ..utils.error_tracker import get_error_tracker, ErrorCategory, ErrorSeverity

# Load environment variables from .env file
load_dotenv()

# Setup file logging
from ..config.logging_config import setup_file_logging
setup_file_logging()

# Configure logging
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance"""
    application = FastAPI(
        title="Code Embedding AI Pipeline",
        description="AI-powered code embedding and semantic search service for Spring Boot codebases",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Add middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request ID middleware
    @application.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        """Add unique request ID to each request"""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Request timing middleware
    @application.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add request processing time to response headers"""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # Request logging middleware
    @application.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests"""
        start_time = time.time()
        request_id = getattr(request.state, 'request_id', 'unknown')

        logger.info("Request started",
                    request_id=request_id,
                    method=request.method,
                    url=str(request.url),
                    client_ip=request.client.host if request.client else None)

        response = await call_next(request)

        process_time = time.time() - start_time

        logger.info("Request completed",
                    request_id=request_id,
                    status_code=response.status_code,
                    process_time=process_time)

        return response

    # Global exception handler
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler with error tracking"""
        request_id = getattr(request.state, 'request_id', 'unknown')

        # Track the error
        error_tracker = get_error_tracker()
        error_tracker.record_error(
            exc,
            category=ErrorCategory.INTERNAL,
            severity=ErrorSeverity.CRITICAL,
            context={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None
            }
        )

        logger.error("Unhandled exception",
                     request_id=request_id,
                     method=request.method,
                     url=str(request.url),
                     error=str(exc),
                     exc_info=True)

        # Build error response
        error_response = ErrorResponseBuilder.from_exception(
            exc,
            error_code=ErrorCode.INTERNAL_ERROR,
            include_traceback=False  # Don't expose traceback to clients
        )
        error_response.request_id = request_id

        return JSONResponse(
            status_code=get_http_status_for_error(error_response.error_code),
            content=error_response.model_dump(),
            headers={"X-Request-ID": request_id}
        )

    # HTTP exception handler
    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with error tracking"""
        request_id = getattr(request.state, 'request_id', 'unknown')

        # Determine severity based on status code
        if exc.status_code >= 500:
            severity = ErrorSeverity.HIGH
        elif exc.status_code >= 400:
            severity = ErrorSeverity.MEDIUM
        else:
            severity = ErrorSeverity.LOW

        # Track the error for 4xx and 5xx responses
        if exc.status_code >= 400:
            error_tracker = get_error_tracker()
            error_tracker.record_error(
                exc,
                category=ErrorCategory.VALIDATION if exc.status_code < 500 else ErrorCategory.INTERNAL,
                severity=severity,
                context={
                    "request_id": request_id,
                    "status_code": exc.status_code,
                    "method": request.method,
                    "url": str(request.url)
                }
            )

        logger.warning("HTTP exception",
                       request_id=request_id,
                       status_code=exc.status_code,
                       detail=exc.detail)

        # Map HTTP status codes to error codes
        error_code = ErrorCode.INTERNAL_ERROR
        if exc.status_code == 400:
            error_code = ErrorCode.VALIDATION_ERROR
        elif exc.status_code == 404:
            error_code = ErrorCode.RESOURCE_NOT_FOUND
        elif exc.status_code == 429:
            error_code = ErrorCode.RATE_LIMIT_EXCEEDED
        elif exc.status_code == 503:
            error_code = ErrorCode.SERVICE_UNAVAILABLE

        error_response = ErrorResponseBuilder.internal_error(
            message=str(exc.detail) if exc.detail else "An error occurred"
        )
        error_response.error_code = error_code
        error_response.request_id = request_id

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(),
            headers={"X-Request-ID": request_id}
        )

    # Include all routers
    for router in all_routers:
        application.include_router(router)

    return application


# Create FastAPI application
app = create_app()


# Custom startup message
@app.on_event("startup")
async def startup_message():
    """Log startup message"""
    logger.info("Code Embedding AI Pipeline API started",
                title=app.title,
                version=app.version,
                docs_url=app.docs_url)


@app.on_event("shutdown")
async def shutdown_message():
    """Log shutdown message"""
    logger.info("Code Embedding AI Pipeline API stopped")


if __name__ == "__main__":
    import uvicorn

    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
