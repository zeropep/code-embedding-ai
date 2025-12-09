# Error Handling Guide

## 📋 Overview

This document describes the comprehensive error handling system implemented in the Code Embedding AI service. The system provides:

- **Retry Logic**: Automatic retries with exponential backoff
- **Error Tracking**: Centralized error monitoring and statistics
- **Standardized Responses**: User-friendly error messages with actionable suggestions
- **Graceful Degradation**: System continues functioning when non-critical services fail

---

## 🎯 Features

### 1. Retry Logic with Exponential Backoff

Automatic retry mechanism for transient failures:

```python
from src.utils.retry import retry_async, RetryConfig

# Use predefined configuration
from src.utils.retry import NETWORK_RETRY_CONFIG

# Or create custom configuration
config = RetryConfig(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)

# Apply retry to async function
result = await retry_async(
    my_async_function,
    arg1, arg2,
    config=config
)
```

**Predefined Configurations:**

| Config | Max Attempts | Initial Delay | Max Delay | Use Case |
|--------|-------------|---------------|-----------|----------|
| `NETWORK_RETRY_CONFIG` | 5 | 1.0s | 30.0s | Network/API calls |
| `DATABASE_RETRY_CONFIG` | 3 | 0.5s | 10.0s | Database operations |
| `API_RETRY_CONFIG` | 3 | 1.0s | 15.0s | External API calls |
| `FAST_RETRY_CONFIG` | 2 | 0.1s | 1.0s | Fast local operations |

### 2. Error Tracking and Monitoring

Centralized error tracking system:

```python
from src.utils.error_tracker import get_error_tracker, ErrorCategory, ErrorSeverity

error_tracker = get_error_tracker()

# Record an error
error_tracker.record_error(
    exception,
    category=ErrorCategory.EMBEDDING,
    severity=ErrorSeverity.HIGH,
    context={"operation": "embedding_generation", "batch_size": 100}
)

# Get statistics
stats = error_tracker.get_statistics(
    category=ErrorCategory.EMBEDDING,
    time_window_seconds=300  # Last 5 minutes
)

# Get health status
health = error_tracker.get_health_status()
```

**Error Categories:**

- `NETWORK`: Network/connection errors
- `DATABASE`: Database operation errors
- `EMBEDDING`: Embedding generation errors
- `VALIDATION`: Input validation errors
- `AUTHENTICATION`: Auth/permission errors
- `INTERNAL`: Internal system errors
- `EXTERNAL_API`: External API errors
- `TIMEOUT`: Timeout errors
- `UNKNOWN`: Uncategorized errors

**Error Severity Levels:**

- `LOW`: Minor issues, no impact on functionality
- `MEDIUM`: Moderate issues, degraded performance
- `HIGH`: Significant issues, feature unavailable
- `CRITICAL`: System-wide failures

### 3. Standardized Error Responses

User-friendly error responses with actionable suggestions:

```python
from src.api.error_models import ErrorResponseBuilder, ErrorCode

# Validation error
error = ErrorResponseBuilder.validation_error(
    message="Invalid input",
    field="query",
    suggestion="Query must be at least 3 characters long"
)

# Not found error
error = ErrorResponseBuilder.not_found(
    resource_type="Project",
    resource_id="project_123"
)

# Service unavailable
error = ErrorResponseBuilder.service_unavailable(
    service_name="Embedding Service",
    retry_after_seconds=60
)

# From exception
error = ErrorResponseBuilder.from_exception(
    exception,
    error_code=ErrorCode.EMBEDDING_ERROR
)
```

**Error Response Format:**

```json
{
  "success": false,
  "error_code": "EMBEDDING_ERROR",
  "message": "Failed to generate embeddings for the provided content",
  "details": [
    {
      "field": "content",
      "message": "Content exceeds maximum length of 8000 characters",
      "code": "CONTENT_TOO_LONG"
    }
  ],
  "suggestion": "Try splitting the content into smaller chunks",
  "timestamp": "2025-12-07T10:30:00.000Z",
  "request_id": "req_123abc"
}
```

---

## 🔧 API Endpoints

### Error Monitoring Endpoints

#### Get Error Statistics

```http
GET /status/errors
```

**Query Parameters:**
- `category` (optional): Filter by error category
- `severity` (optional): Filter by severity level
- `time_window_seconds` (optional): Time window for statistics

**Example:**

```bash
curl "http://localhost:8000/status/errors?category=embedding&time_window_seconds=300"
```

**Response:**

```json
{
  "status": "success",
  "error_statistics": {
    "total_errors": 15,
    "errors_by_category": {
      "embedding": 10,
      "network": 5
    },
    "errors_by_severity": {
      "high": 8,
      "medium": 7
    },
    "error_rate_per_minute": 3.0,
    "recent_errors": [...]
  }
}
```

#### Get Error Health Status

```http
GET /status/errors/health
```

**Example:**

```bash
curl http://localhost:8000/status/errors/health
```

**Response:**

```json
{
  "status": "success",
  "health": {
    "status": "healthy",
    "total_errors_lifetime": 150,
    "errors_last_5_min": 5,
    "error_rate_per_minute": 1.0,
    "critical_errors": 0,
    "high_severity_errors": 2,
    "issues": [],
    "uptime_seconds": 3600
  }
}
```

#### Clear Error Statistics

```http
DELETE /status/errors
```

**Example:**

```bash
curl -X DELETE http://localhost:8000/status/errors
```

---

## 📊 Error Codes

### Client Errors (4xx)

| Code | HTTP Status | Description |
|------|------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid input or parameters |
| `INVALID_REQUEST` | 400 | Malformed request |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | 404 | Resource does not exist |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |

### Server Errors (5xx)

| Code | HTTP Status | Description |
|------|------------|-------------|
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |
| `TIMEOUT_ERROR` | 504 | Request timeout |
| `DATABASE_ERROR` | 500 | Database operation failed |
| `EMBEDDING_ERROR` | 500 | Embedding generation failed |
| `MODEL_LOAD_ERROR` | 500 | Model loading failed |
| `VECTOR_STORE_ERROR` | 500 | Vector store operation failed |

---

## 🔍 Implementation Examples

### Example 1: Adding Retry Logic to a Function

```python
from src.utils.retry import with_retry, RetryConfig

@with_retry(RetryConfig(max_attempts=3, initial_delay=1.0))
async def fetch_data_from_api():
    """Automatically retries on failure"""
    response = await api_client.get("/data")
    return response.json()
```

### Example 2: Tracking Errors in Service Layer

```python
from src.utils.error_tracker import get_error_tracker, ErrorCategory, ErrorSeverity

async def process_embeddings(chunks):
    error_tracker = get_error_tracker()

    try:
        results = await embedding_client.generate(chunks)
        return results
    except Exception as e:
        # Track the error
        error_tracker.record_error(
            e,
            category=ErrorCategory.EMBEDDING,
            severity=ErrorSeverity.HIGH,
            context={
                "operation": "batch_embedding",
                "chunk_count": len(chunks)
            }
        )
        # Re-raise or handle gracefully
        raise
```

### Example 3: Graceful Degradation in API Routes

```python
from src.api.error_models import ErrorResponseBuilder, ErrorCode

@router.post("/search/semantic")
async def semantic_search(request: SearchRequest):
    try:
        # Try primary search method
        results = await vector_store.search(request.query)
        return results

    except EmbeddingError as e:
        # Gracefully degrade to metadata-only search
        logger.warning("Embedding search failed, falling back to metadata search")

        fallback_results = await vector_store.search_by_metadata({
            "language": request.language
        })

        # Return partial results with warning
        return {
            "results": fallback_results,
            "degraded": True,
            "warning": "Embedding search unavailable, showing metadata-based results"
        }

    except Exception as e:
        # Return user-friendly error
        error = ErrorResponseBuilder.embedding_error(
            reason=str(e),
            recoverable=True
        )
        raise HTTPException(
            status_code=500,
            detail=error.model_dump()
        )
```

---

## 📈 Monitoring Best Practices

### 1. Regular Health Checks

Monitor system health using the error health endpoint:

```python
import httpx

async def check_system_health():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/status/errors/health")
        health = response.json()["health"]

        if health["status"] == "critical":
            # Alert administrators
            send_alert(f"System critical: {health['issues']}")
        elif health["status"] == "degraded":
            # Log warning
            logger.warning("System degraded", issues=health["issues"])
```

### 2. Error Rate Monitoring

Track error rates over time:

```python
async def monitor_error_rates():
    error_tracker = get_error_tracker()

    # Get last hour statistics
    stats = error_tracker.get_statistics(time_window_seconds=3600)

    if stats.error_rate_per_minute > 10:
        logger.error("High error rate detected",
                     rate=stats.error_rate_per_minute,
                     total_errors=stats.total_errors)
```

### 3. Category-Specific Monitoring

Monitor specific error categories:

```python
# Monitor embedding errors
embedding_errors = error_tracker.get_errors_by_category(
    ErrorCategory.EMBEDDING,
    limit=10
)

# Monitor critical errors
critical_errors = error_tracker.get_critical_errors(limit=10)
```

---

## 🛠️ Troubleshooting

### High Error Rate

**Symptom**: Error rate exceeds 10 errors/minute

**Solutions**:
1. Check error categories to identify the source
2. Review recent errors for patterns
3. Verify external service availability
4. Check system resources (CPU, memory, disk)

### Frequent Embedding Errors

**Symptom**: High number of `EMBEDDING_ERROR` occurrences

**Solutions**:
1. Check model loading status
2. Verify GPU/CPU availability
3. Review input data quality
4. Check for memory issues
5. See `KNOWN_ISSUES.md` for PyTorch compatibility issues

### Database Connection Errors

**Symptom**: Repeated `DATABASE_ERROR` occurrences

**Solutions**:
1. Verify ChromaDB is running
2. Check database connection settings
3. Review database logs
4. Restart database service if needed

---

## ✅ Testing Error Handling

### Manual Testing

```bash
# Test error endpoint
curl http://localhost:8000/status/errors

# Test with filters
curl "http://localhost:8000/status/errors?severity=critical"

# Test health endpoint
curl http://localhost:8000/status/errors/health

# Clear error statistics
curl -X DELETE http://localhost:8000/status/errors
```

### Programmatic Testing

```python
from src.utils.error_tracker import get_error_tracker, ErrorCategory, ErrorSeverity

def test_error_tracking():
    error_tracker = get_error_tracker()
    error_tracker.clear_errors()

    # Simulate error
    try:
        raise ValueError("Test error")
    except ValueError as e:
        error_tracker.record_error(
            e,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM
        )

    # Verify tracking
    stats = error_tracker.get_statistics()
    assert stats.total_errors == 1
    assert stats.errors_by_category[ErrorCategory.VALIDATION] == 1
```

---

## 📚 Related Documentation

- [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) - Known system issues and limitations
- [Performance Benchmarks](benchmarks/README.md) - Performance optimization guide
- [API Documentation](http://localhost:8000/docs) - Interactive API docs

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-07 | Initial comprehensive error handling system |

---

**Last Updated**: 2025-12-07
