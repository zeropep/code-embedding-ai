# API Documentation

## Overview

The Code Embedding AI Pipeline provides a comprehensive REST API for processing repositories, searching code, and managing the system. All endpoints return JSON responses and support standard HTTP status codes.

**Base URL**: `http://localhost:8000`
**API Version**: v1
**Content-Type**: `application/json`

## Authentication

Authentication is optional but recommended for production deployments.

```http
X-API-Key: your-api-key-here
```

## Endpoints

### Health and Status

#### Health Check
Get the overall health status of the system.

```http
GET /health
```

**Response**:
```json
{
    "status": "healthy",
    "components": {
        "database": {
            "status": "healthy",
            "total_chunks": 15420,
            "response_time": 0.05
        },
        "embedding_service": {
            "status": "healthy",
            "api_accessible": true,
            "response_time": 0.12
        },
        "update_manager": {
            "status": "healthy",
            "monitoring_active": true,
            "last_check_time": "2024-01-20T10:30:00Z"
        }
    },
    "timestamp": "2024-01-20T10:30:15Z"
}
```

**Status Codes**:
- `200`: System is healthy
- `503`: System is unhealthy or degraded

#### System Statistics
Get comprehensive system statistics and metrics.

```http
GET /api/v1/stats
```

**Response**:
```json
{
    "database": {
        "total_chunks": 15420,
        "collection_name": "code_embeddings",
        "index_size": "245.6 MB",
        "avg_chunk_size": 156
    },
    "embedding_service": {
        "total_requests": 1250,
        "successful_requests": 1240,
        "failed_requests": 10,
        "average_processing_time": 1.23,
        "cache_hit_rate": 0.15
    },
    "parser": {
        "files_processed": 892,
        "total_lines_processed": 125430,
        "supported_languages": ["java", "kotlin", "html"],
        "avg_chunks_per_file": 17.3
    },
    "security": {
        "secrets_detected": 23,
        "files_with_secrets": 8,
        "false_positive_rate": 0.05
    }
}
```

### Repository Processing

#### Process Repository
Process a repository to extract code chunks, detect secrets, generate embeddings, and store in vector database.

```http
POST /api/v1/process
```

**Request Body**:
```json
{
    "repository_path": "/path/to/repository",
    "include_security_scan": true,
    "force_reprocess": false,
    "chunk_size_override": null,
    "language_filter": ["java", "html"],
    "exclude_patterns": ["*/test/*", "*/target/*"]
}
```

**Parameters**:
- `repository_path` (required): Absolute path to the Git repository
- `include_security_scan` (optional, default: true): Enable secret detection and masking
- `force_reprocess` (optional, default: false): Reprocess even if files haven't changed
- `chunk_size_override` (optional): Override default chunk size limits
- `language_filter` (optional): Process only specified languages
- `exclude_patterns` (optional): Glob patterns to exclude files

**Response**:
```json
{
    "status": "success",
    "processing_summary": {
        "files_processed": 45,
        "chunks_created": 782,
        "secrets_detected": 3,
        "files_with_secrets": 1,
        "processing_time": 42.5,
        "languages_found": ["java", "html"]
    },
    "security_report": {
        "scan_summary": {
            "total_secrets_found": 3,
            "files_with_secrets": 1,
            "sensitivity_distribution": {
                "high": 1,
                "medium": 2,
                "low": 0
            }
        },
        "secret_types_found": ["password", "api_key"],
        "high_risk_files": [
            "src/main/java/config/DatabaseConfig.java"
        ],
        "recommendations": [
            "Review detected secrets in configuration files",
            "Consider using environment variables for credentials"
        ]
    },
    "chunks": [
        {
            "chunk_id": "chunk_abc123",
            "file_path": "src/main/java/UserService.java",
            "class_name": "UserService",
            "function_name": "createUser",
            "layer_type": "service",
            "language": "java",
            "token_count": 145,
            "has_secrets": false
        }
    ]
}
```

**Status Codes**:
- `200`: Processing completed successfully
- `400`: Invalid request parameters
- `404`: Repository path not found
- `500`: Processing error

### Code Search

#### Semantic Search
Search for code chunks using natural language queries or code snippets.

```http
POST /api/v1/search
```

**Request Body**:
```json
{
    "query": "user authentication and validation logic",
    "limit": 10,
    "similarity_threshold": 0.7,
    "language_filter": "java",
    "layer_filter": "service",
    "file_filter": "*/service/*",
    "include_metadata": true,
    "explain_results": false
}
```

**Parameters**:
- `query` (required): Natural language or code query
- `limit` (optional, default: 10): Maximum number of results
- `similarity_threshold` (optional, default: 0.0): Minimum similarity score
- `language_filter` (optional): Filter by programming language
- `layer_filter` (optional): Filter by architectural layer (controller, service, entity, etc.)
- `file_filter` (optional): Glob pattern to filter files
- `include_metadata` (optional, default: true): Include chunk metadata
- `explain_results` (optional, default: false): Include explanation of results

**Response**:
```json
{
    "results": [
        {
            "chunk_id": "chunk_xyz789",
            "content": "public User authenticate(String username, String password) {\n    if (username == null || password == null) {\n        throw new ValidationException(\"Credentials required\");\n    }\n    // Authentication logic\n}",
            "similarity": 0.92,
            "metadata": {
                "file_path": "src/main/java/AuthService.java",
                "class_name": "AuthService",
                "function_name": "authenticate",
                "layer_type": "service",
                "language": "java",
                "start_line": 45,
                "end_line": 52,
                "token_count": 87
            }
        }
    ],
    "total_results": 1,
    "query_time": 0.15,
    "search_metadata": {
        "query_embedding_time": 0.08,
        "vector_search_time": 0.05,
        "post_processing_time": 0.02,
        "model_version": "jina-code-embeddings-1.5b"
    }
}
```

### Chunk Management

#### Get Specific Chunk
Retrieve detailed information about a specific code chunk.

```http
GET /api/v1/chunks/{chunk_id}
```

**Response**:
```json
{
    "chunk_id": "chunk_abc123",
    "content": "public void updateUser(User user) { ... }",
    "metadata": {
        "file_path": "src/main/java/UserService.java",
        "class_name": "UserService",
        "function_name": "updateUser",
        "layer_type": "service",
        "language": "java",
        "start_line": 78,
        "end_line": 95,
        "token_count": 123,
        "created_at": "2024-01-20T10:15:00Z",
        "last_modified": "2024-01-20T10:15:00Z"
    },
    "embedding_metadata": {
        "model_version": "jina-code-embeddings-1.5b",
        "embedding_time": 0.45,
        "vector_dimension": 768
    },
    "security_metadata": {
        "scanned": true,
        "secrets_found": 0,
        "sensitivity_level": "low"
    }
}
```

**Status Codes**:
- `200`: Chunk found
- `404`: Chunk not found

#### Update Chunk
Update an existing chunk's content and regenerate embedding.

```http
PUT /api/v1/chunks/{chunk_id}
```

**Request Body**:
```json
{
    "content": "updated code content",
    "metadata": {
        "function_name": "newFunctionName"
    }
}
```

#### Delete Chunks
Delete chunks by file path or chunk IDs.

```http
DELETE /api/v1/chunks
```

**Request Body**:
```json
{
    "file_path": "src/main/java/OldClass.java"
}
```

or

```json
{
    "chunk_ids": ["chunk_abc123", "chunk_xyz789"]
}
```

### File Management

#### List Processed Files
Get a list of all files that have been processed.

```http
GET /api/v1/files
```

**Query Parameters**:
- `language`: Filter by language
- `has_secrets`: Filter files with/without secrets
- `limit`: Number of files to return
- `offset`: Pagination offset

**Response**:
```json
{
    "files": [
        {
            "file_path": "src/main/java/UserService.java",
            "language": "java",
            "chunk_count": 12,
            "total_tokens": 1456,
            "has_secrets": false,
            "last_modified": "2024-01-20T09:30:00Z",
            "processed_at": "2024-01-20T10:15:00Z"
        }
    ],
    "total_files": 45,
    "pagination": {
        "limit": 20,
        "offset": 0,
        "total": 45
    }
}
```

### Update Monitoring

#### Start Update Monitoring
Start monitoring a repository for changes.

```http
POST /api/v1/monitoring/start
```

**Request Body**:
```json
{
    "repository_path": "/path/to/repository",
    "check_interval_seconds": 300,
    "enable_file_watching": true
}
```

#### Stop Update Monitoring
Stop monitoring a repository.

```http
POST /api/v1/monitoring/stop
```

**Request Body**:
```json
{
    "repository_path": "/path/to/repository"
}
```

#### Get Monitoring Status
Get the status of update monitoring.

```http
GET /api/v1/monitoring/status
```

**Response**:
```json
{
    "monitoring_active": true,
    "monitored_repositories": [
        {
            "repository_path": "/path/to/repo",
            "last_check_time": "2024-01-20T10:25:00Z",
            "changes_detected": 3,
            "monitoring_since": "2024-01-20T09:00:00Z"
        }
    ]
}
```

#### Get Update Statistics
Get statistics about updates and changes.

```http
GET /api/v1/monitoring/stats
```

**Response**:
```json
{
    "total_updates_processed": 127,
    "files_modified": 89,
    "files_added": 23,
    "files_deleted": 15,
    "last_update_time": "2024-01-20T10:20:00Z",
    "update_frequency": {
        "daily_average": 12.3,
        "weekly_average": 86.2
    }
}
```

## Error Handling

All API endpoints return standard HTTP status codes and error responses:

```json
{
    "error": "ValidationError",
    "message": "Repository path is required",
    "details": {
        "field": "repository_path",
        "code": "MISSING_REQUIRED_FIELD"
    },
    "timestamp": "2024-01-20T10:30:00Z",
    "request_id": "req_abc123"
}
```

### Status Codes
- `200`: Success
- `201`: Created
- `400`: Bad Request (validation error)
- `401`: Unauthorized (invalid API key)
- `403`: Forbidden
- `404`: Not Found
- `429`: Rate Limited
- `500`: Internal Server Error
- `503`: Service Unavailable

## Rate Limiting

API endpoints are rate limited to ensure fair usage:

- **Default**: 100 requests per minute per API key
- **Search**: 50 requests per minute per API key
- **Processing**: 10 requests per hour per API key

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642680000
```

## Pagination

List endpoints support pagination using `limit` and `offset` parameters:

```http
GET /api/v1/files?limit=20&offset=40
```

Response includes pagination metadata:

```json
{
    "pagination": {
        "limit": 20,
        "offset": 40,
        "total": 150,
        "has_next": true,
        "has_previous": true
    }
}
```

## Webhooks

Configure webhooks to receive notifications about processing events:

```http
POST /api/v1/webhooks
```

**Request Body**:
```json
{
    "url": "https://your-app.com/webhook",
    "events": ["processing.completed", "security.secrets_detected"],
    "secret": "webhook-secret-key"
}
```

### Webhook Events
- `processing.started`: Repository processing started
- `processing.completed`: Repository processing finished
- `processing.failed`: Repository processing failed
- `security.secrets_detected`: Secrets detected during scanning
- `monitoring.changes_detected`: File changes detected
- `system.health_degraded`: System health issues

## SDKs and Client Libraries

### Python SDK
```python
from code_embedding_client import CodeEmbeddingClient

client = CodeEmbeddingClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Process repository
result = await client.process_repository("/path/to/repo")

# Search code
results = await client.search("user authentication", limit=5)
```

### JavaScript SDK
```javascript
import { CodeEmbeddingClient } from 'code-embedding-client';

const client = new CodeEmbeddingClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your-api-key'
});

// Process repository
const result = await client.processRepository('/path/to/repo');

// Search code
const results = await client.search('user authentication', { limit: 5 });
```

## Examples

### Complete Repository Processing Workflow

```bash
# 1. Process repository
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "repository_path": "/path/to/spring-boot-app",
    "include_security_scan": true
  }'

# 2. Search for authentication logic
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "user authentication and session management",
    "limit": 5,
    "language_filter": "java"
  }'

# 3. Start monitoring for changes
curl -X POST http://localhost:8000/api/v1/monitoring/start \
  -H "Content-Type: application/json" \
  -d '{
    "repository_path": "/path/to/spring-boot-app",
    "check_interval_seconds": 300
  }'
```