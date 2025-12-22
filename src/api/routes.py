from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Dict, Any, Optional
import uuid
import time
import structlog

from .models import (
    HealthCheckResponse, SearchResponse, SearchRequest, SearchResult,
    RequestStatus, SimilarCodeRequest, ProcessRepositoryResponse,
    ProcessRepositoryRequest, ProcessingMode, BatchProcessRequest,
    SystemStatusResponse, MetricsResponse, BatchDeleteRequest,
    ProjectListResponse, ProjectStatsResponse, ProjectInfo,
    ProjectCreateRequest, ProjectCreateResponse, ProjectDetailResponse,
    ProjectUpdateRequest, ProjectUpdateResponse, ProjectDeleteResponse
)
from .dependencies import get_embedding_pipeline, get_update_service, get_vector_store
from ..embeddings.embedding_pipeline import EmbeddingPipeline
from ..updates.update_service import UpdateService
from ..updates.models import UpdateRequest
from ..database.vector_store import VectorStore
from ..database.project_repository import ProjectRepository
from ..database.project_models import Project, ProjectStatus
from ..cache.cache_manager import get_cache_manager
from datetime import datetime


logger = structlog.get_logger(__name__)

# Create routers
main_router = APIRouter()
search_router = APIRouter(prefix="/search", tags=["search"])
process_router = APIRouter(prefix="/process", tags=["processing"])
admin_router = APIRouter(prefix="/admin", tags=["administration"])
status_router = APIRouter(prefix="/status", tags=["status"])
projects_router = APIRouter(prefix="/projects", tags=["projects"])


# Main Routes
@main_router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Code Embedding AI Pipeline",
        "version": "1.0.0",
        "description": "AI-powered code embedding and semantic search service",
        "docs": "/docs",
        "health": "/health"
    }


@main_router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline),
    update_service: UpdateService = Depends(get_update_service)
):
    """Comprehensive health check"""
    try:
        start_time = time.time()

        # Check pipeline health
        pipeline_health = await pipeline.health_check()

        # Check update service health
        update_health = await update_service.health_check()

        # Combine health information
        components = {
            "pipeline": pipeline_health,
            "update_service": update_health
        }

        # Determine overall status
        overall_status = "healthy"
        for component_health in components.values():
            if component_health.get("status") == "unhealthy":
                overall_status = "unhealthy"
                break

        health_check_time = time.time() - start_time

        return HealthCheckResponse(
            status=overall_status,
            components=components,
            uptime_seconds=health_check_time,
            version="1.0.0"
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Search Routes
@search_router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline)
):
    """Perform semantic search using text query"""
    try:
        start_time = time.time()
        cache_manager = get_cache_manager()

        # Check cache first
        cached_results = cache_manager.get_search_results(
            query=request.query,
            top_k=request.top_k,
            project_id=request.project_id,
            min_similarity=request.min_similarity
        )

        if cached_results is not None:
            logger.info("Returning cached search results",
                       query=request.query,
                       cached=True)
            return cached_results

        logger.info("Semantic search request",
                    query=request.query,
                    top_k=request.top_k)

        # Generate embedding for query
        await pipeline.embedding_service.start()
        # Create a dummy code chunk for embedding generation
        from ..code_parser.models import CodeChunk, CodeLanguage, LayerType

        # Determine query language from project if available
        query_language = CodeLanguage.JAVA  # Default
        if request.project_id:
            project_repo = ProjectRepository()
            project = project_repo.get(request.project_id)
            if project and hasattr(project, 'primary_language') and project.primary_language:
                language_map = {
                    "java": CodeLanguage.JAVA,
                    "python": CodeLanguage.PYTHON,
                    "kotlin": CodeLanguage.KOTLIN,
                    "html": CodeLanguage.HTML,
                    "unknown": CodeLanguage.JAVA,  # unknown은 java로 fallback
                }
                query_language = language_map.get(project.primary_language, CodeLanguage.JAVA)

        query_chunk = CodeChunk(
            content=request.query,
            file_path="query",
            language=query_language,
            start_line=1,
            end_line=1,
            layer_type=LayerType.UNKNOWN,
            metadata={}
        )

        embedded_chunks = await pipeline.embedding_service.generate_chunk_embeddings([query_chunk])

        if not embedded_chunks or 'embedding' not in embedded_chunks[0].metadata:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")

        query_vector = embedded_chunks[0].metadata['embedding']['vector']

        # Prepare filters with project_id if specified
        filters = request.filters or {}
        if request.project_id:
            filters["project_id"] = request.project_id

        # Perform vector search
        search_results = vector_store.search_similar_chunks(
            query_vector=query_vector,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            filters=filters if filters else None
        )

        # Convert to response format
        results = []
        for result in search_results:
            search_result = SearchResult(
                chunk_id=result.chunk_id,
                similarity_score=result.similarity_score,
                file_path=result.metadata.get("file_path", ""),
                function_name=result.metadata.get("function_name"),
                class_name=result.metadata.get("class_name"),
                layer_type=result.metadata.get("layer_type", "Unknown"),
                start_line=result.metadata.get("start_line", 0),
                end_line=result.metadata.get("end_line", 0),
                content=result.content if request.include_content else None,
                embedding_vector=result.embedding_vector if request.include_embeddings else None,
                metadata=result.metadata
            )
            results.append(search_result)

        search_time = (time.time() - start_time) * 1000

        response = SearchResponse(
            status=RequestStatus.SUCCESS,
            message=f"Found {len(results)} results",
            results=results,
            total_results=len(results),
            query_info={
                "query": request.query,
                "search_type": request.search_type,
                "top_k": request.top_k,
                "min_similarity": request.min_similarity
            },
            search_time_ms=search_time
        )

        # Cache the results
        cache_manager.set_search_results(
            results=response,
            query=request.query,
            top_k=request.top_k,
            project_id=request.project_id,
            min_similarity=request.min_similarity
        )

        return response

    except Exception as e:
        logger.error("Semantic search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@search_router.post("/metadata", response_model=SearchResponse)
async def metadata_search(
    filters: Dict[str, Any],
    top_k: int = Query(default=10, ge=1, le=100),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Search by metadata filters only"""
    try:
        start_time = time.time()

        logger.info("Metadata search request", filters=filters, top_k=top_k)

        search_results = vector_store.search_by_metadata(filters, limit=top_k)

        results = []
        for result in search_results:
            search_result = SearchResult(
                chunk_id=result.chunk_id,
                similarity_score=1.0,  # No similarity for metadata search
                file_path=result.metadata.get("file_path", ""),
                function_name=result.metadata.get("function_name"),
                class_name=result.metadata.get("class_name"),
                layer_type=result.metadata.get("layer_type", "Unknown"),
                start_line=result.metadata.get("start_line", 0),
                end_line=result.metadata.get("end_line", 0),
                content=result.content,
                metadata=result.metadata
            )
            results.append(search_result)

        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            status=RequestStatus.SUCCESS,
            message=f"Found {len(results)} results",
            results=results,
            total_results=len(results),
            query_info={
                "filters": filters,
                "search_type": "metadata",
                "top_k": top_k
            },
            search_time_ms=search_time
        )

    except Exception as e:
        logger.error("Metadata search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metadata search failed: {str(e)}")


@search_router.post("/similar-code", response_model=SearchResponse)
async def find_similar_code(
    request: SimilarCodeRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline)
):
    """Find code similar to a given code snippet"""
    try:
        # Generate embedding for code snippet
        await pipeline.embedding_service.start()
        from ..code_parser.models import CodeChunk, CodeLanguage, LayerType

        # Determine language from project or request
        language_map = {
            "java": CodeLanguage.JAVA,
            "python": CodeLanguage.PYTHON,
            "kotlin": CodeLanguage.KOTLIN,
            "html": CodeLanguage.HTML,
            "xml": CodeLanguage.XML
        }

        code_language = CodeLanguage.JAVA  # Default
        if request.project_id:
            project_repo = ProjectRepository()
            project = project_repo.get(request.project_id)
            if project and hasattr(project, 'primary_language') and project.primary_language:
                code_language = language_map.get(project.primary_language, CodeLanguage.JAVA)
        elif request.language:
            code_language = language_map.get(request.language, CodeLanguage.JAVA)

        snippet_chunk = CodeChunk(
            content=request.code_snippet,
            file_path="snippet",
            language=code_language,
            start_line=1,
            end_line=1,
            layer_type=LayerType.UNKNOWN,
            metadata={}
        )

        embedded_chunks = await pipeline.embedding_service.generate_chunk_embeddings([snippet_chunk])

        if not embedded_chunks or 'embedding' not in embedded_chunks[0].metadata:
            raise HTTPException(status_code=500, detail="Failed to generate code embedding")

        query_vector = embedded_chunks[0].metadata['embedding']['vector']

        # Prepare filters with project_id if specified
        filters = {}
        if request.project_id:
            filters["project_id"] = request.project_id

        # Search for similar code
        search_results = vector_store.search_similar_chunks(
            query_vector=query_vector,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            filters=filters if filters else None
        )

        # Filter results if needed
        if not request.include_same_file:
            search_results = [r for r in search_results if r.metadata.get("file_path") != "snippet"]

        results = []
        for result in search_results:
            search_result = SearchResult(
                chunk_id=result.chunk_id,
                similarity_score=result.similarity_score,
                file_path=result.metadata.get("file_path", ""),
                function_name=result.metadata.get("function_name"),
                class_name=result.metadata.get("class_name"),
                layer_type=result.metadata.get("layer_type", "Unknown"),
                start_line=result.metadata.get("start_line", 0),
                end_line=result.metadata.get("end_line", 0),
                content=result.content,
                metadata=result.metadata
            )
            results.append(search_result)

        return SearchResponse(
            status=RequestStatus.SUCCESS,
            message=f"Found {len(results)} similar code chunks",
            results=results,
            total_results=len(results),
            query_info={
                "code_snippet_length": len(request.code_snippet),
                "language": request.language,
                "min_similarity": request.min_similarity
            },
            search_time_ms=0  # Will be calculated
        )

    except Exception as e:
        logger.error("Similar code search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Similar code search failed: {str(e)}")


# Processing Routes
@process_router.post("/repository", response_model=ProcessRepositoryResponse)
async def process_repository(
    request: ProcessRepositoryRequest,
    background_tasks: BackgroundTasks,
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline)
):
    """Process a repository to generate embeddings"""
    try:
        request_id = str(uuid.uuid4())

        logger.info("Repository processing request",
                    request_id=request_id,
                    repo_path=request.repo_path,
                    mode=request.mode)

        if request.mode == ProcessingMode.FULL:
            # Process entire repository
            result = await pipeline.process_repository(
                request.repo_path,
                project_id=request.project_id,
                project_name=request.project_name
            )
        elif request.mode == ProcessingMode.INCREMENTAL:
            # Incremental processing would need UpdateService integration
            raise HTTPException(status_code=501, detail="Incremental mode not implemented in this endpoint")
        else:  # AUTO mode
            # Auto-detect best processing mode
            result = await pipeline.process_repository(
                request.repo_path,
                project_id=request.project_id,
                project_name=request.project_name
            )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))

        return ProcessRepositoryResponse(
            status=RequestStatus.SUCCESS,
            message="Repository processed successfully",
            request_id=request_id,
            processing_summary=result.get("processing_summary"),
            parsing_stats=result.get("parsing_stats"),
            security_stats=result.get("security_stats"),
            embedding_stats=result.get("embedding_stats")
        )

    except Exception as e:
        logger.error("Repository processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@process_router.post("/files", response_model=ProcessRepositoryResponse)
async def process_files(
    request: BatchProcessRequest,
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline)
):
    """Process specific files"""
    try:
        request_id = str(uuid.uuid4())

        logger.info("File processing request",
                    request_id=request_id,
                    file_count=len(request.file_paths))

        result = await pipeline.process_files(
            request.file_paths,
            project_id=request.project_id,
            project_name=request.project_name
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))

        return ProcessRepositoryResponse(
            status=RequestStatus.SUCCESS,
            message="Files processed successfully",
            request_id=request_id,
            processing_summary=result.get("processing_summary"),
            parsing_stats=result.get("parsing_stats"),
            security_stats=result.get("security_stats"),
            embedding_stats=result.get("embedding_stats")
        )

    except Exception as e:
        logger.error("File processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@process_router.post("/update", response_model=Dict[str, Any])
async def trigger_update(
    repo_path: Optional[str] = None,
    force_full: bool = False,
    update_service: Optional[UpdateService] = Depends(get_update_service)
):
    """Trigger an incremental update - DEPRECATED: Use /process/project/{project_id} instead"""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use POST /process/project/{project_id} to process project repositories."
    )


@process_router.post("/project/{project_id}", response_model=ProcessRepositoryResponse)
async def process_project(
    project_id: str,
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline)
):
    """Process all files in a project's repository"""
    try:
        request_id = str(uuid.uuid4())

        logger.info("Processing project repository",
                    request_id=request_id,
                    project_id=project_id)

        # Get project information
        project_repo = ProjectRepository()
        project = project_repo.get(project_id)

        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project '{project_id}' not found"
            )

        # Check if repository path exists
        from pathlib import Path
        repo_path = Path(project.repository_path)
        if not repo_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Repository path does not exist: {project.repository_path}"
            )

        logger.info("Starting repository processing",
                    project_id=project_id,
                    project_name=project.name,
                    repo_path=project.repository_path)

        # Process repository with project metadata
        result = await pipeline.process_repository(
            repo_path=project.repository_path,
            project_id=project.project_id,
            project_name=project.name
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))

        # Update project statistics
        if result.get("processing_summary"):
            summary = result["processing_summary"]
            total_chunks = summary.get("chunks_with_embeddings", 0)
            total_files = summary.get("total_files_parsed", 0)
            project_repo.update(project_id, {
                "total_chunks": total_chunks,
                "total_files": total_files,
                "last_indexed_at": datetime.now(),
                "status": "active"
            })
            logger.info("Project statistics updated",
                       project_id=project_id,
                       total_chunks=total_chunks,
                       total_files=total_files)

        return ProcessRepositoryResponse(
            status=RequestStatus.SUCCESS,
            message=f"Project '{project.name}' processed successfully",
            request_id=request_id,
            processing_summary=result.get("processing_summary"),
            parsing_stats=result.get("parsing_stats"),
            security_stats=result.get("security_stats"),
            embedding_stats=result.get("embedding_stats")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Project processing failed", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# Status Routes
@status_router.get("/system", response_model=SystemStatusResponse)
async def get_system_status(
    update_service: UpdateService = Depends(get_update_service),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get comprehensive system status"""
    try:
        service_status = update_service.get_status()
        database_stats = vector_store.get_statistics().to_dict()

        return SystemStatusResponse(
            service_status=service_status,
            database_stats=database_stats,
            update_metrics=service_status.get("state_summary", {})
        )

    except Exception as e:
        logger.error("Failed to get system status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@status_router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline),
    update_service: UpdateService = Depends(get_update_service),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get system metrics"""
    try:
        embedding_metrics = pipeline.embedding_service.get_metrics()
        update_metrics = update_service.state_manager.get_metrics().to_dict()
        vector_store_stats = vector_store.get_statistics().to_dict()

        system_metrics = {
            "uptime": time.time(),
            "requests_processed": embedding_metrics.get("total_requests", 0)
        }

        return MetricsResponse(
            embedding_metrics=embedding_metrics,
            update_metrics=update_metrics,
            vector_store_stats=vector_store_stats,
            system_metrics=system_metrics
        )

    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@status_router.get("/cache", response_model=Dict[str, Any])
async def get_cache_stats():
    """Get cache statistics"""
    try:
        cache_manager = get_cache_manager()
        stats = cache_manager.get_all_stats()

        return {
            "status": "success",
            "cache_stats": stats,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error("Failed to get cache stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Cache stats retrieval failed: {str(e)}")


@status_router.get("/errors", response_model=Dict[str, Any])
async def get_error_statistics(
    category: Optional[str] = None,
    severity: Optional[str] = None,
    time_window_seconds: Optional[float] = None
):
    """
    Get error statistics with optional filtering

    Query Parameters:
    - category: Filter by error category (network, database, embedding, etc.)
    - severity: Filter by severity (low, medium, high, critical)
    - time_window_seconds: Only include errors from last N seconds
    """
    try:
        from ..utils.error_tracker import get_error_tracker, ErrorCategory, ErrorSeverity

        error_tracker = get_error_tracker()

        # Parse filters
        category_filter = None
        if category:
            try:
                category_filter = ErrorCategory(category.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category. Valid values: {[c.value for c in ErrorCategory]}"
                )

        severity_filter = None
        if severity:
            try:
                severity_filter = ErrorSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid severity. Valid values: {[s.value for s in ErrorSeverity]}"
                )

        # Get statistics
        stats = error_tracker.get_statistics(
            category=category_filter,
            severity=severity_filter,
            time_window_seconds=time_window_seconds
        )

        return {
            "status": "success",
            "error_statistics": stats.to_dict(),
            "timestamp": time.time(),
            "filters": {
                "category": category,
                "severity": severity,
                "time_window_seconds": time_window_seconds
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get error statistics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error statistics retrieval failed: {str(e)}")


@status_router.get("/errors/health", response_model=Dict[str, Any])
async def get_error_health_status():
    """
    Get overall system health status based on error patterns

    Returns health status (healthy, degraded, critical) along with error metrics
    """
    try:
        from ..utils.error_tracker import get_error_tracker

        error_tracker = get_error_tracker()
        health = error_tracker.get_health_status()

        return {
            "status": "success",
            "health": health,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error("Failed to get error health status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error health check failed: {str(e)}")


@status_router.delete("/errors", response_model=Dict[str, Any])
async def clear_error_statistics():
    """Clear all error statistics (admin operation)"""
    try:
        from ..utils.error_tracker import get_error_tracker

        error_tracker = get_error_tracker()
        error_tracker.clear_errors()

        return {
            "status": "success",
            "message": "Error statistics cleared",
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error("Failed to clear error statistics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error clear failed: {str(e)}")


# Admin Routes
@admin_router.delete("/chunks", response_model=Dict[str, Any])
async def delete_chunks(
    request: BatchDeleteRequest,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Delete chunks from vector store"""
    try:
        if request.file_paths:
            # Delete by file paths
            total_deleted = 0
            for file_path in request.file_paths:
                result = vector_store.delete_chunks_by_file(file_path)
                total_deleted += result.successful_items

            return {
                "status": "success",
                "message": f"Deleted {total_deleted} chunks from {len(request.file_paths)} files",
                "deleted_count": total_deleted
            }

        elif request.filters:
            # Delete by metadata filters
            # First find matching chunks
            matching_chunks = vector_store.search_by_metadata(request.filters, limit=10000)
            chunk_ids = [chunk.chunk_id for chunk in matching_chunks]

            if chunk_ids:
                result = vector_store.client.delete_chunks(chunk_ids)
                return {
                    "status": "success",
                    "message": f"Deleted {result.successful_items} chunks matching filters",
                    "deleted_count": result.successful_items
                }
            else:
                return {
                    "status": "success",
                    "message": "No chunks found matching filters",
                    "deleted_count": 0
                }

        else:
            raise HTTPException(status_code=400, detail="Either file_paths or filters must be provided")

    except Exception as e:
        logger.error("Chunk deletion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@admin_router.post("/reset", response_model=Dict[str, Any])
async def reset_database(
    confirm: bool = Query(..., description="Confirmation flag"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Reset the entire database (vector store and project metadata)"""
    if not confirm:
        raise HTTPException(status_code=400, detail="Reset operation must be confirmed")

    try:
        logger.warning("Resetting entire database (vector store + projects)")

        # Reset vector store (ChromaDB)
        vector_success = vector_store.reset_database()

        # Reset project repository
        from ..database.project_repository import ProjectRepository
        project_repo = ProjectRepository()
        project_success = project_repo.reset_all()

        if vector_success and project_success:
            return {
                "status": "success",
                "message": "Database reset successfully (vector store + projects)",
                "timestamp": time.time()
            }
        else:
            error_details = []
            if not vector_success:
                error_details.append("vector store reset failed")
            if not project_success:
                error_details.append("project reset failed")
            raise HTTPException(status_code=500, detail=f"Reset failed: {', '.join(error_details)}")

    except Exception as e:
        logger.error("Database reset failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# Project Routes
@projects_router.get("/", response_model=ProjectListResponse)
async def list_projects(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get list of all projects from the project repository"""
    try:
        start_time = time.time()

        logger.info("Listing all projects")

        # Get projects from ProjectRepository
        project_repo = ProjectRepository()
        all_projects = project_repo.get_all()

        # Build project list with chunk counts from project metadata
        projects = []
        for project in all_projects:
            projects.append(ProjectInfo(
                id=project.project_id,
                name=project.name,
                chunk_count=project.total_chunks
            ))

        logger.info("Projects listed", total_projects=len(projects))

        return ProjectListResponse(
            status=RequestStatus.SUCCESS,
            message=f"Found {len(projects)} projects",
            projects=projects,
            total_projects=len(projects)
        )

    except Exception as e:
        logger.error("Failed to list projects", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")


@projects_router.get("/{project_id}/stats", response_model=ProjectStatsResponse)
async def get_project_statistics(
    project_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get detailed statistics for a specific project"""
    try:
        start_time = time.time()
        cache_manager = get_cache_manager()

        # Check cache first
        cached_stats = cache_manager.get_project_stats(project_id)
        if cached_stats is not None:
            logger.info("Returning cached project stats",
                       project_id=project_id,
                       cached=True)
            return cached_stats

        logger.info("Retrieving project stats", project_id=project_id)

        # First check if project exists in ProjectRepository
        project_repo = ProjectRepository()
        project = project_repo.get(project_id)

        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project '{project_id}' not found"
            )

        # Get stats from vector store
        stats = vector_store.get_project_stats(project_id)

        # If no chunks exist yet, return default stats with project info from repository
        if 'error' in stats:
            logger.info("No chunks found for project, returning default stats",
                       project_id=project_id)
            response = ProjectStatsResponse(
                status=RequestStatus.SUCCESS,
                project_id=project.project_id,
                project_name=project.name,
                total_chunks=0,
                total_files=0,
                total_tokens=0,
                avg_tokens_per_chunk=0.0,
                languages={},
                layer_types={},
                last_updated=project.updated_at.timestamp()
            )
            # Cache the stats
            cache_manager.set_project_stats(project_id, response)
            return response

        logger.info("Project stats retrieved",
                    project_id=project_id,
                    total_chunks=stats.get('total_chunks', 0))

        response = ProjectStatsResponse(
            status=RequestStatus.SUCCESS,
            project_id=stats['project_id'],
            project_name=stats['project_name'],
            total_chunks=stats['total_chunks'],
            total_files=stats['total_files'],
            total_tokens=stats['total_tokens'],
            avg_tokens_per_chunk=stats['avg_tokens_per_chunk'],
            languages=stats['languages'],
            layer_types=stats['layer_types'],
            last_updated=stats['last_updated']
        )

        # Cache the stats
        cache_manager.set_project_stats(project_id, response)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get project stats",
                     project_id=project_id,
                     error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get project stats: {str(e)}")


@projects_router.post("/", response_model=ProjectCreateResponse, status_code=201)
async def create_project(
    request: ProjectCreateRequest
):
    """Create a new project and generate unique project ID"""
    try:
        logger.info("Creating new project", name=request.name)

        # Initialize project repository
        project_repo = ProjectRepository()

        # Generate project ID
        project_id = Project.generate_id()

        # Create project object
        project = Project(
            project_id=project_id,
            name=request.name,
            repository_path=request.repository_path,
            description=request.description,
            git_remote_url=request.git_remote_url,
            git_branch=request.git_branch,
            status=ProjectStatus.INITIALIZING
        )

        # Save to database
        created_project = project_repo.create(project)

        logger.info("Project created successfully",
                    project_id=created_project.project_id,
                    name=created_project.name)

        return ProjectCreateResponse(
            message=f"Project '{request.name}' created successfully",
            project_id=created_project.project_id,
            name=created_project.name,
            repository_path=created_project.repository_path,
            git_remote_url=created_project.git_remote_url,
            git_branch=created_project.git_branch,
            created_at=created_project.created_at.isoformat()
        )

    except ValueError as e:
        logger.error("Project creation failed - validation error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create project", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")


@projects_router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project_details(
    project_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get detailed information about a specific project"""
    try:
        logger.info("Retrieving project details", project_id=project_id)

        # Get from project repository
        project_repo = ProjectRepository()
        project = project_repo.get(project_id)

        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project '{project_id}' not found"
            )

        # Get latest stats from vector store
        stats = vector_store.get_project_stats(project_id)
        if 'error' not in stats:
            # Update project with latest counts
            project.total_chunks = stats.get('total_chunks', 0)
            project.total_files = stats.get('total_files', 0)

        logger.info("Project details retrieved", project_id=project_id)

        return ProjectDetailResponse(
            project_id=project.project_id,
            name=project.name,
            repository_path=project.repository_path,
            description=project.description,
            git_remote_url=project.git_remote_url,
            git_branch=project.git_branch,
            project_status=project.status.value,
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat(),
            total_chunks=project.total_chunks,
            total_files=project.total_files,
            last_indexed_at=project.last_indexed_at.isoformat() if project.last_indexed_at else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get project details",
                     project_id=project_id,
                     error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get project details: {str(e)}")


@projects_router.put("/{project_id}", response_model=ProjectUpdateResponse)
async def update_project(
    project_id: str,
    request: ProjectUpdateRequest
):
    """Update project information"""
    try:
        logger.info("Updating project", project_id=project_id)

        # Initialize project repository
        project_repo = ProjectRepository()

        # Check if project exists
        if not project_repo.exists(project_id):
            raise HTTPException(
                status_code=404,
                detail=f"Project '{project_id}' not found"
            )

        # Prepare updates
        updates = {}
        if request.name is not None:
            updates['name'] = request.name
        if request.repository_path is not None:
            updates['repository_path'] = request.repository_path
        if request.description is not None:
            updates['description'] = request.description
        if request.git_remote_url is not None:
            updates['git_remote_url'] = request.git_remote_url
        if request.git_branch is not None:
            updates['git_branch'] = request.git_branch
        if request.status is not None:
            updates['status'] = request.status

        # Update project
        updated_project = project_repo.update(project_id, updates)

        if not updated_project:
            raise HTTPException(
                status_code=404,
                detail=f"Project '{project_id}' not found"
            )

        logger.info("Project updated successfully", project_id=project_id)

        # Return updated project details
        project_detail = ProjectDetailResponse(
            project_id=updated_project.project_id,
            name=updated_project.name,
            repository_path=updated_project.repository_path,
            description=updated_project.description,
            git_remote_url=updated_project.git_remote_url,
            git_branch=updated_project.git_branch,
            project_status=updated_project.status.value,
            created_at=updated_project.created_at.isoformat(),
            updated_at=updated_project.updated_at.isoformat(),
            total_chunks=updated_project.total_chunks,
            total_files=updated_project.total_files,
            last_indexed_at=updated_project.last_indexed_at.isoformat()
                if updated_project.last_indexed_at else None
        )

        return ProjectUpdateResponse(
            message=f"Project '{project_id}' updated successfully",
            project=project_detail
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update project",
                     project_id=project_id,
                     error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")


@projects_router.delete("/{project_id}", response_model=ProjectDeleteResponse)
async def delete_project(
    project_id: str,
    delete_chunks: bool = Query(False, description="Also delete all chunks belonging to this project"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Delete a project and optionally its embeddings"""
    try:
        logger.info("Deleting project", project_id=project_id, delete_chunks=delete_chunks)

        # Initialize project repository
        project_repo = ProjectRepository()

        # Check if project exists
        if not project_repo.exists(project_id):
            raise HTTPException(
                status_code=404,
                detail=f"Project '{project_id}' not found"
            )

        chunks_deleted = 0

        # Delete chunks if requested
        if delete_chunks:
            logger.info("Deleting chunks for project", project_id=project_id)
            try:
                # Delete chunks from vector store
                result = vector_store.delete_by_metadata({"project_id": project_id})
                chunks_deleted = result.successful_items
                logger.info("Chunks deleted", project_id=project_id, count=chunks_deleted)
            except Exception as e:
                logger.error("Failed to delete chunks", project_id=project_id, error=str(e))
                # Continue with project deletion even if chunk deletion fails

        # Delete project from repository
        deleted = project_repo.delete(project_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Project '{project_id}' not found"
            )

        logger.info("Project deleted successfully",
                    project_id=project_id,
                    chunks_deleted=chunks_deleted)

        return ProjectDeleteResponse(
            message=f"Project '{project_id}' deleted successfully",
            project_id=project_id,
            chunks_deleted=chunks_deleted
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete project",
                     project_id=project_id,
                     error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")


# Include all routers
all_routers = [
    main_router,
    search_router,
    process_router,
    status_router,
    admin_router,
    projects_router
]
