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
    ProjectListResponse, ProjectStatsResponse, ProjectInfo
)
from .dependencies import get_embedding_pipeline, get_update_service, get_vector_store
from ..embeddings.embedding_pipeline import EmbeddingPipeline
from ..updates.update_service import UpdateService
from ..updates.models import UpdateRequest
from ..database.vector_store import VectorStore


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

        logger.info("Semantic search request",
                    query=request.query,
                    top_k=request.top_k)

        # Generate embedding for query
        await pipeline.embedding_service.start()
        try:
            # Create a dummy code chunk for embedding generation
            from ..code_parser.models import CodeChunk, CodeLanguage, LayerType

            query_chunk = CodeChunk(
                content=request.query,
                file_path="query",
                language=CodeLanguage.JAVA,  # Default language
                start_line=1,
                end_line=1,
                layer_type=LayerType.UNKNOWN,
                metadata={}
            )

            embedded_chunks = await pipeline.embedding_service.generate_chunk_embeddings([query_chunk])

            if not embedded_chunks or 'embedding' not in embedded_chunks[0].metadata:
                raise HTTPException(status_code=500, detail="Failed to generate query embedding")

            query_vector = embedded_chunks[0].metadata['embedding']['vector']

        finally:
            await pipeline.embedding_service.stop()

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

        return SearchResponse(
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
        try:
            from ..code_parser.models import CodeChunk, CodeLanguage, LayerType

            # Determine language
            language_map = {
                "java": CodeLanguage.JAVA,
                "kotlin": CodeLanguage.KOTLIN,
                "html": CodeLanguage.HTML,
                "xml": CodeLanguage.XML
            }

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

        finally:
            await pipeline.embedding_service.stop()

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
            result = await pipeline.process_repository(request.repo_path)
        elif request.mode == ProcessingMode.INCREMENTAL:
            # Incremental processing would need UpdateService integration
            raise HTTPException(status_code=501, detail="Incremental mode not implemented in this endpoint")
        else:  # AUTO mode
            # Auto-detect best processing mode
            result = await pipeline.process_repository(request.repo_path)

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

        result = await pipeline.process_files(request.file_paths)

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
    update_service: UpdateService = Depends(get_update_service)
):
    """Trigger an incremental update"""
    try:
        request_id = str(uuid.uuid4())

        update_request = UpdateRequest(
            request_id=request_id,
            repo_path=repo_path or update_service.repo_path,
            force_full_update=force_full
        )

        result = await update_service.request_update(update_request)

        return {
            "status": "success",
            "message": "Update completed",
            "result": result.to_dict()
        }

    except Exception as e:
        logger.error("Update failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


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
    """Reset the entire vector database"""
    if not confirm:
        raise HTTPException(status_code=400, detail="Reset operation must be confirmed")

    try:
        logger.warning("Resetting vector database")
        success = vector_store.reset_database()

        if success:
            return {
                "status": "success",
                "message": "Database reset successfully",
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=500, detail="Database reset failed")

    except Exception as e:
        logger.error("Database reset failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# Project Routes
@projects_router.get("/", response_model=ProjectListResponse)
async def list_projects(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get list of all projects in the database"""
    try:
        start_time = time.time()

        logger.info("Listing all projects")

        projects_data = vector_store.get_all_projects()

        # Convert to ProjectInfo models
        projects = [
            ProjectInfo(
                id=project['id'],
                name=project['name'],
                chunk_count=project['chunk_count']
            )
            for project in projects_data
        ]

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

        logger.info("Retrieving project stats", project_id=project_id)

        stats = vector_store.get_project_stats(project_id)

        # Check if project exists
        if 'error' in stats:
            raise HTTPException(
                status_code=404,
                detail=f"Project not found: {stats.get('error', 'Unknown error')}"
            )

        logger.info("Project stats retrieved",
                    project_id=project_id,
                    total_chunks=stats.get('total_chunks', 0))

        return ProjectStatsResponse(
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get project stats",
                     project_id=project_id,
                     error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get project stats: {str(e)}")


# Include all routers
all_routers = [
    main_router,
    search_router,
    process_router,
    status_router,
    admin_router,
    projects_router
]
