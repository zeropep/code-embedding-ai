import uuid
from typing import List, Dict, Any, Optional
import structlog

from .chroma_client import ChromaDBClient
from .models import (VectorDBConfig, StoredChunk, ChunkMetadata, VectorSearchResult,
                     VectorSearchQuery, DatabaseStats, BulkOperationResult)
from ..code_parser.models import CodeChunk


logger = structlog.get_logger(__name__)


class VectorStore:
    """High-level interface for vector storage operations"""

    def __init__(self, config: VectorDBConfig = None):
        if config is None:
            config = VectorDBConfig()

        if not config.validate():
            raise ValueError("Invalid vector database configuration")

        self.config = config
        self.client = ChromaDBClient(config)
        self._is_connected = False

        logger.info("VectorStore initialized",
                    collection_name=config.collection_name,
                    persistent=config.persistent)

    def connect(self) -> bool:
        """Connect to the vector database"""
        try:
            success = self.client.connect()
            self._is_connected = success
            if success:
                logger.info("VectorStore connected successfully")
            else:
                logger.error("Failed to connect VectorStore")
            return success
        except Exception as e:
            logger.error("VectorStore connection error", error=str(e))
            self._is_connected = False
            return False

    def disconnect(self):
        """Disconnect from the vector database"""
        try:
            self.client.disconnect()
            self._is_connected = False
            logger.info("VectorStore disconnected")
        except Exception as e:
            logger.error("Error during VectorStore disconnect", error=str(e))

    def store_chunks(self, chunks: List[CodeChunk]) -> BulkOperationResult:
        """Store code chunks with embeddings in the database"""
        if not self._ensure_connected():
            return BulkOperationResult(
                operation_type="store",
                total_items=len(chunks),
                successful_items=0,
                failed_items=len(chunks),
                processing_time=0.0,
                errors=["Database not connected"]
            )

        logger.info("Storing code chunks", count=len(chunks))

        # Convert CodeChunk objects to StoredChunk objects
        stored_chunks = []
        conversion_errors = []

        for chunk in chunks:
            try:
                stored_chunk = self._convert_to_stored_chunk(chunk)
                if stored_chunk:
                    stored_chunks.append(stored_chunk)
                else:
                    conversion_errors.append(f"Failed to convert chunk from {chunk.file_path}")
            except Exception as e:
                conversion_errors.append(f"Error converting chunk: {str(e)}")

        if conversion_errors:
            logger.warning("Some chunks failed conversion",
                           failed_count=len(conversion_errors),
                           errors=conversion_errors[:5])  # Log first 5 errors

        if not stored_chunks:
            return BulkOperationResult(
                operation_type="store",
                total_items=len(chunks),
                successful_items=0,
                failed_items=len(chunks),
                processing_time=0.0,
                errors=["No valid chunks to store"] + conversion_errors
            )

        # Store in database
        result = self.client.store_chunks(stored_chunks)

        # Add conversion errors to result
        if conversion_errors:
            result.errors.extend(conversion_errors)
            result.failed_items += len(conversion_errors)

        logger.info("Chunk storage completed",
                    total_requested=len(chunks),
                    converted=len(stored_chunks),
                    stored=result.successful_items,
                    failed=result.failed_items)

        return result

    def search_similar_chunks(self, query_vector: List[float],
                              top_k: int = 10,
                              min_similarity: float = 0.0,
                              filters: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar code chunks"""
        if not self._ensure_connected():
            logger.error("Database not connected for search")
            return []

        query = VectorSearchQuery(
            query_vector=query_vector,
            top_k=min(top_k, self.config.max_top_k),
            min_similarity=min_similarity,
            filters=filters
        )

        logger.debug("Searching similar chunks",
                     top_k=query.top_k,
                     min_similarity=min_similarity,
                     has_filters=filters is not None)

        results = self.client.search_similar(query)

        logger.debug("Search completed", results_count=len(results))
        return results

    def search_by_metadata(self, filters: Dict[str, Any],
                           limit: int = 100) -> List[VectorSearchResult]:
        """Search chunks by metadata filters only"""
        if not self._ensure_connected():
            return []

        try:
            # Use ChromaDB's get method for metadata-only queries
            results = self.client.collection.get(
                where=filters,
                limit=min(limit, self.config.max_top_k),
                include=["metadatas", "documents"]
            )

            # Convert to VectorSearchResult objects
            search_results = []
            if results.get('ids'):
                for i, chunk_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                    content = results['documents'][i] if i < len(results['documents']) else ""

                    result = VectorSearchResult(
                        chunk_id=chunk_id,
                        similarity_score=1.0,  # No similarity for metadata-only search
                        metadata=metadata,
                        content=content
                    )
                    search_results.append(result)

            logger.debug("Metadata search completed", results_count=len(search_results))
            return search_results

        except Exception as e:
            logger.error("Metadata search failed", error=str(e))
            return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[StoredChunk]:
        """Retrieve a specific chunk by ID"""
        if not self._ensure_connected():
            return None

        return self.client.get_chunk_by_id(chunk_id)

    def delete_by_metadata(self, filters: Dict[str, Any]) -> BulkOperationResult:
        """Delete all chunks matching metadata filters"""
        if not self._ensure_connected():
            return BulkOperationResult(
                operation_type="delete",
                total_items=0,
                successful_items=0,
                failed_items=0,
                processing_time=0.0,
                errors=["Database not connected"]
            )

        logger.info("Deleting chunks by metadata", filters=filters)

        try:
            # Find all chunks matching the filters
            matching_chunks = self.search_by_metadata(filters, limit=10000)
            chunk_ids = [chunk.chunk_id for chunk in matching_chunks]

            if not chunk_ids:
                logger.info("No chunks found matching filters", filters=filters)
                return BulkOperationResult(
                    operation_type="delete",
                    total_items=0,
                    successful_items=0,
                    failed_items=0,
                    processing_time=0.0
                )

            # Delete the chunks
            result = self.client.delete_chunks(chunk_ids)
            logger.info("Chunks deleted by metadata",
                        filters=filters,
                        chunks_deleted=result.successful_items)

            return result

        except Exception as e:
            logger.error("Failed to delete chunks by metadata",
                         filters=filters,
                         error=str(e))
            return BulkOperationResult(
                operation_type="delete",
                total_items=0,
                successful_items=0,
                failed_items=0,
                processing_time=0.0,
                errors=[str(e)]
            )

    def delete_chunks_by_file(self, file_path: str) -> BulkOperationResult:
        """Delete all chunks from a specific file"""
        if not self._ensure_connected():
            return BulkOperationResult(
                operation_type="delete",
                total_items=0,
                successful_items=0,
                failed_items=0,
                processing_time=0.0,
                errors=["Database not connected"]
            )

        logger.info("Deleting chunks by file", file_path=file_path)

        try:
            # Find all chunks for the file
            file_chunks = self.search_by_metadata({"file_path": file_path})
            chunk_ids = [chunk.chunk_id for chunk in file_chunks]

            if not chunk_ids:
                logger.info("No chunks found for file", file_path=file_path)
                return BulkOperationResult(
                    operation_type="delete",
                    total_items=0,
                    successful_items=0,
                    failed_items=0,
                    processing_time=0.0
                )

            # Delete the chunks
            result = self.client.delete_chunks(chunk_ids)
            logger.info("File chunks deleted",
                        file_path=file_path,
                        chunks_deleted=result.successful_items)

            return result

        except Exception as e:
            logger.error("Failed to delete file chunks",
                         file_path=file_path,
                         error=str(e))
            return BulkOperationResult(
                operation_type="delete",
                total_items=0,
                successful_items=0,
                failed_items=0,
                processing_time=0.0,
                errors=[str(e)]
            )

    def update_chunk(self, chunk: CodeChunk) -> bool:
        """Update an existing chunk"""
        if not self._ensure_connected():
            return False

        try:
            # Convert to stored chunk
            stored_chunk = self._convert_to_stored_chunk(chunk)
            if not stored_chunk:
                return False

            # Delete old version and insert new
            delete_result = self.client.delete_chunks([stored_chunk.chunk_id])
            if delete_result.successful_items > 0:
                store_result = self.client.store_chunks([stored_chunk])
                return store_result.successful_items > 0

            return False

        except Exception as e:
            logger.error("Failed to update chunk", error=str(e))
            return False

    def get_statistics(self) -> DatabaseStats:
        """Get database statistics"""
        if not self._ensure_connected():
            return DatabaseStats()

        return self.client.get_statistics()

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            "vector_store_status": "healthy",
            "database_health": {},
            "configuration": {
                "collection_name": self.config.collection_name,
                "persistent": self.config.persistent,
                "max_batch_size": self.config.max_batch_size
            }
        }

        try:
            # Check database health
            health["database_health"] = self.client.health_check()

            # Check if we can perform basic operations
            if self._ensure_connected():
                stats = self.get_statistics()
                health["total_chunks"] = stats.total_chunks
                health["total_files"] = stats.total_files
            else:
                health["vector_store_status"] = "unhealthy"
                health["error"] = "Cannot connect to database"

        except Exception as e:
            health["vector_store_status"] = "unhealthy"
            health["error"] = str(e)
            logger.error("VectorStore health check failed", error=str(e))

        return health

    def reset_database(self) -> bool:
        """Reset the entire database (delete all data)"""
        logger.warning("Resetting vector database - all data will be lost")

        if not self._ensure_connected():
            return False

        return self.client.reset_collection()

    def _convert_to_stored_chunk(self, chunk: CodeChunk) -> Optional[StoredChunk]:
        """Convert CodeChunk to StoredChunk"""
        try:
            # Check if chunk has embedding
            if 'embedding' not in chunk.metadata or not chunk.metadata['embedding'].get('vector'):
                logger.warning("Chunk missing embedding vector",
                               file_path=chunk.file_path,
                               function_name=chunk.function_name)
                return None

            embedding_data = chunk.metadata['embedding']
            embedding_vector = embedding_data['vector']

            # Validate embedding vector
            if not isinstance(embedding_vector, list) or len(embedding_vector) == 0:
                logger.warning("Invalid embedding vector type",
                               file_path=chunk.file_path,
                               vector_type=type(embedding_vector).__name__)
                return None

            # Generate unique chunk ID if not present
            chunk_id = embedding_data.get('embedding_id', str(uuid.uuid4()))

            # Create metadata object (filter out None values for ChromaDB compatibility)
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                file_path=chunk.file_path,
                language=chunk.language.value,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                function_name=chunk.function_name or "",
                class_name=chunk.class_name or "",
                layer_type=chunk.layer_type.value,
                token_count=chunk.token_count,
                sensitivity_level=chunk.metadata.get('security', {}).get('sensitivity_level', 'LOW'),
                file_hash=chunk.metadata.get('file_hash', ""),
                embedding_model=embedding_data.get('model_version', 'jina-embeddings-v2-base-code'),
                embedding_dimensions=len(embedding_vector),
                project_id=chunk.project_id or "default",
                project_name=chunk.project_name or "default"
            )

            return StoredChunk(
                chunk_id=chunk_id,
                content=chunk.content,
                embedding_vector=embedding_vector,
                metadata=metadata
            )

        except Exception as e:
            logger.error("Failed to convert chunk to stored chunk",
                         file_path=chunk.file_path,
                         error=str(e))
            return None

    def _ensure_connected(self) -> bool:
        """Ensure database connection"""
        if not self._is_connected:
            return self.connect()
        return True

    def search_by_function_name(self, function_name: str,
                                top_k: int = 10) -> List[VectorSearchResult]:
        """Search chunks by function name"""
        filters = {"function_name": function_name}
        return self.search_by_metadata(filters, limit=top_k)

    def search_by_class_name(self, class_name: str,
                             top_k: int = 10) -> List[VectorSearchResult]:
        """Search chunks by class name"""
        filters = {"class_name": class_name}
        return self.search_by_metadata(filters, limit=top_k)

    def search_by_layer_type(self, layer_type: str,
                             top_k: int = 10) -> List[VectorSearchResult]:
        """Search chunks by layer type (Controller, Service, etc.)"""
        filters = {"layer_type": layer_type}
        return self.search_by_metadata(filters, limit=top_k)

    def search_by_sensitivity(self, sensitivity_level: str,
                              top_k: int = 10) -> List[VectorSearchResult]:
        """Search chunks by sensitivity level"""
        filters = {"sensitivity_level": sensitivity_level}
        return self.search_by_metadata(filters, limit=top_k)

    def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get list of all projects in the database"""
        if not self._ensure_connected():
            return []

        try:
            # Get all chunks and extract unique project_ids
            # This is not the most efficient way, but works for now
            # TODO: Optimize with direct ChromaDB query if possible
            all_results = self.client.collection.get(
                include=["metadatas"]
            )

            if not all_results.get('metadatas'):
                return []

            # Extract unique projects
            projects_dict = {}
            for metadata in all_results['metadatas']:
                project_id = metadata.get('project_id', 'default')
                project_name = metadata.get('project_name', 'default')

                if project_id not in projects_dict:
                    projects_dict[project_id] = {
                        'id': project_id,
                        'name': project_name,
                        'chunk_count': 0
                    }
                projects_dict[project_id]['chunk_count'] += 1

            projects = list(projects_dict.values())
            logger.debug("Retrieved projects", count=len(projects))
            return projects

        except Exception as e:
            logger.error("Failed to get projects", error=str(e))
            return []

    def get_project_stats(self, project_id: str) -> Dict[str, Any]:
        """Get statistics for a specific project"""
        if not self._ensure_connected():
            return {}

        try:
            # Special handling for default project: if project_id is "default",
            # get all chunks (including those without project_id metadata)
            if project_id == "default":
                # Get all chunks
                results = self.client.collection.get(
                    include=["metadatas"]
                )

                # Filter for chunks that either don't have project_id or have project_id="default"
                filtered_metadatas = []
                for metadata in results.get('metadatas', []):
                    chunk_project_id = metadata.get('project_id', 'default')
                    if chunk_project_id == 'default':
                        filtered_metadatas.append(metadata)

                results['metadatas'] = filtered_metadatas
            else:
                # Get chunks for specific project
                results = self.client.collection.get(
                    where={"project_id": project_id},
                    include=["metadatas"]
                )

            if not results.get('metadatas'):
                return {
                    'project_id': project_id,
                    'total_chunks': 0,
                    'total_files': 0,
                    'languages': {},
                    'layer_types': {},
                    'error': 'Project not found'
                }

            # Analyze metadata
            file_paths = set()
            languages = {}
            layer_types = {}
            total_tokens = 0
            project_name = 'Unknown'

            for metadata in results['metadatas']:
                # Count files
                file_path = metadata.get('file_path')
                if file_path:
                    file_paths.add(file_path)

                # Count languages
                lang = metadata.get('language', 'unknown')
                languages[lang] = languages.get(lang, 0) + 1

                # Count layer types
                layer = metadata.get('layer_type', 'Unknown')
                layer_types[layer] = layer_types.get(layer, 0) + 1

                # Sum tokens
                total_tokens += metadata.get('token_count', 0)

                # Get project name
                if metadata.get('project_name'):
                    project_name = metadata['project_name']

            stats = {
                'project_id': project_id,
                'project_name': project_name,
                'total_chunks': len(results['metadatas']),
                'total_files': len(file_paths),
                'total_tokens': total_tokens,
                'avg_tokens_per_chunk': total_tokens / len(results['metadatas']) if results['metadatas'] else 0,
                'languages': languages,
                'layer_types': layer_types,
                'last_updated': max((m.get('last_updated', 0) for m in results['metadatas']), default=0)
            }

            logger.debug("Retrieved project stats", project_id=project_id, total_chunks=stats['total_chunks'])
            return stats

        except Exception as e:
            logger.error("Failed to get project stats", project_id=project_id, error=str(e))
            return {'project_id': project_id, 'error': str(e)}
