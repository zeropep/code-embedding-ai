import chromadb
from chromadb.config import Settings
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
import structlog

from .models import (VectorDBConfig, ChunkMetadata, StoredChunk, VectorSearchResult,
                     VectorSearchQuery, DatabaseStats, BulkOperationResult, VectorDBStatus)


logger = structlog.get_logger(__name__)


class ChromaDBClient:
    """Client for ChromaDB vector database operations"""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[chromadb.Collection] = None
        self.status = VectorDBStatus.DISCONNECTED

        logger.info("ChromaDBClient initialized",
                   host=config.host,
                   port=config.port,
                   collection=config.collection_name)

    def connect(self) -> bool:
        """Connect to ChromaDB"""
        try:
            self.status = VectorDBStatus.INITIALIZING
            logger.info("Connecting to ChromaDB",
                       host=self.config.host,
                       port=self.config.port)

            if self.config.persistent:
                # Persistent client with disk storage
                self.client = chromadb.PersistentClient(
                    path=self.config.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            else:
                # HTTP client for remote ChromaDB
                self.client = chromadb.HttpClient(
                    host=self.config.host,
                    port=self.config.port,
                    settings=Settings(
                        anonymized_telemetry=False
                    )
                )

            # Get or create collection
            self.collection = self._get_or_create_collection()
            self.status = VectorDBStatus.CONNECTED

            logger.info("Connected to ChromaDB successfully",
                       collection_name=self.config.collection_name)
            return True

        except Exception as e:
            self.status = VectorDBStatus.ERROR
            logger.error("Failed to connect to ChromaDB", error=str(e))
            return False

    def disconnect(self):
        """Disconnect from ChromaDB"""
        try:
            if self.client:
                # ChromaDB doesn't have explicit disconnect method
                # Just clear references
                self.client = None
                self.collection = None
                self.status = VectorDBStatus.DISCONNECTED
                logger.info("Disconnected from ChromaDB")
        except Exception as e:
            logger.error("Error during disconnect", error=str(e))

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.config.collection_name
            )
            logger.info("Retrieved existing collection",
                       collection_name=self.config.collection_name)
            return collection

        except Exception:
            # Collection doesn't exist, create it
            logger.info("Creating new collection",
                       collection_name=self.config.collection_name)

            metadata = {
                "hnsw:space": "cosine",  # Use cosine similarity
                "description": "Code embeddings for semantic search",
                "created_at": time.time(),
                "embedding_dimensions": self.config.embedding_function
            }

            collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata=metadata
            )
            return collection

    def store_chunks(self, chunks: List[StoredChunk]) -> BulkOperationResult:
        """Store multiple chunks in the database"""
        if not self._ensure_connected():
            return BulkOperationResult(
                operation_type="insert",
                total_items=len(chunks),
                successful_items=0,
                failed_items=len(chunks),
                processing_time=0.0,
                errors=["Database not connected"]
            )

        start_time = time.time()
        successful = 0
        errors = []

        logger.info("Storing chunks", count=len(chunks))

        try:
            # Process in batches
            for i in range(0, len(chunks), self.config.max_batch_size):
                batch = chunks[i:i + self.config.max_batch_size]
                batch_success = self._store_batch(batch)
                successful += batch_success

                if batch_success < len(batch):
                    failed_count = len(batch) - batch_success
                    errors.append(f"Failed to store {failed_count} chunks in batch {i//self.config.max_batch_size}")

        except Exception as e:
            logger.error("Bulk store operation failed", error=str(e))
            errors.append(str(e))

        processing_time = time.time() - start_time
        failed = len(chunks) - successful

        result = BulkOperationResult(
            operation_type="insert",
            total_items=len(chunks),
            successful_items=successful,
            failed_items=failed,
            processing_time=processing_time,
            errors=errors
        )

        logger.info("Chunk storage completed",
                   total=len(chunks),
                   successful=successful,
                   failed=failed,
                   processing_time=processing_time)

        return result

    def _store_batch(self, chunks: List[StoredChunk]) -> int:
        """Store a batch of chunks"""
        try:
            ids = [chunk.chunk_id for chunk in chunks]
            embeddings = [chunk.embedding_vector for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata.to_dict() for chunk in chunks]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.debug("Batch stored successfully", batch_size=len(chunks))
            return len(chunks)

        except Exception as e:
            logger.error("Failed to store batch", error=str(e), batch_size=len(chunks))
            return 0

    def search_similar(self, query: VectorSearchQuery) -> List[VectorSearchResult]:
        """Search for similar chunks"""
        if not self._ensure_connected():
            logger.error("Database not connected for search")
            return []

        try:
            start_time = time.time()

            # Prepare query
            if query.query_vector:
                query_embeddings = [query.query_vector]
            else:
                # If only text is provided, we need to embed it first
                # This would require integration with embedding service
                logger.error("Text-only queries not implemented yet")
                return []

            # Perform search
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=min(query.top_k, self.config.max_top_k),
                where=query.filters,
                include=["metadatas", "documents", "distances"]
            )

            # Convert to VectorSearchResult objects
            search_results = self._convert_search_results(results, query.min_similarity)

            search_time = (time.time() - start_time) * 1000  # Convert to ms
            logger.debug("Search completed",
                        results_count=len(search_results),
                        search_time_ms=search_time)

            return search_results

        except Exception as e:
            logger.error("Search failed", error=str(e))
            return []

    def _convert_search_results(self, raw_results: Dict, min_similarity: float) -> List[VectorSearchResult]:
        """Convert ChromaDB results to VectorSearchResult objects"""
        results = []

        if not raw_results.get('ids') or not raw_results['ids'][0]:
            return results

        ids = raw_results['ids'][0]
        distances = raw_results.get('distances', [[]])[0]
        metadatas = raw_results.get('metadatas', [[]])[0]
        documents = raw_results.get('documents', [[]])[0]

        for i, chunk_id in enumerate(ids):
            # Convert distance to similarity (ChromaDB returns distances)
            distance = distances[i] if i < len(distances) else 1.0
            similarity = 1.0 - distance  # Assuming cosine distance

            if similarity >= min_similarity:
                metadata = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else ""

                result = VectorSearchResult(
                    chunk_id=chunk_id,
                    similarity_score=similarity,
                    metadata=metadata,
                    content=content
                )
                results.append(result)

        return results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[StoredChunk]:
        """Retrieve a specific chunk by ID"""
        if not self._ensure_connected():
            return None

        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["metadatas", "documents", "embeddings"]
            )

            if results['ids'] and results['ids'][0]:
                metadata_dict = results['metadatas'][0] if results['metadatas'] else {}
                content = results['documents'][0] if results['documents'] else ""
                embedding = results['embeddings'][0] if results['embeddings'] else []

                metadata = ChunkMetadata.from_dict(metadata_dict)

                return StoredChunk(
                    chunk_id=chunk_id,
                    content=content,
                    embedding_vector=embedding,
                    metadata=metadata
                )

        except Exception as e:
            logger.error("Failed to retrieve chunk", chunk_id=chunk_id, error=str(e))

        return None

    def delete_chunks(self, chunk_ids: List[str]) -> BulkOperationResult:
        """Delete multiple chunks"""
        if not self._ensure_connected():
            return BulkOperationResult(
                operation_type="delete",
                total_items=len(chunk_ids),
                successful_items=0,
                failed_items=len(chunk_ids),
                processing_time=0.0,
                errors=["Database not connected"]
            )

        start_time = time.time()
        successful = 0
        errors = []

        logger.info("Deleting chunks", count=len(chunk_ids))

        try:
            # Process in batches
            for i in range(0, len(chunk_ids), self.config.max_batch_size):
                batch_ids = chunk_ids[i:i + self.config.max_batch_size]

                try:
                    self.collection.delete(ids=batch_ids)
                    successful += len(batch_ids)
                    logger.debug("Batch deleted successfully", batch_size=len(batch_ids))
                except Exception as e:
                    errors.append(f"Failed to delete batch: {str(e)}")

        except Exception as e:
            logger.error("Bulk delete operation failed", error=str(e))
            errors.append(str(e))

        processing_time = time.time() - start_time
        failed = len(chunk_ids) - successful

        return BulkOperationResult(
            operation_type="delete",
            total_items=len(chunk_ids),
            successful_items=successful,
            failed_items=failed,
            processing_time=processing_time,
            errors=errors
        )

    def get_statistics(self) -> DatabaseStats:
        """Get database statistics"""
        if not self._ensure_connected():
            return DatabaseStats()

        try:
            # Get collection count
            count_result = self.collection.count()
            total_chunks = count_result if isinstance(count_result, int) else 0

            # Get some sample data to analyze
            sample_size = min(1000, total_chunks)
            if sample_size > 0:
                sample_results = self.collection.get(
                    limit=sample_size,
                    include=["metadatas"]
                )

                # Analyze metadata
                language_counts = {}
                layer_counts = {}
                sensitivity_counts = {}
                file_paths = set()

                if sample_results.get('metadatas'):
                    for metadata in sample_results['metadatas']:
                        # Language distribution
                        lang = metadata.get('language', 'unknown')
                        language_counts[lang] = language_counts.get(lang, 0) + 1

                        # Layer distribution
                        layer = metadata.get('layer_type', 'unknown')
                        layer_counts[layer] = layer_counts.get(layer, 0) + 1

                        # Sensitivity distribution
                        sensitivity = metadata.get('sensitivity_level', 'LOW')
                        sensitivity_counts[sensitivity] = sensitivity_counts.get(sensitivity, 0) + 1

                        # Unique files
                        file_path = metadata.get('file_path')
                        if file_path:
                            file_paths.add(file_path)

                # Extrapolate to full dataset
                if sample_size < total_chunks:
                    scale_factor = total_chunks / sample_size
                    for counts_dict in [language_counts, layer_counts, sensitivity_counts]:
                        for key in counts_dict:
                            counts_dict[key] = int(counts_dict[key] * scale_factor)

                total_files = int(len(file_paths) * (total_chunks / sample_size)) if sample_size < total_chunks else len(file_paths)

            else:
                language_counts = {}
                layer_counts = {}
                sensitivity_counts = {}
                total_files = 0

            stats = DatabaseStats(
                total_chunks=total_chunks,
                total_files=total_files,
                language_counts=language_counts,
                layer_counts=layer_counts,
                sensitivity_counts=sensitivity_counts
            )

            logger.debug("Statistics gathered",
                        total_chunks=total_chunks,
                        total_files=total_files)

            return stats

        except Exception as e:
            logger.error("Failed to get statistics", error=str(e))
            return DatabaseStats()

    def reset_collection(self) -> bool:
        """Reset (delete and recreate) the collection"""
        if not self.client:
            return False

        try:
            logger.warning("Resetting collection", collection_name=self.config.collection_name)

            # Delete existing collection
            try:
                self.client.delete_collection(name=self.config.collection_name)
            except Exception:
                pass  # Collection might not exist

            # Create new collection
            self.collection = self._get_or_create_collection()

            logger.info("Collection reset successfully")
            return True

        except Exception as e:
            logger.error("Failed to reset collection", error=str(e))
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            "status": self.status.value,
            "connected": self._ensure_connected(),
            "collection_exists": False,
            "can_read": False,
            "can_write": False
        }

        if self.collection:
            try:
                # Check if collection exists and is accessible
                count = self.collection.count()
                health["collection_exists"] = True
                health["can_read"] = True
                health["total_chunks"] = count

                # Test write capability with a dummy entry
                test_id = f"health_check_{int(time.time())}"
                try:
                    self.collection.add(
                        ids=[test_id],
                        embeddings=[[0.0] * 1024],
                        documents=["health check"],
                        metadatas=[{"test": True}]
                    )
                    health["can_write"] = True

                    # Clean up test entry
                    self.collection.delete(ids=[test_id])

                except Exception as write_error:
                    health["write_error"] = str(write_error)

            except Exception as e:
                health["error"] = str(e)

        return health

    def _ensure_connected(self) -> bool:
        """Ensure database is connected"""
        if self.status != VectorDBStatus.CONNECTED or not self.collection:
            return self.connect()
        return True