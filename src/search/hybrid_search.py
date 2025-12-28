"""
Hybrid search service combining BM25 keyword search and vector semantic search
"""
from typing import List, Dict, Any, Optional, Tuple
import structlog

from .bm25_index import BM25Index
from ..database.vector_store import VectorStore
from ..database.models import VectorSearchResult
from ..embeddings.embedding_pipeline import EmbeddingPipeline

logger = structlog.get_logger(__name__)


class HybridSearchService:
    """
    Hybrid search combining BM25 (keyword) and vector (semantic) search
    using Reciprocal Rank Fusion (RRF) for result merging
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_pipeline: EmbeddingPipeline
    ):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline
        self.bm25_index = BM25Index()
        self._index_built = False

    async def ensure_index(self, project_id: Optional[str] = None) -> bool:
        """
        Ensure BM25 index is built

        Args:
            project_id: Optional project ID to build index for specific project

        Returns:
            True if index is ready, False otherwise
        """
        if self._index_built and self.bm25_index.is_built():
            logger.debug("BM25 index already built")
            return True

        try:
            logger.info("Building BM25 index", project_id=project_id)

            # Get all documents from vector store
            documents = self.vector_store.get_all_documents(project_id=project_id)

            if not documents:
                logger.warning("No documents found to build index",
                             project_id=project_id)
                return False

            # Build BM25 index
            self.bm25_index.build_index(documents)
            self._index_built = True

            logger.info("BM25 index built successfully",
                       document_count=len(documents))
            return True

        except Exception as e:
            logger.error("Failed to build BM25 index",
                        error=str(e),
                        project_id=project_id)
            return False

    async def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        project_id: Optional[str] = None,
        min_similarity: float = 0.4
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search

        Args:
            query: Search query string
            top_k: Number of results to return
            alpha: Weight for semantic search (0.0 = keyword only, 1.0 = semantic only)
            project_id: Optional project ID filter
            min_similarity: Minimum similarity threshold for vector search

        Returns:
            List of VectorSearchResult objects, ranked by hybrid score
        """
        try:
            logger.info("Hybrid search started",
                       query=query,
                       top_k=top_k,
                       alpha=alpha,
                       project_id=project_id)

            # Ensure BM25 index is built
            if not await self.ensure_index(project_id=project_id):
                logger.warning("BM25 index not available, falling back to semantic only")
                alpha = 1.0  # Force semantic-only search

            # Get BM25 results (keyword search)
            bm25_results = []
            if alpha < 1.0 and self.bm25_index.is_built():
                bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
                logger.debug("BM25 search completed", results_count=len(bm25_results))

            # Get semantic results (vector search)
            semantic_results = []
            if alpha > 0.0:
                # Generate query embedding
                await self.embedding_pipeline.embedding_service.start()

                from ..code_parser.models import CodeChunk, CodeLanguage, LayerType

                # Determine language from project if available
                query_language = CodeLanguage.JAVA  # Default
                if project_id:
                    from ..database.project_repository import ProjectRepository
                    project_repo = ProjectRepository()
                    project = project_repo.get(project_id)
                    if project and hasattr(project, 'primary_language') and project.primary_language:
                        language_map = {
                            "java": CodeLanguage.JAVA,
                            "python": CodeLanguage.PYTHON,
                            "kotlin": CodeLanguage.KOTLIN,
                            "html": CodeLanguage.HTML,
                            "unknown": CodeLanguage.JAVA,
                        }
                        query_language = language_map.get(project.primary_language, CodeLanguage.JAVA)

                query_chunk = CodeChunk(
                    content=query,
                    file_path="query",
                    language=query_language,
                    start_line=1,
                    end_line=1,
                    layer_type=LayerType.UNKNOWN,
                    metadata={}
                )

                embedded_chunks = await self.embedding_pipeline.embedding_service.generate_chunk_embeddings([query_chunk])

                if embedded_chunks and 'embedding' in embedded_chunks[0].metadata:
                    query_vector = embedded_chunks[0].metadata['embedding']['vector']

                    # Prepare filters
                    filters = {}
                    if project_id:
                        filters["project_id"] = project_id

                    # Perform vector search
                    semantic_results = self.vector_store.search_similar_chunks(
                        query_vector=query_vector,
                        top_k=top_k * 2,
                        min_similarity=min_similarity,
                        filters=filters if filters else None
                    )
                    logger.debug("Semantic search completed",
                               results_count=len(semantic_results))

            # Merge results using Reciprocal Rank Fusion
            merged_results = self._reciprocal_rank_fusion(
                bm25_results=bm25_results,
                semantic_results=semantic_results,
                alpha=alpha,
                k=60  # Standard RRF constant
            )

            # Limit to top_k
            final_results = merged_results[:top_k]

            logger.info("Hybrid search completed",
                       query=query,
                       total_results=len(final_results),
                       bm25_count=len(bm25_results),
                       semantic_count=len(semantic_results))

            return final_results

        except Exception as e:
            logger.error("Hybrid search failed", error=str(e), query=query)
            raise

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        semantic_results: List[VectorSearchResult],
        alpha: float,
        k: int = 60
    ) -> List[VectorSearchResult]:
        """
        Merge BM25 and semantic results using Reciprocal Rank Fusion

        RRF formula: score(chunk) = alpha * (1/(k + rank_semantic)) + (1-alpha) * (1/(k + rank_bm25))

        Args:
            bm25_results: List of (chunk_id, bm25_score) tuples
            semantic_results: List of VectorSearchResult objects
            alpha: Weight for semantic search (0.0-1.0)
            k: RRF constant (default 60)

        Returns:
            Merged list of VectorSearchResult objects, sorted by hybrid score
        """
        # Build lookup maps
        bm25_rank_map = {chunk_id: rank for rank, (chunk_id, score) in enumerate(bm25_results)}
        semantic_rank_map = {result.chunk_id: rank for rank, result in enumerate(semantic_results)}
        semantic_result_map = {result.chunk_id: result for result in semantic_results}

        # Collect all unique chunk IDs
        all_chunk_ids = set(bm25_rank_map.keys()) | set(semantic_rank_map.keys())

        # Calculate RRF scores
        hybrid_scores = {}
        for chunk_id in all_chunk_ids:
            semantic_rank = semantic_rank_map.get(chunk_id, float('inf'))
            bm25_rank = bm25_rank_map.get(chunk_id, float('inf'))

            semantic_score = alpha / (k + semantic_rank) if semantic_rank != float('inf') else 0.0
            bm25_score = (1 - alpha) / (k + bm25_rank) if bm25_rank != float('inf') else 0.0

            hybrid_scores[chunk_id] = semantic_score + bm25_score

        # Sort by hybrid score descending
        sorted_chunk_ids = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)

        # Build final result list
        final_results = []
        for chunk_id in sorted_chunk_ids:
            # Prefer semantic result if available (has more metadata)
            if chunk_id in semantic_result_map:
                result = semantic_result_map[chunk_id]
                # Update similarity score with hybrid score
                result.similarity_score = hybrid_scores[chunk_id]
                final_results.append(result)
            else:
                # BM25-only result, need to fetch from vector store
                stored_chunk = self.vector_store.get_chunk_by_id(chunk_id)
                if stored_chunk:
                    result = VectorSearchResult(
                        chunk_id=chunk_id,
                        similarity_score=hybrid_scores[chunk_id],
                        metadata=stored_chunk.metadata.to_dict() if hasattr(stored_chunk.metadata, 'to_dict') else {},
                        content=stored_chunk.content
                    )
                    final_results.append(result)

        return final_results

    def invalidate_index(self) -> None:
        """Invalidate BM25 index (will be rebuilt on next search)"""
        self._index_built = False
        logger.info("BM25 index invalidated")
