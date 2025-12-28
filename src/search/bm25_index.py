"""
BM25 keyword search index for code search
"""
import re
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
import structlog

logger = structlog.get_logger(__name__)


class BM25Index:
    """BM25 keyword search index optimized for code"""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self._is_built = False

    def build_index(self, documents: List[Dict]) -> None:
        """
        Build BM25 index from documents

        Args:
            documents: List of dicts with keys: chunk_id, content, metadata
        """
        try:
            logger.info("Building BM25 index", document_count=len(documents))

            self.chunk_ids = []
            self.tokenized_corpus = []

            for doc in documents:
                chunk_id = doc.get("chunk_id", "")
                content = doc.get("content", "")

                if not chunk_id or not content:
                    continue

                # Tokenize content
                tokens = self.tokenize(content)

                self.chunk_ids.append(chunk_id)
                self.tokenized_corpus.append(tokens)

            # Build BM25 index
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                self._is_built = True
                logger.info("BM25 index built successfully",
                           total_documents=len(self.chunk_ids))
            else:
                logger.warning("No documents to index")
                self._is_built = False

        except Exception as e:
            logger.error("Failed to build BM25 index", error=str(e))
            self._is_built = False
            raise

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using BM25

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score descending
        """
        if not self._is_built or self.bm25 is None:
            logger.warning("BM25 index not built, returning empty results")
            return []

        try:
            # Tokenize query
            query_tokens = self.tokenize(query)

            if not query_tokens:
                logger.warning("Query produced no tokens", query=query)
                return []

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Pair chunk_ids with scores
            scored_chunks = [(self.chunk_ids[i], float(scores[i]))
                           for i in range(len(scores))]

            # Sort by score descending
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            # Return top_k
            results = scored_chunks[:top_k]

            logger.debug("BM25 search completed",
                        query=query,
                        results_count=len(results))

            return results

        except Exception as e:
            logger.error("BM25 search failed", error=str(e), query=query)
            return []

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text optimized for code

        Handles:
        - camelCase splitting (e.g., "getUserId" -> ["get", "user", "id"])
        - snake_case splitting (e.g., "user_name" -> ["user", "name"])
        - Special characters separation
        - Lowercase normalization

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        if not text:
            return []

        tokens = []

        # Split on whitespace and special characters, keeping alphanumeric
        # This regex keeps underscores for now (we'll handle them next)
        words = re.findall(r'[a-zA-Z0-9_]+', text)

        for word in words:
            # Split camelCase: getUserId -> get User Id
            camel_split = re.sub('([a-z])([A-Z])', r'\1 \2', word)

            # Split on underscores: user_name -> user name
            parts = camel_split.replace('_', ' ').split()

            # Lowercase and add to tokens
            for part in parts:
                part_lower = part.lower()
                if part_lower and len(part_lower) > 1:  # Skip single characters
                    tokens.append(part_lower)

        return tokens

    def is_built(self) -> bool:
        """Check if index is built"""
        return self._is_built

    def get_document_count(self) -> int:
        """Get number of indexed documents"""
        return len(self.chunk_ids)
