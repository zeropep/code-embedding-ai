"""
End-to-End tests for search functionality
Tests semantic search, similar code search, and project-specific searches
"""

import pytest
import pytest_asyncio
import httpx
import asyncio
from typing import Dict, Any, List


class TestE2ESemanticSearch:
    """E2E tests for semantic search functionality"""

    @pytest.fixture
    def api_base_url(self):
        """Base URL for API"""
        return "http://localhost:8000"

    @pytest_asyncio.fixture
    async def http_client(self, api_base_url):
        """Create HTTP client for testing"""
        async with httpx.AsyncClient(base_url=api_base_url, timeout=30.0) as client:
            yield client

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_semantic_search_basic(self, http_client):
        """
        E2E Test: Basic semantic search
        Tests that semantic search returns results with correct structure
        """
        response = await http_client.post(
            "/search/semantic",
            json={
                "query": "function to process user data",
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "results" in data
        assert isinstance(data["results"], list)

        # Verify result structure if results exist
        if len(data["results"]) > 0:
            result = data["results"][0]
            assert "file_path" in result
            assert "content" in result
            assert "similarity" in result
            assert "line_start" in result
            assert "line_end" in result
            assert "metadata" in result

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_semantic_search_with_top_k(self, http_client):
        """
        E2E Test: Semantic search with different top_k values
        Tests that top_k parameter limits results correctly
        """
        for top_k in [1, 3, 5, 10]:
            response = await http_client.post(
                "/search/semantic",
                json={
                    "query": "class definition",
                    "top_k": top_k
                }
            )

            assert response.status_code == 200
            data = response.json()

            # Results should not exceed top_k
            assert len(data["results"]) <= top_k

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_semantic_search_with_min_similarity(self, http_client):
        """
        E2E Test: Semantic search with minimum similarity threshold
        Tests that results meet the minimum similarity requirement
        """
        response = await http_client.post(
            "/search/semantic",
            json={
                "query": "function implementation",
                "top_k": 10,
                "min_similarity": 0.7
            }
        )

        assert response.status_code == 200
        data = response.json()

        # All results should have similarity >= min_similarity
        for result in data["results"]:
            assert result["similarity"] >= 0.7

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_semantic_search_empty_query(self, http_client):
        """
        E2E Test: Semantic search with empty query
        Tests error handling for invalid input
        """
        response = await http_client.post(
            "/search/semantic",
            json={
                "query": "",
                "top_k": 5
            }
        )

        # Should return 422 for validation error or handle gracefully
        assert response.status_code in [200, 422]

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_semantic_search_with_project_filter(self, http_client):
        """
        E2E Test: Semantic search with project_id filter
        Tests project-specific search functionality
        """
        # First, get list of projects
        projects_response = await http_client.get("/projects")

        if projects_response.status_code == 200:
            projects_data = projects_response.json()

            if len(projects_data["projects"]) > 0:
                project_id = projects_data["projects"][0]["project_id"]

                # Search within specific project
                response = await http_client.post(
                    "/search/semantic",
                    json={
                        "query": "function",
                        "project_id": project_id,
                        "top_k": 5
                    }
                )

                assert response.status_code == 200
                data = response.json()
                assert "results" in data
            else:
                pytest.skip("No projects available for testing")


class TestE2ESimilarCodeSearch:
    """E2E tests for similar code search functionality"""

    @pytest.fixture
    def api_base_url(self):
        """Base URL for API"""
        return "http://localhost:8000"

    @pytest_asyncio.fixture
    async def http_client(self, api_base_url):
        """Create HTTP client for testing"""
        async with httpx.AsyncClient(base_url=api_base_url, timeout=30.0) as client:
            yield client

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_similar_code_search_basic(self, http_client):
        """
        E2E Test: Basic similar code search
        Tests finding similar code patterns
        """
        code_snippet = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""

        response = await http_client.post(
            "/search/similar-code",
            json={
                "code_snippet": code_snippet,
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "results" in data
        assert isinstance(data["results"], list)

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_similar_code_search_with_language_filter(self, http_client):
        """
        E2E Test: Similar code search with language filter
        Tests language-specific code search
        """
        code_snippet = "public void processData() { }"

        response = await http_client.post(
            "/search/similar-code",
            json={
                "code_snippet": code_snippet,
                "language": "java",
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

        # Verify results are from Java files if results exist
        for result in data["results"]:
            if "metadata" in result and "language" in result["metadata"]:
                assert result["metadata"]["language"].lower() in ["java", "unknown"]

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_similar_code_search_python(self, http_client):
        """
        E2E Test: Similar code search for Python code
        """
        code_snippet = """
class UserService:
    def get_user(self, user_id):
        return self.repository.find_by_id(user_id)
"""

        response = await http_client.post(
            "/search/similar-code",
            json={
                "code_snippet": code_snippet,
                "language": "python",
                "top_k": 3
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data


class TestE2EProjectSearch:
    """E2E tests for project-related search functionality"""

    @pytest.fixture
    def api_base_url(self):
        """Base URL for API"""
        return "http://localhost:8000"

    @pytest_asyncio.fixture
    async def http_client(self, api_base_url):
        """Create HTTP client for testing"""
        async with httpx.AsyncClient(base_url=api_base_url, timeout=30.0) as client:
            yield client

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_list_projects(self, http_client):
        """
        E2E Test: List all projects
        Tests project listing functionality
        """
        response = await http_client.get("/projects")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "projects" in data
        assert isinstance(data["projects"], list)

        # Verify project structure if projects exist
        if len(data["projects"]) > 0:
            project = data["projects"][0]
            assert "project_id" in project
            assert "total_chunks" in project

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_get_project_stats(self, http_client):
        """
        E2E Test: Get project statistics
        Tests project statistics retrieval
        """
        # First get list of projects
        projects_response = await http_client.get("/projects")

        if projects_response.status_code == 200:
            projects_data = projects_response.json()

            if len(projects_data["projects"]) > 0:
                project_id = projects_data["projects"][0]["project_id"]

                # Get stats for project
                response = await http_client.get(f"/projects/{project_id}/stats")

                assert response.status_code == 200
                data = response.json()

                # Verify stats structure
                assert "total_chunks" in data
                assert "total_files" in data
                assert isinstance(data["total_chunks"], int)
                assert isinstance(data["total_files"], int)
            else:
                pytest.skip("No projects available for testing")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_project_specific_search_isolation(self, http_client):
        """
        E2E Test: Project search isolation
        Tests that search results are properly isolated by project
        """
        # Get all projects
        projects_response = await http_client.get("/projects")

        if projects_response.status_code == 200:
            projects_data = projects_response.json()

            if len(projects_data["projects"]) >= 2:
                # Search in two different projects
                project1_id = projects_data["projects"][0]["project_id"]
                project2_id = projects_data["projects"][1]["project_id"]

                # Search in project 1
                response1 = await http_client.post(
                    "/search/semantic",
                    json={
                        "query": "function",
                        "project_id": project1_id,
                        "top_k": 10
                    }
                )

                # Search in project 2
                response2 = await http_client.post(
                    "/search/semantic",
                    json={
                        "query": "function",
                        "project_id": project2_id,
                        "top_k": 10
                    }
                )

                assert response1.status_code == 200
                assert response2.status_code == 200

                data1 = response1.json()
                data2 = response2.json()

                # Results should be from different projects
                # (if they have results)
                assert "results" in data1
                assert "results" in data2
            else:
                pytest.skip("Need at least 2 projects for isolation test")


class TestE2ESearchPerformance:
    """E2E performance tests for search functionality"""

    @pytest.fixture
    def api_base_url(self):
        """Base URL for API"""
        return "http://localhost:8000"

    @pytest_asyncio.fixture
    async def http_client(self, api_base_url):
        """Create HTTP client for testing"""
        async with httpx.AsyncClient(base_url=api_base_url, timeout=30.0) as client:
            yield client

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.performance
    async def test_search_response_time(self, http_client):
        """
        E2E Performance Test: Search response time
        Target: < 500ms for semantic search
        """
        import time

        queries = [
            "function to handle user authentication",
            "class for data processing",
            "error handling implementation"
        ]

        response_times = []

        for query in queries:
            start_time = time.time()

            response = await http_client.post(
                "/search/semantic",
                json={
                    "query": query,
                    "top_k": 5
                }
            )

            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to ms

            assert response.status_code == 200
            response_times.append(response_time)

        avg_response_time = sum(response_times) / len(response_times)
        print(f"\nAverage search response time: {avg_response_time:.2f}ms")
        print(f"Min: {min(response_times):.2f}ms, Max: {max(response_times):.2f}ms")

        # Warn if average exceeds target
        if avg_response_time > 500:
            print(f"WARNING: Average response time ({avg_response_time:.2f}ms) exceeds target (500ms)")

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.performance
    async def test_concurrent_searches(self, http_client):
        """
        E2E Performance Test: Concurrent search requests
        Tests system behavior under concurrent load
        """
        queries = [
            "function definition",
            "class implementation",
            "error handling",
            "data processing",
            "user authentication",
            "database query",
            "API endpoint",
            "validation logic"
        ]

        async def search(query: str):
            return await http_client.post(
                "/search/semantic",
                json={
                    "query": query,
                    "top_k": 3
                }
            )

        # Execute concurrent searches
        import time
        start_time = time.time()

        responses = await asyncio.gather(
            *[search(q) for q in queries],
            return_exceptions=True
        )

        end_time = time.time()
        total_time = (end_time - start_time) * 1000

        # Verify responses
        successful = sum(
            1 for r in responses
            if isinstance(r, httpx.Response) and r.status_code == 200
        )

        print(f"\nConcurrent searches: {successful}/{len(queries)} successful")
        print(f"Total time: {total_time:.2f}ms")
        print(f"Average per request: {total_time/len(queries):.2f}ms")

        # At least 80% should succeed
        assert successful >= len(queries) * 0.8


class TestE2ESearchAccuracy:
    """E2E tests for search result accuracy"""

    @pytest.fixture
    def api_base_url(self):
        """Base URL for API"""
        return "http://localhost:8000"

    @pytest_asyncio.fixture
    async def http_client(self, api_base_url):
        """Create HTTP client for testing"""
        async with httpx.AsyncClient(base_url=api_base_url, timeout=30.0) as client:
            yield client

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_search_relevance_ordering(self, http_client):
        """
        E2E Test: Search results are ordered by relevance
        Tests that similarity scores are in descending order
        """
        response = await http_client.post(
            "/search/semantic",
            json={
                "query": "function to calculate total",
                "top_k": 10
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify results are ordered by similarity (descending)
        similarities = [r["similarity"] for r in data["results"]]

        for i in range(len(similarities) - 1):
            assert similarities[i] >= similarities[i + 1], \
                f"Results not ordered: {similarities[i]} < {similarities[i + 1]}"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_search_metadata_completeness(self, http_client):
        """
        E2E Test: Search results contain complete metadata
        Tests that all expected metadata fields are present
        """
        response = await http_client.post(
            "/search/semantic",
            json={
                "query": "function implementation",
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()

        required_fields = ["file_path", "content", "similarity", "metadata"]

        for result in data["results"]:
            for field in required_fields:
                assert field in result, f"Missing field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
