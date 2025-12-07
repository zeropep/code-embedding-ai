"""
Search Performance Benchmark
벤치마크 검색 성능 측정 및 분석
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
import httpx
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    query: str
    top_k: int
    response_time_ms: float
    num_results: int
    success: bool
    error: str = None


@dataclass
class BenchmarkSummary:
    """벤치마크 요약"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    success_rate: float


class SearchBenchmark:
    """검색 성능 벤치마크"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []

    async def run_single_search(
        self,
        query: str,
        top_k: int = 5,
        project_id: str = None,
        min_similarity: float = None
    ) -> BenchmarkResult:
        """단일 검색 실행 및 측정"""
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {"query": query, "top_k": top_k}
                if project_id:
                    payload["project_id"] = project_id
                if min_similarity is not None:
                    payload["min_similarity"] = min_similarity

                response = await client.post(
                    f"{self.base_url}/search/semantic",
                    json=payload
                )

                elapsed_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    return BenchmarkResult(
                        query=query,
                        top_k=top_k,
                        response_time_ms=elapsed_ms,
                        num_results=len(data.get("results", [])),
                        success=True
                    )
                else:
                    return BenchmarkResult(
                        query=query,
                        top_k=top_k,
                        response_time_ms=elapsed_ms,
                        num_results=0,
                        success=False,
                        error=f"HTTP {response.status_code}"
                    )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return BenchmarkResult(
                query=query,
                top_k=top_k,
                response_time_ms=elapsed_ms,
                num_results=0,
                success=False,
                error=str(e)
            )

    async def benchmark_response_time(self, queries: List[str], iterations: int = 5):
        """응답 시간 벤치마크"""
        print(f"\n{'='*60}")
        print("검색 응답 시간 벤치마크")
        print(f"{'='*60}")
        print(f"쿼리 수: {len(queries)}")
        print(f"반복 횟수: {iterations}")
        print(f"{'='*60}\n")

        for query in queries:
            print(f"쿼리: '{query}'")
            query_results = []

            for i in range(iterations):
                result = await self.run_single_search(query)
                query_results.append(result)
                self.results.append(result)
                print(f"  실행 {i+1}: {result.response_time_ms:.2f}ms")

            # 통계 계산
            response_times = [r.response_time_ms for r in query_results if r.success]
            if response_times:
                print(f"  평균: {statistics.mean(response_times):.2f}ms")
                print(f"  중앙값: {statistics.median(response_times):.2f}ms")
                print(f"  최소: {min(response_times):.2f}ms")
                print(f"  최대: {max(response_times):.2f}ms")
            print()

    async def benchmark_top_k_variations(self, query: str, top_k_values: List[int]):
        """top_k 값에 따른 성능 측정"""
        print(f"\n{'='*60}")
        print("top_k 값별 성능 벤치마크")
        print(f"{'='*60}")
        print(f"쿼리: '{query}'")
        print(f"top_k 값: {top_k_values}")
        print(f"{'='*60}\n")

        for top_k in top_k_values:
            result = await self.run_single_search(query, top_k=top_k)
            self.results.append(result)

            print(f"top_k={top_k:3d}: {result.response_time_ms:7.2f}ms, "
                  f"결과: {result.num_results}개")

    async def benchmark_concurrent_requests(
        self,
        queries: List[str],
        concurrent_levels: List[int]
    ):
        """동시 요청 성능 측정"""
        print(f"\n{'='*60}")
        print("동시 요청 벤치마크")
        print(f"{'='*60}")
        print(f"쿼리 수: {len(queries)}")
        print(f"동시성 레벨: {concurrent_levels}")
        print(f"{'='*60}\n")

        for concurrent in concurrent_levels:
            print(f"\n동시 요청 수: {concurrent}")

            # 쿼리 반복하여 동시 요청 수만큼 생성
            test_queries = (queries * (concurrent // len(queries) + 1))[:concurrent]

            start_time = time.time()
            tasks = [self.run_single_search(q) for q in test_queries]
            results = await asyncio.gather(*tasks)
            total_time = (time.time() - start_time) * 1000

            self.results.extend(results)

            successful = sum(1 for r in results if r.success)
            avg_response = statistics.mean([r.response_time_ms for r in results if r.success])

            print(f"  총 시간: {total_time:.2f}ms")
            print(f"  성공: {successful}/{concurrent}")
            print(f"  평균 응답 시간: {avg_response:.2f}ms")
            print(f"  처리량: {concurrent / (total_time / 1000):.2f} req/sec")

    async def benchmark_min_similarity(self, query: str, thresholds: List[float]):
        """min_similarity 임계값별 성능 측정"""
        print(f"\n{'='*60}")
        print("min_similarity 임계값별 벤치마크")
        print(f"{'='*60}")
        print(f"쿼리: '{query}'")
        print(f"임계값: {thresholds}")
        print(f"{'='*60}\n")

        for threshold in thresholds:
            result = await self.run_single_search(query, min_similarity=threshold)
            self.results.append(result)

            print(f"min_similarity={threshold:.2f}: {result.response_time_ms:7.2f}ms, "
                  f"결과: {result.num_results}개")

    def generate_summary(self) -> BenchmarkSummary:
        """벤치마크 요약 생성"""
        if not self.results:
            return None

        successful_results = [r for r in self.results if r.success]
        response_times = [r.response_time_ms for r in successful_results]

        if not response_times:
            return None

        response_times.sort()

        return BenchmarkSummary(
            total_requests=len(self.results),
            successful_requests=len(successful_results),
            failed_requests=len(self.results) - len(successful_results),
            avg_response_time_ms=statistics.mean(response_times),
            min_response_time_ms=min(response_times),
            max_response_time_ms=max(response_times),
            median_response_time_ms=statistics.median(response_times),
            p95_response_time_ms=response_times[int(len(response_times) * 0.95)],
            p99_response_time_ms=response_times[int(len(response_times) * 0.99)],
            success_rate=len(successful_results) / len(self.results)
        )

    def print_summary(self):
        """요약 출력"""
        summary = self.generate_summary()

        if not summary:
            print("벤치마크 결과가 없습니다.")
            return

        print(f"\n{'='*60}")
        print("벤치마크 요약")
        print(f"{'='*60}")
        print(f"총 요청 수:        {summary.total_requests}")
        print(f"성공 요청:         {summary.successful_requests}")
        print(f"실패 요청:         {summary.failed_requests}")
        print(f"성공률:            {summary.success_rate*100:.2f}%")
        print(f"\n응답 시간 통계:")
        print(f"  평균:            {summary.avg_response_time_ms:.2f}ms")
        print(f"  중앙값:          {summary.median_response_time_ms:.2f}ms")
        print(f"  최소:            {summary.min_response_time_ms:.2f}ms")
        print(f"  최대:            {summary.max_response_time_ms:.2f}ms")
        print(f"  P95:             {summary.p95_response_time_ms:.2f}ms")
        print(f"  P99:             {summary.p99_response_time_ms:.2f}ms")
        print(f"{'='*60}")

        # 목표 대비 평가
        print(f"\n목표 대비 평가:")
        target_response_time = 500  # ms
        if summary.avg_response_time_ms < target_response_time:
            print(f"✅ 평균 응답 시간 목표 달성 ({summary.avg_response_time_ms:.2f}ms < {target_response_time}ms)")
        else:
            print(f"❌ 평균 응답 시간 목표 미달성 ({summary.avg_response_time_ms:.2f}ms >= {target_response_time}ms)")

        if summary.p95_response_time_ms < target_response_time:
            print(f"✅ P95 응답 시간 목표 달성 ({summary.p95_response_time_ms:.2f}ms < {target_response_time}ms)")
        else:
            print(f"❌ P95 응답 시간 목표 미달성 ({summary.p95_response_time_ms:.2f}ms >= {target_response_time}ms)")

    def save_results(self, output_path: str = None):
        """결과를 JSON 파일로 저장"""
        if output_path is None:
            output_path = "benchmark_results.json"

        summary = self.generate_summary()

        data = {
            "summary": asdict(summary) if summary else None,
            "results": [asdict(r) for r in self.results]
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n결과 저장: {output_file}")


async def run_comprehensive_benchmark():
    """종합 벤치마크 실행"""
    benchmark = SearchBenchmark()

    # 테스트 쿼리
    test_queries = [
        "function to process user data",
        "class definition for user service",
        "error handling implementation",
        "database connection code",
        "API endpoint for user authentication"
    ]

    print("=" * 60)
    print("검색 성능 종합 벤치마크")
    print("=" * 60)
    print(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 응답 시간 벤치마크
    await benchmark.benchmark_response_time(test_queries[:3], iterations=5)

    # 2. top_k 값별 성능
    await benchmark.benchmark_top_k_variations(
        test_queries[0],
        top_k_values=[5, 10, 20, 50, 100]
    )

    # 3. 동시 요청 벤치마크
    await benchmark.benchmark_concurrent_requests(
        test_queries,
        concurrent_levels=[1, 5, 10, 20]
    )

    # 4. min_similarity 임계값별 성능
    await benchmark.benchmark_min_similarity(
        test_queries[0],
        thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]
    )

    # 요약 출력
    benchmark.print_summary()

    # 결과 저장
    benchmark.save_results("code-embedding-ai/benchmarks/benchmark_results.json")

    print(f"\n종료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")


async def run_quick_benchmark():
    """빠른 벤치마크 (개발용)"""
    benchmark = SearchBenchmark()

    test_queries = [
        "function implementation",
        "class definition",
        "error handling"
    ]

    print("빠른 벤치마크 실행 중...")

    await benchmark.benchmark_response_time(test_queries, iterations=3)
    await benchmark.benchmark_top_k_variations(test_queries[0], [5, 10, 20])

    benchmark.print_summary()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(run_quick_benchmark())
    else:
        asyncio.run(run_comprehensive_benchmark())
