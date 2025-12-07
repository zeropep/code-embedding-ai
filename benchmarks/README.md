# 성능 벤치마크 및 최적화

## 📊 개요

이 디렉토리는 검색 성능 벤치마크 및 최적화와 관련된 도구를 포함합니다.

## 🎯 성능 목표

- **검색 응답 시간**: < 500ms (평균)
- **P95 응답 시간**: < 500ms
- **임베딩 생성 속도**: > 100 chunks/sec
- **메모리 사용량**: < 2GB
- **캐시 히트율**: > 30%

## 🚀 벤치마크 실행

### 종합 벤치마크 (권장)

```bash
cd code-embedding-ai
python benchmarks/search_benchmark.py
```

**포함 내용**:
- 응답 시간 측정 (5회 반복)
- top_k 값별 성능 (5, 10, 20, 50, 100)
- 동시 요청 테스트 (1, 5, 10, 20)
- min_similarity 임계값별 성능 (0.5, 0.6, 0.7, 0.8, 0.9)

### 빠른 벤치마크 (개발용)

```bash
python benchmarks/search_benchmark.py quick
```

**포함 내용**:
- 응답 시간 측정 (3회 반복)
- top_k 값별 성능 (5, 10, 20)

## 📈 벤치마크 결과

### 결과 저장 위치
```
code-embedding-ai/benchmarks/benchmark_results.json
```

### 결과 포맷

```json
{
  "summary": {
    "total_requests": 50,
    "successful_requests": 48,
    "failed_requests": 2,
    "avg_response_time_ms": 324.5,
    "min_response_time_ms": 145.2,
    "max_response_time_ms": 892.3,
    "median_response_time_ms": 298.7,
    "p95_response_time_ms": 645.1,
    "p99_response_time_ms": 823.4,
    "success_rate": 0.96
  },
  "results": [...]
}
```

## ⚡ 성능 최적화

### 1. 캐싱 전략

#### 구현된 캐시
- **검색 결과 캐시**: 5분 TTL, 최대 500개
- **임베딩 캐시**: 1시간 TTL, 최대 1000개
- **프로젝트 통계 캐시**: 10분 TTL, 최대 100개

#### 캐시 통계 확인
```bash
curl http://localhost:8000/status/cache
```

**응답 예시**:
```json
{
  "status": "success",
  "cache_stats": {
    "embedding_cache": {
      "size": 234,
      "max_size": 1000,
      "hits": 1250,
      "misses": 780,
      "evictions": 45,
      "hit_rate": 0.616,
      "ttl_seconds": 3600
    },
    "search_results_cache": {
      "size": 156,
      "max_size": 500,
      "hits": 892,
      "misses": 445,
      "evictions": 23,
      "hit_rate": 0.667,
      "ttl_seconds": 300
    },
    "stats_cache": {
      "size": 12,
      "max_size": 100,
      "hits": 456,
      "misses": 23,
      "evictions": 0,
      "hit_rate": 0.952,
      "ttl_seconds": 600
    }
  }
}
```

### 2. 캐시 특징

- **LRU (Least Recently Used)**: 가장 오래 사용되지 않은 항목 제거
- **TTL (Time To Live)**: 자동 만료
- **Thread-Safe**: 멀티스레드 환경에서 안전
- **통계 수집**: 히트율, 미스율, Eviction 추적

### 3. 배치 처리

임베딩 생성은 배치로 처리되어 성능 향상:
- **기본 배치 크기**: 100 chunks
- **병렬 처리**: 가능한 경우 병렬 임베딩 생성

### 4. 데이터베이스 최적화

- **ChromaDB 인덱싱**: 자동 벡터 인덱싱
- **메타데이터 필터링**: 효율적인 필터 쿼리

## 📊 성능 모니터링

### 실시간 메트릭

```bash
curl http://localhost:8000/status/metrics
```

### 캐시 통계

```bash
curl http://localhost:8000/status/cache
```

### 시스템 상태

```bash
curl http://localhost:8000/status/system
```

## 🔧 튜닝 가이드

### 캐시 크기 조정

`src/cache/cache_manager.py`에서 캐시 크기 및 TTL 조정:

```python
self.embedding_cache = LRUCache(max_size=1000, ttl_seconds=3600)
self.search_results_cache = LRUCache(max_size=500, ttl_seconds=300)
self.stats_cache = LRUCache(max_size=100, ttl_seconds=600)
```

### 배치 크기 조정

`src/database/models.py`에서 `VectorDBConfig.max_batch_size` 조정:

```python
max_batch_size: int = 100  # 기본값
```

### ChromaDB 설정

`src/database/vector_store.py`에서 ChromaDB 설정:

```python
# 지속성 디렉토리
persist_directory = "chromadb_data"

# 컬렉션 이름
collection_name = "code_embeddings"
```

## 📝 벤치마크 예시

### 1. 기본 응답 시간 측정

```
쿼리: 'function to process user data'
  실행 1: 234.56ms
  실행 2: 189.23ms
  실행 3: 212.45ms
  실행 4: 198.76ms
  실행 5: 205.34ms
  평균: 208.07ms
  중앙값: 205.34ms
  최소: 189.23ms
  최대: 234.56ms
```

### 2. top_k 영향

```
top_k=  5:  185.32ms, 결과: 5개
top_k= 10:  198.45ms, 결과: 10개
top_k= 20:  215.67ms, 결과: 20개
top_k= 50:  278.92ms, 결과: 50개
top_k=100:  356.23ms, 결과: 100개
```

### 3. 동시 요청 처리

```
동시 요청 수: 10
  총 시간: 892.34ms
  성공: 10/10
  평균 응답 시간: 215.67ms
  처리량: 11.21 req/sec
```

## 🎯 최적화 체크리스트

- [x] 검색 결과 캐싱 구현
- [x] 임베딩 캐싱 구현
- [x] 프로젝트 통계 캐싱 구현
- [x] LRU 캐시 전략
- [x] TTL 기반 자동 만료
- [x] 캐시 통계 엔드포인트
- [x] 벤치마크 도구
- [ ] Redis 캐싱 (선택적, 미구현)
- [ ] 쿼리 쿼리 최적화 (필요시)
- [ ] 인덱스 튜닝 (필요시)

## 🔍 문제 해결

### 캐시 히트율이 낮은 경우

1. TTL 값 증가
2. 캐시 크기 증가
3. 쿼리 정규화 개선

### 메모리 사용량이 높은 경우

1. 캐시 크기 감소
2. TTL 값 감소
3. max_batch_size 감소

### 응답 시간이 느린 경우

1. 캐시 설정 확인
2. ChromaDB 인덱스 상태 확인
3. top_k 값 최적화
4. 동시 요청 수 제한

## 📚 참고 자료

- [ChromaDB 문서](https://docs.trychroma.com/)
- [FastAPI 성능 최적화](https://fastapi.tiangolo.com/deployment/concepts/)
- [Python 프로파일링](https://docs.python.org/3/library/profile.html)
