# 코드 임베딩 AI 파이프라인

Spring Boot + Thymeleaf 코드베이스를 처리하고, 시맨틱 임베딩을 생성하며, 보안 기능을 갖춘 지능형 코드 검색을 지원하는 종합적인 AI 기반 코드 분석 파이프라인입니다.

## 주요 기능

- **다중 언어 코드 파싱**: AST 기반 분석을 통한 Java, Kotlin, HTML/Thymeleaf, Python 지원
- **지능형 청킹**: 설정 가능한 토큰 제한과 오버랩을 가진 시맨틱 코드 분할
- **보안 스캐닝**: 시크릿, 자격증명, 민감 데이터의 자동 탐지 및 마스킹
- **벡터 임베딩**: jina-code-embeddings-1.5b를 사용한 고품질 코드 임베딩
- **벡터 저장소**: 효율적인 유사도 검색을 제공하는 ChromaDB 통합
- **증분 업데이트**: 변경된 파일만 처리하는 Git diff 기반 모니터링
- **REST API**: 외부 시스템 통합을 위한 완전한 기능의 웹 API
- **CLI 인터페이스**: 배치 처리 및 관리를 위한 명령줄 도구
- **모니터링**: 포괄적인 로깅, 메트릭, 상태 모니터링

## 빠른 시작

### 사전 요구사항

- Python 3.9+
- Java/Kotlin/HTML 코드가 있는 Git 저장소
- Jina AI API 키

### 설치

```bash
# 저장소 클론
git clone <repository-url>
cd code-embedding-ai

# 의존성 설치
uv sync

# 환경 변수 설정
export JINA_API_KEY="your-jina-api-key"
export CHROMADB_PERSIST_DIR="/path/to/vector/storage"
```

### 기본 사용법

#### CLI 인터페이스

```bash
# 저장소 처리
uv run python -m src.cli process /path/to/spring-boot-repo

# 코드 검색
uv run python -m src.cli search "사용자 인증 로직"

# 변경 사항 모니터링 시작
uv run python -m src.cli monitor /path/to/repo

# 웹 서버 시작
uv run python -m src.cli server start --port 8000
```

#### Python API

```python
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.code_parser.models import ParserConfig
from src.security.models import SecurityConfig
from src.embeddings.models import EmbeddingConfig

# 파이프라인 구성
parser_config = ParserConfig(
    min_tokens=50,
    max_tokens=500,
    excluded_dirs=[".venv", "node_modules", "__pycache__"]  # 디렉토리 제외
)
security_config = SecurityConfig(enabled=True)
embedding_config = EmbeddingConfig(api_key="your-api-key")

# 파이프라인 생성
pipeline = EmbeddingPipeline(
    parser_config=parser_config,
    security_config=security_config,
    embedding_config=embedding_config,
    chunk_batch_size=100  # 한 번에 100개 청크 처리
)

# 저장소 처리
result = await pipeline.process_repository("/path/to/repo")
```

## 아키텍처

파이프라인은 여러 핵심 구성 요소로 이루어져 있습니다:

### 1. 코드 파서 (`src/code_parser/`)
- **언어 지원**: Java, Kotlin, HTML/Thymeleaf, Python
- **AST 분석**: 메서드/클래스 추출, 레이어 감지
- **청킹 전략**: 설정 가능한 매개변수로 시맨틱 분할
- **Python 프레임워크**: Django, Flask, FastAPI 감지 및 레이어 분류

### 2. 보안 스캐너 (`src/security/`)
- **시크릿 탐지**: 패스워드, API 키, 토큰, 데이터베이스 URL
- **콘텐츠 마스킹**: 민감 데이터 숨기면서 구문 보존
- **패턴 매칭**: 설정 가능한 정규식 패턴 및 화이트리스트

### 3. 임베딩 서비스 (`src/embeddings/`)
- **Jina AI 통합**: jina-code-embeddings-1.5b 모델
- **배치 처리**: 재시도 로직으로 최적화된 API 호출
- **캐싱**: 성능을 위한 선택적 임베딩 캐싱

### 4. 벡터 데이터베이스 (`src/database/`)
- **ChromaDB 백엔드**: 영구 벡터 저장소
- **메타데이터 인덱싱**: 필터링 및 검색을 위한 풍부한 메타데이터
- **유사도 검색**: 설정 가능한 임계값으로 코사인 유사도

### 5. 업데이트 시스템 (`src/updates/`)
- **Git 모니터링**: 파일 변경 자동 감지
- **증분 처리**: 수정된 파일만 처리
- **파일 워칭**: 실시간 모니터링 옵션

### 6. API & CLI (`src/api/`, `src/cli.py`)
- **REST 엔드포인트**: 처리, 검색, 모니터링, 통계
- **CLI 명령어**: 배치 처리 및 관리
- **인증**: 선택적 API 키 인증

### 7. 모니터링 (`src/monitoring/`)
- **구조화된 로깅**: 컨텍스트가 있는 JSON 형식 로그
- **메트릭 수집**: 성능 및 사용 통계
- **상태 확인**: 서비스 가용성 모니터링
- **알림**: 오류 및 성능에 대한 구성 가능한 알림

## 구성

### 환경 변수

```bash
# 필수
JINA_API_KEY=your-jina-api-key

# Jina AI 임베딩 설정
JINA_API_URL=https://api.jina.ai/v1/embeddings
EMBEDDING_MODEL=jina-code-embeddings-1.5b
EMBEDDING_DIMENSIONS=1024
EMBEDDING_BATCH_SIZE=8              # API 호출당 텍스트 수 (권장: 8-20)
MAX_CONCURRENT_EMBEDDINGS=5         # 최대 동시 API 요청 수
EMBEDDING_TIMEOUT=60                # API 타임아웃(초)
ENABLE_EMBEDDING_CACHE=true         # 인메모리 임베딩 캐시

# ChromaDB 설정
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
CHROMADB_COLLECTION_NAME=code_embeddings
CHROMADB_PERSISTENT=true
CHROMADB_PERSIST_DIR=./chroma_db
CHROMADB_BATCH_SIZE=100

# 파서 설정
CHUNK_MIN_TOKENS=50                 # 청크당 최소 토큰
CHUNK_MAX_TOKENS=500                # 청크당 최대 토큰
CHUNK_OVERLAP_TOKENS=20             # 청크 간 오버랩
INCLUDE_COMMENTS=false
SUPPORTED_EXTENSIONS=.java,.kt,.html,.xml,.properties,.yml,.yaml,.py

# 보안 설정
ENABLE_SECRET_SCANNING=true
PRESERVE_SYNTAX=true
SENSITIVITY_THRESHOLD=0.7
SCAN_COMMENTS=true
SCAN_STRINGS=true

# 업데이트 서비스 설정
UPDATE_STATE_DIR=./update_state
UPDATE_CHECK_INTERVAL=300
MAX_CONCURRENT_UPDATES=3
ENABLE_FILE_WATCHING=false

# 로깅 설정
LOG_LEVEL=INFO
```

### 구성 파일

상세한 구성을 위해 `config.yaml`을 생성하세요:

```yaml
parser:
  min_tokens: 50
  max_tokens: 500
  overlap_tokens: 50
  supported_extensions: [".java", ".kt", ".html", ".py"]
  excluded_dirs:
    - ".venv"
    - "venv"
    - "node_modules"
    - "__pycache__"
    - ".pytest_cache"
    - "chroma_db"
    - "dist"
    - "build"
    - ".git"
    - ".idea"
    - ".vscode"

security:
  enabled: true
  preserve_syntax: true
  sensitivity_threshold: 0.7
  whitelist_patterns: ["test_", "example_"]

embedding:
  model_name: "jina-code-embeddings-1.5b"
  batch_size: 32
  timeout: 30
  enable_caching: true

database:
  collection_name: "code_embeddings"
  persistent: true
  max_batch_size: 100

monitoring:
  enable_metrics: true
  enable_alerting: true
  log_level: "INFO"
```

### 파서 구성 상세 설정

#### 제외 디렉토리

기본적으로 파서는 임베딩하지 말아야 할 일반적인 디렉토리를 자동으로 제외합니다:

**개발 환경:**
- `.venv`, `venv`, `.env` - Python 가상환경
- `node_modules`, `bower_components` - JavaScript 의존성

**버전 관리:**
- `.git`, `.svn`, `.hg` - VCS 디렉토리

**Python 캐시:**
- `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`

**빌드 결과물:**
- `dist`, `build`, `target`, `out`

**IDE 설정:**
- `.idea`, `.vscode`, `.vs`

**테스트/커버리지:**
- `coverage`, `.coverage`, `htmlcov`, `.tox`, `.nox`

**데이터베이스:**
- `chroma_db`, `chromadb` - 벡터 데이터베이스 저장소

**기타:**
- `logs`, `log` - 로그 파일
- `.DS_Store`, `Thumbs.db` - OS 파일

`excluded_dirs` 매개변수를 사용하여 이 목록을 커스터마이징할 수 있습니다:

```python
from src.code_parser.models import ParserConfig

# 커스텀 제외 설정
parser_config = ParserConfig(
    excluded_dirs=[
        ".venv", "node_modules",  # 기본 제외
        "generated", "migrations",  # 커스텀 제외
        "vendor", "third_party"
    ]
)
```

#### 배치 크기 구성

한 번에 처리할 코드 청크 수를 제어합니다:

```python
from src.embeddings.embedding_pipeline import EmbeddingPipeline

pipeline = EmbeddingPipeline(
    chunk_batch_size=100  # GPU용 권장값
)

# CPU 처리 시 작은 배치 사용:
# chunk_batch_size=50

# 고성능 GPU (24GB+):
# chunk_batch_size=200
```

**배치 크기 가이드라인:**
- **CPU**: 10-50 청크 (느리지만 메모리 안전)
- **GPU (12GB)**: 100-150 청크 (권장)
- **GPU (24GB+)**: 200-500 청크 (고성능)

## API 참조

### 저장소 처리
```http
POST /api/v1/process
{
    "repository_path": "/path/to/repo",
    "include_security_scan": true,
    "force_reprocess": false
}
```

### 코드 검색
```http
POST /api/v1/search
{
    "query": "사용자 인증 로직",
    "limit": 10,
    "similarity_threshold": 0.7,
    "language_filter": "java",
    "layer_filter": "service"
}
```

### 상태 확인
```http
GET /health
```

### 통계 조회
```http
GET /api/v1/stats
```

## 보안 기능

### 시크릿 탐지
파이프라인은 다음을 자동으로 탐지하고 마스킹합니다:
- 패스워드 및 자격증명
- API 키 및 토큰
- 자격증명이 포함된 데이터베이스 URL
- 개인 키 및 인증서
- OAuth 토큰 및 시크릿

### 콘텐츠 마스킹
- 코드 구문 및 구조 보존
- 민감한 값을 타입별 플레이스홀더로 교체
- 분석을 위한 코드 기능 유지
- 구성 가능한 민감도 레벨

### 보안 보고서
```python
# 보안 보고서 생성
scanner = SecurityScanner(config)
report = scanner.generate_security_report(chunks)

print(f"발견된 시크릿: {report['scan_summary']['total_secrets_found']}개")
print(f"고위험 파일: {len(report['high_risk_files'])}개")
```

## 성능 최적화

### 배치 처리
- API 호출을 위한 구성 가능한 배치 크기
- 여러 파일에 대한 병렬 처리
- 메모리 효율적인 청크 처리

### 캐싱
- 선택적 임베딩 캐싱
- 파일 해시 기반 변경 감지
- 영구 벡터 저장소

### 증분 업데이트
- Git diff 기반 변경 감지
- 수정된 파일만 처리
- 효율적인 데이터베이스 업데이트

## 모니터링 및 관찰 가능성

### 메트릭
- 처리 시간 및 처리량
- API 응답 시간 및 오류율
- 리소스 사용량 (CPU, 메모리)
- 캐시 히트/미스 비율

### 로깅
- 구조화된 JSON 로그
- 요청/응답 추적
- 보안 이벤트 로깅
- 성능 메트릭

### 상태 확인
- 서비스 가용성 모니터링
- 데이터베이스 연결성
- API 엔드포인트 상태
- 리소스 활용도

### 알림
- 오류율 임계값
- 성능 저하
- 리소스 고갈
- 보안 이벤트

## 개발

### 테스트 실행
```bash
# 모든 테스트 실행
python run_tests.py

# 특정 테스트 스위트 실행
python run_tests.py --unit
python run_tests.py --integration

# 커버리지와 함께 실행
python run_tests.py --coverage --html
```

### 코드 품질
```bash
# 타입 체킹
mypy src/

# 코드 스타일
flake8 src/ tests/

# 보안 스캔
bandit -r src/
```

## 배포

### Docker
```dockerfile
FROM python:3.9-slim

# Install uv
RUN pip install uv

WORKDIR /app
COPY requirements.txt .
RUN uv sync

COPY src/ src/
COPY config.yaml .

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 프로덕션 구성
- 환경별 구성 파일 사용
- 적절한 로깅 및 모니터링 설정
- 보안 설정 및 API 키 구성
- 백업 및 복구 절차 설정

## 사용 사례

### 1. 코드 리뷰 지원
```bash
# 특정 기능과 관련된 코드 찾기
uv run python -m src.cli search "사용자 등록 프로세스"

# 보안 취약점 검사
uv run python -m src.cli process --security-scan /path/to/repo
```

### 2. 레거시 코드 이해
```bash
# 비슷한 패턴의 코드 검색
uv run python -m src.cli search "데이터베이스 트랜잭션 처리"

# 특정 클래스의 사용법 찾기
uv run python -m src.cli search "UserService 메서드 호출"
```

### 3. 아키텍처 분석
```bash
# 레이어별 코드 검색
uv run python -m src.cli search "컨트롤러 레이어 인증" --layer controller

# 언어별 코드 분석
uv run python -m src.cli search "비즈니스 로직" --language java
```

## 문제 해결

### 일반적인 문제들

#### 1. 임베딩 생성 실패
```bash
# API 키 확인
echo $JINA_API_KEY

# 연결 테스트
curl -H "Authorization: Bearer $JINA_API_KEY" https://api.jina.ai/v1/embeddings
```

#### 2. 데이터베이스 연결 오류
```bash
# ChromaDB 디렉토리 권한 확인
ls -la $CHROMADB_PERSIST_DIR

# 디스크 공간 확인
df -h $CHROMADB_PERSIST_DIR
```

#### 3. 파싱 오류
```bash
# 지원되는 파일 형식 확인
uv run python -m src.cli --help

# 로그 레벨 증가
export LOG_LEVEL=DEBUG
uv run python -m src.cli process /path/to/repo
```

## 기여하기

1. 저장소를 포크합니다
2. 기능 브랜치를 생성합니다
3. 테스트와 함께 변경 사항을 만듭니다
4. 테스트 스위트를 실행합니다
5. Pull Request를 제출합니다

## 라이센스

[라이센스 정보]

## 지원

문제 및 질문이 있으시면:
- GitHub에서 이슈를 생성하세요
- 문서를 확인하세요
- 테스트 예제를 검토하세요

## 추가 리소스

- [API 문서](api_documentation.md) - 완전한 API 참조
- [Python 지원 가이드](python_support.md) - Python/Django/Flask/FastAPI 지원
- [아키텍처 가이드](architecture.md) - 상세한 시스템 설계
- [보안 가이드](security.md) - 보안 기능 및 모범 사례
- [성능 튜닝](performance.md) - 최적화 가이드
- [배포 가이드](deployment.md) - 프로덕션 배포 가이드