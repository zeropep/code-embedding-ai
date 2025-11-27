"""
디버그: test_project 임베딩 테스트
"""
import asyncio
import os
from pathlib import Path

# 환경 변수 설정
os.environ["USE_LOCAL_EMBEDDING_MODEL"] = "true"
os.environ["EMBEDDING_MODEL"] = "jinaai/jina-embeddings-v2-base-code"
os.environ["EMBEDDING_DIMENSIONS"] = "768"

from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.models import EmbeddingConfig
from src.code_parser.python_parser import PythonParser
from src.code_parser.base_parser import ParserConfig


async def main():
    # 설정
    config = EmbeddingConfig()
    print(f"[OK] 설정 로드 완료:")
    print(f"  - use_local_model: {config.use_local_model}")
    print(f"  - model_name: {config.model_name}")
    print(f"  - dimensions: {config.dimensions}")

    # 임베딩 서비스 시작
    embedding_service = EmbeddingService(config)
    await embedding_service.start()
    print(f"[OK] 임베딩 서비스 시작됨")

    # 파일 읽기
    test_file = Path(r"C:\bin\work\vector\test_project\sample.py")
    print(f"\n[OK] 파일 읽기: {test_file}")

    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 파싱
    parser_config = ParserConfig()
    parser = PythonParser(parser_config)
    chunks = parser.parse_file(content, str(test_file), "test_project")
    print(f"[OK] 파싱 완료: {len(chunks)}개 청크")

    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. {chunk.layer_type.value}: {chunk.function_name or chunk.class_name or '(anonymous)'}")
        print(f"     라인 {chunk.start_line}-{chunk.end_line}, {chunk.token_count} 토큰")

    # 임베딩 생성
    print(f"\n[임베딩 생성 시작...]")
    embedded_chunks = await embedding_service.generate_chunk_embeddings(chunks)
    print(f"[OK] 임베딩 생성 완료: {len(embedded_chunks)}개")

    # 결과 확인
    for i, chunk in enumerate(embedded_chunks):
        has_embedding = 'embedding' in chunk.metadata and chunk.metadata['embedding'].get('vector')
        has_error = 'embedding_error' in chunk.metadata

        status = "[OK]" if has_embedding else "[ERROR]"
        print(f"{status} 청크 {i+1}: {chunk.function_name or chunk.class_name or '(anonymous)'}")

        if has_embedding:
            emb = chunk.metadata['embedding']
            print(f"     벡터 길이: {len(emb['vector'])}, 모델: {emb.get('model_version', 'unknown')}")
        elif has_error:
            print(f"     오류: {chunk.metadata['embedding_error']}")

    # 정리
    await embedding_service.stop()
    print(f"\n[OK] 테스트 완료")


if __name__ == "__main__":
    asyncio.run(main())
