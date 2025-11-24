#!/usr/bin/env python3
"""
Python 지원 기능 테스트 스크립트
사용법: python test_python_support.py [Python 파일 경로]
"""

import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.code_parser.parser_factory import ParserFactory
from src.code_parser.python_parser import PythonParser
from src.code_parser.python_framework_detector import PythonFrameworkDetector
from src.code_parser.python_metadata import PythonMetadataExtractor
from src.code_parser.models import ParserConfig
from src.security.secret_detector import SecretDetector
from src.security.models import SecurityConfig


def test_parser_factory():
    """파서 팩토리 테스트"""
    print("=" * 60)
    print("1. 파서 팩토리 테스트")
    print("=" * 60)

    config = ParserConfig(min_tokens=10, max_tokens=500)
    factory = ParserFactory(config)

    print(f"지원 확장자: {factory.get_supported_extensions()}")
    print(f"파서 통계: {factory.get_parser_stats()}")
    print()


def test_python_parser(file_path: str = None):
    """Python 파서 테스트"""
    print("=" * 60)
    print("2. Python 파서 테스트")
    print("=" * 60)

    # 샘플 코드 (또는 파일에서 읽기)
    if file_path and Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            sample_code = f.read()
        print(f"파일: {file_path}")
    else:
        sample_code = '''
from django.db import models
from rest_framework import serializers

class Article(models.Model):
    """블로그 게시물 모델"""
    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    def get_summary(self) -> str:
        """처음 100자 반환"""
        return self.content[:100]


class ArticleSerializer(serializers.ModelSerializer):
    """Article 직렬화"""
    class Meta:
        model = Article
        fields = ['id', 'title', 'content']
'''
        print("샘플 Django 코드 사용")

    print()

    # 파서 생성 및 파싱
    config = ParserConfig(min_tokens=10, max_tokens=500)
    parser = PythonParser(config)

    # 임시 파일로 파싱 테스트
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as f:
        f.write(sample_code)
        temp_path = f.name

    try:
        result = parser.parse_file(Path(temp_path))

        if result:
            print(f"✅ 파싱 성공!")
            print(f"  - 언어: {result.language.value}")
            print(f"  - 총 라인: {result.total_lines}")
            print(f"  - 청크 수: {result.total_chunks}")
            print(f"  - 총 토큰: {result.total_tokens}")
            print()

            print("청크 목록:")
            for i, chunk in enumerate(result.chunks, 1):
                print(f"  [{i}] {chunk.class_name or ''}.{chunk.function_name or '(전체)'}")
                print(f"      라인: {chunk.start_line}-{chunk.end_line}")
                print(f"      레이어: {chunk.layer_type.value}")
                print(f"      토큰: {chunk.token_count}")
                if chunk.metadata:
                    if chunk.metadata.get('decorators'):
                        print(f"      데코레이터: {chunk.metadata['decorators']}")
                    if chunk.metadata.get('docstring'):
                        print(f"      문서: {chunk.metadata['docstring'][:50]}...")
                print()
        else:
            print("❌ 파싱 실패")
    finally:
        import os
        os.unlink(temp_path)


def test_framework_detector():
    """프레임워크 감지 테스트"""
    print("=" * 60)
    print("3. 프레임워크 감지 테스트")
    print("=" * 60)

    detector = PythonFrameworkDetector()

    # Django 테스트
    django_imports = {'modules': {'django', 'rest_framework'}, 'from_imports': {}}
    framework, confidence = detector.detect_framework_from_imports(django_imports)
    print(f"Django imports 감지: {framework.value} (신뢰도: {confidence:.2f})")

    # Flask 테스트
    flask_imports = {'modules': {'flask', 'flask_sqlalchemy'}, 'from_imports': {}}
    framework, confidence = detector.detect_framework_from_imports(flask_imports)
    print(f"Flask imports 감지: {framework.value} (신뢰도: {confidence:.2f})")

    # FastAPI 테스트
    fastapi_imports = {'modules': {'fastapi', 'pydantic'}, 'from_imports': {}}
    framework, confidence = detector.detect_framework_from_imports(fastapi_imports)
    print(f"FastAPI imports 감지: {framework.value} (신뢰도: {confidence:.2f})")
    print()


def test_security_scanner():
    """보안 스캐너 테스트"""
    print("=" * 60)
    print("4. 보안 스캐너 테스트")
    print("=" * 60)

    config = SecurityConfig(enabled=True, sensitivity_threshold=0.5)
    detector = SecretDetector(config)

    # 시크릿이 포함된 샘플 코드
    sample_settings = '''
SECRET_KEY = 'django-insecure-abcdefghijklmnopqrstuvwxyz123456'
DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'PASSWORD': 'super_secret_password_123',
    }
}

AWS_ACCESS_KEY_ID = 'AKIAIOSFODNN7EXAMPLE'
'''

    secrets = detector.detect_python_secrets(sample_settings, 'settings.py', 'django')

    print(f"발견된 시크릿: {len(secrets)}개")
    for secret in secrets:
        print(f"  - 타입: {secret.secret_type.value}")
        print(f"    라인: {secret.line_number}")
        print(f"    패턴: {secret.pattern_name}")
        print(f"    신뢰도: {secret.confidence:.2f}")
        print(f"    내용: {secret.content[:20]}..." if len(secret.content) > 20 else f"    내용: {secret.content}")
        print()


def test_metadata_extractor():
    """메타데이터 추출 테스트"""
    print("=" * 60)
    print("5. 메타데이터 추출 테스트")
    print("=" * 60)

    extractor = PythonMetadataExtractor()

    sample_code = '''
from typing import List, Optional

def calculate_sum(numbers: List[int]) -> int:
    """숫자 합계 계산"""
    return sum(numbers)

class Calculator:
    """계산기 클래스"""

    def add(self, a: int, b: int) -> int:
        """두 수를 더함"""
        return a + b

    def divide(self, a: float, b: float) -> Optional[float]:
        """나눗셈 (0으로 나누면 None)"""
        if b == 0:
            return None
        return a / b
'''

    metadata = extractor.extract_metadata(sample_code, 'calculator.py')

    print("imports:")
    print(f"  표준 라이브러리: {metadata['imports']['standard_library']}")
    print(f"  서드파티: {metadata['imports']['third_party']}")

    print("\n타입 힌트 분석:")
    th = metadata['type_hints']
    print(f"  함수 수: {th['total_functions']}")
    print(f"  반환 타입 커버리지: {th['return_type_coverage']:.0%}")
    print(f"  파라미터 타입 커버리지: {th['parameter_type_coverage']:.0%}")

    print("\n복잡도 분석:")
    cx = metadata['complexity']
    print(f"  코드 라인: {cx['lines_of_code']}")
    print(f"  순환 복잡도: {cx['cyclomatic_complexity']}")
    print(f"  함수 수: {cx['number_of_functions']}")
    print(f"  클래스 수: {cx['number_of_classes']}")
    print()


def main():
    print()
    print("🐍 Python 지원 기능 테스트")
    print("=" * 60)
    print()

    # 명령행 인자로 파일 경로 받기
    file_path = sys.argv[1] if len(sys.argv) > 1 else None

    test_parser_factory()
    test_python_parser(file_path)
    test_framework_detector()
    test_security_scanner()
    test_metadata_extractor()

    print("=" * 60)
    print("✅ 모든 테스트 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
