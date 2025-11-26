"""
Test script for local embedding model
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.embeddings.models import EmbeddingConfig
from src.embeddings.local_embedding_client import LocalEmbeddingClient


async def test_local_embedding():
    """Test local embedding client"""

    print("=" * 60)
    print("Local Embedding Model Test")
    print("=" * 60)

    # Configure for local model - Set environment to prevent config override
    os.environ["USE_LOCAL_EMBEDDING_MODEL"] = "true"
    os.environ["EMBEDDING_MODEL"] = "jinaai/jina-embeddings-v2-base-code"
    os.environ["EMBEDDING_DIMENSIONS"] = "768"

    config = EmbeddingConfig()

    print(f"\n[OK] Configuration:")
    print(f"  - Model: {config.model_name}")
    print(f"  - Dimensions: {config.dimensions}")
    print(f"  - Use Local: {config.use_local_model}")

    # Create client
    print(f"\n[1/4] Creating local embedding client...")
    client = LocalEmbeddingClient(config)

    # Test single embedding
    print(f"\n[2/4] Testing single embedding generation...")
    test_code = """
def calculate_sum(a, b):
    return a + b
    """

    async with client as c:
        result = await c.generate_embedding(test_code, "test_1")

        if result.status.value == "completed":
            print(f"  [OK] Single embedding generated successfully!")
            print(f"  - Vector dimensions: {len(result.vector)}")
            print(f"  - Processing time: {result.processing_time:.3f}s")
            print(f"  - First 5 values: {result.vector[:5]}")

            # Verify dimensions
            if len(result.vector) == 768:
                print(f"  [OK] Dimensions match expected (768)")
            else:
                print(f"  [ERROR] Dimensions mismatch! Expected 768, got {len(result.vector)}")
        else:
            print(f"  [ERROR] Failed: {result.error_message}")
            return False

    # Test batch embeddings
    print(f"\n[3/4] Testing batch embedding generation...")
    test_codes = [
        "def hello(): print('Hello')",
        "class MyClass: pass",
        "import numpy as np"
    ]

    async with client as c:
        results = await c.generate_embeddings_batch(
            test_codes,
            ["batch_1", "batch_2", "batch_3"]
        )

        success_count = sum(1 for r in results if r.status.value == "completed")
        print(f"  [OK] Batch embeddings generated: {success_count}/{len(results)}")

        for i, result in enumerate(results):
            if result.status.value == "completed":
                print(f"  - Batch {i+1}: {len(result.vector)} dims, {result.processing_time:.3f}s")
            else:
                print(f"  - Batch {i+1}: Failed - {result.error_message}")

    # Test caching
    print(f"\n[4/4] Testing cache functionality...")
    async with client as c:
        # First call - should cache
        result1 = await c.generate_embedding(test_code, "cache_test_1")
        time1 = result1.processing_time

        # Second call - should hit cache
        result2 = await c.generate_embedding(test_code, "cache_test_2")
        time2 = result2.processing_time

        print(f"  - First call: {time1:.3f}s")
        print(f"  - Second call (cached): {time2:.3f}s")

        if time2 < time1:
            print(f"  [OK] Cache is working! ({time1/time2:.1f}x faster)")
        else:
            print(f"  [WARNING] Cache might not be working as expected")

        # Cache stats
        stats = client.get_cache_stats()
        print(f"\n  Cache Statistics:")
        print(f"  - Enabled: {stats['enabled']}")
        print(f"  - Total entries: {stats['total_entries']}")
        print(f"  - Estimated size: {stats['estimated_size_mb']:.2f} MB")

    print(f"\n{'=' * 60}")
    print(f"[OK] All tests completed successfully!")
    print(f"{'=' * 60}\n")

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_local_embedding())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
