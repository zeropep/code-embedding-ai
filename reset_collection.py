"""Reset ChromaDB collection for new dimensions"""
import chromadb

# Use PersistentClient for local storage
client = chromadb.PersistentClient(path="./chroma_db")

# Delete the old collection
try:
    client.delete_collection("code_embeddings")
    print("Successfully deleted 'code_embeddings' collection")
except Exception as e:
    print(f"Note: {e}")

# List remaining collections
collections = client.list_collections()
print(f"\nCurrent collections: {[c.name for c in collections]}")

# Create new collection with 1536 dimensions
try:
    from chromadb.utils import embedding_functions
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="jinaai/jina-code-embeddings-1.5b"
    )

    collection = client.create_collection(
        name="code_embeddings",
        embedding_function=embedding_function,
        metadata={
            "hnsw:space": "cosine",
            "description": "Code embeddings (1536 dimensions)"
        }
    )
    print(f"Created new collection 'code_embeddings' with 1536 dimensions")
except Exception as e:
    print(f"Error creating collection: {e}")
