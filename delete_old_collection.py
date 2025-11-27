"""Delete old ChromaDB collection to recreate with new dimensions"""
import chromadb

# Connect to ChromaDB
client = chromadb.HttpClient(host="localhost", port=8000)

# Delete the old collection
try:
    client.delete_collection("code_embeddings")
    print("Successfully deleted 'code_embeddings' collection")
except Exception as e:
    print(f"Error deleting collection: {e}")

# List remaining collections
collections = client.list_collections()
print(f"\nRemaining collections: {[c.name for c in collections]}")
