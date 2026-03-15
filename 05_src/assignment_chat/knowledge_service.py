"""
Service 2: Semantic Knowledge Base
Uses ChromaDB with file persistence + OpenAI embeddings (text-embedding-3-small).
The knowledge base covers 25 AI/ML concepts from data/ai_concepts.csv.

On first use, call initialize_knowledge_base() to create embeddings and persist them.
Subsequent runs load directly from the persisted ChromaDB directory.
"""

import os
import pathlib
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# Paths (relative to this file's parent directory)
_HERE      = pathlib.Path(__file__).parent.parent   # assignment_chat/
DB_PATH    = str(_HERE / "chroma_db")
CSV_PATH   = str(_HERE / "data" / "ai_concepts.csv")
COLLECTION = "ai_concepts"

_collection_cache = None   # module-level singleton


def _get_embedding_function():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )


def initialize_knowledge_base() -> bool:
    """
    Create (or recreate) the ChromaDB collection from ai_concepts.csv.
    Called automatically by get_collection() if the DB is empty.
    Returns True on success, False on failure.
    """
    global _collection_cache
    try:
        print("[KnowledgeService] Building knowledge base from CSV …")
        df = pd.read_csv(CSV_PATH)
        ef = _get_embedding_function()

        client = chromadb.PersistentClient(path=DB_PATH)

        # Drop existing collection if present so we get a clean slate
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        collection.add(
            ids=df["id"].astype(str).tolist(),
            documents=df["description"].tolist(),
            metadatas=[
                {"title": row["title"], "category": row["category"]}
                for _, row in df.iterrows()
            ],
        )

        print(f"[KnowledgeService] Added {len(df)} documents. DB saved to {DB_PATH}")
        _collection_cache = collection
        return True

    except Exception as exc:
        print(f"[KnowledgeService] Initialization failed: {exc}")
        return False


def get_collection():
    """Return the ChromaDB collection, initializing if necessary."""
    global _collection_cache
    if _collection_cache is not None:
        return _collection_cache

    ef = _get_embedding_function()
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        col = client.get_collection(name=COLLECTION, embedding_function=ef)
        # Verify it actually has data
        if col.count() == 0:
            raise ValueError("Collection is empty")
        _collection_cache = col
        print(f"[KnowledgeService] Loaded existing collection ({col.count()} docs).")
        return col
    except Exception:
        # Collection missing or empty — build it now
        success = initialize_knowledge_base()
        if not success:
            raise RuntimeError("Could not load or build the knowledge base.")
        return _collection_cache


def search_knowledge_base(query: str, n_results: int = 3) -> dict:
    """
    Semantic search over the AI/ML knowledge base.

    Args:
        query:     Natural language question or topic.
        n_results: Number of results to return (default 3).

    Returns:
        A dict with 'results' list (each entry: title, category, description, distance)
        or 'error' key if something went wrong.
    """
    try:
        collection = get_collection()
        raw = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append({
                "title":       meta.get("title", ""),
                "category":    meta.get("category", ""),
                "description": doc,
                "relevance":   round(1 - dist, 3),   # convert cosine distance → similarity
            })

        return {"results": results, "query": query}

    except Exception as exc:
        print(f"[KnowledgeService] Search error: {exc}")
        return {"error": str(exc)}
