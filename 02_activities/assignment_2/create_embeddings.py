"""
create_embeddings.py
====================
Run this script ONCE to generate and persist the ChromaDB knowledge base.

Usage:
    cd 05_src/assignment_chat
    python create_embeddings.py

Prerequisites:
    - OPENAI_API_KEY must be set in your environment or in ../05_src/.secrets
    - pip packages: chromadb, openai, pandas (included in course setup)

What it does:
    1. Reads data/ai_concepts.csv (25 AI/ML concept entries)
    2. Sends each description to OpenAI text-embedding-3-small
    3. Stores embeddings + metadata in chroma_db/ (file-persistent ChromaDB)

After running this script, the app.py will load the knowledge base
automatically without re-generating embeddings.

Embedding model: text-embedding-3-small
    - 1536-dimensional dense vectors
    - Chosen for cost efficiency and strong semantic quality
    - Batch ingestion via ChromaDB's built-in OpenAI embedding function
"""

import os
import sys
import pathlib

# Support loading from .secrets file if dotenv is available
try:
    from dotenv import load_dotenv
    secrets_path = pathlib.Path(__file__).parent.parent / ".secrets"
    if secrets_path.exists():
        load_dotenv(secrets_path)
        print(f"Loaded secrets from {secrets_path}")
except ImportError:
    pass  # dotenv not available, rely on env vars


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Set it with:  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Ensure we can import from the services package
    here = pathlib.Path(__file__).parent
    sys.path.insert(0, str(here))

    from services.knowledge_service import initialize_knowledge_base

    print("=" * 50)
    print("NEXUS Knowledge Base — Embedding Generator")
    print("=" * 50)

    success = initialize_knowledge_base()

    if success:
        print("\n✅  Knowledge base created successfully.")
        print(f"    ChromaDB stored at: {here / 'chroma_db'}")
        print("\nYou can now run:  python app.py")
    else:
        print("\n❌  Failed to create knowledge base. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
