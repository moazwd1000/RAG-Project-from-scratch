from pathlib import Path

import nltk
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL_NAME,
    ARTIFACTS_DIR,
)

from src.indexing import build_index


def main():
    # -----------------------------------------------------
    # 1. Make sure sentence tokenizer data exists
    # -----------------------------------------------------

    nltk.download(
        "punkt",
        quiet=True
    )

    nltk.download(
        "punkt_tab",
        quiet=True
    )


    # -----------------------------------------------------
    # 2. Find PDFs
    # -----------------------------------------------------

    data_dir = Path("data")

    documents = sorted(
        data_dir.glob("*.pdf")
    )

    if not documents:
        raise FileNotFoundError(
            "No PDF files found inside data/"
        )

    print(
        f"Found {len(documents)} PDF files:"
    )

    for document in documents:
        print(
            f"- {document}"
        )


    # -----------------------------------------------------
    # 3. Load embedding model
    # -----------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


    # -----------------------------------------------------
    # 4. Build + save index
    # -----------------------------------------------------

    print(
        "\nBuilding index..."
    )

    (
        active_corpus,
        corpus_embeddings
    ) = build_index(
        documents=documents,
        embedding_model=embedding_model,
        chunk_size=1000,
        save=True,
        output_dir=ARTIFACTS_DIR
    )


    # -----------------------------------------------------
    # 5. Summary
    # -----------------------------------------------------

    print(
        "\nIndexing complete."
    )

    print(
        "Chunks:",
        len(active_corpus)
    )

    print(
        "Embedding shape:",
        corpus_embeddings.shape
    )


if __name__ == "__main__":
    main()