from src.ingestion import (
    load_document,
    cleaning,
)

from src.chunking import (
    chunk_documents_sentences,
)

from src.embeddings import (
    create_embeddings,
)

from src.persistence import (
    save_index,
)


def build_index(
    documents,
    embedding_model,
    chunk_size=1000,
    save=True,
    output_dir="artifacts"
):
    # 1. Load PDF pages
    loaded_documents = load_document(
        documents
    )

    # 2. Clean extracted text
    cleaned_documents = cleaning(
        loaded_documents
    )

    # 3. Create sentence-aware chunks
    active_corpus = (
        chunk_documents_sentences(
            cleaned_documents,
            chunk_size=chunk_size
        )
    )

    # 4. Embed chunks
    corpus_embeddings = create_embeddings(
        embedding_model,
        active_corpus
    )

    # 5. Persist artifacts
    if save:
        save_index(
            active_corpus,
            corpus_embeddings,
            output_dir=output_dir
        )

    return (
        active_corpus,
        corpus_embeddings
    )