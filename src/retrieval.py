import numpy as np


def compute_cos_similarity(
    query_vector,
    document_vectors
):
    return (
        np.dot(
            document_vectors,
            query_vector
        )
        /
        (
            np.linalg.norm(query_vector)
            *
            np.linalg.norm(
                document_vectors,
                axis=1
            )
        )
    )


def retrieve_top_k(
    queries,
    embedded_vectors,
    model,
    top_k=20
):
    ranked_indices = []

    for query in queries:
        query_vector = model.encode(
            query
        )

        cos_similarity = (
            compute_cos_similarity(
                query_vector,
                embedded_vectors
            )
        )

        sorted_indices = np.argsort(
            cos_similarity
        )[::-1]

        top_indices = sorted_indices[
            :top_k
        ]

        ranked_indices.append(
            top_indices
        )

    return ranked_indices


def context_from_indices(
    active_corpus,
    indices
):
    context_items = []

    for index in indices:
        context_items.append({
            "text": active_corpus[
                index
            ]["chunk_text"],
            "document": active_corpus[
                index
            ]["document"],
            "page": active_corpus[
                index
            ]["page"]
        })

    return context_items