def reranking(
    query,
    active_corpus,
    retrieval_results,
    reranker,
    top_k=5
):
    cross_encoder_list = []
    candidate_indices = []

    for result in retrieval_results:
        index = result[0]

        candidate_indices.append(
            index
        )

        text = active_corpus[
            index
        ]["chunk_text"]

        cross_encoder_list.append(
            [query, text]
        )

    scores = reranker.predict(
        cross_encoder_list
    )

    sorted_positions = (
        scores.argsort()[::-1]
    )

    reranked_results = []

    for position in sorted_positions[
        :top_k
    ]:
        chunk_index = (
            candidate_indices[
                position
            ]
        )

        reranker_score = scores[
            position
        ]

        reranked_results.append(
            (
                chunk_index,
                float(reranker_score)
            )
        )

    return reranked_results


