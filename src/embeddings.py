def create_embeddings(
    model,
    active_corpus
):
    chunk_list = []

    for chunk in active_corpus:
        chunk_list.append(
            chunk["chunk_text"]
        )

    vectors = model.encode(
        chunk_list
    )

    return vectors