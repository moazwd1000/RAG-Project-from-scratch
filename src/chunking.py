from nltk.tokenize import sent_tokenize


def chunk_documents_sentences(
    document_list,
    chunk_size=1000
):
    results = []

    for doc in document_list:
        page_text = doc["text"]

        sentences = sent_tokenize(page_text)

        current_chunk = []
        chunk_id = 0

        for sentence in sentences:

            # If chunk is empty, add the sentence first
            if not current_chunk:
                current_chunk.append(sentence)
                continue

            current_text = " ".join(current_chunk)

            if (
                len(current_text)
                + len(sentence)
                + 1
                <= chunk_size
            ):
                current_chunk.append(sentence)

            else:
                results.append({
                    "document": doc["document"],
                    "page": doc["page"],
                    "chunk_id": chunk_id,
                    "chunk_text": " ".join(
                        current_chunk
                    )
                })

                chunk_id += 1

                # Preserve one-sentence overlap
                last_sentence = current_chunk[-1]

                current_chunk = [
                    last_sentence,
                    sentence
                ]

        # Save final chunk
        if current_chunk:
            results.append({
                "document": doc["document"],
                "page": doc["page"],
                "chunk_id": chunk_id,
                "chunk_text": " ".join(
                    current_chunk
                )
            })

    return results