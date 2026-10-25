import json
from pathlib import Path

import numpy as np


def save_index(
    active_corpus,
    corpus_embeddings,
    output_dir="artifacts"
):
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    corpus_path = (
        output_path / "corpus.json"
    )

    embeddings_path = (
        output_path / "embeddings.npy"
    )

    # Save corpus + metadata
    with corpus_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            active_corpus,
            file,
            ensure_ascii=False,
            indent=2
        )

    # Save embeddings
    np.save(
        embeddings_path,
        corpus_embeddings
    )

    print(
        f"Saved corpus to: {corpus_path}"
    )

    print(
        f"Saved embeddings to: "
        f"{embeddings_path}"
    )


def load_index(
    input_dir="artifacts"
):
    input_path = Path(input_dir)

    corpus_path = (
        input_path / "corpus.json"
    )

    embeddings_path = (
        input_path / "embeddings.npy"
    )

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: "
            f"{corpus_path}"
        )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: "
            f"{embeddings_path}"
        )

    with corpus_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        active_corpus = json.load(file)

    corpus_embeddings = np.load(
        embeddings_path
    )

    return (
        active_corpus,
        corpus_embeddings
    )