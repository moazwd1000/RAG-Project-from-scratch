from src.pipeline import RAGPipeline


def main():
    rag = RAGPipeline()

    question = (
        "Why is masking necessary "
        "in the Transformer decoder?"
    )

    result = rag.retrieve(question)

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(result["question"])

    print("\n" + "=" * 80)
    print("DENSE TOP 20")
    print("=" * 80)
    print(result["dense_top_k"])

    print("\n" + "=" * 80)
    print("RERANKED TOP 5")
    print("=" * 80)
    print(result["reranked_top_k"])

    print("\n" + "=" * 80)
    print("FINAL CONTEXT")
    print("=" * 80)

    for i, item in enumerate(
        result["context"],
        start=1
    ):
        print(f"\n--- Context {i} ---")
        print("Document:", item["document"])
        print("Page:", item["page"])
        print(item["text"])


if __name__ == "__main__":
    main()