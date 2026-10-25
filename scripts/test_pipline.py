from src.pipeline import RAGPipeline

pipeline = RAGPipeline()

result = pipeline.query(
    "Why is masking necessary in the Transformer decoder?"
)

print("\nQUESTION:")
print(result["question"])

print("\nANSWER:")
print(result["answer"])

print("\nCONTEXT:")
for i, item in enumerate(result["context"], start=1):
    print(f"\n--- Context {i} ---")
    print("Document:", item["document"])
    print("Page:", item["page"])
    print(item["text"])