from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

from src.config import (
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    ARTIFACTS_DIR,
    MODEL_SERVER_URL, 
    MODEL_API_KEY
)

from src.persistence import load_index
from src.retrieval import (
    retrieve_top_k,
    context_from_indices,
)

from src.reranking import reranking
from src.generation import generation

class RAGPipeline:

    def __init__(
        self,
        artifacts_dir=ARTIFACTS_DIR
    ):
        print("Loading embedding model...")

        self.embedding_model = (
            SentenceTransformer(
                EMBEDDING_MODEL_NAME
            )
        )

        print("Loading reranker...")

        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME
        )

        print("Loading saved index...")

        (
            self.active_corpus,
            self.corpus_embeddings
        ) = load_index(
            artifacts_dir
        )

        print(
            "RAG pipeline ready."
        )


    def retrieve(
        self,
        question,
        retrieval_top_k=RETRIEVAL_TOP_K,
        rerank_top_k=RERANK_TOP_K
    ):
        # -----------------------------------------
        # 1. Dense retrieval
        # -----------------------------------------

        dense_results = retrieve_top_k(
            [question],
            self.corpus_embeddings,
            self.embedding_model,
            top_k=retrieval_top_k
        )

        dense_indices = [
            int(index)
            for index in dense_results[0]
        ]


        # -----------------------------------------
        # 2. Prepare candidates for reranker
        # -----------------------------------------

        candidates = [
            (index, 0.0)
            for index in dense_indices
        ]


        # -----------------------------------------
        # 3. Cross-encoder reranking
        # -----------------------------------------

        reranked_results = reranking(
            question,
            self.active_corpus,
            candidates,
            self.reranker,
            top_k=rerank_top_k
        )

        final_indices = [
            int(result[0])
            for result in reranked_results
        ]


        # -----------------------------------------
        # 4. Build final context
        # -----------------------------------------

        context = context_from_indices(
            self.active_corpus,
            final_indices
        )


        return {
            "question": question,
            "dense_top_k": dense_indices,
            "reranked_top_k": final_indices,
            "context": context,
        }
    
    def query(self,question):
        retrieval_result = self.retrieve(question)
        context = retrieval_result["context"]
        response = generation(question,context,MODEL_SERVER_URL,MODEL_API_KEY)
        return {"question": question, "context": context, "answer": response}
    
    


