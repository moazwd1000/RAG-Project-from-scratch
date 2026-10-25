from dotenv import load_dotenv
import os

load_dotenv()
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")

if(not MODEL_SERVER_URL or not MODEL_API_KEY):
    raise ValueError("MODEL_SERVER_URL and MODEL_API_KEY must be set in the .env file")


EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 5

ARTIFACTS_DIR = "artifacts"