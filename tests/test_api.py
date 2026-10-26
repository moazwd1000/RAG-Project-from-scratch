from fastapi.testclient import TestClient
from api.main import app,rag_pipeline
from src.generation import ModelServerError

client = TestClient(app)

def test_empty_question_returns_400():
    response = client.post("/query",json={"question": " "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


def test_model_server_unavailable_returns_503(monkeypatch):
    def mock_post(question):
        raise ModelServerError("Model server unavailable")
    monkeypatch.setattr(rag_pipeline,"query",mock_post)
    response = client.post("/query",json={"question": "What is AI?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model server unavailable"
    

def test_successful_query_returns_200(monkeypatch):

    def mock_post(question):
        return {
            "question": question,
            "answer": "Artificial Intelligence is the simulation of human intelligence in machines.",
            "sources": [
                {
                    "document": "data\\1706.03762v7.pdf",
                    "page": 3
                }
            ],
            "latency": {
                "retrieval": 0.1,
                "generation": 0.2,
                "total": 0.3
            }
        }

    monkeypatch.setattr(
        rag_pipeline,
        "query",
        mock_post
    )

    response = client.post(
        "/query",
        json={"question": "What is AI?"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is AI?"
    assert data["answer"] == "Artificial Intelligence is the simulation of human intelligence in machines."

    assert "sources" in data
    assert isinstance(data["sources"], list)

    assert "latency" in data
    assert "retrieval" in data["latency"]
    assert "generation" in data["latency"]
    assert "total" in data["latency"]

def test_internal_error_returns_500(monkeypatch):

    def mock_post(question):
        raise Exception("Something broke")

    monkeypatch.setattr(
        rag_pipeline,
        "query",
        mock_post
    )

    response = client.post(
        "/query",
        json={"question": "What is AI?"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"