from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline import RAGPipeline

rag_pipeline = RAGPipeline()
app = FastAPI()



class Question(BaseModel):
    question:str


@app.post("/query")
def query(request:Question):
    response = rag_pipeline.query(request.question)
    return {"question": request.question, "context": response["context"], "answer": response["answer"]}


