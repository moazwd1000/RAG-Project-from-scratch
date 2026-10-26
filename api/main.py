from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from src.pipeline import RAGPipeline
from src.generation import ModelServerError

rag_pipeline = RAGPipeline()
app = FastAPI()



class Question(BaseModel):
    question:str


@app.post("/query")
def query(request:Question):
    if request.question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        response = rag_pipeline.query(request.question)
    except ModelServerError as e:
        raise HTTPException(status_code=503,detail="Model server unavailable")
    except Exception as e:
        raise HTTPException(status_code=500,detail="Internal server error")
    return response


