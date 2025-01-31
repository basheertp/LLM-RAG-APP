from fastapi import FastAPI, HTTPException
import uvicorn
import os
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel
from typing import Optional
from utils.inference import predict_rag

load_dotenv()

app = FastAPI()

class InputData(BaseModel):
    question: str

@app.post("/predict")
async def predict(data: InputData):
    """
    Accepts a JSON input with a "text" field and returns a processed response.
    """
    try:
        question = data.question
        response = predict_rag(question)
        return response
    except Exception as e:
        # Handle errors and return a 500 status code
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # mounting at the root path
    uvicorn.run(
        app="api:app",
        host=os.getenv("UVICORN_HOST"),  
        port=int(os.getenv("UVICORN_PORT"))
    )