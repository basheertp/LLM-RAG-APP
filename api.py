from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel
from typing import Optional
from utils.inference import predict_rag
from utils.build_rag import RAG

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_FOLDER = "source_data"
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
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