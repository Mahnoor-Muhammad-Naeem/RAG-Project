"""
FastAPI app to serve the RAG model.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import yaml
import uvicorn

import sys
import os
sys.path.append(os.path.abspath("../.."))
from src.models.rag_model import RAGModel

# Create FastAPI app
app = FastAPI(
    title="RAG API",
    description="API for the Retrieval-Augmented Generation system",
    version="0.1.0",
)

# Load configuration
def load_config(config_path: str = "../../config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# Setup CORS
config = load_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG model
rag_model = RAGModel()

# Define request models
class QueryRequest(BaseModel):
    query: str
    include_metadata: Optional[bool] = True

class IndexRequest(BaseModel):
    directory_path: str

# Define response models
class DocumentMetadata(BaseModel):
    source: str
    similarity: float

class QueryResponse(BaseModel):
    response: str
    context: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG model.
    
    Args:
        request: The query request containing the query text and options.
        
    Returns:
        The RAG model's response.
    """
    try:
        result = rag_model.generate_response(request.query)
        
        # Exclude metadata if not requested
        if not request.include_metadata:
            result["context"] = []
            result["metadata"] = {}
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_documents(request: IndexRequest):
    """
    Index documents from a directory.
    
    Args:
        request: The index request containing the directory path.
        
    Returns:
        A confirmation message.
    """
    try:
        # Check if directory exists
        if not os.path.isdir(request.directory_path):
            raise HTTPException(status_code=400, detail="Directory not found")
        
        rag_model.index_documents(request.directory_path)
        return {"message": f"Successfully indexed documents from {request.directory_path}"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the app
    config = load_config()
    uvicorn.run(
        "app:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True
    ) 