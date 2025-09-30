"""
Module for retrieving relevant documents based on queries.
"""
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import yaml
import chromadb
from src.embeddings.embedding_generator import EmbeddingGenerator

class Retriever:
    """
    Class for retrieving relevant documents based on queries.
    """
    
    def __init__(self, config_path: str = "../../config/config.yaml"):
        """
        Initialize the Retriever with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.top_k = self.config["retrieval"]["top_k"]
        self.similarity_threshold = self.config["retrieval"]["similarity_threshold"]
        self.embedding_generator = EmbeddingGenerator(config_path)
        self.vector_db_type = self.config["embeddings"]["vector_db"]
        
        # Set up the vector database connection
        if self.vector_db_type == "chromadb":
            self.vector_db = self._setup_chromadb()
        else:
            raise ValueError(f"Unsupported vector database type: {self.vector_db_type}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Dict containing configuration parameters.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    
    def _setup_chromadb(self) -> Any:
        """
        Set up a connection to ChromaDB.
        
        Returns:
            ChromaDB client instance.
        """
        client = chromadb.Client()
        # Create a collection for storing documents
        collection = client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        return collection
    
    def add_documents(self, texts: List[str], 
                     metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add documents to the vector database.
        
        Args:
            texts: List of document texts.
            metadata: Optional metadata for each document.
        """
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        if self.vector_db_type == "chromadb":
            # Format for ChromaDB
            ids = [f"doc_{i}" for i in range(len(texts))]
            
            # Convert embeddings to the correct format if needed
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            # Add to collection
            self.vector_db.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=texts,
                metadatas=metadata if metadata else None
            )
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query text.
            top_k: Number of results to return. If None, use the default from config.
            
        Returns:
            List of retrieved documents with metadata and relevance scores.
        """
        # Use the default top_k if not specified
        if top_k is None:
            top_k = self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        if self.vector_db_type == "chromadb":
            # Search in ChromaDB
            results = self.vector_db.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            for i, (doc_id, document, distance) in enumerate(zip(
                results.get("ids", [[]])[0],
                results.get("documents", [[]])[0],
                results.get("distances", [[]])[0]
            )):
                # Convert distance to similarity score (assuming cosine distance)
                similarity = 1 - distance
                
                # Only include documents above the similarity threshold
                if similarity >= self.similarity_threshold:
                    metadata = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                    
                    formatted_results.append({
                        "id": doc_id,
                        "document": document,
                        "similarity": similarity,
                        "metadata": metadata
                    })
            
            return formatted_results
        
        # This should not be reached due to the check in __init__
        raise ValueError(f"Unsupported vector database type: {self.vector_db_type}")
    
    def rerank_results(self, query: str, initial_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank initial retrieval results using more sophisticated methods.
        
        Args:
            query: The original query text.
            initial_results: List of initially retrieved documents.
            
        Returns:
            Reranked list of retrieved documents.
        """
        if not self.config["retrieval"]["use_reranker"]:
            return initial_results
        
        # This is a placeholder for actual reranking logic
        # In a real implementation, you might use a cross-encoder model here
        print("Reranking not implemented yet, returning original results")
        return initial_results


if __name__ == "__main__":
    # Example usage
    retriever = Retriever()
    
    # Add some example documents
    documents = [
        "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models with external knowledge.",
        "RAG systems combine retrieval mechanisms with text generation for more accurate and contextually grounded responses.",
        "Vector embeddings convert text into numerical vectors that capture semantic meaning.",
        "ChromaDB is a vector database for storing and efficiently searching embeddings.",
        "Document chunking is important for balancing context preservation and retrieval efficiency."
    ]
    
    metadata = [
        {"source": "research_paper", "topic": "RAG"},
        {"source": "documentation", "topic": "RAG"},
        {"source": "textbook", "topic": "embeddings"},
        {"source": "documentation", "topic": "vector_db"},
        {"source": "blog", "topic": "document_processing"}
    ]
    
    # Add documents to the retriever
    retriever.add_documents(documents, metadata)
    
    # Test retrieval
    query = "How does RAG work?"
    results = retriever.retrieve(query)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} relevant documents:")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Document: {result['document']}")
        print(f"Similarity Score: {result['similarity']:.4f}")
        print(f"Metadata: {result['metadata']}") 