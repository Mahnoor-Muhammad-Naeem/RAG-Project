"""
Module implementing the RAG model that combines retrieval and generation.
"""
import os
from typing import List, Dict, Any, Optional
import yaml
from openai import OpenAI
import dotenv

from src.data.document_loader import DocumentLoader
from src.retrieval.retriever import Retriever

# Load environment variables
dotenv.load_dotenv()

class RAGModel:
    """
    RAG model that combines retrieval and generation for enhanced responses.
    """
    
    def __init__(self, config_path: str = "../../config/config.yaml"):
        """
        Initialize the RAG model with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.retriever = Retriever(config_path)
        self.document_loader = DocumentLoader(config_path)
        
        # OpenAI client setup
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = self.config["llm"]["model"]
        self.temperature = self.config["llm"]["temperature"]
        self.max_tokens = self.config["llm"]["max_tokens"]
        self.system_prompt = self.config["llm"]["system_prompt"]
    
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
    
    def index_documents(self, directory_path: str) -> None:
        """
        Index documents from a directory for retrieval.
        
        Args:
            directory_path: Path to the directory containing documents.
        """
        # Load documents
        document_dict = self.document_loader.load_directory(directory_path)
        
        # Process each document
        for file_path, content in document_dict.items():
            # Create basic metadata
            metadata = {"source": file_path}
            
            # Add documents to retriever (in a real system, you'd do chunking here)
            self.retriever.add_documents([content], [metadata])
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to a query using RAG.
        
        Args:
            query: The query text.
            
        Returns:
            Dict containing the response, retrieved context, and metadata.
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query)
        
        # If no relevant documents were found
        if not retrieved_docs:
            return {
                "response": "I don't have enough information to answer that question.",
                "context": [],
                "metadata": {}
            }
        
        # Create context from retrieved documents
        context = "\n\n".join([doc["document"] for doc in retrieved_docs])
        
        # Generate response using LLM
        prompt = f"""
        Based on the following context, answer the question. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer that question."
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content
        
        # Return response with context and metadata
        return {
            "response": response_text,
            "context": [doc["document"] for doc in retrieved_docs],
            "metadata": {
                "sources": [doc["metadata"] for doc in retrieved_docs],
                "similarity_scores": [doc["similarity"] for doc in retrieved_docs]
            }
        }


if __name__ == "__main__":
    # Example usage
    rag_model = RAGModel()
    
    # Index sample documents
    sample_documents = [
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
    rag_model.retriever.add_documents(sample_documents, metadata)
    
    # Test the model with a query
    query = "Can you explain how RAG works?"
    response_data = rag_model.generate_response(query)
    
    print(f"Query: {query}")
    print(f"\nResponse: {response_data['response']}")
    print("\nRetrieved Context:")
    for i, context in enumerate(response_data['context']):
        print(f"  {i+1}. {context}")
    
    print("\nMetadata:")
    for i, (source, score) in enumerate(zip(
        response_data['metadata']['sources'], 
        response_data['metadata']['similarity_scores']
    )):
        print(f"  {i+1}. Source: {source}, Similarity: {score:.4f}") 