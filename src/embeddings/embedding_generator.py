"""
Module for generating embeddings from text using various models.
"""
import os
from typing import List, Dict, Any, Union, Optional
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """
    Class for generating embeddings from text using various models.
    """
    
    def __init__(self, config_path: str = "../../config/config.yaml"):
        """
        Initialize the EmbeddingGenerator with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.model_name = self.config["embeddings"]["model"]
        self.model = self._load_model()
        
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
    
    def _load_model(self) -> Any:
        """
        Load the embedding model.
        
        Returns:
            The loaded model.
        """
        return SentenceTransformer(self.model_name)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text.
        
        Args:
            text: The text to generate an embedding for.
            
        Returns:
            The embedding vector as a numpy array.
        """
        return self.model.encode(text)
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for.
            
        Returns:
            List of embedding vectors as numpy arrays.
        """
        return self.model.encode(texts)
    
    def generate_and_save_embeddings(self, texts: List[str], 
                                    output_dir: str, 
                                    metadata: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate embeddings for a list of texts and save them to a file.
        
        Args:
            texts: List of texts to generate embeddings for.
            output_dir: Directory to save the embeddings to.
            metadata: Optional metadata for each text.
            
        Returns:
            Path to the saved embeddings file.
        """
        embeddings = self.generate_embeddings(texts)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings
        output_path = os.path.join(output_dir, "embeddings.npz")
        if metadata:
            np.savez(output_path, embeddings=embeddings, metadata=metadata)
        else:
            np.savez(output_path, embeddings=embeddings)
        
        return output_path


if __name__ == "__main__":
    # Example usage
    generator = EmbeddingGenerator()
    
    # Test with some example texts
    texts = [
        "This is a sample sentence for embedding.",
        "Another example of text that will be converted to an embedding vector.",
    ]
    
    embeddings = generator.generate_embeddings(texts)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings[0].shape}")
    
    # Example of saving embeddings
    output_dir = "../../data/embeddings"
    metadata = [{"id": 1, "source": "example"}, {"id": 2, "source": "example"}]
    
    output_path = generator.generate_and_save_embeddings(texts, output_dir, metadata)
    print(f"Embeddings saved to {output_path}") 