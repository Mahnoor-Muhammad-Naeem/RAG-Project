"""
Document loader module for processing various file types.
"""
import os
from typing import List, Dict, Any, Optional
import yaml

class DocumentLoader:
    """
    Class for loading and processing documents from various sources.
    """
    
    def __init__(self, config_path: str = "../../config/config.yaml"):
        """
        Initialize the DocumentLoader with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.supported_file_types = self.config["document_processing"]["supported_file_types"]
        
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
    
    def load_document(self, file_path: str) -> str:
        """
        Load a document from a file path.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            The text content of the document.
            
        Raises:
            ValueError: If the file type is not supported.
        """
        file_extension = os.path.splitext(file_path)[1].lower().replace(".", "")
        
        if file_extension not in self.supported_file_types:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        if file_extension == "pdf":
            return self._load_pdf(file_path)
        elif file_extension == "txt":
            return self._load_text(file_path)
        elif file_extension == "docx":
            return self._load_docx(file_path)
        elif file_extension == "html":
            return self._load_html(file_path)
        
        raise ValueError(f"No handler implemented for supported file type: {file_extension}")
    
    def _load_pdf(self, file_path: str) -> str:
        """
        Load text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            The text content of the PDF.
        """
        from PyPDF2 import PdfReader
        
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def _load_text(self, file_path: str) -> str:
        """
        Load text content from a text file.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            The text content of the file.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    
    def _load_docx(self, file_path: str) -> str:
        """
        Load text content from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file.
            
        Returns:
            The text content of the DOCX file.
        """
        import docx2txt
        return docx2txt.process(file_path)
    
    def _load_html(self, file_path: str) -> str:
        """
        Load text content from an HTML file.
        
        Args:
            file_path: Path to the HTML file.
            
        Returns:
            The text content of the HTML file.
        """
        from bs4 import BeautifulSoup
        
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file.read(), "html.parser")
            return soup.get_text(separator="\n", strip=True)
    
    def load_directory(self, directory_path: str) -> Dict[str, str]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory.
            
        Returns:
            Dictionary mapping file paths to their text content.
        """
        result = {}
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1].lower().replace(".", "")
                
                if file_extension in self.supported_file_types:
                    try:
                        result[file_path] = self.load_document(file_path)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        return result


if __name__ == "__main__":
    # Example usage
    loader = DocumentLoader()
    
    # Test with a sample text file
    sample_text_path = "../../data/raw/sample.txt"
    
    # Create a sample file if it doesn't exist
    if not os.path.exists(sample_text_path):
        os.makedirs(os.path.dirname(sample_text_path), exist_ok=True)
        with open(sample_text_path, "w", encoding="utf-8") as f:
            f.write("This is a sample text document for testing the document loader.")
    
    text_content = loader.load_document(sample_text_path)
    print(f"Loaded text content: {text_content}") 