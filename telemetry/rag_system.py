"""
RAG (Retrieval-Augmented Generation) System
Enhances LLM responses by retrieving relevant documents
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import pickle

# Try to import optional dependencies
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️ python-docx not available. .docx files will not be supported.")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️ tiktoken not available. Using simple text splitting.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available. Using simple similarity.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not available. RAG system will not work.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunker:
    """Document chunking utility"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if TIKTOKEN_AVAILABLE:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI tokenizer
        else:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Simple character-based approximation
            return len(text) // 4
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            chunks = []
            
            for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
            
            return chunks
        else:
            # Simple character-based chunking
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
            return chunks
    
    def chunk_document(self, file_path: str) -> List[Dict[str, str]]:
        """Chunk a document file (supports .docx, .txt)"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.docx':
            return self._chunk_docx(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._chunk_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    def _chunk_docx(self, file_path: Path) -> List[Dict[str, str]]:
        """Chunk a Word document"""
        if not DOCX_AVAILABLE:
            raise ValueError("python-docx not available. Cannot process .docx files.")
        
        doc = Document(file_path)
        full_text = ""
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text += paragraph.text + "\n"
        
        chunks = self.chunk_text(full_text)
        return [{"content": chunk, "source": str(file_path)} for chunk in chunks]
    
    def _chunk_txt(self, file_path: Path) -> List[Dict[str, str]]:
        """Chunk a text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = self.chunk_text(text)
        return [{"content": chunk, "source": str(file_path)} for chunk in chunks]

class EmbeddingManager:
    """Manages text embeddings using free local models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager with a free local model
        
        Args:
            model_name: Name of the sentence-transformers model to use
                       Options: "all-MiniLM-L6-v2" (fast, good quality)
                               "all-mpnet-base-v2" (slower, better quality)
                               "paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ValueError("sentence-transformers not available. Cannot initialize EmbeddingManager.")
        
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded successfully")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {e}")
            return [[] for _ in texts]

class VectorStore:
    """Simple in-memory vector store for document retrieval"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_documents(self, documents: List[Dict[str, str]], embeddings: List[List[float]]):
        """Add documents and their embeddings to the store"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        self.documents.extend([doc["content"] for doc in documents])
        self.embeddings.extend(embeddings)
        self.metadata.extend([{"source": doc["source"]} for doc in documents])
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """Search for similar documents"""
        if not self.embeddings:
            return []
        
        if not SKLEARN_AVAILABLE:
            # Simple fallback: return first few documents
            logger.warning("scikit-learn not available. Using simple document selection.")
            return [{"content": self.documents[i], "similarity": 0.5, "metadata": self.metadata[i]} 
                   for i in range(min(top_k, len(self.documents)))]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "content": self.documents[idx],
                "similarity": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })
        
        return results
    
    def save(self, file_path: str):
        """Save vector store to file"""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "metadata": self.metadata
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved vector store to {file_path}")
    
    def load(self, file_path: str):
        """Load vector store from file"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        logger.info(f"Loaded vector store from {file_path}")

class RAGSystem:
    """Main RAG system for document retrieval and LLM enhancement"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000):
        self.chunker = DocumentChunker(chunk_size=chunk_size)
        self.embedding_manager = EmbeddingManager(model_name=model_name)
        self.vector_store = VectorStore()
        self.cache_file = "vector_store_cache.pkl"
    
    def load_documents(self, file_path: str, force_reload: bool = False):
        """Load and process documents"""
        # Try to load from cache first
        if not force_reload and os.path.exists(self.cache_file):
            try:
                self.vector_store.load(self.cache_file)
                logger.info("Loaded vector store from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Process documents
        logger.info(f"Processing document: {file_path}")
        chunks = self.chunker.chunk_document(file_path)
        
        if not chunks:
            logger.warning("No chunks generated from document")
            return
        
        # Get embeddings
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_manager.get_embeddings_batch(texts)
        
        # Filter out empty embeddings
        valid_chunks = []
        valid_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:  # Check if embedding is not empty
                valid_chunks.append(chunk)
                valid_embeddings.append(embedding)
        
        # Add to vector store
        self.vector_store.add_documents(valid_chunks, valid_embeddings)
        
        # Save to cache
        self.vector_store.save(self.cache_file)
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        # Get query embedding
        query_embedding = self.embedding_manager.get_embedding(query)
        
        if not query_embedding:
            logger.warning("Failed to get query embedding")
            return []
        
        # Search for similar documents
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return results
    
    def enhance_prompt(self, user_prompt: str, system_prompt: str = "", top_k: int = 3) -> str:
        """Enhance prompt with relevant documents"""
        # Combine user prompt and system prompt for retrieval
        retrieval_query = f"{system_prompt} {user_prompt}".strip()
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(retrieval_query, top_k=top_k)
        
        if not relevant_docs:
            logger.warning("No relevant documents found")
            return user_prompt
        
        # Build enhanced prompt
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        
        enhanced_prompt = f"""參考以下相關文檔：

{context}

基於以上文檔，請回答：

{user_prompt}"""
        
        logger.info(f"Enhanced prompt with {len(relevant_docs)} relevant documents")
        return enhanced_prompt
    
    def get_document_info(self) -> Dict:
        """Get information about loaded documents"""
        return {
            "total_documents": len(self.vector_store.documents),
            "cache_file": self.cache_file,
            "cache_exists": os.path.exists(self.cache_file),
            "embedding_model": self.embedding_manager.model_name
        }

# Global RAG system instance
rag_system = None

def get_rag_system(model_name: str = "all-MiniLM-L6-v2") -> RAGSystem:
    """Get global RAG system instance"""
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem(model_name=model_name)
    return rag_system 