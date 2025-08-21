#!/usr/bin/env python3
"""
Vector Store Module
FAISS-based vector database for similarity search
"""

import os
import pickle
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
import warnings

from ..utils.config import VECTOR_DB_CONFIG

warnings.filterwarnings("ignore")

class VectorStore:
    """FAISS-based vector database for similarity search"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or VECTOR_DB_CONFIG.storage_path
        self.indexes = {}
        self.product_mapping = []
        self.metadata = {}
        
        self._initialize_storage()
        self._load_indexes()
        
        print(f" Vector Store initialized")
        print(f"   Storage: {self.storage_path}")
        print(f"   Indexes: {len(self.indexes)}")
        print(f"   Products: {len(self.product_mapping)}")
    
    def _initialize_storage(self) -> None:
        """Initialize storage directory"""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _load_indexes(self) -> None:
        """Load FAISS indexes from disk"""
        try:
            # Load FAISS indexes
            index_configs = [
                ('visual', VECTOR_DB_CONFIG.visual_index_file),
                ('textual', VECTOR_DB_CONFIG.text_index_file),
                ('combined', VECTOR_DB_CONFIG.combined_index_file)
            ]
            
            for index_name, index_file in index_configs:
                index_path = os.path.join(self.storage_path, index_file)
                if os.path.exists(index_path):
                    self.indexes[index_name] = faiss.read_index(index_path)
                    print(f" Loaded {index_name} index: {self.indexes[index_name].ntotal} vectors")
            
            # Load product mapping
            mapping_path = os.path.join(self.storage_path, VECTOR_DB_CONFIG.product_mapping_file)
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    self.product_mapping = pickle.load(f)
                print(f" Loaded product mapping: {len(self.product_mapping)} products")
            
            # Load metadata
            metadata_path = os.path.join(self.storage_path, VECTOR_DB_CONFIG.metadata_file)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"Loaded metadata")
            
        except Exception as e:
            print(f" Vector store loading failed: {e}")
            self.indexes = {}
            self.product_mapping = []
            self.metadata = {}
    
    def search_similar(self, query_embedding: np.ndarray, index_type: str = "combined", 
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        
        if index_type not in self.indexes:
            print(f" Index '{index_type}' not available")
            return []
        
        try:
            # Ensure query embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search
            index = self.indexes[index_type]
            distances, indices = index.search(query_embedding.astype(np.float32), top_k)
            
            # Prepare results
            results = []
            for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.product_mapping):
                    result = {
                        'rank': rank + 1,
                        'product_id': self.product_mapping[idx],
                        'similarity_score': float(distance),
                        'index_type': index_type
                    }
                    results.append(result)
            
            print(f" Found {len(results)} matches using {index_type} index")
            return results
            
        except Exception as e:
            print(f" Vector search failed: {e}")
            return []
    
    def add_vectors(self, embeddings: Dict[str, List[np.ndarray]], product_ids: List[str]) -> bool:
        """Add new vectors to indexes"""
        
        try:
            print(f" Adding {len(product_ids)} vectors to indexes...")
            
            for index_type, vectors in embeddings.items():
                if index_type not in self.indexes:
                    # Create new index
                    dimension = vectors[0].shape[0]
                    self.indexes[index_type] = faiss.IndexFlatIP(dimension)
                    print(f" Created new {index_type} index (dim: {dimension})")
                
                # Convert to numpy array
                vectors_array = np.array(vectors, dtype=np.float32)
                
                # Add to index
                self.indexes[index_type].add(vectors_array)
                print(f" Added {len(vectors)} vectors to {index_type} index")
            
            # Update product mapping
            self.product_mapping.extend(product_ids)
            
            # Save indexes
            self.save_indexes()
            
            return True
            
        except Exception as e:
            print(f" Adding vectors failed: {e}")
            return False
    
    def save_indexes(self) -> bool:
        """Save indexes to disk"""
        
        try:
            print(f" Saving vector indexes...")
            
            # Save FAISS indexes
            index_configs = [
                ('visual', VECTOR_DB_CONFIG.visual_index_file),
                ('textual', VECTOR_DB_CONFIG.text_index_file),
                ('combined', VECTOR_DB_CONFIG.combined_index_file)
            ]
            
            for index_name, index_file in index_configs:
                if index_name in self.indexes:
                    index_path = os.path.join(self.storage_path, index_file)
                    faiss.write_index(self.indexes[index_name], index_path)
                    print(f"Saved {index_name} index")
            
            # Save product mapping
            mapping_path = os.path.join(self.storage_path, VECTOR_DB_CONFIG.product_mapping_file)
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.product_mapping, f)
            print(f" Saved product mapping")
            
            # Update and save metadata
            self.metadata.update({
                'total_products': len(self.product_mapping),
                'indexes': list(self.indexes.keys()),
                'last_updated': np.datetime64('now').isoformat()
            })
            
            metadata_path = os.path.join(self.storage_path, VECTOR_DB_CONFIG.metadata_file)
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"✅ Saved metadata")
            
            return True
            
        except Exception as e:
            print(f" Saving indexes failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        
        stats = {
            'storage_path': self.storage_path,
            'total_products': len(self.product_mapping),
            'indexes': {}
        }
        
        for index_name, index in self.indexes.items():
            stats['indexes'][index_name] = {
                'total_vectors': index.ntotal,
                'dimension': index.d
            }
        
        return stats
    
    def rebuild_index(self, index_type: str, vectors: List[np.ndarray]) -> bool:
        """Rebuild a specific index"""
        
        try:
            print(f"🔄 Rebuilding {index_type} index...")
            
            if not vectors:
                print(f" No vectors provided for {index_type}")
                return False
            
            # Create new index
            dimension = vectors[0].shape[0]
            new_index = faiss.IndexFlatIP(dimension)
            
            # Add all vectors
            vectors_array = np.array(vectors, dtype=np.float32)
            new_index.add(vectors_array)
            
            # Replace old index
            self.indexes[index_type] = new_index
            
            print(f"Rebuilt {index_type} index: {new_index.ntotal} vectors")
        
            # Save
            self.save_indexes()
            
            return True
            
        except Exception as e:
            print(f" Rebuilding {index_type} index failed: {e}")
            return False
    
    def clear_indexes(self) -> None:
        """Clear all indexes"""
        print(" Clearing all indexes...")
        self.indexes = {}
        self.product_mapping = []
        self.metadata = {}
    
    def is_available(self) -> bool:
        """Check if vector store is available"""
        return len(self.indexes) > 0 and len(self.product_mapping) > 0
    
    def get_available_indexes(self) -> List[str]:
        """Get list of available index types"""
        return list(self.indexes.keys())
    
    def close(self) -> None:
        """Close vector store (for compatibility)"""
        print(" Vector store closed")

class VectorSearchEngine:
    """High-level vector search engine"""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
    
    def search(self, embeddings: Dict[str, np.ndarray], strategy: str = "combined", 
               top_k: int = 5) -> List[Dict[str, Any]]:
        """Execute search with strategy selection"""
        
        # Strategy selection logic
        if strategy == "visual" and 'visual' in embeddings:
            query_embedding = embeddings['visual']
            index_type = 'visual'
        elif strategy == "textual" and 'textual' in embeddings:
            query_embedding = embeddings['textual']
            index_type = 'textual'
        elif strategy == "combined":
            if 'visual' in embeddings and 'textual' in embeddings:
                query_embedding = np.concatenate([embeddings['visual'], embeddings['textual']])
                index_type = 'combined'
            elif 'visual' in embeddings:
                query_embedding = embeddings['visual']
                index_type = 'visual'
            elif 'textual' in embeddings:
                query_embedding = embeddings['textual']
                index_type = 'textual'
            else:
                print(" No embeddings available for search")
                return []
        else:
            print(f" Invalid strategy: {strategy}")
            return []
        
        return self.vector_store.search_similar(query_embedding, index_type, top_k)
    
    def is_ready(self) -> bool:
        """Check if search engine is ready"""
        return self.vector_store.is_available()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return self.vector_store.get_statistics()

# Convenience functions
def get_vector_store(storage_path: Optional[str] = None) -> VectorStore:
    """Get vector store instance"""
    return VectorStore(storage_path)

def get_search_engine(vector_store: Optional[VectorStore] = None) -> VectorSearchEngine:
    """Get vector search engine instance"""
    return VectorSearchEngine(vector_store)