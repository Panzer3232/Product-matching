#!/usr/bin/env python3
"""
Product Matching Engine Module
Main engine that coordinates all components for product matching
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import warnings

from ..models.ai_models import MultiModalEmbeddingGenerator
from ..database.manager import DatabaseManager
from ..database.vector_store import VectorSearchEngine
from ..utils.config import get_config, DATA_CONFIG

warnings.filterwarnings("ignore")

class ProductMatchingEngine:
    """
    Advanced product matching engine with multi-modal AI capabilities
    Integrates BLIP vision-language model, CLIP, and SentenceTransformers
    """
    
    def __init__(self, triton_endpoint: Optional[str] = None):
        self.config = get_config()
        self.triton_endpoint = triton_endpoint or self.config.triton.endpoint
        
        # Initialize components
        self.ai_models = None
        self.database = None
        self.vector_search = None
        self.products_catalog = {}
        
        print("PRODUCT MATCHING ENGINE")
        print("=" * 50)
        print(f" Triton Endpoint: {self.triton_endpoint}")
        print(f" Device: {self.config.ai_models.device}")
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all system components"""
        
        print("\n🔧 Initializing components...")
        
        try:
            # Initialize AI models
            print("Loading AI models...")
            self.ai_models = MultiModalEmbeddingGenerator()
            
            # Initialize database
            print("Initializing database...")
            self.database = DatabaseManager()
            
            # Initialize vector search
            print(" Initializing vector search...")
            self.vector_search = VectorSearchEngine()
            
            # Load product catalog
            print("Loading product catalog...")
            self._load_product_catalog()
            
            print("All components initialized successfully")
            self._print_system_status()
            
        except Exception as e:
            print(f" Component initialization failed: {e}")
            raise
    
    def _load_product_catalog(self) -> None:
        """Load product catalog from JSON file"""
        
        catalog_files = [
            DATA_CONFIG.products_catalog,
            DATA_CONFIG.sample_products
        ]
        
        for catalog_file in catalog_files:
            if os.path.exists(catalog_file):
                print(f" Loading catalog: {catalog_file}")
                with open(catalog_file, 'r', encoding='utf-8') as f:
                    products_list = json.load(f)
                
                # Convert to indexed catalog
                for product in products_list:
                    self.products_catalog[product['product_id']] = product
                
                print(f" Catalog loaded: {len(self.products_catalog)} products")
                return
        
        print(" No product catalog found")
    
    def _print_system_status(self) -> None:
        """Print comprehensive system status"""
        
        print(f"\n📋 SYSTEM STATUS")
        print("=" * 30)
        print(f"   AI Models: ✅")
        print(f"   Database: ✅ ({len(self.database.products_collection)} products)")
        print(f"    Vector Search: {'✅' if self.vector_search.is_ready() else 'not worked'}")
        print(f"   Product Catalog: ✅ ({len(self.products_catalog)} items)")
        print(f"    BLIP Available: {'✅' if self.ai_models.blip_handler.is_available() else 'not worked'}")
    
    def match_product(self, image_path: Optional[str] = None, text_query: str = "", 
                     search_strategy: str = "combined", max_results: int = 1) -> Dict[str, Any]:
        """Execute complete product matching pipeline"""
        
        operation_start_time = datetime.now(timezone.utc)
        request_id = str(uuid.uuid4())
        
        print(f"\n PRODUCT MATCHING PIPELINE")
        print("=" * 40)
        print(f" Image: {image_path if image_path else 'None'}")
        print(f" Query: '{text_query}'")
        print(f" Strategy: {search_strategy}")
        print(f"Max Results: {max_results}")
        print(f" Request ID: {request_id}")
        
        # Validate inputs
        validation_result = self._validate_inputs(image_path, text_query)
        if validation_result['status'] == 'error':
            self._log_operation(request_id, validation_result, operation_start_time)
            return validation_result
        
        try:
            # Step 1: Generate image description
            image_description = ""
            description_source = "None"
            
            if image_path:
                print(f"\n Step 1: Generating image description...")
                image_description = self.ai_models.generate_image_description(image_path)
                description_source = "BLIP" if self.ai_models.blip_handler.is_available() else "CLIP"
                print(f" Description: '{image_description}'")
            
            # Step 2: Combine text inputs
            print(f"\n Step 2: Processing text inputs...")
            combined_text = self._combine_text_inputs(image_description, text_query)
            print(f" Combined text: '{combined_text}'")
            
            # Step 3: Generate embeddings
            print(f"\n Step 3: Generating embeddings...")
            embeddings = self.ai_models.generate_embeddings(image_path, combined_text)
            
            # Step 4: Search for matches
            print(f"\n Step 4: Searching for matches...")
            search_results = self.vector_search.search(embeddings, search_strategy, max_results)
            
            # Step 5: Enrich results with metadata
            print(f"\n Step 5: Enriching results...")
            enriched_results = self._enrich_search_results(search_results)
            
            # Step 6: Calculate execution time and prepare response
            execution_time = (datetime.now(timezone.utc) - operation_start_time).total_seconds()
            
            # Display results
            self._display_results(enriched_results, execution_time, description_source)
            
            # Prepare response
            response = {
                'status': 'success',
                'request_id': request_id,
                'image_path': image_path,
                'text_query': text_query,
                'combined_description': combined_text,
                'description_source': description_source,
                'search_strategy': search_strategy,
                'results': enriched_results,
                'execution_time': execution_time
            }
            
            # Log operation
            log_id = self._log_operation(request_id, response, operation_start_time)
            response['log_id'] = log_id
            
            return response
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - operation_start_time).total_seconds()
            error_response = {
                'status': 'error',
                'request_id': request_id,
                'error': str(e),
                'execution_time': execution_time
            }
            
            print(f" Matching failed: {e}")
            log_id = self._log_operation(request_id, error_response, operation_start_time)
            error_response['log_id'] = log_id
            
            return error_response
    
    def _validate_inputs(self, image_path: Optional[str], text_query: str) -> Dict[str, Any]:
        """Validate input parameters"""
        
        if not image_path and not text_query.strip():
            return {
                'status': 'error',
                'error': 'Either image path or text query must be provided'
            }
        
        if image_path and not os.path.exists(image_path):
            return {
                'status': 'error',
                'error': f'Image file not found: {image_path}'
            }
        
        return {'status': 'valid'}
    
    def _combine_text_inputs(self, image_description: str, text_query: str) -> str:
        """Combine image description and text query"""
        
        if image_description and text_query.strip():
            return f"{image_description}. {text_query}".strip()
        elif image_description:
            return image_description
        elif text_query.strip():
            return text_query.strip()
        else:
            return ""
    
    def _enrich_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich search results with product metadata"""
        
        enriched_results = []
        
        for result in search_results:
            product_id = result['product_id']
            product_metadata = self.products_catalog.get(product_id, {})
            
            enriched_result = {
                'rank': result['rank'],
                'product_id': product_id,
                'similarity_score': result['similarity_score'],
                'search_strategy': result['index_type'],
                'metadata': {
                    'name': product_metadata.get('name', 'Unknown'),
                    'brand': product_metadata.get('brand', 'Unknown'),
                    'category': product_metadata.get('category', 'Unknown'),
                    'price': product_metadata.get('price', 0),
                    'description': product_metadata.get('description', ''),
                    'image_path': product_metadata.get('image_path', '')
                }
            }
            
            enriched_results.append(enriched_result)
        
        return enriched_results
    
    def _display_results(self, results: List[Dict[str, Any]], execution_time: float, description_source: str) -> None:
        """Display search results"""
        
        print(f"\n🏆 SEARCH RESULTS:")
        if results:
            for result in results:
                metadata = result['metadata']
                print(f"\n   🥇 Rank {result['rank']}: {metadata['name']}")
                print(f"      🏷️ Brand: {metadata['brand']} | 📂 Category: {metadata['category']}")
                print(f"      💰 Price: ${metadata['price']} | 🎯 Score: {result['similarity_score']:.4f}")
                print(f"      📝 Description: {metadata['description']}")
                print(f"      🖼️ Image: {metadata['image_path']}")
                print(f"      🔍 Strategy: {result['search_strategy']}")
        else:
            print("    No matching products found")
        
        print(f"\n⏱ Execution time: {execution_time:.3f}s")
        print(f"🔧 Description source: {description_source}")
    
    def _log_operation(self, request_id: str, response: Dict[str, Any], start_time: datetime) -> Optional[str]:
        """Log operation to database"""
        
        try:
            operation_data = {
                "request_id": request_id,
                "inputs": {
                    "image_path": response.get('image_path'),
                    "text_query": response.get('text_query'),
                    "search_strategy": response.get('search_strategy')
                },
                "results": {
                    "status": response['status'],
                    "total_matches": len(response.get('results', [])),
                    "execution_time_seconds": response.get('execution_time', 0)
                },
                "system_info": {
                    "blip_available": self.ai_models.blip_handler.is_available(),
                    "vector_db_available": self.vector_search.is_ready(),
                    "products_in_catalog": len(self.products_catalog)
                }
            }
            
            if response['status'] == 'error':
                operation_data['error'] = response.get('error')
            
            log_id = self.database.insert_operation_log(
                operation_type="product_matching",
                operation_data=operation_data,
                status=response['status']
            )
            
            print(f" Operation logged: {log_id}")
            return log_id
            
        except Exception as e:
            print(f" Logging failed: {e}")
            return None
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics"""
        
        print(f"\n SYSTEM DIAGNOSTICS")
        print("=" * 30)
        
        # Database statistics
        db_stats = self.database.get_collection_statistics()
        print(f" Database: {db_stats['products_count']} products, {db_stats['logs_count']} logs")
        
        # Vector search statistics
        search_stats = self.vector_search.get_stats()
        print(f"🔍 Vector DB: {search_stats['total_products']} products")
        for index_name, index_info in search_stats['indexes'].items():
            print(f"   {index_name}: {index_info['total_vectors']} vectors")
        
        # AI model status
        print(f" AI Models:")
        print(f"   CLIP: ")
        print(f"   SentenceTransformer: ")
        print(f"   BLIP: {'' if self.ai_models.blip_handler.is_available() else 'not worked'}")
        
        return {
            'database': db_stats,
            'vector_search': search_stats,
            'ai_models': {
                'clip_available': True,
                'sentence_transformer_available': True,
                'blip_available': self.ai_models.blip_handler.is_available()
            },
            'product_catalog_size': len(self.products_catalog)
        }
    
    def close(self) -> None:
        """Clean up system resources"""
        
        print(" Shutting down Product Matching Engine...")
        
        if self.ai_models:
            self.ai_models.close()
        
        if self.database:
            self.database.close()
        
        if self.vector_search:
            self.vector_search.vector_store.close()
        
        print(" System shutdown complete")

# Convenience function
def get_matching_engine(triton_endpoint: Optional[str] = None) -> ProductMatchingEngine:
    """Create and return a product matching engine instance"""
    return ProductMatchingEngine(triton_endpoint)