#!/usr/bin/env python3
"""
Database Manager Module
MongoDB-compatible JSON database manager for product metadata and logging
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import warnings

from ..utils.config import DATABASE_CONFIG

warnings.filterwarnings("ignore")

class DatabaseManager:
    """
    MongoDB-compatible JSON database manager for product metadata and logging
    Production-ready with full CRUD operations and transaction logging
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or DATABASE_CONFIG.storage_path
        self.products_file = os.path.join(self.storage_path, DATABASE_CONFIG.products_file)
        self.logs_file = os.path.join(self.storage_path, DATABASE_CONFIG.logs_file)
        self.metadata_file = os.path.join(self.storage_path, DATABASE_CONFIG.metadata_file)
        
        self._initialize_storage()
        self._load_collections()
        
        print(f" Database Manager initialized")
        print(f"   Storage: {self.storage_path}")
        print(f"   Products: {len(self.products_collection)} items")
        print(f"   Logs: {len(self.logs_collection)} entries")
    
    def _initialize_storage(self) -> None:
        """Initialize storage directories and files"""
        
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Initialize files if they don't exist
            for file_path in [self.products_file, self.logs_file]:
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        json.dump([], f)
            
            # Initialize metadata
            if not os.path.exists(self.metadata_file):
                metadata = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "database_type": "json_mongodb_mock",
                    "version": "1.0.0",
                    "collections": ["products", "logs"],
                    "last_accessed": datetime.now(timezone.utc).isoformat()
                }
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
        except Exception as e:
            print(f" Storage initialization failed: {e}")
            raise
    
    def _load_collections(self) -> None:
        """Load data collections from JSON files"""
        
        try:
            # Load products collection
            with open(self.products_file, 'r') as f:
                self.products_collection = json.load(f)
            
            # Load logs collection
            with open(self.logs_file, 'r') as f:
                self.logs_collection = json.load(f)
            
            # Update last accessed time
            self._update_metadata({"last_accessed": datetime.now(timezone.utc).isoformat()})

            # Auto-populate if empty
            if len(self.products_collection) == 0:
               self._populate_from_existing_data() 
            
        except Exception as e:
            print(f" Collection loading failed: {e}")
            self.products_collection = []
            self.logs_collection = []
    
    def _populate_from_existing_data(self) -> None:
        """Populate products collection from existing embeddings file"""
        
        try:
            embeddings_file = "data/products_with_embeddings.json"
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'r') as f:
                    products_data = json.load(f)
                
                # Convert to database format
                for product in products_data:
                    product_doc = {
                        "product_id": product["product_id"],
                        "name": product["name"],
                        "brand": product["brand"],
                        "category": product["category"],
                        "price": float(product["price"]),
                        "description": product["description"],
                        "image_path": product["image_path"],
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                    self.products_collection.append(product_doc)
                
                # Save to products file
                self._save_collection("products", self.products_collection)
                
                print(f"Populated database with {len(products_data)} products from embeddings file")
            
        except Exception as e:
            print(f" Failed to populate from existing data: {e}")

    def _save_collection(self, collection_name: str, data: List[Dict[str, Any]]) -> bool:
        """Save collection data to JSON file"""
        
        try:
            if collection_name == "products":
                file_path = self.products_file
            elif collection_name == "logs":
                file_path = self.logs_file
            else:
                raise ValueError(f"Unknown collection: {collection_name}")
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f" Failed to save {collection_name}: {e}")
            return False
    
    def _update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update database metadata"""
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata.update(updates)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            print(f" Metadata update warning: {e}")
    
    def find_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Find product by ID (MongoDB-style query)"""
        
        for product in self.products_collection:
            if product.get("product_id") == product_id:
                return product
        return None
    
    def find_products_by_category(self, category: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find products by category with optional limit"""
        
        results = []
        for product in self.products_collection:
            if product.get("category") == category:
                results.append(product)
                if limit and len(results) >= limit:
                    break
        
        return results
    
    def find_products_by_price_range(self, min_price: float, max_price: float, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find products within price range"""
        
        results = []
        for product in self.products_collection:
            price = float(product.get("price", 0))
            if min_price <= price <= max_price:
                results.append(product)
                if limit and len(results) >= limit:
                    break
        
        return results
    
    def insert_operation_log(self, operation_type: str, operation_data: Dict[str, Any], status: str = "success") -> Optional[str]:
        """Insert operation log with MongoDB-style structure"""
        
        try:
            log_entry = {
                "_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation_type": operation_type,
                "operation_data": operation_data,
                "status": status,
                "session_id": getattr(self, '_session_id', str(uuid.uuid4())[:8]),
                "environment": "production"
            }
            
            self.logs_collection.append(log_entry)
            
            # Keep only last 5000 logs to prevent memory issues
            if len(self.logs_collection) > 5000:
                self.logs_collection = self.logs_collection[-5000:]
            
            self._save_collection("logs", self.logs_collection)
            
            return log_entry["_id"]
            
        except Exception as e:
            print(f" Logging failed: {e}")
            return None
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        stats = {
            "products_count": len(self.products_collection),
            "logs_count": len(self.logs_collection),
            "storage_path": self.storage_path,
            "last_accessed": datetime.now(timezone.utc).isoformat()
        }
        
        # Category breakdown
        categories = {}
        for product in self.products_collection:
            category = product.get("category", "Unknown")
            categories[category] = categories.get(category, 0) + 1
        
        stats["categories"] = categories
        
        # Recent operations
        recent_ops = []
        for log in self.logs_collection[-10:]:  # Last 10 operations
            recent_ops.append({
                "operation": log.get("operation_type"),
                "timestamp": log.get("timestamp"),
                "status": log.get("status")
            })
        
        stats["recent_operations"] = recent_ops
        
        return stats
    
    def get_all_products(self) -> List[Dict[str, Any]]:
        """Get all products"""
        return self.products_collection.copy()
    
    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent logs"""
        return self.logs_collection[-limit:] if self.logs_collection else []
    
    def close(self) -> None:
        """Close database connections (for compatibility)"""
        print(" Database manager closed")

# Convenience function for creating database manager
def get_database_manager(storage_path: Optional[str] = None) -> DatabaseManager:
    """Create and return a database manager instance"""
    return DatabaseManager(storage_path)