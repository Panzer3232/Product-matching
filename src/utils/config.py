#!/usr/bin/env python3
"""
Configuration Management Module
Centralized configuration for the product matching system
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

# Load environment variables
load_dotenv()

@dataclass
class AIModelConfig:
    """AI model configuration"""
    clip_model_name: str = "ViT-B/32"
    sentence_model_name: str = "all-MiniLM-L6-v2"
    blip_model_name: str = "Salesforce/blip-image-captioning-base"
    device: str = "cuda"  # or "cpu"

@dataclass
class TritonConfig:
    """Triton server configuration"""
    endpoint: str = "localhost:8000"
    http_port: int = 8000
    grpc_port: int = 8001
    metrics_port: int = 8002
    max_batch_size: int = 4
    log_verbose: int = 1

@dataclass
class DatabaseConfig:
    """Database configuration"""
    storage_path: str = "data/database"
    products_file: str = "products.json"
    logs_file: str = "operation_logs.json"
    metadata_file: str = "database_metadata.json"
    
    # MongoDB configuration (if used)
    mongodb_connection_string: Optional[str] = None
    mongodb_database_name: str = "product_matcher_db"
    products_collection: str = "products"
    logs_collection: str = "logs"

@dataclass
class VectorDatabaseConfig:
    """Vector database configuration"""
    storage_path: str = "data/vector_db"
    visual_index_file: str = "visual_index.faiss"
    text_index_file: str = "text_index.faiss"
    combined_index_file: str = "combined_index.faiss"
    product_mapping_file: str = "product_mapping.pkl"
    metadata_file: str = "metadata.json"

@dataclass
class DataConfig:
    """Data paths configuration"""
    products_catalog: str = "data/products_with_embeddings.json"
    sample_products: str = "data/sample_products.json"
    products_images_dir: str = "data/products"
    test_images_dir: str = "data/test_images"

@dataclass
class SystemConfig:
    """Complete system configuration"""
    ai_models: AIModelConfig
    triton: TritonConfig
    database: DatabaseConfig
    vector_db: VectorDatabaseConfig
    data: DataConfig
    
    # System settings
    environment: str = "production"
    debug: bool = False
    max_workers: int = 4
    timeout_seconds: int = 30

def load_config() -> SystemConfig:
    """Load configuration from environment variables and defaults"""
    
    # Load from environment variables
    ai_models = AIModelConfig(
        clip_model_name=os.getenv('CLIP_MODEL_NAME', 'ViT-B/32'),
        sentence_model_name=os.getenv('SENTENCE_MODEL_NAME', 'all-MiniLM-L6-v2'),
        blip_model_name=os.getenv('BLIP_MODEL_NAME', 'Salesforce/blip-image-captioning-base'),
        device=os.getenv('DEVICE', 'cuda')
    )
    
    triton = TritonConfig(
        endpoint=os.getenv('TRITON_ENDPOINT', 'localhost:8000'),
        http_port=int(os.getenv('TRITON_HTTP_PORT', '8000')),
        grpc_port=int(os.getenv('TRITON_GRPC_PORT', '8001')),
        metrics_port=int(os.getenv('TRITON_METRICS_PORT', '8002')),
        max_batch_size=int(os.getenv('MAX_BATCH_SIZE', '4'))
    )
    
    database = DatabaseConfig(
        storage_path=os.getenv('DATABASE_STORAGE_PATH', 'data/database'),
        mongodb_connection_string=os.getenv('MONGODB_CONNECTION_STRING'),
        mongodb_database_name=os.getenv('MONGODB_DATABASE_NAME', 'product_matcher_db'),
        products_collection=os.getenv('PRODUCTS_COLLECTION', 'products'),
        logs_collection=os.getenv('LOGS_COLLECTION', 'logs')
    )
    
    vector_db = VectorDatabaseConfig(
        storage_path=os.getenv('VECTOR_DB_PATH', 'data/vector_db')
    )
    
    data = DataConfig(
        products_catalog=os.getenv('PRODUCTS_CATALOG', 'data/products_with_embeddings.json'),
        products_images_dir=os.getenv('PRODUCTS_IMAGES_DIR', 'data/products'),
        test_images_dir=os.getenv('TEST_IMAGES_DIR', 'data/test_images')
    )
    
    return SystemConfig(
        ai_models=ai_models,
        triton=triton,
        database=database,
        vector_db=vector_db,
        data=data,
        environment=os.getenv('ENVIRONMENT', 'production'),
        debug=os.getenv('DEBUG', 'false').lower() == 'true',
        max_workers=int(os.getenv('MAX_WORKERS', '4')),
        timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', '30'))
    )

# Global configuration instance
config = load_config()

def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    return config

def update_config(**kwargs) -> None:
    """Update configuration values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

# Export commonly used configs for convenience
AI_CONFIG = config.ai_models
TRITON_CONFIG = config.triton
DATABASE_CONFIG = config.database
VECTOR_DB_CONFIG = config.vector_db
DATA_CONFIG = config.data