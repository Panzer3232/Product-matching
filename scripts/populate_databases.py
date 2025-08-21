"""
Populate Vector Database and MongoDB
Creates FAISS vector database and populates MongoDB with product data
"""

import os
import json
import numpy as np
import faiss
import pickle
from datetime import datetime
from dotenv import load_dotenv
import pymongo
from pymongo import MongoClient

# Load environment variables
load_dotenv()

def load_products_with_embeddings():
    """Load products with generated embeddings"""
    
    try:
        with open('data/products_with_embeddings.json', 'r', encoding='utf-8') as f:
            products = json.load(f)
        print(f"📁 Loaded {len(products)} products with embeddings")
        return products
    except FileNotFoundError:
        print("❌ products_with_embeddings.json not found")
        print("💡 Run generate_embeddings.py first")
        return None

def create_vector_database(products):
    """Create FAISS vector database for similarity search"""
    
    print("\n🔍 CREATING VECTOR DATABASE")
    print("=" * 40)
    
    if not products:
        return None, None, None
    
    # Extract embeddings
    visual_embeddings = []
    text_embeddings = []
    combined_embeddings = []
    product_ids = []
    
    for product in products:
        visual_embeddings.append(product['visual_embedding'])
        text_embeddings.append(product['text_embedding'])
        combined_embeddings.append(product['combined_embedding'])
        product_ids.append(product['product_id'])
    
    # Convert to numpy arrays
    visual_vectors = np.array(visual_embeddings, dtype=np.float32)
    text_vectors = np.array(text_embeddings, dtype=np.float32)
    combined_vectors = np.array(combined_embeddings, dtype=np.float32)
    
    print(f"✅ Visual vectors: {visual_vectors.shape}")
    print(f"✅ Text vectors: {text_vectors.shape}")
    print(f"✅ Combined vectors: {combined_vectors.shape}")
    
    # Create FAISS indexes
    visual_dim = visual_vectors.shape[1]
    text_dim = text_vectors.shape[1]
    combined_dim = combined_vectors.shape[1]
    
    # Create indexes (using L2 distance, normalized vectors = cosine similarity)
    visual_index = faiss.IndexFlatIP(visual_dim)  # Inner Product for normalized vectors
    text_index = faiss.IndexFlatIP(text_dim)
    combined_index = faiss.IndexFlatIP(combined_dim)
    
    # Add vectors to indexes
    visual_index.add(visual_vectors)
    text_index.add(text_vectors)
    combined_index.add(combined_vectors)
    
    print(f"✅ Visual index: {visual_index.ntotal} vectors")
    print(f"✅ Text index: {text_index.ntotal} vectors")
    print(f"✅ Combined index: {combined_index.ntotal} vectors")
    
    # Create metadata for vector database
    vector_metadata = {
        'product_ids': product_ids,
        'created_at': datetime.utcnow().isoformat(),
        'visual_dim': visual_dim,
        'text_dim': text_dim,
        'combined_dim': combined_dim,
        'total_products': len(products)
    }
    
    return {
        'visual_index': visual_index,
        'text_index': text_index,
        'combined_index': combined_index,
        'metadata': vector_metadata
    }

def save_vector_database(vector_db):
    """Save FAISS indexes and metadata to disk"""
    
    print("\n💾 SAVING VECTOR DATABASE")
    print("=" * 30)
    
    if not vector_db:
        return False
    
    # Create vector database directory
    vector_db_dir = "data/vector_db"
    os.makedirs(vector_db_dir, exist_ok=True)
    
    try:
        # Save FAISS indexes
        faiss.write_index(vector_db['visual_index'], 
                         os.path.join(vector_db_dir, 'visual_index.faiss'))
        faiss.write_index(vector_db['text_index'], 
                         os.path.join(vector_db_dir, 'text_index.faiss'))
        faiss.write_index(vector_db['combined_index'], 
                         os.path.join(vector_db_dir, 'combined_index.faiss'))
        
        # Save metadata
        with open(os.path.join(vector_db_dir, 'metadata.json'), 'w') as f:
            json.dump(vector_db['metadata'], f, indent=2)
        
        # Save product ID mapping
        with open(os.path.join(vector_db_dir, 'product_mapping.pkl'), 'wb') as f:
            pickle.dump(vector_db['metadata']['product_ids'], f)
        
        print(f"✅ Visual index saved: {vector_db_dir}/visual_index.faiss")
        print(f"✅ Text index saved: {vector_db_dir}/text_index.faiss")
        print(f"✅ Combined index saved: {vector_db_dir}/combined_index.faiss")
        print(f"✅ Metadata saved: {vector_db_dir}/metadata.json")
        print(f"✅ Product mapping saved: {vector_db_dir}/product_mapping.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save vector database: {e}")
        return False

def connect_to_mongodb():
    """Connect to MongoDB Atlas"""
    
    print("\n🗄️ CONNECTING TO MONGODB")
    print("=" * 30)
    
    try:
        connection_string = os.getenv('MONGODB_CONNECTION_STRING')
        if not connection_string:
            print("❌ MongoDB connection string not found in .env")
            return None, None
        
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas")
        
        # Get database
        db_name = os.getenv('MONGODB_DATABASE_NAME', 'product_matcher_db')
        db = client[db_name]
        print(f"✅ Database: {db_name}")
        
        return client, db
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return None, None

def populate_mongodb(products, db):
    """Populate MongoDB with product data"""
    
    print("\n📊 POPULATING MONGODB")
    print("=" * 25)
    
    if not products or db is None:
        return False
    
    try:
        # Get products collection
        products_collection = db.products
        
        # Clear existing data (for clean setup)
        existing_count = products_collection.count_documents({})
        if existing_count > 0:
            print(f"🔄 Clearing {existing_count} existing products...")
            products_collection.delete_many({})
        
        # Insert products
        print(f"📝 Inserting {len(products)} products...")
        
        # Prepare documents for insertion
        documents = []
        for product in products:
            # Create MongoDB document
            doc = {
                '_id': product['product_id'],  # Use product_id as MongoDB _id
                'product_id': product['product_id'],
                'name': product['name'],
                'category': product['category'],
                'brand': product['brand'],
                'price': product['price'],
                'description': product['description'],
                'image_path': product['image_path'],
                'generated_description': product['generated_description'],
                'visual_embedding': product['visual_embedding'],
                'text_embedding': product['text_embedding'],
                'combined_embedding': product['combined_embedding'],
                'embedding_model_info': product['embedding_model_info'],
                'created_at': product['created_at'],
                'updated_at': datetime.utcnow().isoformat()
            }
            documents.append(doc)
        
        # Insert documents
        result = products_collection.insert_many(documents)
        print(f"✅ Inserted {len(result.inserted_ids)} products")
        
        # Verify insertion
        inserted_count = products_collection.count_documents({})
        print(f"✅ Total products in database: {inserted_count}")
        
        # Create some sample logs
        logs_collection = db.logs
        
        log_entry = {
            'timestamp': datetime.utcnow(),
            'operation': 'database_population',
            'status': 'success',
            'products_inserted': len(result.inserted_ids),
            'message': 'Initial database population completed'
        }
        
        logs_collection.insert_one(log_entry)
        print(f"✅ Log entry created")
        
        return True
        
    except Exception as e:
        print(f"❌ MongoDB population failed: {e}")
        return False

def test_vector_search(vector_db):
    """Test vector database search functionality"""
    
    print("\n🔍 TESTING VECTOR SEARCH")
    print("=" * 30)
    
    if not vector_db:
        return False
    
    try:
        # Get the first product's actual embedding for testing
        with open('data/products_with_embeddings.json', 'r') as f:
            products = json.load(f)
        
        if not products:
            print("❌ No products available for testing")
            return False
        
        # Use first product's combined embedding as test query
        test_embedding = np.array([products[0]['combined_embedding']], dtype=np.float32)
        
        # Search for similar vectors
        k = min(3, len(products))  # Top 3 or all available products
        distances, indices = vector_db['combined_index'].search(test_embedding, k)
        
        print(f"✅ Search completed")
        print(f"Test query: {products[0]['name']}")
        print(f"Top {k} similar products:")
        
        product_ids = vector_db['metadata']['product_ids']
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(product_ids):
                print(f"  {i+1}. {product_ids[idx]} (similarity: {distance:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector search test failed: {e}")
        return False

def display_database_summary(db):
    """Display summary of populated databases"""
    
    print("\n📊 DATABASE SUMMARY")
    print("=" * 30)
    
    try:
        # MongoDB summary
        products_collection = db.products
        logs_collection = db.logs
        
        product_count = products_collection.count_documents({})
        log_count = logs_collection.count_documents({})
        
        print(f"MongoDB Atlas:")
        print(f"  📦 Products: {product_count}")
        print(f"  📝 Logs: {log_count}")
        
        # Sample product
        sample_product = products_collection.find_one()
        if sample_product:
            print(f"  🔍 Sample product: {sample_product['name']}")
        
        # Vector database summary
        vector_db_dir = "data/vector_db"
        if os.path.exists(vector_db_dir):
            print(f"\nVector Database (FAISS):")
            print(f"  📁 Location: {vector_db_dir}")
            
            files = os.listdir(vector_db_dir)
            for file in files:
                file_path = os.path.join(vector_db_dir, file)
                size_kb = os.path.getsize(file_path) / 1024
                print(f"  📄 {file}: {size_kb:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Database summary failed: {e}")
        return False

def main():
    print("🗄️ DATABASE POPULATION")
    print("=" * 50)
    
    # Load products with embeddings
    products = load_products_with_embeddings()
    if not products:
        return
    
    # Create vector database
    vector_db = create_vector_database(products)
    if not vector_db:
        print("❌ Vector database creation failed")
        return
    
    # Save vector database
    vector_save_success = save_vector_database(vector_db)
    if not vector_save_success:
        print("❌ Vector database save failed")
        return
    
    # Test vector search
    vector_test_success = test_vector_search(vector_db)
    
    # Connect to MongoDB
    client, db = connect_to_mongodb()
    if client is None or db is None:
        print("❌ MongoDB connection failed")
        return
    
    # Populate MongoDB
    mongo_success = populate_mongodb(products, db)
    if not mongo_success:
        print("❌ MongoDB population failed")
        if client:
            client.close()
        return
    
    # Display summary
    summary_success = display_database_summary(db)
    
    # Close MongoDB connection
    if client:
        client.close()
    
    print("\n" + "=" * 50)
    if vector_save_success and mongo_success and vector_test_success:
        print("🎉 DATABASE POPULATION COMPLETE!")
        print("✅ FAISS vector database created and saved")
        print("✅ MongoDB Atlas populated with products")
        print("✅ Vector search functionality tested")
        print("✅ All systems ready for product matching!")
        print("\n🚀 Next: Build the matching pipeline!")
    else:
        print("⚠️ Database population completed with some issues")
        print("💡 Check error messages above")

if __name__ == "__main__":
    main()