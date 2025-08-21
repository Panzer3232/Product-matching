"""
Generate Embeddings for Sample Products
Creates visual and text embeddings using CLIP and SentenceTransformers
"""

import os
import json
import torch
import clip
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

def load_models():
    """Load CLIP and SentenceTransformer models"""
    
    print("🤖 Loading AI models...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")
    
    try:
        # Load CLIP model
        clip_model_name = os.getenv('CLIP_MODEL_NAME', 'ViT-B/32')
        print(f"Loading CLIP model: {clip_model_name}")
        
        clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
        print(f"✅ CLIP model loaded successfully")
        
        # Load SentenceTransformer model
        sentence_model_name = os.getenv('SENTENCE_MODEL_NAME', 'all-MiniLM-L6-v2')
        print(f"Loading Sentence model: {sentence_model_name}")
        
        sentence_model = SentenceTransformer(sentence_model_name, device=device)
        print(f"✅ Sentence model loaded successfully")
        
        return clip_model, clip_preprocess, sentence_model, device
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None, None, None

def generate_visual_embedding(image_path, clip_model, clip_preprocess, device):
    """Generate visual embedding using CLIP"""
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            visual_features = clip_model.encode_image(image_tensor)
            # Normalize the features
            visual_features = visual_features / visual_features.norm(dim=1, keepdim=True)
        
        # Convert to numpy array
        embedding = visual_features.cpu().numpy().flatten()
        
        return embedding.tolist()  # Convert to list for JSON serialization
        
    except Exception as e:
        print(f"❌ Visual embedding failed for {image_path}: {e}")
        return None

def generate_text_embedding(text, sentence_model):
    """Generate text embedding using SentenceTransformers"""
    
    try:
        # Generate embedding
        embedding = sentence_model.encode(text, convert_to_numpy=True)
        
        return embedding.tolist()  # Convert to list for JSON serialization
        
    except Exception as e:
        print(f"❌ Text embedding failed for '{text}': {e}")
        return None

def create_product_description(product):
    """Create a comprehensive text description for each product"""
    
    # Create rich description combining multiple fields
    description_parts = [
        product['name'],
        product['brand'],
        product['category'],
        product.get('subcategory', ''),
        product['description'],
        product.get('color', ''),
        product.get('storage', ''),
        product.get('size', ''),
        product.get('connectivity', ''),
    ]
    
    # Filter out empty parts and join
    description = ' '.join([part for part in description_parts if part])
    
    return description

def process_products():
    """Process all products and generate embeddings"""
    
    print("\n🔄 PROCESSING PRODUCTS")
    print("=" * 40)
    
    # Load sample data
    try:
        with open('data/sample_products.json', 'r', encoding='utf-8') as f:
            products = json.load(f)
        print(f"📁 Loaded {len(products)} products from sample_products.json")
    except FileNotFoundError:
        print("❌ sample_data.json not found. Run create_sample_products.py first.")
        return None
    
    # Load models
    clip_model, clip_preprocess, sentence_model, device = load_models()
    if not all([clip_model, clip_preprocess, sentence_model]):
        print("❌ Failed to load models")
        return None
    
    # Process each product
    processed_products = []
    
    for i, product in enumerate(products, 1):
        print(f"\n📦 Processing product {i}/{len(products)}: {product['name']}")
        
        # Generate visual embedding
        image_path = product['image_path']
        if os.path.exists(image_path):
            print(f"   🖼️  Generating visual embedding from {image_path}")
            visual_embedding = generate_visual_embedding(
                image_path, clip_model, clip_preprocess, device
            )
            if visual_embedding:
                print(f"   ✅ Visual embedding: {len(visual_embedding)} dimensions")
            else:
                print(f"   ❌ Visual embedding failed")
                continue
        else:
            print(f"   ❌ Image not found: {image_path}")
            continue
        
        # Generate text description and embedding
        description = create_product_description(product)
        print(f"   📝 Description: {description[:50]}...")
        
        print(f"   🔤 Generating text embedding")
        text_embedding = generate_text_embedding(description, sentence_model)
        if text_embedding:
            print(f"   ✅ Text embedding: {len(text_embedding)} dimensions")
        else:
            print(f"   ❌ Text embedding failed")
            continue
        
        # Add embeddings to product data
        product_with_embeddings = product.copy()
        product_with_embeddings.update({
            'visual_embedding': visual_embedding,
            'text_embedding': text_embedding,
            'combined_embedding': visual_embedding + text_embedding,  # Concatenate
            'generated_description': description,
            'embedding_model_info': {
                'clip_model': os.getenv('CLIP_MODEL_NAME', 'ViT-B/32'),
                'sentence_model': os.getenv('SENTENCE_MODEL_NAME', 'all-MiniLM-L6-v2'),
                'visual_dim': len(visual_embedding),
                'text_dim': len(text_embedding),
                'combined_dim': len(visual_embedding) + len(text_embedding)
            }
        })
        
        processed_products.append(product_with_embeddings)
        
        print(f"   ✅ Product {product['product_id']} processed successfully")
    
    return processed_products

def save_embeddings(products_with_embeddings):
    """Save products with embeddings to JSON file"""
    
    if not products_with_embeddings:
        print("❌ No products to save")
        return False
    
    try:
        # Save to JSON file
        output_path = "data/products_with_embeddings.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(products_with_embeddings, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved products with embeddings: {output_path}")
        
        # Print summary
        sample_product = products_with_embeddings[0]
        print(f"\n📊 Embedding Summary:")
        print(f"   - Visual dimensions: {len(sample_product['visual_embedding'])}")
        print(f"   - Text dimensions: {len(sample_product['text_embedding'])}")
        print(f"   - Combined dimensions: {len(sample_product['combined_embedding'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save embeddings: {e}")
        return False

def verify_embeddings():
    """Verify the generated embeddings"""
    
    print("\n🔍 VERIFYING EMBEDDINGS")
    print("=" * 30)
    
    try:
        with open('data/products_with_embeddings.json', 'r') as f:
            products = json.load(f)
        
        for product in products:
            name = product['name']
            visual_dim = len(product['visual_embedding'])
            text_dim = len(product['text_embedding'])
            combined_dim = len(product['combined_embedding'])
            
            print(f"✅ {name}:")
            print(f"   Visual: {visual_dim}D, Text: {text_dim}D, Combined: {combined_dim}D")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    print("🧠 EMBEDDING GENERATION FOR SAMPLE PRODUCTS")
    print("=" * 60)
    
    # Process products and generate embeddings
    products_with_embeddings = process_products()
    
    if products_with_embeddings:
        # Save embeddings
        save_success = save_embeddings(products_with_embeddings)
        
        if save_success:
            # Verify embeddings
            verify_success = verify_embeddings()
            
            if verify_success:
                print("\n" + "=" * 60)
                print("🎉 EMBEDDING GENERATION COMPLETE!")
                print("✅ All products processed successfully")
                print("✅ Visual and text embeddings generated")
                print("✅ Combined embeddings created")
                print("✅ Data saved and verified")
                print("\n🚀 Ready for database population!")
            else:
                print("\n⚠️ Embeddings generated but verification failed")
        else:
            print("\n❌ Failed to save embeddings")
    else:
        print("\n❌ No products processed successfully")
        print("💡 Check if sample images exist and models load correctly")

if __name__ == "__main__":
    main()