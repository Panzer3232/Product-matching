"""
Simple Product Data Creation
Creates product metadata for real downloaded images
"""

import os
import json
import datetime

def create_product_data():
    """Create product data for real downloaded images"""
    
    # Product data (assumes you downloaded real images)
    products = [
        {
            "product_id": "prod_001",
            "name": "iPhone 15 Pro",
            "category": "Electronics",
            "brand": "Apple",
            "price": 1199.99,
            "description": "Latest iPhone with titanium design and A17 Pro chip",
            "image_path": "data/products/iphone15_pro.jpeg",
            "created_at": datetime.datetime.utcnow().isoformat()
        },
        {
            "product_id": "prod_002", 
            "name": "Samsung Galaxy S24 Ultra",
            "category": "Electronics",
            "brand": "Samsung",
            "price": 1299.99,
            "description": "Premium Android smartphone with S Pen and advanced cameras",
            "image_path": "data/products/galaxy_s24_ultra.jpeg",
            "created_at": datetime.datetime.utcnow().isoformat()
        },
        {
            "product_id": "prod_003",
            "name": "Nike Air Force 1",
            "category": "Fashion",
            "brand": "Nike",
            "price": 110.00,
            "description": "Classic white leather sneakers with iconic Nike design",
            "image_path": "data/products/nike_air_force1.jpeg",
            "created_at": datetime.datetime.utcnow().isoformat()
        },
        {
            "product_id": "prod_004",
            "name": "MacBook Pro M3",
            "category": "Electronics", 
            "brand": "Apple",
            "price": 1999.99,
            "description": "Professional laptop with M3 chip and Liquid Retina XDR display",
            "image_path": "data/products/macbook_pro_m3.jpeg",
            "created_at": datetime.datetime.utcnow().isoformat()
        },
        {
            "product_id": "prod_005",
            "name": "Sony WH-1000XM5 Headphones",
            "category": "Electronics",
            "brand": "Sony", 
            "price": 399.99,
            "description": "Industry-leading noise canceling wireless headphones",
            "image_path": "data/products/sony_headphones.jpeg",
            "created_at": datetime.datetime.utcnow().isoformat()
        }
    ]
    
    return products

def verify_images():
    """Check if all required images exist"""
    
    required_images = [
        "iphone15_pro.jpeg",
        "galaxy_s24_ultra.jpeg", 
        "nike_air_force1.jpeg",
        "macbook_pro_m3.jpeg",
        "sony_headphones.jpeg"
    ]
    
    print("📊 Checking for product images...")
    
    missing_images = []
    for img_name in required_images:
        img_path = os.path.join("data/products", img_name)
        if os.path.exists(img_path):
            print(f"✅ Found: {img_name}")
        else:
            print(f"❌ Missing: {img_name}")
            missing_images.append(img_name)
    
    return len(missing_images) == 0, missing_images

def save_product_data(products):
    """Save product data to JSON"""
    
    os.makedirs("data", exist_ok=True)
    
    json_path = "data/sample_products.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved product data: {json_path}")

def display_instructions():
    """Display instructions for downloading images"""
    
    print("\n📝 IMAGE DOWNLOAD INSTRUCTIONS:")
    print("=" * 50)
    print("Please download these 5 product images and save them in data/products/:")
    print()
    
    instructions = [
        ("iPhone 15 Pro", "iphone15_pro.jpeg", "Search: 'iPhone 15 Pro official product image'"),
        ("Samsung Galaxy S24 Ultra", "galaxy_s24_ultra.jpeg", "Search: 'Samsung Galaxy S24 Ultra product photo'"),
        ("Nike Air Force 1", "nike_air_force1.jpeg", "Search: 'Nike Air Force 1 white sneakers'"),
        ("MacBook Pro M3", "macbook_pro_m3.jpeg", "Search: 'MacBook Pro M3 official image'"),
        ("Sony WH-1000XM5", "sony_headphones.jpeg", "Search: 'Sony WH-1000XM5 headphones product image'")
    ]
    
    for i, (product, filename, search_tip) in enumerate(instructions, 1):
        print(f"{i}. {product}")
        print(f"   Save as: data/products/{filename}")
        print(f"   {search_tip}")
        print()
    
    print("💡 Tips:")
    print("- Use high-quality product images (preferably 400x400 or larger)")
    print("- Save as .jpeg format")
    print("- Use official product photos when possible")
    print("- Avoid images with complex backgrounds")

def main():
    print("🛍️ PRODUCT DATA CREATION (with Real Images)")
    print("=" * 60)
    
    # Create products directory
    os.makedirs("data/products", exist_ok=True)
    
    # Check if images exist
    images_exist, missing_images = verify_images()
    
    if images_exist:
        print("✅ All product images found!")
        
        # Create product data
        products = create_product_data()
        
        # Save to JSON
        save_product_data(products)
        
        print("\n" + "=" * 60)
        print("🎉 PRODUCT DATA CREATION COMPLETE!")
        print("✅ All 5 product images verified")
        print("✅ Product metadata created")
        print("✅ Data saved to sample_products.json")
        print("\n🚀 Ready for embedding generation!")
        
    else:
        print(f"\n❌ Missing {len(missing_images)} image(s)")
        display_instructions()
        print("\n📋 Next Steps:")
        print("1. Download the missing images")
        print("2. Save them in data/products/ with exact filenames above")
        print("3. Run this script again")

if __name__ == "__main__":
    main()