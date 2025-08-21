#!/usr/bin/env python3
"""
Startup script for Product Matching Web UI
Handles directory creation and dependency checking
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if required dependencies are installed"""
    
    required_packages = [
        'fastapi',
        'uvicorn', 
        'jinja2',
        'PIL'  # Pillow
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install missing packages with:")
        print("   pip install fastapi uvicorn jinja2 pillow python-multipart aiofiles")
        return False
    
    return True

def create_directories():
    """Create required directories for web UI"""
    
    directories = [
        'templates',
        'static', 
        'temp_images'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory exists: {directory}")

def check_triton_server():
    """Check if Triton server is running"""
    
    try:
        import requests
        response = requests.get('http://localhost:8000/v2/health/ready', timeout=2)
        if response.status_code == 200:
            print("✅ Triton server is running")
            return True
        else:
            print("⚠️ Triton server responded but not ready")
            return False
    except Exception:
        print("❌ Triton server is not running")
        print("💡 Start Triton server first with: python scripts/start_server.py")
        return False

def main():
    """Main startup function"""
    
    print("🚀 PRODUCT MATCHING WEB UI STARTUP")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies satisfied")
    
    # Create directories
    print("\n📁 Setting up directories...")
    create_directories()
    
    # Check if core files exist
    print("\n📋 Checking core files...")
    required_files = [
        'app.py',
        'templates/chat.html',
        'src/matching/engine.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        sys.exit(1)
    
    # Check Triton server (optional)
    print("\n🌐 Checking Triton server...")
    triton_running = check_triton_server()
    
    if not triton_running:
        response = input("\n❓ Continue without Triton server? (BLIP features will be disabled) [y/N]: ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Start web server
    print("\n🚀 Starting web server...")
    print("🌐 Access the interface at: http://localhost:8080")
    print("📖 API docs at: http://localhost:8080/docs")
    print("⏹️ Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        import uvicorn
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8080,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n⏹️ Web server stopped")
    except Exception as e:
        print(f"\n❌ Failed to start web server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()