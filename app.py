#!/usr/bin/env python3
"""
Web UI for Product Matching System
Chat-style interface with async processing and queue management
"""

import asyncio
import base64
import io
import os
import sys
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import warnings

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import uvicorn

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.matching.engine import ProductMatchingEngine
from src.utils.config import get_config

warnings.filterwarnings("ignore")

# Global variables for async queue management
processing_queue = asyncio.Queue()
current_processing = False
engine_instance = None

class ChatMessage:
    """Chat message structure"""
    def __init__(self, message_type: str, content: str, timestamp: datetime = None, 
                 metadata: Dict = None, image_data: str = None, request_id: str = None):
        self.message_type = message_type  # 'user', 'assistant', 'system'
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.image_data = image_data  # base64 encoded image
        self.request_id = request_id or str(uuid.uuid4())

class AsyncProductMatcher:
    """Async wrapper for product matching with queue management"""
    
    def __init__(self):
        self.engine = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the matching engine asynchronously"""
        if not self.is_initialized:
            print("🚀 Initializing Product Matching Engine for Web UI...")
            
            # Run initialization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.engine = await loop.run_in_executor(None, ProductMatchingEngine)
            self.is_initialized = True
            print("✅ Engine initialized for web interface")
    
    async def match_product_async(self, image_file: Optional[UploadFile] = None, 
                                 text_query: str = "", search_strategy: str = "combined") -> Dict[str, Any]:
        """Async product matching with proper error handling"""
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Handle image upload
            image_path = None
            if image_file:
                # Save uploaded image temporarily
                image_content = await image_file.read()
                image_path = f"temp_images/{uuid.uuid4()}.{image_file.filename.split('.')[-1]}"
                
                # Create temp directory if it doesn't exist
                os.makedirs("temp_images", exist_ok=True)
                
                with open(image_path, "wb") as f:
                    f.write(image_content)
            
            # Run matching in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.engine.match_product,
                image_path,
                text_query,
                search_strategy,
                1  # max_results
            )
            
            # Clean up temporary image
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
            
            # Enhance result with image data
            if result['status'] == 'success' and result['results']:
                await self._add_product_images(result['results'])
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'request_id': str(uuid.uuid4())
            }
    
    async def _add_product_images(self, results: List[Dict[str, Any]]):
        """Add base64 encoded product images to results"""
        
        for result in results:
            image_path = result['metadata'].get('image_path', '')
            if image_path and os.path.exists(image_path):
                try:
                    # Load and encode image
                    with Image.open(image_path) as img:
                        # Resize for web display
                        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                        
                        # Convert to base64
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=85)
                        img_data = base64.b64encode(buffer.getvalue()).decode()
                        
                        result['metadata']['image_base64'] = f"data:image/jpeg;base64,{img_data}"
                        
                except Exception as e:
                    print(f"⚠️ Failed to load image {image_path}: {e}")
                    result['metadata']['image_base64'] = None
            else:
                result['metadata']['image_base64'] = None

# Initialize FastAPI app
app = FastAPI(title="Product Matching System", description="AI-powered product search and matching")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global async matcher instance
matcher = AsyncProductMatcher()

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("🌐 Starting Product Matching Web UI...")
    # Initialize matcher
    await matcher.initialize()
    print("✅ Web UI ready!")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/api/match")
async def api_match_product(
    request: Request,
    image: Optional[UploadFile] = File(None),
    query: str = Form(""),
    strategy: str = Form("combined")
):
    """API endpoint for product matching"""
    
    global current_processing
    
    # Check if system is currently processing
    if current_processing:
        raise HTTPException(
            status_code=429, 
            detail="System is currently processing another request. Please wait."
        )
    
    # Validate inputs
    if not image and not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Either image or text query must be provided"
        )
    
    try:
        # Set processing flag
        current_processing = True
        
        # Process the request
        result = await matcher.match_product_async(
            image_file=image,
            text_query=query,
            search_strategy=strategy
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            content={
                'status': 'error', 
                'error': str(e),
                'request_id': str(uuid.uuid4())
            },
            status_code=500
        )
    
    finally:
        # Always reset processing flag
        current_processing = False

@app.get("/api/status")
async def api_status():
    """Get system status"""
    
    try:
        if not matcher.is_initialized:
            return {"status": "initializing", "processing": current_processing}
        
        # Get system diagnostics
        loop = asyncio.get_event_loop()
        diagnostics = await loop.run_in_executor(None, matcher.engine.get_system_diagnostics)
        
        return {
            "status": "ready",
            "processing": current_processing,
            "diagnostics": diagnostics
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "processing": current_processing
        }

@app.get("/api/config")
async def api_config():
    """Get system configuration"""
    
    try:
        config = get_config()
        return {
            "ai_models": {
                "clip_model": config.ai_models.clip_model_name,
                "sentence_model": config.ai_models.sentence_model_name,
                "blip_model": config.ai_models.blip_model_name,
                "device": config.ai_models.device
            },
            "triton": {
                "endpoint": config.triton.endpoint,
                "max_batch_size": config.triton.max_batch_size
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting Product Matching Web Server...")
    print("🌐 Access the interface at: http://localhost:8080")
    print("📖 API docs at: http://localhost:8080/docs")
    
    # Create required directories
    os.makedirs("temp_images", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,  # Disable reload to prevent engine reinitialization
        log_level="info"
    )