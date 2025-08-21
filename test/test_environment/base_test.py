"""
Complete Environment Test
Tests all components needed for the product matching system
"""

import torch
import sys
import warnings
warnings.filterwarnings("ignore")

def test_basic_setup():
    print("=== BASIC SETUP TEST ===")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        print("✅ Basic setup OK")
    else:
        print("❌ CUDA not available")
    print()

def test_core_libraries():
    print("=== CORE LIBRARIES TEST ===")
    
    libraries = [
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence_transformers"), 
        ("PIL", "PIL"),
        ("opencv", "cv2"),
        ("numpy", "numpy"),
        ("clip", "clip"),
        ("faiss", "faiss"),
        ("pymongo", "pymongo"),
        ("fastapi", "fastapi")
    ]
    
    results = {}
    for name, import_name in libraries:
        try:
            __import__(import_name)
            print(f"✅ {name}: OK")
            results[name] = True
        except ImportError as e:
            print(f"❌ {name}: {str(e)[:60]}...")
            results[name] = False
    
    return results

def test_sentence_transformers_detailed():
    print("\n=== SENTENCE TRANSFORMERS DETAILED TEST ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try loading a small model
        model_name = "all-MiniLM-L6-v2"
        print(f"Loading model: {model_name}")
        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        model = SentenceTransformer(model_name, device=device)
        print(f"✅ Model loaded successfully on {device}")
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        print(f"✅ Text encoding successful. Embedding shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sentence transformers detailed test failed: {e}")
        return False

def test_tensorrt():
    print("\n=== TENSORRT TEST ===")
    
    try:
        import tensorrt as trt
        print(f"✅ TensorRT imported successfully")
        print(f"TensorRT Version: {trt.__version__}")
        
        # Test builder
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        print(f"✅ TensorRT Builder created")
        
        return True
        
    except ImportError as e:
        print(f"❌ TensorRT not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ TensorRT error: {e}")
        return False

def test_triton():
    print("\n=== TRITON CLIENT TEST ===")
    
    try:
        import tritonclient.http as httpclient
        print(f"✅ Triton HTTP client OK")
        
        import tritonclient.grpc as grpcclient
        print(f"✅ Triton gRPC client OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Triton client not installed: {e}")
        return False

def test_clip_detailed():
    print("\n=== CLIP DETAILED TEST ===")
    
    try:
        import clip
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        print(f"✅ CLIP ViT-B/32 loaded on {device}")
        
        # Test with dummy image
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(dummy_image)
            
        print(f"✅ Image encoding successful. Shape: {image_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLIP detailed test failed: {e}")
        return False

def main():
    print("🔍 COMPLETE ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    # Run all tests
    test_basic_setup()
    lib_results = test_core_libraries()
    st_ok = test_sentence_transformers_detailed()
    trt_ok = test_tensorrt()
    triton_ok = test_triton()
    clip_ok = test_clip_detailed()
    
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY:")
    print("=" * 60)
    
    # Core requirements
    core_ready = (
        torch.cuda.is_available() and 
        lib_results.get('transformers', False) and
        lib_results.get('clip', False) and
        lib_results.get('faiss', False) and
        lib_results.get('pymongo', False)
    )
    
    # Advanced requirements
    advanced_ready = st_ok and trt_ok and triton_ok
    
    if core_ready and advanced_ready:
        print("🎉 ALL SYSTEMS GO!")
        print("✅ Core ML libraries: Working")
        print("✅ GPU support: Working") 
        print("✅ Sentence transformers: Working")
        print("✅ TensorRT: Working")
        print("✅ Triton client: Working")
        print("\n🚀 Ready to proceed with MongoDB setup!")
        
    elif core_ready:
        print("✅ CORE SYSTEM READY")
        print("✅ Basic ML pipeline can work")
        print("⚠️ Missing components:")
        if not st_ok:
            print("   - Sentence transformers needs fixing")
        if not trt_ok:
            print("   - TensorRT needs installation")
        if not triton_ok:
            print("   - Triton client needs installation")
        print("\n📝 Can proceed with basic setup, add missing components later")
        
    else:
        print("❌ CORE ISSUES DETECTED")
        print("❌ Need to fix basic libraries before proceeding")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()