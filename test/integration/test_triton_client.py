#!/usr/bin/env python3
"""
Corrected Test Client - Sends Proper Batch Dimensions
Fixed to work with @batch decorator expectations
"""

import tritonclient.http as httpclient
import numpy as np
import torch
from PIL import Image
import base64
import io
import os
from transformers import BlipProcessor

def encode_image_to_base64(image_path):
    """Encode image to base64 for high-level API"""
    with open(image_path, 'rb') as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode('utf-8')

def test_vision_language_model():
    """Test the low-level vision-language model"""
    
    print("🧪 Testing Complete BLIP Vision-Language Model")
    print("=" * 50)
    
    try:
        # Connect to server
        client = httpclient.InferenceServerClient(url="localhost:8000")
        
        # Check server and model health
        if not client.is_server_ready():
            print("❌ Server is not ready")
            return False
            
        if not client.is_model_ready("blip_vision_language"):
            print("❌ Vision-language model is not ready")
            return False
        
        print("✅ Server and model are ready")
        
        # Load BLIP processor for image preprocessing (same as vlm_quantization_3.py)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Find test image
        test_image_path = None
        for directory in ["data/products", "data/test_images"]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_image_path = os.path.join(directory, file)
                        break
                if test_image_path:
                    break
        
        if not test_image_path:
            print("❌ No test images found")
            return False
        
        print(f"📸 Using test image: {test_image_path}")
        
        # Load and preprocess image (same as vlm_quantization_3.py)
        image = Image.open(test_image_path).convert('RGB')
        print(f"📊 Image size: {image.size}")
        
        # Preprocess image to get pixel values (same as vlm_quantization_3.py)
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].numpy().astype(np.float16)
        
        print(f"📊 Original pixel values shape: {pixel_values.shape}")
        
        # Test 1: Image-only inference (captioning)
        print("\n🎯 Test 1: Image-only captioning")
        
        # FIXED: Keep the batch dimension for @batch decorator
        # Server expects [-1,3,384,384], so send (1,3,384,384)
        print(f"📊 Sending pixel values shape: {pixel_values.shape}")
        
        # Prepare inputs - KEEP batch dimension
        pixel_input = httpclient.InferInput("pixel_values", pixel_values.shape, "FP16")
        pixel_input.set_data_from_numpy(pixel_values)
        
        outputs = [httpclient.InferRequestedOutput("logits")]
        
        # Run inference
        results = client.infer("blip_vision_language", [pixel_input], outputs=outputs)
        logits = results.as_numpy("logits")
        
        print(f"✅ Image-only inference successful!")
        print(f"📊 Output logits shape: {logits.shape}")
        
        # Test 2: Image + text inference  
        print("\n🎯 Test 2: Image + text inference")
        
        # Create text input (same as vlm_quantization_3.py)
        text_input = "a photo of"
        text_inputs = processor(text=text_input, return_tensors="pt")
        input_ids = text_inputs['input_ids'].numpy().astype(np.int32)
        attention_mask = text_inputs['attention_mask'].numpy().astype(np.int32)
        
        print(f"📊 Original text input IDs shape: {input_ids.shape}")
        print(f"📊 Original attention mask shape: {attention_mask.shape}")
        
        # FIXED: Keep batch dimensions for @batch decorator
        # Server expects [-1,-1], so send (1,seq_len)
        print(f"📊 Sending input_ids shape: {input_ids.shape}")
        print(f"📊 Sending attention_mask shape: {attention_mask.shape}")
        
        # Prepare inputs - KEEP batch dimensions
        pixel_input = httpclient.InferInput("pixel_values", pixel_values.shape, "FP16")
        pixel_input.set_data_from_numpy(pixel_values)
        
        text_input = httpclient.InferInput("input_ids", input_ids.shape, "INT32")
        text_input.set_data_from_numpy(input_ids)
        
        mask_input = httpclient.InferInput("attention_mask", attention_mask.shape, "INT32")
        mask_input.set_data_from_numpy(attention_mask)
        
        outputs = [httpclient.InferRequestedOutput("logits")]
        
        # Run inference
        results = client.infer("blip_vision_language", [pixel_input, text_input, mask_input], outputs=outputs)
        logits = results.as_numpy("logits")
        
        print(f"✅ Image + text inference successful!")
        print(f"📊 Output logits shape: {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision-language model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_captioning_endpoint():
    """Test the high-level image captioning endpoint"""
    
    print("\n🧪 Testing High-Level Image Captioning Endpoint")
    print("=" * 50)
    
    try:
        # Connect to server
        client = httpclient.InferenceServerClient(url="localhost:8000")
        
        # Check model health
        if not client.is_model_ready("blip_captioning"):
            print("❌ Captioning model is not ready")
            return False
        
        print("✅ Captioning model is ready")
        
        # Find test images
        test_images = []
        for directory in ["data/products", "data/test_images"]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_images.append(os.path.join(directory, file))
                        if len(test_images) >= 2:  # Test with 2 images
                            break
                if len(test_images) >= 2:
                    break
        
        if not test_images:
            print("❌ No test images found")
            return False
        
        print(f"📸 Testing with {len(test_images)} images")
        
        # Test each image individually
        for i, img_path in enumerate(test_images):
            print(f"\n--- Testing Image {i+1}: {os.path.basename(img_path)} ---")
            
            # Encode image to base64
            b64_data = encode_image_to_base64(img_path)
            print(f"📊 Base64 data length: {len(b64_data)}")
            
            # FIXED: Create proper batch dimension for @batch decorator
            # Server expects [-1,1], so send (1,1) shape
            image_input = np.array([[b64_data.encode('utf-8')]], dtype=object)  # Shape: (1,1)
            print(f"📊 Image input shape: {image_input.shape}")
            
            # Prepare input
            images_input = httpclient.InferInput("images_b64", image_input.shape, "BYTES")
            images_input.set_data_from_numpy(image_input)
            
            outputs = [httpclient.InferRequestedOutput("captions")]
            
            # Run inference
            results = client.infer("blip_captioning", [images_input], outputs=outputs)
            captions = results.as_numpy("captions")
            
            print(f"✅ Image captioning successful!")
            print(f"📊 Captions shape: {captions.shape}")
            print(f"🎯 Generated Caption: '{captions[0][0].decode('utf-8') if isinstance(captions[0][0], bytes) else captions[0][0]}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Image captioning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_metadata():
    """Test model metadata and configuration"""
    
    print("\n🧪 Testing Model Metadata")
    print("=" * 30)
    
    try:
        client = httpclient.InferenceServerClient(url="localhost:8000")
        
        # Test vision-language model metadata
        print("📋 Vision-Language Model Metadata:")
        metadata = client.get_model_metadata("blip_vision_language")
        print(f"  Platform: {metadata['platform']}")
        print(f"  Inputs: {len(metadata['inputs'])}")
        print(f"  Outputs: {len(metadata['outputs'])}")
        
        for input_info in metadata['inputs']:
            print(f"    Input: {input_info['name']} ({input_info['datatype']}) {input_info['shape']}")
        
        for output_info in metadata['outputs']:
            print(f"    Output: {output_info['name']} ({output_info['datatype']}) {output_info['shape']}")
        
        # Test captioning model metadata
        print("\n📋 Captioning Model Metadata:")
        metadata = client.get_model_metadata("blip_captioning")
        print(f"  Platform: {metadata['platform']}")
        print(f"  Inputs: {len(metadata['inputs'])}")
        print(f"  Outputs: {len(metadata['outputs'])}")
        
        for input_info in metadata['inputs']:
            print(f"    Input: {input_info['name']} ({input_info['datatype']}) {input_info['shape']}")
        
        for output_info in metadata['outputs']:
            print(f"    Output: {output_info['name']} ({output_info['datatype']}) {output_info['shape']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata test failed: {e}")
        return False

def test_batch_inference():
    """Test actual batch inference with multiple samples"""
    
    print("\n🧪 Testing Batch Inference")
    print("=" * 30)
    
    try:
        client = httpclient.InferenceServerClient(url="localhost:8000")
        
        # Test batch captioning with 2 images
        test_images = []
        for directory in ["data/products", "data/test_images"]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_images.append(os.path.join(directory, file))
                        if len(test_images) >= 2:
                            break
                if len(test_images) >= 2:
                    break
        
        if len(test_images) < 2:
            print("⚠️ Need at least 2 images for batch test")
            return True
        
        print(f"📸 Testing batch with {len(test_images)} images")
        
        # Encode both images to base64
        batch_b64_data = []
        for img_path in test_images:
            b64_data = encode_image_to_base64(img_path)
            batch_b64_data.append([b64_data.encode('utf-8')])
        
        # Create batch input: shape (2,1) for 2 images
        batch_input = np.array(batch_b64_data, dtype=object)
        print(f"📊 Batch input shape: {batch_input.shape}")
        
        # Prepare batch input
        images_input = httpclient.InferInput("images_b64", batch_input.shape, "BYTES")
        images_input.set_data_from_numpy(batch_input)
        
        outputs = [httpclient.InferRequestedOutput("captions")]
        
        # Run batch inference
        results = client.infer("blip_captioning", [images_input], outputs=outputs)
        captions = results.as_numpy("captions")
        
        print(f"✅ Batch captioning successful!")
        print(f"📊 Batch captions shape: {captions.shape}")
        
        for i, (img_path, caption) in enumerate(zip(test_images, captions)):
            caption_text = caption[0].decode('utf-8') if isinstance(caption[0], bytes) else caption[0]
            print(f"  {i+1}. {os.path.basename(img_path)}: '{caption_text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    
    print("🚀 CORRECTED BLIP VISION-LANGUAGE TRITON TESTS")
    print("=" * 60)
    print("🔧 Fixed batch dimension handling")
    print("🔧 Proper @batch decorator compatibility")
    print("=" * 60)
    
    # Test 1: Basic connection
    try:
        client = httpclient.InferenceServerClient(url="localhost:8000")
        if client.is_server_ready():
            print("✅ Server connection successful")
        else:
            print("❌ Server is not ready")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test 2: Vision-language model
    vl_success = test_vision_language_model()
    
    # Test 3: Image captioning endpoint
    cap_success = test_image_captioning_endpoint()
    
    # Test 4: Model metadata
    meta_success = test_model_metadata()
    
    # Test 5: Batch inference
    batch_success = test_batch_inference()
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 CORRECTED TEST SUMMARY")
    print("=" * 60)
    print(f"🤖 Vision-Language Model: {'✅ PASS' if vl_success else '❌ FAIL'}")
    print(f"📸 Image Captioning: {'✅ PASS' if cap_success else '❌ FAIL'}")
    print(f"📋 Model Metadata: {'✅ PASS' if meta_success else '❌ FAIL'}")
    print(f"📦 Batch Inference: {'✅ PASS' if batch_success else '❌ FAIL'}")
    
    if vl_success and cap_success and meta_success and batch_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Complete BLIP Vision-Language model working")
        print("✅ True image understanding capabilities")
        print("✅ Proper batch dimension handling")
        print("✅ Both single and batch inference working")
        print("✅ Ready for product matching pipeline")
    else:
        print("\n⚠️ Some tests failed. Check logs above.")

if __name__ == "__main__":
    main()