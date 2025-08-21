# 🤖 AI-Powered Product Matching System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.11+-green.svg)](https://developer.nvidia.com/tensorrt)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.105+-green.svg)](https://fastapi.tiangolo.com)


> **Advanced multi-modal AI system for product matching using Vision-Language Models, TensorRT quantization, and NVIDIA Triton Inference Server**

## 🎯 Overview

This project implements a complete end-to-end product matching pipeline that leverages cutting-edge AI technologies to find the closest product match for given images and text descriptions. The system combines multiple AI models in a production-ready architecture optimized for performance and scalability.

### 🏆 Key Achievements

- ✅ **Multi-Modal AI**: CLIP + BLIP + SentenceTransformers integration
- ✅ **TensorRT Optimization**: FP16 quantization for 2x speed improvement  
- ✅ **Production Deployment**: NVIDIA Triton Inference Server
- ✅ **Vector Search**: FAISS-based similarity search with 3 search strategies
- ✅ **Modern Web UI**: Chat-style interface with async processing
- ✅ **Modular Architecture**: Clean, maintainable, and scalable codebase

## 🛠️ Technical Stack

### AI & Machine Learning
- **Vision-Language Model**: BLIP (Salesforce/blip-image-captioning-base)
- **Vision Encoder**: CLIP ViT-B/32 for visual embeddings
- **Text Encoder**: SentenceTransformers (all-MiniLM-L6-v2)
- **Quantization**: TensorRT FP16 optimization
- **Inference Server**: NVIDIA Triton Inference Server

### Databases & Search
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Metadata Storage**: MongoDB-compatible JSON database
- **Search Strategies**: Visual, Textual, and Combined matching

### Web Framework & UI
- **Backend**: FastAPI with async processing
- **Frontend**: Modern chat-style interface with real-time updates
- **Image Processing**: PIL + OpenCV for image handling
- **Deployment**: Uvicorn ASGI server

## 🚀 Quick Start

### Prerequisites
- **GPU**: NVIDIA RTX 30/40 series (CUDA 12.1+)
- **Python**: 3.10+
- **Memory**: 8GB+ GPU VRAM, 16GB+ system RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Panzer3232/Product-matching.git
   cd Product-matching
   ```

2. **Create virtual environment**
   ```bash
   conda create -n product_matching python=3.10
   conda activate product_matching
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test dependecies** (first run only)
   ```bash
   python tests/test_quantization/base_test.py
   ```

### Running the System

#### 1. Start Triton Inference Server
```bash
python scripts/start_server.py
```

#### 2. Launch Web Interface
```bash
python start_web_ui.py
```

#### 3. Access the Application
- **Web UI**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

#### 4. Command Line Interface
```bash
# Interactive demo
python main.py demo

# Direct matching
python main.py match --image data/products/phone.jpg --query "smartphone"

# System diagnostics  
python main.py diagnostics
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │  FastAPI        │    │  Triton Server  │
│  (Chat Style)   │◄──►│  (Async API)    │◄──►│  (BLIP + TRT)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  FAISS Vector   │◄──►│  Matching       │◄──►│  MongoDB Mock   │
│  Database       │    │  Engine         │    │  (JSON Store)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Search Strategies

1. **Visual Search**: Pure image-based matching using CLIP embeddings
2. **Textual Search**: Description-based matching using SentenceTransformers  
3. **Combined Search**: Multi-modal fusion for best accuracy

### Model Pipeline

```
Input Image ──► BLIP ──► Image Description ──┐
                                             ├──► Combined Text ──► SentenceTransformer ──► Text Embedding
Text Query ─────────────────────────────────┘
                                             
Input Image ──► CLIP ──► Visual Embedding

[Visual Embedding + Text Embedding] ──► FAISS Search ──► Top Matches
```

## 📊 Performance Metrics

| Component | Latency | Throughput | Optimization |
|-----------|---------|------------|--------------|
| BLIP (TensorRT) | ~200ms | 5 imgs/sec | FP16 Quantization |
| CLIP Encoding | ~50ms | 20 imgs/sec | GPU Acceleration |
| Vector Search | ~5ms | 1000 queries/sec | FAISS Indexing |
| End-to-End | ~300ms | 3 queries/sec | Async Processing |

## 🎮 Web Interface Features

### Chat-Style UI
- **Modern Design**: Gradient backgrounds, smooth animations
- **Real-time Status**: Processing indicators and system health
- **Mobile Responsive**: Works on desktop, tablet, and mobile
- **Async Queue**: Prevents concurrent processing conflicts

### Multi-Modal Input
- **Image Upload**: Drag & drop or click to upload
- **Text Queries**: Auto-resizing text input  
- **Strategy Selection**: Choose visual, textual, or combined search
- **Real-time Preview**: Image thumbnails and file validation

### Rich Results Display
- **Product Cards**: Professional layout with images and metadata
- **Similarity Scores**: Color-coded confidence levels
- **Complete Info**: Name, brand, category, price, description
- **Performance Metrics**: Execution time and request tracking

## 🧪 Testing & Validation

### System Tests
```bash
# Environment verification
python tests/test_environment.py

# Model quantization test
python tests/test_quantization.py

# Integration test
python main.py demo
```

### Sample Data
The repository includes:
- **5 sample products** with metadata and embeddings
- **Product images** in multiple categories (electronics, fashion)
- **Pre-built FAISS indexes** for immediate testing
- **Test images** for validation

## 🚀 Deployment Options

### Local Development
```bash
# Quick start
python start_web_ui.py
```

## 📈 API Reference

### Core Endpoints

#### POST `/api/match`
Product matching endpoint with multi-modal input support.


#### GET `/api/status`
System health and diagnostics.

#### GET `/api/config`
Current system configuration.

## 🔧 Configuration

### Environment Variables
```bash
# AI Models
CLIP_MODEL_NAME=ViT-B/32
SENTENCE_MODEL_NAME=all-MiniLM-L6-v2
BLIP_MODEL_NAME=Salesforce/blip-image-captioning-base
DEVICE=cuda

# Triton Server
TRITON_ENDPOINT=localhost:8000
MAX_BATCH_SIZE=4

# Database
DATABASE_STORAGE_PATH=data/database
VECTOR_DB_PATH=data/vector_db

# Web Server
WEB_PORT=8080
DEBUG=false
```

## 🐛 Troubleshooting

### Common Issues

**1. GPU Memory Error**
```bash
# Reduce batch size or use CPU fallback
export MAX_BATCH_SIZE=1
export DEVICE=cpu
```

**2. Triton Server Connection Failed**
```bash
# Check server status
curl http://localhost:8000/v2/health/ready

# Restart server
python scripts/start_server.py
```

**3. Model Loading Issues**
```bash
# Regenerate TensorRT models
python tests/test_quantization.py
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📞 Contact

**Anmol Ashri** - [GitHub](https://github.com/Panzer3232)

**Project Link**: [https://github.com/Panzer3232/Product-matching](https://github.com/Panzer3232/Product-matching)

---

⭐ **Star this repository if it helped you!**