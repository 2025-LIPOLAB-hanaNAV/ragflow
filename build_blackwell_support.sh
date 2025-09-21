#!/bin/bash

# RAGFlow Docker Build Script with RTX 5080 Blackwell Support
# This script builds RAGFlow with PyTorch 2.7.1+cu128 for RTX 5080 compatibility

echo "🚀 Starting RAGFlow Docker build with Blackwell RTX 5080 support..."
echo "📦 PyTorch version: 2.7.1+cu128 (direct install from PyTorch official)"
echo "🔧 CUDA version: 12.8"
echo "🎯 Target GPU: RTX 5080 (sm_120)"
echo "🔗 PyTorch CUDA 12.8 index: https://download.pytorch.org/whl/cu128"
echo "   • torch==2.7.1+cu128"
echo "   • torchvision==0.22.1+cu128"
echo "   • torchaudio==2.7.1+cu128"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check current directory
if [ ! -f "Dockerfile" ]; then
    echo "❌ Dockerfile not found. Please run this script from the RAGFlow root directory."
    exit 1
fi

# Build the Docker image
echo "🏗️ Building Docker image..."
docker build --build-arg LIGHTEN=0 --build-arg NEED_MIRROR=0 -t jjkim110523/ragflow-local:latest .

# Check build result
if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully!"
    echo ""
    echo "🎉 RAGFlow with RTX 5080 Blackwell support is ready!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Stop existing RAGFlow containers: docker-compose down"
    echo "   2. Update docker-compose.yml to use the new image: ragflow:latest"
    echo "   3. Start RAGFlow: docker-compose up -d"
    echo "   4. Test BAAI/bge-reranker-v2-m3 model functionality"
    echo ""
    echo "🔍 Features enabled:"
    echo "   • PyTorch 2.7.1+cu128 with RTX 5080 support"
    echo "   • GPU compatibility detection for sm_120 architecture"
    echo "   • Automatic fallback to CPU if needed"
    echo "   • BAAI/bge-reranker-v2-m3 model optimization"
else
    echo "❌ Build failed. Please check the error messages above."
    exit 1
fi
