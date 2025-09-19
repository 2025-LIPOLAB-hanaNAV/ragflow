# 외부 모델 디렉토리 설정 가이드

## 개요

RAGFlow에서 모델을 별도 디렉토리로 관리할 수 있도록 설정했습니다.
이제 모델을 미리 다운로드해서 로컬에 저장하고, 컨테이너에서 사용할 수 있습니다.

## 디렉토리 구조

```
ragflow/docker/
├── models/                     # 외부 모델 저장 디렉토리
│   ├── BAAI/                  # Hugging Face 모델 구조
│   │   └── bge-reranker-v2-m3/
│   ├── sentence-transformers/
│   └── microsoft/
└── docker-compose-gpu.yml     # 모델 마운트 설정 포함
```

## 마운트 설정

Docker Compose에서 다음과 같이 마운트됩니다:

```yaml
volumes:
  - ./models:/ragflow/models              # RAGFlow 모델 디렉토리
  - ./models:/root/.cache/huggingface     # Hugging Face 캐시
  - ./models:/root/.cache/modelscope      # ModelScope 캐시

environment:
  - HF_HOME=/root/.cache/huggingface
  - TRANSFORMERS_CACHE=/root/.cache/huggingface
  - MODEL_PATH=/ragflow/models
```

## 모델 사용 방법

### 1. 모델 다운로드

로컬에서 원하는 모델을 미리 다운로드:

```bash
# models 디렉토리로 이동
cd /mnt/c/Users/JJ/Projects/ragflow/docker/models

# 1) Git LFS로 직접 다운로드
git clone https://huggingface.co/BAAI/bge-reranker-v2-m3

# 2) Python으로 다운로드
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('BAAI/bge-reranker-v2-m3', cache_dir='./BAAI')
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3', cache_dir='./BAAI')
"

# 3) Hugging Face CLI로 다운로드
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ./BAAI/bge-reranker-v2-m3
```

### 2. 컨테이너 재시작

모델을 다운로드한 후 컨테이너를 재시작:

```bash
docker-compose -f docker-compose-gpu.yml down
docker-compose -f docker-compose-gpu.yml up -d
```

### 3. 모델 경로 확인

컨테이너 내에서 모델이 올바르게 마운트되었는지 확인:

```bash
docker exec ragflow-server ls -la /ragflow/models
docker exec ragflow-server ls -la /root/.cache/huggingface
```

## 모델 설정 예시

### 기본 설정 (자동 다운로드)
```python
# RAGFlow가 자동으로 Hugging Face에서 다운로드
model_name = "BAAI/bge-reranker-v2-m3"
```

### 로컬 모델 사용
```python
# 미리 다운로드된 로컬 모델 사용
model_name = "/ragflow/models/BAAI/bge-reranker-v2-m3"
# 또는
model_name = "/root/.cache/huggingface/BAAI/bge-reranker-v2-m3"
```

## 장점

1. **빠른 시작**: 모델을 미리 다운로드해두면 컨테이너 시작 시간 단축
2. **오프라인 사용**: 인터넷 연결 없이도 모델 사용 가능
3. **버전 관리**: 특정 버전의 모델을 고정해서 사용 가능
4. **용량 절약**: 여러 컨테이너가 같은 모델 디렉토리 공유 가능

## 주의사항

1. **디스크 용량**: 모델 파일은 크기가 클 수 있으므로 충분한 디스크 공간 확보
2. **권한 설정**: 모델 파일의 읽기 권한 확인
3. **모델 호환성**: RAGFlow에서 지원하는 모델 형식인지 확인

## 예시: bge-reranker-v2-m3 설치

```bash
# 1. models 디렉토리로 이동
cd /mnt/c/Users/JJ/Projects/ragflow/docker/models

# 2. 모델 다운로드
git clone https://huggingface.co/BAAI/bge-reranker-v2-m3

# 3. 디렉토리 구조 확인
ls -la BAAI/bge-reranker-v2-m3/
# config.json, pytorch_model.bin, tokenizer.json 등이 있어야 함

# 4. 컨테이너 재시작
cd ..
docker-compose -f docker-compose-gpu.yml restart ragflow

# 5. 확인
docker exec ragflow-server ls -la /ragflow/models/BAAI/bge-reranker-v2-m3/
```

이제 모델이 로컬에서 로드되어 네트워크 다운로드 시간이 없고, GPU 가속도 정상적으로 작동합니다.