# RTX 5080 Blackwell Support Fixes

## 수정된 파일들

### 1. rag/llm/rerank_model.py

#### 변경사항 1: RTX 5080 GPU 지원 활성화 (라인 90-92)
```python
# 수정 전:
if compute_cap >= 120:
    print(f"⚠️ GPU {props.name} (sm_{props.major}{props.minor}) not supported, using CPU")
    use_gpu = False
else:
    use_gpu = True

# 수정 후:
# Enable GPU for all compatible architectures including Blackwell (sm_120+)
use_gpu = True
print(f"✅ Using GPU {props.name} (sm_{props.major}{props.minor})")
```

#### 변경사항 2: YoudaoRerank GPU 지원 활성화 (라인 213-215)
```python
# 수정 전:
if compute_cap >= 120:
    print(f"⚠️ GPU {props.name} (sm_{props.major}{props.minor}) not supported for YoudaoRerank, using CPU")
    device = 'cpu'
else:
    device = 'cuda'

# 수정 후:
# Enable GPU for all compatible architectures including Blackwell (sm_120+)
device = 'cuda'
print(f"✅ YoudaoRerank using GPU {props.name} (sm_{props.major}{props.minor})")
```

#### 변경사항 3: 배열 초기화 오류 수정 (라인 122)
```python
# 수정 전:
res = np.array(len(pairs), dtype=float)  # 스칼라 생성 (오류)

# 수정 후:
res = np.zeros(len(pairs), dtype=float)  # 올바른 배열 생성
```

#### 변경사항 4: 0차원 배열 처리 추가 (라인 160-161)
```python
# 추가된 코드:
# Ensure scores is always a list/array, even for single batch
if hasattr(scores, 'ndim') and scores.ndim == 0:
    scores = [float(scores)]
```

### 2. conf/llm_factories.json

#### 변경사항: Ollama bge-reranker-v2-m3 모델 타입 수정 (라인 4025-4029)
```json
// 수정 전:
{
    "llm_name": "bge-reranker-v2-m3",
    "tags": "TEXT EMBEDDING,TEXT RE-RANK",
    "max_tokens": 8192,
    "model_type": "embedding",  // 잘못된 타입
    "is_tools": false
}

// 수정 후:
{
    "llm_name": "bge-reranker-v2-m3",
    "tags": "TEXT RE-RANK",
    "max_tokens": 8192,
    "model_type": "rerank",  // 올바른 타입
    "is_tools": false
}
```

### 3. docker/docker-compose-gpu.yml

#### 변경사항: 수정된 파일 마운트 추가
```yaml
volumes:
  - ./ragflow-logs:/ragflow/logs
  - ./nginx/ragflow.conf:/etc/nginx/conf.d/ragflow.conf
  - ./nginx/proxy.conf:/etc/nginx/proxy.conf
  - ./nginx/nginx.conf:/etc/nginx/nginx.conf
  - ../deepdoc:/ragflow/deepdoc
  - ../rag/llm/rerank_model.py:/ragflow/rag/llm/rerank_model.py  # 추가
  - ../conf/llm_factories.json:/ragflow/conf/llm_factories.json  # 추가
```

### 4. docker/.env

#### 변경사항: 포트 변경 (충돌 해결)
```bash
# 수정 전:
SVR_HTTP_PORT=9380

# 수정 후:
SVR_HTTP_PORT=19380
```

## 테스트 결과

### GPU 지원 확인
```
GPU available: True
GPU: NVIDIA GeForce RTX 5080 (sm_120)
✅ Using GPU NVIDIA GeForce RTX 5080 (sm_120)
```

### 리랭커 성능 테스트
```
Query: What is machine learning?

Results:
1. Score: 0.9992 - Machine learning is a subset of artificial intelligence
2. Score: 0.0032 - Python is a programming language used for ML
3. Score: 0.0000 - The weather is nice today
4. Score: 0.0000 - Dogs are loyal pets

Tokens used: 26
```

## Docker 이미지 빌드 시 주의사항

1. **기본 이미지**: `jjkim110523/ragflow:blackwell` 사용 중
2. **필수 파일들**: 위 4개 파일의 수정사항이 모두 포함되어야 함
3. **GPU 지원**: `docker-compose-gpu.yml` 사용 필수
4. **포트**: 19380 포트 사용 (충돌 방지)

## 접속 정보

- **웹 인터페이스**: http://localhost:19380
- **GPU 가속**: RTX 5080 Blackwell 아키텍처 완전 지원
- **리랭커 모델**: BAAI/bge-reranker-v2-m3 (GPU 가속)

## 성능 개선 효과

✅ RTX 5080 GPU 완전 지원
✅ Blackwell 아키텍처(sm_120) 인식
✅ 리랭커 GPU 가속 활성화
✅ 배열 처리 오류 해결
✅ 안정적인 유사도 점수 계산