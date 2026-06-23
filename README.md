# Project 3: Context-Aware Neural Recommendation Engine
**Zaalima Development — Production DSML Project**

A Two-Tower deep learning recommendation system for personalized fashion recommendations using the H&M dataset.

---

## Architecture Overview

```
[H&M Dataset]
     │
     ▼
[Week 1: PySpark Pipeline]
 └── data_processing.py
     ├── Missing value handling & cold-start detection
     ├── Contextual feature engineering (recency, popularity, time)
     └── Vocabulary generation (user & item)
     │
     ▼
[Week 2: Two-Tower Neural Network]
 └── two_tower_model.py
     ├── QueryTower  → encodes user context
     ├── CandidateTower → encodes item context
     ├── Negative sampling dataset builder
     └── Recall@K & NDCG evaluation
     │
     ▼
[Week 3: Feature Store & ANN Index]
 └── feature_store.py
     ├── Redis → user profiles + item embeddings
     ├── FAISS IVFFlat index → ANN search
     └── EmbeddingExporter → batch item embedding generation
     │
     ▼
[Week 4: FastAPI + Airflow]
 ├── api.py → real-time recommendation endpoint
 └── airflow_dag.py
     ├── Weekly model retraining DAG
     └── Daily embedding refresh DAG
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download H&M Dataset
```bash
# Download from Kaggle: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
# Place files in ./data/raw/
# articles.csv, customers.csv, transactions_train.csv
```

### 3. Run Week 1 — Data Processing
```bash
python week1/data_processing.py
```

### 4. Run Week 2 — Train Model
```bash
python week2/two_tower_model.py
```

### 5. Run Week 3 — Setup Feature Store
```bash
# Requires Redis running:
docker run -d -p 6379:6379 redis:7-alpine
python week3/feature_store.py
```

### 6. Run Week 4 — Start API
```bash
# Option A: Direct
uvicorn week4.api:app --host 0.0.0.0 --port 8000

# Option B: Docker Compose (full stack)
docker-compose up --build
```

### 7. Test the API
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657", "top_k": 10}'
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/recommend` | Get top-K recommendations |
| `GET` | `/user/{id}/profile` | Fetch user feature profile |
| `GET` | `/user/{id}/history` | Fetch user interaction history |
| `GET` | `/items/{id}` | Fetch item metadata |

---

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin / zaalima2025)

---

## Key Metrics
- **Recall@10** — primary retrieval quality metric
- **NDCG@10** — ranking quality
- **API Latency** — target < 50ms p99
- **ANN Index Size** — scales to millions of items

---

## GitHub Workflow (Zaalima Requirement)
```bash
git init
git checkout -b main
git add .
git commit -m "feat(week1): PySpark data processing pipeline"
# Push daily — commit history is your evaluation metric
git push origin main
```

---

*Zaalima Development Confidential*
