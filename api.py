"""
Week 4: ML Pipeline Scheduling & API
FastAPI microservice for real-time recommendations
Project 3: Context-Aware Neural Recommendation Engine
Zaalima Development
"""

import os
import json
import time
import logging
import numpy as np
import tensorflow as tf
from contextlib import asynccontextmanager
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
import faiss
import pickle
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  GLOBAL STATE (loaded at startup)
# ─────────────────────────────────────────────
class AppState:
    query_tower: tf.keras.Model = None
    redis_client: redis.Redis = None
    faiss_index: faiss.Index = None
    article_ids: List[str] = []
    embedding_dim: int = 64


state = AppState()


# ─────────────────────────────────────────────
#  STARTUP / SHUTDOWN
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and connections on startup."""
    logger.info("Loading Query Tower model...")
    state.query_tower = tf.keras.models.load_model(
        os.getenv("QUERY_TOWER_PATH", "./models/query_tower")
    )

    logger.info("Connecting to Redis...")
    state.redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", None),
        decode_responses=False,
    )
    state.redis_client.ping()

    logger.info("Loading FAISS index...")
    faiss_path = os.getenv("FAISS_INDEX_PATH", "./models/faiss_index")
    state.faiss_index = faiss.read_index(faiss_path + ".bin")
    state.faiss_index.nprobe = 10
    with open(faiss_path + "_ids.json") as f:
        state.article_ids = json.load(f)

    logger.info(f"API ready. FAISS index has {state.faiss_index.ntotal} items.")
    yield

    logger.info("Shutting down...")
    state.redis_client.close()


# ─────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Zaalima Recommendation Engine API",
    description="Context-Aware Neural Recommendation Engine — Production API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations to return")
    context: Optional[Dict] = Field(default=None, description="Optional real-time context override")


class RecommendationItem(BaseModel):
    article_id: str
    score: float
    product_group_name: Optional[str] = None
    colour_group_name: Optional[str] = None


class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[RecommendationItem]
    latency_ms: float
    is_cold_start: bool


class HealthResponse(BaseModel):
    status: str
    faiss_index_size: int
    redis_memory: str
    model_loaded: bool


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def get_user_profile_from_redis(customer_id: str) -> Optional[Dict]:
    key = f"user:{customer_id}:profile"
    data = state.redis_client.get(key)
    return json.loads(data) if data else None


def get_item_metadata_from_redis(article_id: str) -> Optional[Dict]:
    key = f"item:{article_id}:meta"
    data = state.redis_client.get(key)
    return json.loads(data) if data else None


def build_cold_start_features(customer_id: str) -> Dict:
    """Fallback features for cold-start users."""
    return {
        "customer_id": customer_id,
        "club_member_status": "UNKNOWN",
        "age_normalized": 0.5,
        "days_since_last_purchase_normalized": 1.0,
        "purchase_hour_normalized": 0.5,
    }


def generate_user_embedding(user_features: Dict) -> np.ndarray:
    """Run query tower inference to get user embedding."""
    tf_input = {k: tf.constant([v]) for k, v in user_features.items()}
    embedding = state.query_tower(tf_input).numpy()[0]
    return embedding.astype(np.float32)


def ann_search(query_vector: np.ndarray, top_k: int) -> List[tuple]:
    """FAISS ANN search returning (article_id, score) pairs."""
    query = query_vector.reshape(1, -1)
    faiss.normalize_L2(query)
    distances, indices = state.faiss_index.search(query, top_k)
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(state.article_ids):
            results.append((state.article_ids[idx], float(score)))
    return results


def log_interaction_async(customer_id: str, article_ids: List[str]):
    """Log recommended items to user history in Redis (background task)."""
    pipe = state.redis_client.pipeline()
    for article_id in article_ids:
        key = f"user:{customer_id}:history"
        pipe.lpush(key, article_id)
        pipe.ltrim(key, 0, 49)  # keep last 50
    pipe.execute()


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check."""
    try:
        redis_info = state.redis_client.info()
        return HealthResponse(
            status="healthy",
            faiss_index_size=state.faiss_index.ntotal,
            redis_memory=redis_info["used_memory_human"],
            model_loaded=state.query_tower is not None,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {str(e)}")


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
):
    """
    Primary recommendation endpoint.
    - Fetches user profile from Redis
    - Runs query tower inference
    - ANN search over item embeddings
    - Returns top-K recommendations
    """
    start_time = time.time()

    # Fetch user profile
    user_profile = get_user_profile_from_redis(request.customer_id)
    is_cold_start = user_profile is None

    if is_cold_start:
        logger.info(f"Cold-start user: {request.customer_id}")
        user_features = build_cold_start_features(request.customer_id)
    else:
        # Apply optional real-time context overrides
        if request.context:
            user_profile.update(request.context)
        user_features = user_profile

    # Generate user embedding
    user_embedding = generate_user_embedding(user_features)

    # ANN retrieval
    ann_results = ann_search(user_embedding, top_k=request.top_k * 2)  # oversample, then rerank

    # Build response items
    recommendations = []
    for article_id, score in ann_results[:request.top_k]:
        meta = get_item_metadata_from_redis(article_id) or {}
        recommendations.append(RecommendationItem(
            article_id=article_id,
            score=round(score, 4),
            product_group_name=meta.get("product_group_name"),
            colour_group_name=meta.get("colour_group_name"),
        ))

    # Log recommended items to user history (non-blocking)
    background_tasks.add_task(
        log_interaction_async,
        request.customer_id,
        [r.article_id for r in recommendations],
    )

    latency_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"Recommendations for {request.customer_id}: {len(recommendations)} items in {latency_ms}ms")

    return RecommendationResponse(
        customer_id=request.customer_id,
        recommendations=recommendations,
        latency_ms=latency_ms,
        is_cold_start=is_cold_start,
    )


@app.get("/user/{customer_id}/history", tags=["Users"])
async def get_user_history(customer_id: str, limit: int = Query(default=20, le=50)):
    """Retrieve a user's recent interaction history."""
    key = f"user:{customer_id}:history"
    history = state.redis_client.lrange(key, 0, limit - 1)
    return {
        "customer_id": customer_id,
        "history": [item.decode() for item in history],
    }


@app.get("/user/{customer_id}/profile", tags=["Users"])
async def get_user_profile(customer_id: str):
    """Retrieve a user's stored feature profile."""
    profile = get_user_profile_from_redis(customer_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"No profile found for customer_id: {customer_id}")
    return {"customer_id": customer_id, "profile": profile}


@app.get("/items/{article_id}", tags=["Items"])
async def get_item_details(article_id: str):
    """Retrieve stored metadata for a given item."""
    meta = get_item_metadata_from_redis(article_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"No metadata found for article_id: {article_id}")
    return {"article_id": article_id, "metadata": meta}


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=False,
        log_level="info",
    )
