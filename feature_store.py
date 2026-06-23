"""
Week 3: Model Serving & Feature Store Setup
- Export embeddings
- Redis as low-latency Feature Store
- Approximate Nearest Neighbor (ANN) search using FAISS
Project 3: Context-Aware Neural Recommendation Engine
Zaalima Development
"""

import os
import json
import time
import pickle
import numpy as np
import redis
import faiss
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  REDIS FEATURE STORE
# ─────────────────────────────────────────────
class RedisFeatureStore:
    """
    Low-latency Feature Store backed by Redis.
    Stores:
      - user profiles (customer_id → feature dict)
      - pre-computed item vectors (article_id → embedding)
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # binary for numpy arrays
        )
        self._test_connection()

    def _test_connection(self):
        try:
            self.client.ping()
            logger.info("Redis connection established.")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    # ── User Profiles ──────────────────────────────────────────────────────

    def set_user_profile(self, customer_id: str, profile: Dict, ttl_seconds: int = 86400):
        """Store a user feature profile. TTL = 24h by default."""
        key = f"user:{customer_id}:profile"
        self.client.setex(key, ttl_seconds, json.dumps(profile))

    def get_user_profile(self, customer_id: str) -> Optional[Dict]:
        """Retrieve a user feature profile."""
        key = f"user:{customer_id}:profile"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def batch_set_user_profiles(self, profiles: Dict[str, Dict], ttl_seconds: int = 86400):
        """Batch upsert user profiles using Redis pipeline."""
        pipe = self.client.pipeline()
        for customer_id, profile in profiles.items():
            key = f"user:{customer_id}:profile"
            pipe.setex(key, ttl_seconds, json.dumps(profile))
        pipe.execute()
        logger.info(f"Stored {len(profiles)} user profiles in Redis.")

    # ── Item Embeddings ────────────────────────────────────────────────────

    def set_item_embedding(self, article_id: str, embedding: np.ndarray, ttl_seconds: int = 604800):
        """Store a pre-computed item embedding. TTL = 7 days by default."""
        key = f"item:{article_id}:embedding"
        self.client.setex(key, ttl_seconds, pickle.dumps(embedding))

    def get_item_embedding(self, article_id: str) -> Optional[np.ndarray]:
        """Retrieve a pre-computed item embedding."""
        key = f"item:{article_id}:embedding"
        data = self.client.get(key)
        if data:
            return pickle.loads(data)
        return None

    def batch_set_item_embeddings(self, embeddings: Dict[str, np.ndarray], ttl_seconds: int = 604800):
        """Batch upsert item embeddings using Redis pipeline."""
        pipe = self.client.pipeline()
        for article_id, emb in embeddings.items():
            key = f"item:{article_id}:embedding"
            pipe.setex(key, ttl_seconds, pickle.dumps(emb))
        pipe.execute()
        logger.info(f"Stored {len(embeddings)} item embeddings in Redis.")

    # ── User Interaction History ───────────────────────────────────────────

    def append_user_interaction(self, customer_id: str, article_id: str, max_history: int = 50):
        """Append a user interaction. Keeps last `max_history` items only."""
        key = f"user:{customer_id}:history"
        self.client.lpush(key, article_id)
        self.client.ltrim(key, 0, max_history - 1)

    def get_user_history(self, customer_id: str) -> List[str]:
        key = f"user:{customer_id}:history"
        return [item.decode() for item in self.client.lrange(key, 0, -1)]

    def health_check(self) -> Dict:
        info = self.client.info()
        return {
            "used_memory_human": info["used_memory_human"],
            "connected_clients": info["connected_clients"],
            "keyspace_hits": info["keyspace_hits"],
            "keyspace_misses": info["keyspace_misses"],
        }


# ─────────────────────────────────────────────
#  EMBEDDING EXPORTER
# ─────────────────────────────────────────────
class EmbeddingExporter:
    """
    Day 1-3: Loads the trained candidate tower and generates
    embeddings for all items in the catalog.
    """

    def __init__(self, candidate_tower_path: str):
        logger.info(f"Loading candidate tower from: {candidate_tower_path}")
        self.candidate_tower = tf.keras.models.load_model(candidate_tower_path)

    def generate_item_embeddings(
        self,
        item_features: Dict[str, List],
        batch_size: int = 512,
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all items.
        item_features: {"article_id": [...], "product_group_name": [...], ...}
        Returns dict: {article_id: embedding_vector}
        """
        article_ids = item_features["article_id"]
        n = len(article_ids)
        logger.info(f"Generating embeddings for {n} items...")

        all_embeddings = {}
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = {k: v[start:end] for k, v in item_features.items()}
            batch_tf = {k: tf.constant(v) for k, v in batch.items()}
            embeddings = self.candidate_tower(batch_tf).numpy()
            for i, article_id in enumerate(article_ids[start:end]):
                all_embeddings[article_id] = embeddings[i]

        logger.info(f"Generated {len(all_embeddings)} item embeddings.")
        return all_embeddings

    def generate_user_embedding(
        self,
        query_tower: tf.keras.Model,
        user_features: Dict,
    ) -> np.ndarray:
        """Generate a single user embedding at inference time."""
        user_tf = {k: tf.constant([v]) for k, v in user_features.items()}
        return query_tower(user_tf).numpy()[0]


# ─────────────────────────────────────────────
#  FAISS ANN INDEX
# ─────────────────────────────────────────────
class FAISSANNIndex:
    """
    Day 7: Approximate Nearest Neighbor (ANN) search using FAISS.
    Used for rapid candidate retrieval during inference.
    """

    def __init__(self, embedding_dim: int = 64, use_gpu: bool = False):
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index = None
        self.article_ids = []
        self._build_index()

    def _build_index(self):
        """Build a FAISS IVF-Flat index for scalable ANN search."""
        # IVFFlat: inverted file index for millions of vectors
        quantizer = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine after L2 norm)
        nlist = 100  # number of Voronoi cells
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        logger.info(f"FAISS IVFFlat index created (dim={self.embedding_dim}, nlist={nlist})")

    def train_and_add(self, embeddings: Dict[str, np.ndarray]):
        """Train the index and add all item embeddings."""
        self.article_ids = list(embeddings.keys())
        vectors = np.array(list(embeddings.values()), dtype=np.float32)

        # L2 normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        logger.info(f"Training FAISS index on {len(vectors)} vectors...")
        self.index.train(vectors)
        self.index.add(vectors)
        self.index.nprobe = 10  # number of cells to visit during search
        logger.info(f"FAISS index ready: {self.index.ntotal} vectors indexed.")

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Search for top-K nearest neighbors.
        Returns list of (article_id, score) tuples.
        """
        query = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.article_ids) and idx >= 0:
                results.append((self.article_ids[idx], float(score)))
        return results

    def save(self, path: str = "./models/faiss_index"):
        """Persist the FAISS index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path + ".bin")
        with open(path + "_ids.json", "w") as f:
            json.dump(self.article_ids, f)
        logger.info(f"FAISS index saved to {path}")

    def load(self, path: str = "./models/faiss_index"):
        """Load a persisted FAISS index from disk."""
        self.index = faiss.read_index(path + ".bin")
        with open(path + "_ids.json") as f:
            self.article_ids = json.load(f)
        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")


# ─────────────────────────────────────────────
#  SETUP SCRIPT
# ─────────────────────────────────────────────
def setup_feature_store_and_ann(
    candidate_tower_path: str,
    item_features: Dict[str, List],
    user_profiles: Dict[str, Dict],
    redis_host: str = "localhost",
    redis_port: int = 6379,
    embedding_dim: int = 64,
):
    """
    Full Week 3 pipeline:
    1. Generate item embeddings from trained candidate tower
    2. Push embeddings + user profiles to Redis
    3. Build FAISS index for ANN retrieval
    """
    # Step 1: Generate embeddings
    exporter = EmbeddingExporter(candidate_tower_path)
    item_embeddings = exporter.generate_item_embeddings(item_features)

    # Step 2: Push to Redis
    store = RedisFeatureStore(host=redis_host, port=redis_port)
    store.batch_set_item_embeddings(item_embeddings)
    store.batch_set_user_profiles(user_profiles)

    # Step 3: Build FAISS ANN index
    ann = FAISSANNIndex(embedding_dim=embedding_dim)
    ann.train_and_add(item_embeddings)
    ann.save("./models/faiss_index")

    logger.info("Week 3 setup complete: Feature store & ANN index ready.")
    return store, ann


if __name__ == "__main__":
    logger.info("Week 3: Feature Store setup (requires trained model from Week 2)")
    # This is run after Week 2 training completes
    # setup_feature_store_and_ann(...)
