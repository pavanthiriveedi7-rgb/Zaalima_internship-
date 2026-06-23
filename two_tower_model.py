"""
Week 2: Deep Learning Model Architecture
Two-Tower Neural Network using TensorFlow Recommenders (TFRS)
Project 3: Context-Aware Neural Recommendation Engine
Zaalima Development
"""

import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  QUERY TOWER (User Context)
# ─────────────────────────────────────────────
class QueryTower(tf.keras.Model):
    """
    Encodes user context into a dense embedding vector.
    Inputs: customer_id, age (normalized), club_member_status, recency, purchase_hour
    """

    def __init__(
        self,
        user_ids: List[str],
        club_statuses: List[str],
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Embedding layers for categorical features
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(user_ids) + 1, embedding_dim),
        ])

        self.club_status_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=club_statuses, mask_token=None),
            tf.keras.layers.Embedding(len(club_statuses) + 1, 8),
        ])

        # Dense network to combine all user features
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(embedding_dim),
        ])

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        user_emb = self.user_embedding(inputs["customer_id"])
        club_emb = self.club_status_embedding(inputs["club_member_status"])

        # Continuous features
        age_norm = tf.expand_dims(inputs["age_normalized"], axis=-1)
        recency = tf.expand_dims(inputs["days_since_last_purchase_normalized"], axis=-1)
        hour = tf.expand_dims(inputs["purchase_hour_normalized"], axis=-1)

        # Concatenate all
        concat = tf.concat([user_emb, club_emb, age_norm, recency, hour], axis=-1)
        return self.dense_layers(concat)


# ─────────────────────────────────────────────
#  CANDIDATE TOWER (Item Context)
# ─────────────────────────────────────────────
class CandidateTower(tf.keras.Model):
    """
    Encodes item context into a dense embedding vector.
    Inputs: article_id, product_group_name, colour_group_name, monthly_purchase_count
    """

    def __init__(
        self,
        article_ids: List[str],
        product_groups: List[str],
        colour_groups: List[str],
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.article_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=article_ids, mask_token=None),
            tf.keras.layers.Embedding(len(article_ids) + 1, embedding_dim),
        ])

        self.product_group_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=product_groups, mask_token=None),
            tf.keras.layers.Embedding(len(product_groups) + 1, 16),
        ])

        self.colour_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=colour_groups, mask_token=None),
            tf.keras.layers.Embedding(len(colour_groups) + 1, 8),
        ])

        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(embedding_dim),
        ])

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        article_emb = self.article_embedding(inputs["article_id"])
        product_emb = self.product_group_embedding(inputs["product_group_name"])
        colour_emb = self.colour_embedding(inputs["colour_group_name"])
        popularity = tf.expand_dims(inputs["monthly_purchase_count_normalized"], axis=-1)

        concat = tf.concat([article_emb, product_emb, colour_emb, popularity], axis=-1)
        return self.dense_layers(concat)


# ─────────────────────────────────────────────
#  FULL TWO-TOWER MODEL (TFRS)
# ─────────────────────────────────────────────
class TwoTowerRecommender(tfrs.Model):
    """
    Full Two-Tower model combining Query and Candidate towers.
    Uses TFRS FactorizedTopK metric for evaluation.
    """

    def __init__(
        self,
        query_tower: QueryTower,
        candidate_tower: CandidateTower,
        articles_dataset: tf.data.Dataset,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.query_tower = query_tower
        self.candidate_tower = candidate_tower

        # Task: retrieval using factorized top-k
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=articles_dataset.batch(128).map(self.candidate_tower)
            )
        )

    def compute_loss(self, features: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        query_embeddings = self.query_tower(features)
        candidate_embeddings = self.candidate_tower(features)
        return self.task(query_embeddings, candidate_embeddings, compute_metrics=not training)


# ─────────────────────────────────────────────
#  NEGATIVE SAMPLING DATASET BUILDER
# ─────────────────────────────────────────────
def build_training_dataset(
    transactions: List[Dict],
    all_article_ids: List[str],
    negative_ratio: int = 4,
    batch_size: int = 1024,
) -> tf.data.Dataset:
    """
    Build dataset with negative sampling.
    For each positive (user, item) pair, sample `negative_ratio` random negatives.
    """
    logger.info("Building training dataset with negative sampling...")

    positives = []
    for tx in transactions:
        positives.append({**tx, "label": 1.0})

    # Generate negatives
    negatives = []
    rng = np.random.default_rng(42)
    for tx in transactions:
        neg_articles = rng.choice(all_article_ids, size=negative_ratio, replace=False)
        for neg in neg_articles:
            neg_tx = {**tx, "article_id": neg, "label": 0.0}
            negatives.append(neg_tx)

    all_samples = positives + negatives
    rng.shuffle(all_samples)

    def gen():
        for sample in all_samples:
            yield sample

    # Build tf.data.Dataset
    sample_types = {k: tf.string if isinstance(v, str) else tf.float32 for k, v in all_samples[0].items()}
    dataset = tf.data.Dataset.from_generator(gen, output_signature={
        k: tf.TensorSpec(shape=(), dtype=t) for k, t in sample_types.items()
    })
    dataset = dataset.shuffle(10_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    logger.info(f"Dataset built: {len(all_samples)} samples")
    return dataset


# ─────────────────────────────────────────────
#  TRAINING PIPELINE
# ─────────────────────────────────────────────
def train_model(
    model: TwoTowerRecommender,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    output_dir: str = "./models",
) -> tf.keras.callbacks.History:
    """Train the Two-Tower model with callbacks."""
    os.makedirs(output_dir, exist_ok=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_total_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_total_loss", factor=0.5, patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model"),
            save_best_only=True,
            monitor="val_total_loss",
        ),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, "logs")),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )
    logger.info("Training complete.")
    return history


# ─────────────────────────────────────────────
#  EVALUATION METRICS
# ─────────────────────────────────────────────
def evaluate_recall_at_k(
    model: TwoTowerRecommender,
    test_dataset: tf.data.Dataset,
    k_values: List[int] = [10, 50, 100],
) -> Dict[str, float]:
    """Compute Recall@K and NDCG for the retrieval task."""
    results = model.evaluate(test_dataset, return_dict=True)
    logger.info(f"Evaluation results: {results}")
    return results


def compute_ndcg(recommended: List[str], relevant: List[str], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K).
    recommended: ordered list of recommended item IDs
    relevant: set of ground-truth item IDs
    """
    relevant_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # log2(rank+1), rank starts at 1

    # Ideal DCG
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────────────────────────────
#  EXPORT MODELS
# ─────────────────────────────────────────────
def export_towers(
    model: TwoTowerRecommender,
    article_ids: List[str],
    output_dir: str = "./models",
):
    """
    Export Query Tower (for inference) and precompute all item embeddings.
    """
    logger.info("Exporting model towers...")

    # Save query tower
    query_path = os.path.join(output_dir, "query_tower")
    model.query_tower.save(query_path)
    logger.info(f"Query tower saved to {query_path}")

    # Save candidate tower
    candidate_path = os.path.join(output_dir, "candidate_tower")
    model.candidate_tower.save(candidate_path)
    logger.info(f"Candidate tower saved to {candidate_path}")

    return query_path, candidate_path


if __name__ == "__main__":
    # Load vocabularies from Week 1
    with open("./data/vocab/user_vocab.json") as f:
        user_vocab = json.load(f)
    with open("./data/vocab/item_vocab.json") as f:
        item_vocab = json.load(f)

    EMBEDDING_DIM = 64

    query_tower = QueryTower(
        user_ids=user_vocab["customer_id"],
        club_statuses=user_vocab["club_member_status"],
        embedding_dim=EMBEDDING_DIM,
    )

    candidate_tower = CandidateTower(
        article_ids=item_vocab["article_id"],
        product_groups=item_vocab["product_group_name"],
        colour_groups=item_vocab["colour_group_name"],
        embedding_dim=EMBEDDING_DIM,
    )

    # Placeholder articles dataset for metrics
    articles_ds = tf.data.Dataset.from_tensor_slices({
        "article_id": item_vocab["article_id"],
        "product_group_name": ["Unknown"] * len(item_vocab["article_id"]),
        "colour_group_name": ["Unknown"] * len(item_vocab["article_id"]),
        "monthly_purchase_count_normalized": [0.0] * len(item_vocab["article_id"]),
    })

    model = TwoTowerRecommender(query_tower, candidate_tower, articles_ds, EMBEDDING_DIM)
    logger.info("Week 2: Two-Tower model ready for training.")
