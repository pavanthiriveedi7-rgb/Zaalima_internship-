"""
Week 4: Apache Airflow DAG
- Weekly model retraining
- Daily embedding updates
Project 3: Context-Aware Neural Recommendation Engine
Zaalima Development
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.email import send_email
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  DEFAULT ARGS
# ─────────────────────────────────────────────
default_args = {
    "owner": "zaalima-ml-team",
    "depends_on_past": False,
    "email": ["ml-alerts@zaalima.dev"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=6),
}


# ─────────────────────────────────────────────
#  TASK FUNCTIONS
# ─────────────────────────────────────────────
def extract_new_transactions(**context):
    """Pull last 7 days of transactions from the data warehouse."""
    logger.info("Extracting new transactions from data warehouse...")
    # In production: query Redshift/BigQuery or Spark job
    # spark-submit week1/data_processing.py --mode incremental


def validate_data_quality(**context):
    """Run Great Expectations data quality checks."""
    logger.info("Running data quality validation...")
    # Check for: missing customer_ids, null prices, future timestamps
    # Raise AirflowException if quality checks fail


def run_feature_engineering(**context):
    """Re-run feature engineering on new data."""
    logger.info("Running feature engineering pipeline...")
    # spark-submit week1/data_processing.py --mode features


def retrain_model(**context):
    """Retrain Two-Tower model with latest data."""
    logger.info("Starting model retraining...")
    # python week2/two_tower_model.py --mode retrain --epochs 5


def evaluate_model(**context):
    """Evaluate new model vs current production model."""
    logger.info("Evaluating new model...")
    # Compare Recall@K and NDCG — only promote if improvement > 1%
    new_model_recall = 0.85   # placeholder — real value from eval
    prod_model_recall = 0.83  # placeholder — from model registry
    if new_model_recall <= prod_model_recall:
        raise ValueError(f"New model ({new_model_recall}) not better than prod ({prod_model_recall}). Aborting.")
    logger.info(f"New model is better. Recall: {new_model_recall} vs {prod_model_recall}")


def promote_model(**context):
    """Copy new model artifacts to production path."""
    logger.info("Promoting new model to production...")
    # cp -r ./models/new/ ./models/production/
    # Trigger rolling restart of FastAPI pods


def update_item_embeddings(**context):
    """Daily: regenerate item embeddings and refresh FAISS index."""
    logger.info("Updating item embeddings...")
    # python week3/feature_store.py --mode update_embeddings


def refresh_faiss_index(**context):
    """Daily: rebuild FAISS index with updated embeddings."""
    logger.info("Rebuilding FAISS ANN index...")
    # python week3/feature_store.py --mode rebuild_index


def notify_success(**context):
    logger.info("Pipeline completed successfully. Sending notification...")


# ─────────────────────────────────────────────
#  DAG 1: WEEKLY MODEL RETRAINING
# ─────────────────────────────────────────────
with DAG(
    dag_id="weekly_model_retraining",
    default_args=default_args,
    description="Full model retraining pipeline — runs every Sunday at 2 AM",
    schedule_interval="0 2 * * 0",  # Every Sunday at 02:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "retraining", "recommendation"],
) as weekly_dag:

    t_extract = PythonOperator(
        task_id="extract_new_transactions",
        python_callable=extract_new_transactions,
    )

    t_validate = PythonOperator(
        task_id="validate_data_quality",
        python_callable=validate_data_quality,
    )

    t_features = PythonOperator(
        task_id="run_feature_engineering",
        python_callable=run_feature_engineering,
    )

    t_retrain = BashOperator(
        task_id="retrain_model",
        bash_command=(
            "cd /opt/recommendation_engine && "
            "python week2/two_tower_model.py --mode retrain --epochs 10"
        ),
    )

    t_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    t_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    t_notify = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
    )

    # DAG flow
    t_extract >> t_validate >> t_features >> t_retrain >> t_evaluate >> t_promote >> t_notify


# ─────────────────────────────────────────────
#  DAG 2: DAILY EMBEDDING UPDATES
# ─────────────────────────────────────────────
with DAG(
    dag_id="daily_embedding_updates",
    default_args=default_args,
    description="Daily refresh of item embeddings and FAISS index",
    schedule_interval="0 3 * * *",  # Every day at 03:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "embeddings", "recommendation"],
) as daily_dag:

    t_update_embeddings = PythonOperator(
        task_id="update_item_embeddings",
        python_callable=update_item_embeddings,
    )

    t_refresh_index = PythonOperator(
        task_id="refresh_faiss_index",
        python_callable=refresh_faiss_index,
    )

    t_health_check = BashOperator(
        task_id="api_health_check",
        bash_command="curl -f http://localhost:8000/health || exit 1",
    )

    t_notify_daily = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
    )

    t_update_embeddings >> t_refresh_index >> t_health_check >> t_notify_daily
