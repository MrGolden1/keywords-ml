"""Configuration for transaction category classification model."""

# ========== Model Configuration ==========
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, widely used
# Alternative: "BAAI/bge-small-en-v1.5"

# ========== Data Configuration ==========
CSV_PATH = "Plaid_transactions_for_categorization.csv"
TEXT_COLUMN = "clean_description"  # Use cleaned transaction descriptions
TYPE_COLUMN = "type"  # Transaction type: debit or credit
CATEGORY_COLUMN = "category"
SUBCATEGORY_COLUMN = "sub_category"

# ========== Data Filtering ==========
MIN_SUBCATEGORY_SAMPLES = 100  # Filter out rare subcategories
MAX_ROWS = None  # Set to None for full dataset, or limit for testing (e.g., 10_000)

# ========== Training Configuration ==========
TEST_SIZE = 0.2  # 20% validation split
RANDOM_STATE = 42  # For reproducibility
EMB_BATCH_SIZE = 2048  # Batch size for embedding generation (L40S GPU can handle up to 2048)

# ========== Classifier Configuration ==========
# XGBoost is better for handling class imbalance and non-linear patterns
USE_XGBOOST = True  # Set to False to use SGDClassifier instead

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'tree_method': 'hist',  # Fast histogram-based
    'device': 'cuda',  # Use GPU if available, falls back to CPU
    'verbosity': 1,
    'random_state': 42,
}

SGD_PARAMS = {
    'loss': 'log_loss',
    'penalty': 'l2',
    'alpha': 0.0001,
    'max_iter': 200,
    'class_weight': 'balanced',
    'verbose': 1,
    'n_jobs': 1,
    'random_state': 42,
}

# ========== File Paths ==========
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
DATA_CLEANED = "data_cleaned.parquet"
DATA_TRAIN = "data_train.parquet"
DATA_VAL = "data_val.parquet"

# Model artifacts
CAT_ENCODER_PATH = ARTIFACTS_DIR / "category_label_encoder.joblib"
SUBCAT_ENCODER_PATH = ARTIFACTS_DIR / "subcategory_label_encoder.joblib"
CAT_TO_SUB_MAPPING_PATH = ARTIFACTS_DIR / "cat_to_sub_mapping.joblib"

# Embeddings
X_TRAIN_EMB_PATH = ARTIFACTS_DIR / "X_train_embeddings.npy"
X_VAL_EMB_PATH = ARTIFACTS_DIR / "X_val_embeddings.npy"
Y_CAT_TRAIN_PATH = ARTIFACTS_DIR / "y_cat_train.npy"
Y_CAT_VAL_PATH = ARTIFACTS_DIR / "y_cat_val.npy"
Y_SUB_TRAIN_PATH = ARTIFACTS_DIR / "y_sub_train.npy"
Y_SUB_VAL_PATH = ARTIFACTS_DIR / "y_sub_val.npy"

# Trained models (embedding-based)
CAT_CLASSIFIER_PATH = ARTIFACTS_DIR / "category_classifier_logreg.joblib"
SUB_CLASSIFIER_PATH = ARTIFACTS_DIR / "subcategory_classifier_logreg.joblib"

# TF-IDF artifacts (for keyword extraction)
TFIDF_VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
CAT_TFIDF_SVC_PATH = ARTIFACTS_DIR / "cat_tfidf_linearsvc.joblib"
SUB_TFIDF_SVC_PATH = ARTIFACTS_DIR / "sub_tfidf_linearsvc.joblib"
TFIDF_KEYWORDS_JSON = ARTIFACTS_DIR / "tfidf_keywords.json"
