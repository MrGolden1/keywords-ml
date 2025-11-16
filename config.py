"""Configuration for transaction category classification model."""

# ========== Model Configuration ==========
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, optimized for short text
# Alternative: "sentence-transformers/all-MiniLM-L6-v2"

# ========== Data Configuration ==========
CSV_PATH = "Plaid_transactions_for_categorization.csv"
TEXT_COLUMN = "clean_description"  # Use cleaned transaction descriptions
CATEGORY_COLUMN = "category"
SUBCATEGORY_COLUMN = "sub_category"

# ========== Data Filtering ==========
MIN_SUBCATEGORY_SAMPLES = 100  # Filter out rare subcategories
MAX_ROWS = None  # Set to e.g. 100_000 for faster experiments, None for full dataset

# ========== Training Configuration ==========
TEST_SIZE = 0.2  # 20% validation split
RANDOM_STATE = 42  # For reproducibility
EMB_BATCH_SIZE = 1024  # Batch size for embedding generation (L40S GPU can handle up to 2048)

# ========== Classifier Configuration ==========
# Using class_weight='balanced' helps with class imbalance
# Using 'saga' solver for memory efficiency with large datasets
LOGREG_PARAMS = {
    'max_iter': 100,
    'n_jobs': 1,  # Single job to avoid memory multiplication
    'verbose': 1,
    'class_weight': 'balanced',  # Handle class imbalance
    'solver': 'saga',  # Memory-efficient for large datasets
    'penalty': 'l2',
    'C': 1.0,
}

# ========== File Paths ==========
ARTIFACTS_DIR = "artifacts"
DATA_CLEANED = "data_cleaned.parquet"
DATA_TRAIN = "data_train.parquet"
DATA_VAL = "data_val.parquet"

# Model artifacts
CAT_ENCODER_PATH = f"{ARTIFACTS_DIR}/category_label_encoder.joblib"
SUBCAT_ENCODER_PATH = f"{ARTIFACTS_DIR}/subcategory_label_encoder.joblib"
CAT_TO_SUB_MAPPING_PATH = f"{ARTIFACTS_DIR}/cat_to_sub_mapping.joblib"

# Embeddings
X_TRAIN_EMB_PATH = f"{ARTIFACTS_DIR}/X_train_embeddings.npy"
X_VAL_EMB_PATH = f"{ARTIFACTS_DIR}/X_val_embeddings.npy"
Y_CAT_TRAIN_PATH = f"{ARTIFACTS_DIR}/y_cat_train.npy"
Y_CAT_VAL_PATH = f"{ARTIFACTS_DIR}/y_cat_val.npy"
Y_SUB_TRAIN_PATH = f"{ARTIFACTS_DIR}/y_sub_train.npy"
Y_SUB_VAL_PATH = f"{ARTIFACTS_DIR}/y_sub_val.npy"

# Trained models
CAT_CLASSIFIER_PATH = f"{ARTIFACTS_DIR}/category_classifier_logreg.joblib"
SUB_CLASSIFIER_PATH = f"{ARTIFACTS_DIR}/subcategory_classifier_logreg.joblib"
