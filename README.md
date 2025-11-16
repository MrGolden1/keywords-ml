# Transaction Categorization ML Pipeline

Machine learning pipeline for categorizing bank transaction descriptions into categories and subcategories.

## Dataset
- **File**: `Plaid_transactions_for_categorization.csv`
- **Size**: ~6.3M transactions
- **Categories**: 12 (Transfer, Shops, Service, Food and Drink, etc.)
- **Subcategories**: 141 → 111 (after filtering rare ones)

## Model Architecture
- **Embedding**: BAAI/bge-small-en-v1.5 (384-dim, optimized for short text)
- **Classifiers**: Logistic Regression with balanced class weights
  - Category classifier (12 classes)
  - Subcategory classifier (111 classes)
- **Hierarchical Masking**: Ensures predicted subcategory is valid for predicted category

## Key Features
✅ Handles massive class imbalance (1.7M x ratio)  
✅ Filters rare subcategories (<100 samples)  
✅ Stratified train/val split  
✅ GPU acceleration for embeddings  
✅ Hierarchical prediction with masking  
✅ Production-ready inference API

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn sentence-transformers torch joblib
```

### 2. Run Full Pipeline
```bash
python run_pipeline.py
```

This will:
1. ✅ Clean and prepare data
2. ✅ Generate embeddings
3. ✅ Train models
4. ✅ Evaluate performance
5. ✅ Run inference demo

### 3. Or Run Steps Individually
```bash
# Step 1: Prepare data
python prepare_data.py

# Step 2: Generate embeddings (uses GPU if available)
python embed_dataset.py

# Step 3: Train classifiers
python train_models.py

# Step 4: Evaluate models
python evaluate.py

# Step 5: Test inference
python inference.py
```

## Project Structure
```
keywords-ml/
├── config.py                    # Configuration and hyperparameters
├── prepare_data.py              # Data cleaning and preprocessing
├── embed_dataset.py             # Embedding generation
├── train_models.py              # Model training
├── evaluate.py                  # Model evaluation
├── inference.py                 # Production inference API
├── run_pipeline.py              # Master script
├── explore_dataset.py           # Dataset exploration
│
├── artifacts/                   # Saved models and encoders
│   ├── category_classifier_logreg.joblib
│   ├── subcategory_classifier_logreg.joblib
│   ├── category_label_encoder.joblib
│   ├── subcategory_label_encoder.joblib
│   ├── cat_to_sub_mapping.joblib
│   ├── X_train_embeddings.npy
│   ├── X_val_embeddings.npy
│   └── ...
│
├── data_cleaned.parquet         # Cleaned dataset
├── data_train.parquet           # Training set
└── data_val.parquet             # Validation set
```

## Usage Example

### Python API
```python
from inference import TransactionClassifier

# Initialize classifier (loads models once)
classifier = TransactionClassifier()

# Classify single transaction
result = classifier.classify_transaction("WALMART SUPERCENTER")
print(result)
# {
#     'category': 'Shops',
#     'subcategory': 'Supermarkets and Groceries',
#     'category_confidence': 0.9854,
#     'subcategory_confidence': 0.8923
# }

# Batch classification
transactions = ["NETFLIX", "SHELL GAS", "SALARY DEPOSIT"]
results = classifier.batch_classify(transactions)
```

## Class Imbalance Strategy

**Problem**: Extreme class imbalance
- Categories: 4370x ratio (max: 3.7M, min: 847)
- Subcategories: 1.7M x ratio (max: 1.7M, min: 1)

**Solutions Applied**:
1. ✅ **Filter rare subcategories**: Remove classes with <100 samples
2. ✅ **Stratified sampling**: Maintain class distribution in train/val split
3. ✅ **Class weighting**: `class_weight='balanced'` in LogisticRegression
4. ✅ **Strong embeddings**: Use specialized model for short text

## Expected Performance

Based on the guide and similar projects:
- **Category accuracy**: ~95-98% (12 classes, relatively balanced)
- **Subcategory accuracy**: ~85-92% (111 classes, more challenging)
- **Joint accuracy**: ~82-88% (both must be correct)

## Configuration

Edit `config.py` to customize:
- Embedding model (try `sentence-transformers/all-MiniLM-L6-v2`)
- Minimum subcategory samples threshold
- Train/val split ratio
- Classifier hyperparameters
- Batch sizes

## Next Steps

Once the basic pipeline works:
- [ ] Try alternative embedding models
- [ ] Experiment with XGBoost for subcategory classification
- [ ] Add k-fold cross-validation
- [ ] Include additional features (transaction type, amount)
- [ ] Deploy as REST API
- [ ] Add confidence thresholding for uncertain predictions

## System Requirements

- **RAM**: 30GB+ (you're good!)
- **GPU**: Optional but recommended for faster embedding (detected automatically)
- **Disk**: ~2GB for embeddings and models
- **Python**: 3.8+
