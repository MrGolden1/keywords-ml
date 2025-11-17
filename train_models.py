"""
Train category and subcategory classifiers using LogisticRegression.

Uses pre-computed embeddings to train two separate classifiers:
1. Category classifier (12 classes)
2. Subcategory classifier (111 classes after filtering)
"""

import numpy as np
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import config
from datetime import datetime
import os

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available. Install with: pip install xgboost")
    print("  Falling back to SGDClassifier")

print("=" * 80)
print("TRANSACTION CATEGORIZATION - MODEL TRAINING")
print("=" * 80)

# ========== Load Embeddings & Labels ==========
print("\n[1/3] Loading embeddings and labels...")
X_train = np.load(config.X_TRAIN_EMB_PATH)
X_val = np.load(config.X_VAL_EMB_PATH)
y_cat_train = np.load(config.Y_CAT_TRAIN_PATH)
y_cat_val = np.load(config.Y_CAT_VAL_PATH)
y_sub_train = np.load(config.Y_SUB_TRAIN_PATH)
y_sub_val = np.load(config.Y_SUB_VAL_PATH)

print(f"✓ Train embeddings: {X_train.shape}")
print(f"✓ Val embeddings: {X_val.shape}")
print(f"✓ Category train labels: {y_cat_train.shape} (unique: {len(np.unique(y_cat_train))})")
print(f"✓ Subcategory train labels: {y_sub_train.shape} (unique: {len(np.unique(y_sub_train))})")

# ========== Train Category Classifier ==========
print("\n[2/3] Training CATEGORY classifier...")

# Choose classifier based on configuration
use_xgboost = config.USE_XGBOOST and XGBOOST_AVAILABLE

if use_xgboost:
    print(f"  Model: XGBClassifier (better for imbalanced data)")
    print(f"  Parameters: {config.XGBOOST_PARAMS}")
    
    cat_clf = XGBClassifier(**config.XGBOOST_PARAMS)
else:
    print(f"  Model: SGDClassifier (memory-efficient online learning)")
    print(f"  Parameters: {config.SGD_PARAMS}")
    
    cat_clf = SGDClassifier(**config.SGD_PARAMS)

print(f"  Classes: {len(np.unique(y_cat_train))}")
print(f"  Training samples: {len(X_train):,}")

start_time = datetime.now()
print(f"\n  Training started at {start_time.strftime('%H:%M:%S')}...")
cat_clf.fit(X_train, y_cat_train)
train_time = (datetime.now() - start_time).total_seconds()
print(f"✓ Training completed in {train_time:.1f} seconds")

# Quick validation
y_cat_pred = cat_clf.predict(X_val)
cat_acc = accuracy_score(y_cat_val, y_cat_pred)
print(f"  Validation accuracy: {cat_acc:.4f} ({cat_acc*100:.2f}%)")

# Save model
joblib.dump(cat_clf, config.CAT_CLASSIFIER_PATH)
print(f"✓ Saved category classifier to {config.CAT_CLASSIFIER_PATH}")

# ========== Train Subcategory Classifier ==========
print("\n[3/3] Training SUBCATEGORY classifier...")

if use_xgboost:
    print(f"  Model: XGBClassifier (better for imbalanced data)")
    print(f"  Parameters: {config.XGBOOST_PARAMS}")
    
    sub_clf = XGBClassifier(**config.XGBOOST_PARAMS)
else:
    print(f"  Model: SGDClassifier (memory-efficient online learning)")
    print(f"  Parameters: {config.SGD_PARAMS}")
    
    sub_clf = SGDClassifier(**config.SGD_PARAMS)

print(f"  Classes: {len(np.unique(y_sub_train))}")
print(f"  Training samples: {len(X_train):,}")

start_time = datetime.now()
print(f"\n  Training started at {start_time.strftime('%H:%M:%S')}...")
sub_clf.fit(X_train, y_sub_train)
train_time = (datetime.now() - start_time).total_seconds()
print(f"✓ Training completed in {train_time:.1f} seconds")

# Quick validation
y_sub_pred = sub_clf.predict(X_val)
sub_acc = accuracy_score(y_sub_val, y_sub_pred)
print(f"  Validation accuracy: {sub_acc:.4f} ({sub_acc*100:.2f}%)")

# Save model
joblib.dump(sub_clf, config.SUB_CLASSIFIER_PATH)
print(f"✓ Saved subcategory classifier to {config.SUB_CLASSIFIER_PATH}")

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\nQuick Results:")
print(f"  • Category accuracy: {cat_acc:.4f} ({cat_acc*100:.2f}%)")
print(f"  • Subcategory accuracy: {sub_acc:.4f} ({sub_acc*100:.2f}%)")
print(f"\nNext step: Run evaluate.py for detailed metrics")
