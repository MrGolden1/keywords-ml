"""
Evaluate trained classifiers with detailed metrics.

Includes:
- Category & subcategory accuracy
- Hierarchical/joint accuracy (both must be correct)
- Per-class metrics
- Confusion matrix analysis
"""

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict
import config

print("=" * 80)
print("TRANSACTION CATEGORIZATION - MODEL EVALUATION")
print("=" * 80)

# ========== Load Everything ==========
print("\n[1/4] Loading models, encoders, and data...")
X_val = np.load(config.X_VAL_EMB_PATH)
y_cat_val = np.load(config.Y_CAT_VAL_PATH)
y_sub_val = np.load(config.Y_SUB_VAL_PATH)

cat_clf = joblib.load(config.CAT_CLASSIFIER_PATH)
sub_clf = joblib.load(config.SUB_CLASSIFIER_PATH)
cat_encoder = joblib.load(config.CAT_ENCODER_PATH)
subcat_encoder = joblib.load(config.SUBCAT_ENCODER_PATH)
cat_to_sub = joblib.load(config.CAT_TO_SUB_MAPPING_PATH)

print(f"✓ Loaded validation data: {X_val.shape}")
print(f"✓ Loaded models and encoders")

# ========== Basic Predictions ==========
print("\n[2/4] Making predictions...")
y_cat_pred = cat_clf.predict(X_val)
y_sub_pred = sub_clf.predict(X_val)

# Get probabilities for confidence analysis
y_cat_proba = cat_clf.predict_proba(X_val)
y_sub_proba = sub_clf.predict_proba(X_val)

print("✓ Predictions complete")

# ========== Basic Metrics ==========
print("\n[3/4] Computing metrics...")
print("\n" + "=" * 80)
print("CATEGORY CLASSIFICATION")
print("=" * 80)

cat_acc = accuracy_score(y_cat_val, y_cat_pred)
cat_f1_macro = f1_score(y_cat_val, y_cat_pred, average='macro')
cat_f1_weighted = f1_score(y_cat_val, y_cat_pred, average='weighted')

print(f"\nAccuracy: {cat_acc:.4f} ({cat_acc*100:.2f}%)")
print(f"Macro F1: {cat_f1_macro:.4f}")
print(f"Weighted F1: {cat_f1_weighted:.4f}")

print("\n=== Per-Category Metrics ===")
cat_report = classification_report(
    y_cat_val, 
    y_cat_pred, 
    target_names=cat_encoder.classes_,
    digits=4,
    zero_division=0
)
print(cat_report)

print("\n" + "=" * 80)
print("SUBCATEGORY CLASSIFICATION")
print("=" * 80)

sub_acc = accuracy_score(y_sub_val, y_sub_pred)
sub_f1_macro = f1_score(y_sub_val, y_sub_pred, average='macro')
sub_f1_weighted = f1_score(y_sub_val, y_sub_pred, average='weighted')

print(f"\nAccuracy: {sub_acc:.4f} ({sub_acc*100:.2f}%)")
print(f"Macro F1: {sub_f1_macro:.4f}")
print(f"Weighted F1: {sub_f1_weighted:.4f}")

# Show top and bottom performing subcategories
print("\n=== Top 10 Subcategories (by F1 score) ===")
sub_report_dict = classification_report(
    y_sub_val, 
    y_sub_pred, 
    target_names=subcat_encoder.classes_,
    output_dict=True,
    zero_division=0
)

# Extract f1 scores
subcat_f1 = {k: v['f1-score'] for k, v in sub_report_dict.items() 
             if k not in ['accuracy', 'macro avg', 'weighted avg']}
sorted_subcats = sorted(subcat_f1.items(), key=lambda x: x[1], reverse=True)

for subcat, f1 in sorted_subcats[:10]:
    support = sub_report_dict[subcat]['support']
    print(f"  {subcat:40s}: F1={f1:.4f} (n={int(support):,})")

print("\n=== Bottom 10 Subcategories (by F1 score) ===")
for subcat, f1 in sorted_subcats[-10:]:
    support = sub_report_dict[subcat]['support']
    print(f"  {subcat:40s}: F1={f1:.4f} (n={int(support):,})")

# ========== Hierarchical Evaluation ==========
print("\n[4/4] Computing hierarchical metrics...")
print("\n" + "=" * 80)
print("HIERARCHICAL (JOINT) ACCURACY")
print("=" * 80)

# Both category and subcategory must be correct
joint_correct = (y_cat_pred == y_cat_val) & (y_sub_pred == y_sub_val)
joint_acc = joint_correct.mean()

print(f"\nJoint Accuracy: {joint_acc:.4f} ({joint_acc*100:.2f}%)")
print(f"  (Both category AND subcategory must be correct)")

# Hierarchical accuracy with masking
print("\n=== Hierarchical Prediction with Masking ===")
hierarchical_correct = 0

for i in range(len(X_val)):
    # Get predicted category
    pred_cat = y_cat_pred[i]
    
    # Get allowed subcategories for this category
    allowed_subs = set(cat_to_sub.get(pred_cat, []))
    
    # Mask subcategory probabilities
    sub_probs = y_sub_proba[i].copy()
    if allowed_subs:
        mask = np.array([j in allowed_subs for j in range(len(sub_probs))])
        sub_probs = np.where(mask, sub_probs, 0.0)
    
    # Get masked prediction
    masked_sub_pred = np.argmax(sub_probs)
    
    # Check if both are correct
    if pred_cat == y_cat_val[i] and masked_sub_pred == y_sub_val[i]:
        hierarchical_correct += 1

hierarchical_acc = hierarchical_correct / len(X_val)
print(f"Hierarchical Accuracy (with masking): {hierarchical_acc:.4f} ({hierarchical_acc*100:.2f}%)")

# ========== Confidence Analysis ==========
print("\n" + "=" * 80)
print("CONFIDENCE ANALYSIS")
print("=" * 80)

cat_confidences = np.max(y_cat_proba, axis=1)
sub_confidences = np.max(y_sub_proba, axis=1)

print(f"\nCategory Prediction Confidence:")
print(f"  Mean: {cat_confidences.mean():.4f}")
print(f"  Median: {np.median(cat_confidences):.4f}")
print(f"  Min: {cat_confidences.min():.4f}")
print(f"  Max: {cat_confidences.max():.4f}")

print(f"\nSubcategory Prediction Confidence:")
print(f"  Mean: {sub_confidences.mean():.4f}")
print(f"  Median: {np.median(sub_confidences):.4f}")
print(f"  Min: {sub_confidences.min():.4f}")
print(f"  Max: {sub_confidences.max():.4f}")

# Low confidence predictions
low_conf_threshold = 0.5
low_cat_conf = (cat_confidences < low_conf_threshold).sum()
low_sub_conf = (sub_confidences < low_conf_threshold).sum()

print(f"\nLow confidence predictions (< {low_conf_threshold}):")
print(f"  Category: {low_cat_conf:,} ({low_cat_conf/len(X_val)*100:.2f}%)")
print(f"  Subcategory: {low_sub_conf:,} ({low_sub_conf/len(X_val)*100:.2f}%)")

# ========== Summary ==========
print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)
print(f"\nValidation Set Size: {len(X_val):,} samples")
print(f"\nAccuracy Metrics:")
print(f"  • Category accuracy:              {cat_acc:.4f} ({cat_acc*100:.2f}%)")
print(f"  • Subcategory accuracy:           {sub_acc:.4f} ({sub_acc*100:.2f}%)")
print(f"  • Joint accuracy (both correct):  {joint_acc:.4f} ({joint_acc*100:.2f}%)")
print(f"  • Hierarchical accuracy (masked): {hierarchical_acc:.4f} ({hierarchical_acc*100:.2f}%)")

print(f"\nF1 Scores:")
print(f"  • Category (macro):               {cat_f1_macro:.4f}")
print(f"  • Category (weighted):            {cat_f1_weighted:.4f}")
print(f"  • Subcategory (macro):            {sub_f1_macro:.4f}")
print(f"  • Subcategory (weighted):         {sub_f1_weighted:.4f}")

print("\n" + "=" * 80)
