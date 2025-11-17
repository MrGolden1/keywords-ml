"""Train TF-IDF + LinearSVC models for interpretable keyword extraction.

This script trains separate category and subcategory classifiers using
TF-IDF features instead of embeddings. The linear models allow extraction
of interpretable keywords for rule-based categorization systems.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import config


def load_data():
    """Load training and validation data."""
    print("Loading data...")
    df_train = pd.read_parquet(config.DATA_TRAIN)
    df_val = pd.read_parquet(config.DATA_VAL)
    
    # Extract text and labels
    X_train_text = df_train[config.TEXT_COLUMN].astype(str)
    X_val_text = df_val[config.TEXT_COLUMN].astype(str)
    
    y_cat_train = df_train["category_id"].to_numpy()
    y_cat_val = df_val["category_id"].to_numpy()
    
    y_sub_train = df_train["subcategory_id"].to_numpy()
    y_sub_val = df_val["subcategory_id"].to_numpy()
    
    print(f"Train samples: {len(X_train_text)}")
    print(f"Val samples: {len(X_val_text)}")
    print(f"Categories: {len(np.unique(y_cat_train))}")
    print(f"Subcategories: {len(np.unique(y_sub_train))}")
    
    return X_train_text, X_val_text, y_cat_train, y_cat_val, y_sub_train, y_sub_val


def build_vectorizer():
    """Create TF-IDF vectorizer with optimized settings for transaction text."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),  # unigrams + bigrams
        min_df=10,  # must appear in at least 10 documents (reduces noise)
        max_features=200_000,  # limit vocabulary to 200k most important features
        sublinear_tf=True,  # use 1 + log(tf) instead of raw tf
        token_pattern=r"(?u)\b\w+\b",  # keep pure numeric tokens
        strip_accents="unicode",
        lowercase=True,
    )


def train_sgd_classifier(X_train, y_train, class_name="classifier"):
    """Train SGDClassifier with balanced class weights (10-20x faster than LinearSVC)."""
    print(f"\nTraining {class_name}...")
    
    # Compute class weights to handle imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Train SGDClassifier (much faster, uses all CPU cores)
    clf = SGDClassifier(
        loss="hinge",  # equivalent to LinearSVC
        alpha=0.0001,  # L2 regularization (similar to C=1.0 in LinearSVC)
        class_weight=class_weight_dict,
        max_iter=1000,
        early_stopping=True,  # stop when validation score doesn't improve
        validation_fraction=0.1,  # use 10% of training data for early stopping
        n_iter_no_change=5,  # stop if no improvement for 5 iterations
        random_state=config.RANDOM_STATE,
        n_jobs=-1,  # use all CPU cores
        verbose=1,  # show progress
    )
    
    clf.fit(X_train, y_train)
    print(f"✓ {class_name} trained successfully")
    
    return clf


def evaluate_classifier(clf, X_val, y_val, label_encoder, classifier_name="Classifier"):
    """Evaluate classifier and print detailed metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {classifier_name}")
    print(f"{'='*60}")
    
    y_pred = clf.predict(X_val)
    
    # Overall metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision_macro = precision_score(y_val, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_val, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision_macro:.4f} (macro)")
    print(f"  Recall:    {recall_macro:.4f} (macro)")
    print(f"  F1-Score:  {f1_macro:.4f} (macro)")
    
    # Per-class report
    print(f"\nPer-Class Report:")
    target_names = label_encoder.classes_
    report = classification_report(
        y_val,
        y_pred,
        target_names=target_names,
        zero_division=0,
        digits=3,
    )
    print(report)
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


def main():
    """Main training pipeline."""
    print("="*60)
    print("TF-IDF + LinearSVC Training Pipeline")
    print("="*60)
    
    # Load data
    X_train_text, X_val_text, y_cat_train, y_cat_val, y_sub_train, y_sub_val = load_data()
    
    # Build and fit vectorizer
    print("\nBuilding TF-IDF vectorizer...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)
    
    print(f"✓ Vectorizer fitted")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"  Train matrix shape: {X_train_tfidf.shape}")
    print(f"  Val matrix shape: {X_val_tfidf.shape}")
    
    # Save vectorizer
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, config.TFIDF_VECTORIZER_PATH)
    print(f"✓ Vectorizer saved to {config.TFIDF_VECTORIZER_PATH}")
    
    # Load label encoders
    cat_encoder = joblib.load(config.CAT_ENCODER_PATH)
    sub_encoder = joblib.load(config.SUBCAT_ENCODER_PATH)
    
    # Train category classifier
    cat_clf = train_sgd_classifier(X_train_tfidf, y_cat_train, "Category Classifier")
    joblib.dump(cat_clf, config.CAT_TFIDF_SVC_PATH)
    print(f"✓ Category classifier saved to {config.CAT_TFIDF_SVC_PATH}")
    
    # Evaluate category classifier
    cat_metrics = evaluate_classifier(
        cat_clf,
        X_val_tfidf,
        y_cat_val,
        cat_encoder,
        classifier_name="Category Classifier"
    )
    
    # Train subcategory classifier
    sub_clf = train_sgd_classifier(X_train_tfidf, y_sub_train, "Subcategory Classifier")
    joblib.dump(sub_clf, config.SUB_TFIDF_SVC_PATH)
    print(f"✓ Subcategory classifier saved to {config.SUB_TFIDF_SVC_PATH}")
    
    # Evaluate subcategory classifier
    sub_metrics = evaluate_classifier(
        sub_clf,
        X_val_tfidf,
        y_sub_val,
        sub_encoder,
        classifier_name="Subcategory Classifier"
    )
    
    # Summary
    print("\n" + "="*60)
    print("Training Complete - Summary")
    print("="*60)
    print(f"\nCategory Classifier:")
    print(f"  Accuracy: {cat_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {cat_metrics['f1_macro']:.4f} (macro)")
    
    print(f"\nSubcategory Classifier:")
    print(f"  Accuracy: {sub_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {sub_metrics['f1_macro']:.4f} (macro)")
    
    print(f"\nArtifacts saved to: {config.ARTIFACTS_DIR}")
    print("✓ Ready for keyword extraction!")


if __name__ == "__main__":
    main()
