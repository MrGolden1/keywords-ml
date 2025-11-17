"""Extract interpretable keywords from TF-IDF + LinearSVC models.

This script extracts top weighted terms (keywords) from trained LinearSVC models
to build rule-based categorization systems. For each class, it extracts:
- Including keywords: terms with highest positive weights (strong indicators)
- Excluding keywords: terms with most negative weights (strong anti-indicators)

The output JSON can be used to design manual rules or validate model decisions.
"""

import json
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict
import config


def get_top_terms_for_class(clf, feature_names, class_index, top_n=30):
    """Extract top positive and negative weighted terms for a given class.
    
    Args:
        clf: Trained LinearSVC classifier
        feature_names: Array of feature names from vectorizer
        class_index: Index of the class to extract terms for
        top_n: Number of top terms to extract for each direction
        
    Returns:
        Tuple of (positive_terms, negative_terms), where each is a list of (term, weight) tuples
    """
    coefs = clf.coef_[class_index]
    
    # Top positive terms (strongest indicators)
    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
    pos_terms = feature_names[top_pos_idx]
    pos_weights = coefs[top_pos_idx]
    
    # Top negative terms (strongest anti-indicators)
    top_neg_idx = np.argsort(coefs)[:top_n]
    neg_terms = feature_names[top_neg_idx]
    neg_weights = coefs[top_neg_idx]
    
    return list(zip(pos_terms, pos_weights)), list(zip(neg_terms, neg_weights))


def compute_term_statistics(X_text, vectorizer, y_labels, n_classes):
    """Compute support and dominance statistics for terms across classes.
    
    This helps filter out noisy terms that have low support or are not specific
    to any particular class.
    
    Args:
        X_text: Series of text documents
        vectorizer: Fitted TfidfVectorizer
        y_labels: Array of class labels
        n_classes: Number of classes
        
    Returns:
        Dictionary mapping (class_id, term) -> statistics dict
    """
    print("Computing term statistics for filtering...")
    
    # Transform text to get term-document matrix
    X_tfidf = vectorizer.transform(X_text)
    X_binary = (X_tfidf > 0).astype(int)  # Convert to binary presence/absence
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    stats = {}
    
    for class_id in range(n_classes):
        # Get documents in this class
        class_mask = (y_labels == class_id)
        n_class = class_mask.sum()
        
        if n_class == 0:
            continue
        
        # Binary matrix for this class
        X_class_binary = X_binary[class_mask]
        
        # Support in this class
        support_in_class = np.array(X_class_binary.sum(axis=0)).flatten()
        
        # Support in all documents
        support_total = np.array(X_binary.sum(axis=0)).flatten()
        
        for term_idx, term in enumerate(feature_names):
            sup_class = support_in_class[term_idx]
            sup_total = support_total[term_idx]
            
            if sup_total > 0:
                dominance = sup_class / sup_total
                p_in_class = sup_class / n_class
                
                stats[(class_id, term)] = {
                    'support_class': int(sup_class),
                    'support_total': int(sup_total),
                    'n_class': int(n_class),
                    'dominance': float(dominance),
                    'p_in_class': float(p_in_class),
                }
    
    return stats


def filter_terms(terms, class_id, stats, min_support_class=3, min_dominance=0.05):
    """Filter terms based on support and dominance statistics.
    
    Keeps terms that are either:
    1. Highly specific to the class (high dominance)
    2. Reasonably common in the class (decent support)
    
    Args:
        terms: List of (term, weight) tuples
        class_id: The class index
        stats: Statistics dictionary from compute_term_statistics
        min_support_class: Minimum number of times term must appear in class
        min_dominance: Minimum dominance ratio (class_support / total_support)
        
    Returns:
        Filtered list of (term, weight) tuples
    """
    filtered = []
    
    for term, weight in terms:
        key = (class_id, term)
        
        if key not in stats:
            # Term not found in stats (shouldn't happen, but skip if it does)
            continue
        
        s = stats[key]
        
        # Keep if:
        # 1. Has minimum support in class AND reasonable dominance
        # 2. OR has very high dominance (even if rare) - important for specific IDs
        if s['support_class'] >= min_support_class and s['dominance'] >= min_dominance:
            filtered.append((term, weight, s))
        elif s['dominance'] >= 0.5:  # Very class-specific
            filtered.append((term, weight, s))
    
    return filtered


def build_keywords_for_classifier(
    clf,
    vectorizer,
    label_encoder,
    X_text,
    y_labels,
    top_n=40,
    apply_filtering=True,
    classifier_type="classifier"
):
    """Build keyword dictionary for a classifier.
    
    Args:
        clf: Trained LinearSVC classifier
        vectorizer: Fitted TfidfVectorizer
        label_encoder: LabelEncoder for class names
        X_text: Training text (for statistics)
        y_labels: Training labels (for statistics)
        top_n: Number of top terms to extract
        apply_filtering: Whether to apply statistical filtering
        classifier_type: Name of the classifier for logging
        
    Returns:
        Dictionary mapping class_name -> {include: [...], exclude: [...]}
    """
    print(f"\nExtracting keywords for {classifier_type}...")
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    n_classes = len(label_encoder.classes_)
    
    # Compute statistics if filtering is enabled
    stats = None
    if apply_filtering:
        stats = compute_term_statistics(X_text, vectorizer, y_labels, n_classes)
    
    all_keywords = {}
    
    for class_id in range(n_classes):
        class_name = label_encoder.inverse_transform([class_id])[0]
        
        # Get raw top terms
        pos_terms, neg_terms = get_top_terms_for_class(
            clf, feature_names, class_id, top_n=top_n
        )
        
        # Apply filtering if enabled
        if apply_filtering and stats:
            pos_filtered = filter_terms(pos_terms, class_id, stats)
            neg_filtered = filter_terms(neg_terms, class_id, stats)
            
            # Format with statistics
            include_list = [
                {
                    "term": t,
                    "weight": float(w),
                    "support_class": s['support_class'],
                    "dominance": round(s['dominance'], 3),
                }
                for t, w, s in pos_filtered
            ]
            
            exclude_list = [
                {
                    "term": t,
                    "weight": float(w),
                    "support_class": s['support_class'],
                    "dominance": round(s['dominance'], 3),
                }
                for t, w, s in neg_filtered
            ]
        else:
            # No filtering - just return top terms by weight
            include_list = [
                {"term": t, "weight": float(w)}
                for t, w in pos_terms
            ]
            
            exclude_list = [
                {"term": t, "weight": float(w)}
                for t, w in neg_terms
            ]
        
        all_keywords[class_name] = {
            "include": include_list,
            "exclude": exclude_list,
        }
        
        print(f"  {class_name}: {len(include_list)} include, {len(exclude_list)} exclude keywords")
    
    return all_keywords


def main():
    """Main keyword extraction pipeline."""
    print("="*60)
    print("TF-IDF Keyword Extraction Pipeline")
    print("="*60)
    
    # Load artifacts
    print("\nLoading artifacts...")
    vectorizer = joblib.load(config.TFIDF_VECTORIZER_PATH)
    cat_clf = joblib.load(config.CAT_TFIDF_SVC_PATH)
    sub_clf = joblib.load(config.SUB_TFIDF_SVC_PATH)
    cat_encoder = joblib.load(config.CAT_ENCODER_PATH)
    sub_encoder = joblib.load(config.SUBCAT_ENCODER_PATH)
    
    print(f"✓ Loaded vectorizer with {len(vectorizer.vocabulary_):,} features")
    print(f"✓ Loaded category classifier ({len(cat_encoder.classes_)} classes)")
    print(f"✓ Loaded subcategory classifier ({len(sub_encoder.classes_)} classes)")
    
    # Load training data for statistics
    print("\nLoading training data for statistics...")
    df_train = pd.read_parquet(config.DATA_TRAIN)
    X_train_text = df_train[config.TEXT_COLUMN].astype(str)
    y_cat_train = df_train["category_id"].to_numpy()
    y_sub_train = df_train["subcategory_id"].to_numpy()
    
    # Extract keywords for category classifier
    cat_keywords = build_keywords_for_classifier(
        cat_clf,
        vectorizer,
        cat_encoder,
        X_train_text,
        y_cat_train,
        top_n=40,
        apply_filtering=True,
        classifier_type="Category Classifier"
    )
    
    # Extract keywords for subcategory classifier
    sub_keywords = build_keywords_for_classifier(
        sub_clf,
        vectorizer,
        sub_encoder,
        X_train_text,
        y_sub_train,
        top_n=40,
        apply_filtering=True,
        classifier_type="Subcategory Classifier"
    )
    
    # Build final output
    output = {
        "metadata": {
            "description": "TF-IDF keywords extracted from LinearSVC models",
            "vocabulary_size": len(vectorizer.vocabulary_),
            "n_categories": len(cat_encoder.classes_),
            "n_subcategories": len(sub_encoder.classes_),
            "top_n_per_class": 40,
            "filtering": "applied (min_support_class=3, min_dominance=0.05)",
        },
        "category": cat_keywords,
        "subcategory": sub_keywords,
    }
    
    # Save to JSON
    print(f"\nSaving keywords to {config.TFIDF_KEYWORDS_JSON}...")
    with open(config.TFIDF_KEYWORDS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Keywords saved successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("Extraction Complete - Summary")
    print("="*60)
    
    total_cat_include = sum(len(v['include']) for v in cat_keywords.values())
    total_cat_exclude = sum(len(v['exclude']) for v in cat_keywords.values())
    total_sub_include = sum(len(v['include']) for v in sub_keywords.values())
    total_sub_exclude = sum(len(v['exclude']) for v in sub_keywords.values())
    
    print(f"\nCategory Keywords:")
    print(f"  Classes: {len(cat_keywords)}")
    print(f"  Total including keywords: {total_cat_include}")
    print(f"  Total excluding keywords: {total_cat_exclude}")
    
    print(f"\nSubcategory Keywords:")
    print(f"  Classes: {len(sub_keywords)}")
    print(f"  Total including keywords: {total_sub_include}")
    print(f"  Total excluding keywords: {total_sub_exclude}")
    
    print(f"\n✓ Output saved to: {config.TFIDF_KEYWORDS_JSON}")
    print("\nUse these keywords to build rule-based categorization systems!")


if __name__ == "__main__":
    main()
