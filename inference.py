"""
Inference pipeline for transaction categorization with type feature support.

Loads trained models and provides a classify_transaction() function
that predicts category and subcategory for a single transaction description.

Uses hierarchical masking to ensure predicted subcategory is valid for
the predicted category.
"""

from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import config

def encode_transaction_type(transaction_type):
    """
    Encode transaction type as numerical feature.
    credit -> +1.0, debit -> -1.0, unknown -> 0.0
    """
    if not transaction_type or transaction_type is None:
        return 0.0
    transaction_type = str(transaction_type).lower().strip()
    if 'credit' in transaction_type:
        return 1.0
    elif 'debit' in transaction_type:
        return -1.0
    else:
        return 0.0

class TransactionClassifier:
    """Production-ready transaction categorization classifier with type feature."""
    
    def __init__(self):
        """Load all models and encoders at initialization."""
        print("Loading transaction classifier...")
        
        # Load embedding model
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        # Load classifiers
        self.cat_clf = joblib.load(config.CAT_CLASSIFIER_PATH)
        self.sub_clf = joblib.load(config.SUB_CLASSIFIER_PATH)
        
        # Load encoders
        self.cat_encoder = joblib.load(config.CAT_ENCODER_PATH)
        self.subcat_encoder = joblib.load(config.SUBCAT_ENCODER_PATH)
        
        # Load category -> subcategory mapping
        self.cat_to_sub = joblib.load(config.CAT_TO_SUB_MAPPING_PATH)
        
        print("✓ Classifier loaded successfully")
        print(f"  Categories: {len(self.cat_encoder.classes_)}")
        print(f"  Subcategories: {len(self.subcat_encoder.classes_)}")
    
    def classify_transaction(self, text: str, transaction_type: str = None, return_probabilities=False):
        """
        Classify a single transaction description.
        
        Args:
            text: Transaction description (clean_description)
            transaction_type: 'debit', 'credit', or None
            return_probabilities: If True, include top-k predictions with probabilities
            
        Returns:
            Dictionary with predicted category, subcategory, and confidence scores
        """
        # 1) Embed the text
        emb = self.model.encode([text], normalize_embeddings=True)
        
        # 2) Add transaction type feature
        type_feature = encode_transaction_type(transaction_type)
        emb = np.hstack([emb, [[type_feature]]])  # Shape: (1, 385)
        
        # 3) Category prediction
        cat_proba = self.cat_clf.predict_proba(emb)[0]
        cat_id = int(np.argmax(cat_proba))
        cat_confidence = float(cat_proba[cat_id])
        
        # 4) Subcategory prediction with hierarchical masking
        sub_proba = self.sub_clf.predict_proba(emb)[0]
        
        # Mask by allowed subcategories for the predicted category
        allowed_sub_ids = self.cat_to_sub.get(cat_id, None)
        if allowed_sub_ids is not None:
            mask = np.zeros_like(sub_proba, dtype=bool)
            mask[list(allowed_sub_ids)] = True
            masked_proba = np.where(mask, sub_proba, 0.0)
        else:
            masked_proba = sub_proba
        
        sub_id = int(np.argmax(masked_proba))
        sub_confidence = float(masked_proba[sub_id])
        
        # 5) Decode labels
        category = self.cat_encoder.inverse_transform([cat_id])[0]
        subcategory = self.subcat_encoder.inverse_transform([sub_id])[0]
        
        result = {
            "category": category,
            "subcategory": subcategory,
            "category_confidence": cat_confidence,
            "subcategory_confidence": sub_confidence,
        }
        
        # Optional: include top-k predictions
        if return_probabilities:
            # Top 3 categories
            top_cat_ids = np.argsort(cat_proba)[-3:][::-1]
            result["top_categories"] = [
                {
                    "category": self.cat_encoder.inverse_transform([i])[0],
                    "confidence": float(cat_proba[i])
                }
                for i in top_cat_ids
            ]
            
            # Top 3 masked subcategories
            top_sub_ids = np.argsort(masked_proba)[-3:][::-1]
            result["top_subcategories"] = [
                {
                    "subcategory": self.subcat_encoder.inverse_transform([i])[0],
                    "confidence": float(masked_proba[i])
                }
                for i in top_sub_ids if masked_proba[i] > 0
            ]
        
        return result
    
    def batch_classify(self, texts, transaction_types=None, batch_size=256):
        """
        Classify multiple transactions efficiently.
        
        Args:
            texts: List of transaction descriptions
            transaction_types: List of transaction types ('debit'/'credit') or None
            batch_size: Batch size for embedding
            
        Returns:
            List of prediction dictionaries
        """
        # Embed all texts at once
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Add type features
        if transaction_types is None:
            transaction_types = [None] * len(texts)
        
        type_features = np.array([encode_transaction_type(t) for t in transaction_types]).reshape(-1, 1)
        embeddings = np.hstack([embeddings, type_features])
        
        # Get predictions
        cat_proba = self.cat_clf.predict_proba(embeddings)
        sub_proba = self.sub_clf.predict_proba(embeddings)
        
        results = []
        for i in range(len(texts)):
            # Category
            cat_id = int(np.argmax(cat_proba[i]))
            cat_confidence = float(cat_proba[i][cat_id])
            
            # Masked subcategory
            allowed_sub_ids = self.cat_to_sub.get(cat_id, None)
            if allowed_sub_ids is not None:
                mask = np.zeros_like(sub_proba[i], dtype=bool)
                mask[list(allowed_sub_ids)] = True
                masked_proba = np.where(mask, sub_proba[i], 0.0)
            else:
                masked_proba = sub_proba[i]
            
            sub_id = int(np.argmax(masked_proba))
            sub_confidence = float(masked_proba[sub_id])
            
            results.append({
                "category": self.cat_encoder.inverse_transform([cat_id])[0],
                "subcategory": self.subcat_encoder.inverse_transform([sub_id])[0],
                "category_confidence": cat_confidence,
                "subcategory_confidence": sub_confidence,
            })
        
        return results


# ========== Example Usage ==========
if __name__ == "__main__":
    print("=" * 80)
    print("TRANSACTION CATEGORIZATION - INFERENCE DEMO")
    print("=" * 80)
    
    # Initialize classifier
    classifier = TransactionClassifier()
    
    # Example transactions with types
    test_transactions = [
        ("WALMART SUPERCENTER", "debit"),
        ("SHELL GAS STATION", "debit"),
        ("NETFLIX SUBSCRIPTION", "debit"),
        ("SALARY DEPOSIT COMPANY XYZ", "credit"),
        ("ATM WITHDRAWAL", "debit"),
        ("UBER TRIP", "debit"),
        ("CVS PHARMACY", "debit"),
        ("RENT PAYMENT", "debit"),
        ("CHASE CREDIT CARD PAYMENT", "debit"),
        ("IRS TAX REFUND", "credit"),
    ]
    
    print("\n" + "=" * 80)
    print("SINGLE TRANSACTION CLASSIFICATION (with transaction type)")
    print("=" * 80)
    
    for i, (text, txn_type) in enumerate(test_transactions[:3], 1):
        print(f"\n[{i}] Transaction: {text} ({txn_type})")
        result = classifier.classify_transaction(text, txn_type, return_probabilities=True)
        
        print(f"  ➜ Category: {result['category']}")
        print(f"    Confidence: {result['category_confidence']:.4f}")
        print(f"  ➜ Subcategory: {result['subcategory']}")
        print(f"    Confidence: {result['subcategory_confidence']:.4f}")
        
        if 'top_categories' in result:
            print(f"\n  Top 3 Categories:")
            for cat in result['top_categories']:
                print(f"    - {cat['category']:30s} {cat['confidence']:.4f}")
            
            print(f"\n  Top 3 Subcategories (masked):")
            for sub in result['top_subcategories']:
                print(f"    - {sub['subcategory']:30s} {sub['confidence']:.4f}")
    
    print("\n" + "=" * 80)
    print("BATCH CLASSIFICATION")
    print("=" * 80)
    
    texts = [t[0] for t in test_transactions]
    types = [t[1] for t in test_transactions]
    
    print(f"\nClassifying {len(test_transactions)} transactions...")
    results = classifier.batch_classify(texts, types)
    
    print("\nResults:")
    for (text, txn_type), result in zip(test_transactions, results):
        print(f"\n  {text:40s} [{txn_type}]")
        print(f"    → {result['category']:20s} | {result['subcategory']:30s}")
        print(f"      Conf: {result['category_confidence']:.3f} / {result['subcategory_confidence']:.3f}")
    
    print("\n" + "=" * 80)
