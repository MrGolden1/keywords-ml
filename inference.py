"""
Inference pipeline for transaction categorization.

Loads trained models and provides a classify_transaction() function
that predicts category and subcategory for a single transaction description.

Uses hierarchical masking to ensure predicted subcategory is valid for
the predicted category.
"""

from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import config

class TransactionClassifier:
    """Production-ready transaction categorization classifier."""
    
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
    
    def classify_transaction(self, text: str, return_probabilities=False):
        """
        Classify a single transaction description.
        
        Args:
            text: Transaction description (clean_description)
            return_probabilities: If True, include top-k predictions with probabilities
            
        Returns:
            Dictionary with predicted category, subcategory, and confidence scores
        """
        # 1) Embed the text
        emb = self.model.encode([text], normalize_embeddings=True)
        
        # 2) Category prediction
        cat_proba = self.cat_clf.predict_proba(emb)[0]
        cat_id = int(np.argmax(cat_proba))
        cat_confidence = float(cat_proba[cat_id])
        
        # 3) Subcategory prediction with hierarchical masking
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
        
        # 4) Decode labels
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
    
    def batch_classify(self, texts, batch_size=256):
        """
        Classify multiple transactions efficiently.
        
        Args:
            texts: List of transaction descriptions
            batch_size: Batch size for embedding
            
        Returns:
            List of prediction dictionaries
        """
      