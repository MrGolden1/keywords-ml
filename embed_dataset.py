"""
Generate embeddings for transaction descriptions using sentence-transformers.

Uses BAAI/bge-small-en-v1.5 model to convert text to 384-dimensional embeddings.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import config
import torch

print("=" * 80)
print("TRANSACTION CATEGORIZATION - EMBEDDING GENERATION")
print("=" * 80)

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
    print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = "cpu"
    print("\n⚠ GPU not available, using CPU (this will be slower)")

# ========== Load Model ==========
print(f"\n[1/4] Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
print(f"✓ Model loaded on {device}")
print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

# ========== Define Embedding Function ==========
def embed_texts(texts, batch_size=256, normalize=True):
    """Embed texts in batches."""
    all_embs = []
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size].tolist()
        embs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        all_embs.append(embs)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i:,}/{total:,} texts ({i/total*100:.1f}%)")
    
    return np.vstack(all_embs)

# ========== Load Train Data ==========
print(f"\n[2/4] Loading training data from {config.DATA_TRAIN}...")
train_df = pd.read_parquet(config.DATA_TRAIN)
print(f"Train samples: {len(train_df):,}")

print(f"\n  Embedding training texts...")
X_train = embed_texts(train_df[config.TEXT_COLUMN], batch_size=config.EMB_BATCH_SIZE)
y_cat_train = train_df["category_id"].values
y_sub_train = train_df["subcategory_id"].values

print(f"✓ Train embeddings shape: {X_train.shape}")
print(f"  Category labels shape: {y_cat_train.shape}")
print(f"  Subcategory labels shape: {y_sub_train.shape}")

# ========== Load Val Data ==========
print(f"\n[3/4] Loading validation data from {config.DATA_VAL}...")
val_df = pd.read_parquet(config.DATA_VAL)
print(f"Val samples: {len(val_df):,}")

print(f"\n  Embedding validation texts...")
X_val = embed_texts(val_df[config.TEXT_COLUMN], batch_size=config.EMB_BATCH_SIZE)
y_cat_val = val_df["category_id"].values
y_sub_val = val_df["subcategory_id"].values

print(f"✓ Val embeddings shape: {X_val.shape}")
print(f"  Category labels shape: {y_cat_val.shape}")
print(f"  Subcategory labels shape: {y_sub_val.shape}")

# ========== Save Embeddings ==========
print(f"\n[4/4] Saving embeddings and labels...")
np.save(config.X_TRAIN_EMB_PATH, X_train)
np.save(config.X_VAL_EMB_PATH, X_val)
np.save(config.Y_CAT_TRAIN_PATH, y_cat_train)
np.save(config.Y_CAT_VAL_PATH, y_cat_val)
np.save(config.Y_SUB_TRAIN_PATH, y_sub_train)
np.save(config.Y_SUB_VAL_PATH, y_sub_val)

print(f"✓ Saved train embeddings to {config.X_TRAIN_EMB_PATH}")
print(f"✓ Saved val embeddings to {config.X_VAL_EMB_PATH}")
print(f"✓ Saved category labels to {config.Y_CAT_TRAIN_PATH} and {config.Y_CAT_VAL_PATH}")
print(f"✓ Saved subcategory labels to {config.Y_SUB_TRAIN_PATH} and {config.Y_SUB_VAL_PATH}")

# Memory info
total_size_mb = (X_train.nbytes + X_val.nbytes) / (1024**2)
print(f"\n  Total embeddings size: {total_size_mb:.1f} MB")

print("\n" + "=" * 80)
print("EMBEDDING GENERATION COMPLETE!")
print("=" * 80)
print(f"\nNext step: Run train_models.py to train classifiers")
