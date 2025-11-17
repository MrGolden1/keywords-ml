"""
Generate embeddings for transaction descriptions using sentence-transformers.

Uses BAAI/bge-small-en-v1.5 model to convert text to 384-dimensional embeddings.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import config
import torch

def encode_transaction_type(transaction_type):
    """
    Encode transaction type as numerical feature.
    credit -> +1.0
    debit -> -1.0
    unknown/other -> 0.0
    """
    if pd.isna(transaction_type):
        return 0.0
    transaction_type = str(transaction_type).lower().strip()
    if 'credit' in transaction_type:
        return 1.0
    elif 'debit' in transaction_type:
        return -1.0
    else:
        return 0.0

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
def embed_texts_chunked(texts, types_series, batch_size=256, chunk_size=50000, normalize=True):
    """
    Embed texts in batches and save in chunks to avoid OOM.
    Processes embeddings and type features together in chunks.
    
    Args:
        texts: pandas Series of text to embed
        types_series: pandas Series of transaction types
        batch_size: batch size for model.encode()
        chunk_size: number of samples to process before yielding
        normalize: whether to normalize embeddings
        
    Yields:
        (embeddings_chunk, indices) tuples where embeddings_chunk is (chunk_size, 385)
    """
    total = len(texts)
    
    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_texts = texts.iloc[chunk_start:chunk_end]
        chunk_types = types_series.iloc[chunk_start:chunk_end]
        
        print(f"  Processing chunk {chunk_start:,} to {chunk_end:,} ({chunk_end/total*100:.1f}%)")
        
        # Embed text in batches within this chunk
        chunk_embs = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts.iloc[i:i + batch_size].tolist()
            embs = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            chunk_embs.append(embs)
        
        # Stack embeddings for this chunk
        chunk_text_embs = np.vstack(chunk_embs).astype(np.float32)
        
        # Add type features for this chunk
        type_features = chunk_types.apply(encode_transaction_type).values.astype(np.float32).reshape(-1, 1)
        chunk_full_embs = np.hstack([chunk_text_embs, type_features])
        
        print(f"    Chunk shape: {chunk_full_embs.shape}, size: {chunk_full_embs.nbytes / (1024**2):.1f} MB")
        
        yield chunk_full_embs, (chunk_start, chunk_end)

# ========== Load Train Data ==========
print(f"\n[2/4] Loading training data from {config.DATA_TRAIN}...")
train_df = pd.read_parquet(config.DATA_TRAIN)
n_train = len(train_df)
print(f"Train samples: {n_train:,}")

print(f"\n  Embedding training texts in chunks...")
# Pre-allocate array for all training embeddings
X_train = np.zeros((n_train, 385), dtype=np.float32)

for chunk_embs, (start_idx, end_idx) in embed_texts_chunked(
    train_df[config.TEXT_COLUMN], 
    train_df[config.TYPE_COLUMN],
    batch_size=config.EMB_BATCH_SIZE,
    chunk_size=50000  # Process 50K samples at a time
):
    X_train[start_idx:end_idx] = chunk_embs
    # Free memory
    del chunk_embs

y_cat_train = train_df["category_id"].values
y_sub_train = train_df["subcategory_id"].values

print(f"✓ Train embeddings shape: {X_train.shape}")
print(f"  Text embedding: 384-dim + Type feature: 1-dim = 385-dim total")
print(f"  Category labels shape: {y_cat_train.shape}")
print(f"  Subcategory labels shape: {y_sub_train.shape}")

# ========== Load Val Data ==========
print(f"\n[3/4] Loading validation data from {config.DATA_VAL}...")
val_df = pd.read_parquet(config.DATA_VAL)
n_val = len(val_df)
print(f"Val samples: {n_val:,}")

print(f"\n  Embedding validation texts in chunks...")
# Pre-allocate array for all validation embeddings
X_val = np.zeros((n_val, 385), dtype=np.float32)

for chunk_embs, (start_idx, end_idx) in embed_texts_chunked(
    val_df[config.TEXT_COLUMN], 
    val_df[config.TYPE_COLUMN],
    batch_size=config.EMB_BATCH_SIZE,
    chunk_size=50000
):
    X_val[start_idx:end_idx] = chunk_embs
    # Free memory
    del chunk_embs

y_cat_val = val_df["category_id"].values
y_sub_val = val_df["subcategory_id"].values

print(f"✓ Val embeddings shape: {X_val.shape}")
print(f"  Text embedding: 384-dim + Type feature: 1-dim = 385-dim total")
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
