"""
Data preparation script for transaction categorization.

Steps:
1. Load CSV
2. Clean data (remove missing values, invalid text)
3. Filter rare subcategories (< MIN_SUBCATEGORY_SAMPLES)
4. Create label encoders
5. Build category -> subcategory mapping
6. Stratified train/val split
"""

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict
import config

print("=" * 80)
print("TRANSACTION CATEGORIZATION - DATA PREPARATION")
print("=" * 80)

# ========== Step 1: Load CSV ==========
print("\n[1/6] Loading CSV...")
df = pd.read_csv(config.CSV_PATH, nrows=config.MAX_ROWS)
print(f"Initial dataset shape: {df.shape}")
print(f"Total rows: {len(df):,}")

# ========== Step 2: Data Cleaning ==========
print("\n[2/6] Cleaning data...")
print(f"Missing clean_description: {df[config.TEXT_COLUMN].isna().sum():,}")
print(f"Missing category: {df[config.CATEGORY_COLUMN].isna().sum():,}")
print(f"Missing sub_category: {df[config.SUBCATEGORY_COLUMN].isna().sum():,}")

# Remove rows with missing values
initial_count = len(df)
df = df.dropna(subset=[config.TEXT_COLUMN, config.CATEGORY_COLUMN, config.SUBCATEGORY_COLUMN])
print(f"Removed {initial_count - len(df):,} rows with missing values")

# Clean text column
df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str).str.strip()
df = df[df[config.TEXT_COLUMN] != ""]
print(f"Final rows after cleaning: {len(df):,}")

# ========== Step 3: Filter Rare Subcategories ==========
print(f"\n[3/6] Filtering rare subcategories (min samples: {config.MIN_SUBCATEGORY_SAMPLES})...")
sub_counts = df[config.SUBCATEGORY_COLUMN].value_counts()
print(f"Total unique subcategories before filtering: {len(sub_counts)}")
print(f"Subcategories with < {config.MIN_SUBCATEGORY_SAMPLES} samples: {len(sub_counts[sub_counts < config.MIN_SUBCATEGORY_SAMPLES])}")

valid_subcats = sub_counts[sub_counts >= config.MIN_SUBCATEGORY_SAMPLES].index
df = df[df[config.SUBCATEGORY_COLUMN].isin(valid_subcats)]
print(f"Remaining subcategories: {df[config.SUBCATEGORY_COLUMN].nunique()}")
print(f"Remaining rows: {len(df):,}")

# Show final distribution stats
print("\n=== Final Category Distribution ===")
cat_dist = df[config.CATEGORY_COLUMN].value_counts()
print(cat_dist)

print("\n=== Final Subcategory Distribution (Top 10) ===")
subcat_dist = df[config.SUBCATEGORY_COLUMN].value_counts()
print(subcat_dist.head(10))
print(f"\nMedian samples per subcategory: {subcat_dist.median():.1f}")
print(f"Mean samples per subcategory: {subcat_dist.mean():.1f}")

# Save cleaned data
df.to_parquet(config.DATA_CLEANED, index=False)
print(f"\n✓ Saved cleaned data to {config.DATA_CLEANED}")

# ========== Step 4: Create Label Encoders ==========
print("\n[4/6] Creating label encoders...")
cat_encoder = LabelEncoder()
subcat_encoder = LabelEncoder()

df["category_id"] = cat_encoder.fit_transform(df[config.CATEGORY_COLUMN])
df["subcategory_id"] = subcat_encoder.fit_transform(df[config.SUBCATEGORY_COLUMN])

print(f"Category classes: {len(cat_encoder.classes_)}")
print(f"Subcategory classes: {len(subcat_encoder.classes_)}")
print(f"\nCategories: {sorted(cat_encoder.classes_)}")

# Save encoders
joblib.dump(cat_encoder, config.CAT_ENCODER_PATH)
joblib.dump(subcat_encoder, config.SUBCAT_ENCODER_PATH)
print(f"✓ Saved category encoder to {config.CAT_ENCODER_PATH}")
print(f"✓ Saved subcategory encoder to {config.SUBCAT_ENCODER_PATH}")

# ========== Step 5: Build Category -> Subcategory Mapping ==========
print("\n[5/6] Building category -> subcategory mapping...")
cat_to_sub = defaultdict(set)
for cat_id, sub_id in zip(df["category_id"], df["subcategory_id"]):
    cat_to_sub[cat_id].add(sub_id)

# Convert sets to sorted lists for consistency
cat_to_sub = {k: sorted(list(v)) for k, v in cat_to_sub.items()}

print("Category -> Subcategory mapping:")
for cat_id in sorted(cat_to_sub.keys()):
    cat_name = cat_encoder.inverse_transform([cat_id])[0]
    num_subcats = len(cat_to_sub[cat_id])
    print(f"  {cat_name} (id={cat_id}): {num_subcats} subcategories")

joblib.dump(cat_to_sub, config.CAT_TO_SUB_MAPPING_PATH)
print(f"✓ Saved mapping to {config.CAT_TO_SUB_MAPPING_PATH}")

# ========== Step 6: Train/Val Split ==========
print(f"\n[6/6] Creating train/val split (test_size={config.TEST_SIZE})...")
print("Using stratified split on subcategory_id to maintain class distribution...")

train_df, val_df = train_test_split(
    df,
    test_size=config.TEST_SIZE,
    stratify=df["subcategory_id"],
    random_state=config.RANDOM_STATE,
)

print(f"\nTrain set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val set: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")

# Verify stratification worked
print("\n=== Verifying Stratification ===")
print("Train category distribution:")
print(train_df[config.CATEGORY_COLUMN].value_counts(normalize=True).head())
print("\nVal category distribution:")
print(val_df[config.CATEGORY_COLUMN].value_counts(normalize=True).head())

train_df.to_parquet(config.DATA_TRAIN, index=False)
val_df.to_parquet(config.DATA_VAL, index=False)
print(f"\n✓ Saved train data to {config.DATA_TRAIN}")
print(f"✓ Saved val data to {config.DATA_VAL}")

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE!")
print("=" * 80)
print(f"\nSummary:")
print(f"  • Final dataset: {len(df):,} transactions")
print(f"  • Categories: {len(cat_encoder.classes_)}")
print(f"  • Subcategories: {len(subcat_encoder.classes_)}")
print(f"  • Train samples: {len(train_df):,}")
print(f"  • Val samples: {len(val_df):,}")
print(f"\nNext step: Run embed_dataset.py to generate embeddings")
