"""Quick exploration of the transaction dataset to understand categories and class distribution."""

import pandas as pd
import numpy as np

# Load the CSV
print("Loading CSV...")
df = pd.read_csv("Plaid_transactions_for_categorization.csv")

print(f"\nDataset shape: {df.shape}")
print(f"Total rows: {len(df):,}")

# Check for missing values in key columns
print("\n=== Missing Values ===")
print(f"clean_description: {df['clean_description'].isna().sum():,}")
print(f"category: {df['category'].isna().sum():,}")
print(f"sub_category: {df['sub_category'].isna().sum():,}")

# Categories
print("\n=== CATEGORIES ===")
categories = df['category'].unique()
print(f"Number of unique categories: {len(categories)}")
print(f"Categories: {sorted([c for c in categories if pd.notna(c)])}")

# Category distribution
print("\n=== Category Distribution ===")
cat_counts = df['category'].value_counts()
print(cat_counts)

# Subcategories
print("\n=== SUBCATEGORIES ===")
subcategories = df['sub_category'].unique()
print(f"Number of unique subcategories: {len(subcategories)}")

# Subcategory distribution (top 20 and bottom 20)
print("\n=== Subcategory Distribution (Top 20) ===")
subcat_counts = df['sub_category'].value_counts()
print(subcat_counts.head(20))

print("\n=== Subcategory Distribution (Bottom 20) ===")
print(subcat_counts.tail(20))

# Statistics on subcategory distribution
print("\n=== Subcategory Statistics ===")
print(f"Mean samples per subcategory: {subcat_counts.mean():.1f}")
print(f"Median samples per subcategory: {subcat_counts.median():.1f}")
print(f"Min samples: {subcat_counts.min()}")
print(f"Max samples: {subcat_counts.max():,}")
print(f"Std dev: {subcat_counts.std():.1f}")

# How many subcategories have < 100 samples?
rare_subcats = subcat_counts[subcat_counts < 100]
print(f"\nSubcategories with < 100 samples: {len(rare_subcats)}")
print(f"Subcategories with < 50 samples: {len(subcat_counts[subcat_counts < 50])}")
print(f"Subcategories with < 10 samples: {len(subcat_counts[subcat_counts < 10])}")

# Category -> Subcategory mapping
print("\n=== Category -> Subcategory Mapping ===")
for cat in sorted(cat_counts.index):
    if pd.notna(cat):
        subcats = df[df['category'] == cat]['sub_category'].nunique()
        print(f"{cat}: {subcats} subcategories")

# Check clean_description
print("\n=== Clean Description Sample ===")
valid_desc = df[df['clean_description'].notna()]['clean_description']
print(f"Valid descriptions: {len(valid_desc):,}")
print("\nSample descriptions:")
for desc in valid_desc.head(10):
    print(f"  - {desc[:80]}")

# Class imbalance ratio
print("\n=== Class Imbalance Analysis ===")
max_cat = cat_counts.max()
min_cat = cat_counts.min()
print(f"Category imbalance ratio (max/min): {max_cat/min_cat:.1f}x")

max_sub = subcat_counts.max()
min_sub = subcat_counts.min()
print(f"Subcategory imbalance ratio (max/min): {max_sub/min_sub:.1f}x")
