"""
Master script to run the entire transaction categorization pipeline.

Steps:
1. Prepare data (clean, filter, split)
2. Generate embeddings
3. Train models
4. Evaluate models
5. Run inference demo
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {script_name}")
    print(f"Description: {description}")
    print("=" * 80 + "\n")
    
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script_name} failed with code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {script_name} completed successfully")
    return True

def main():
    print("=" * 80)
    print("TRANSACTION CATEGORIZATION - FULL PIPELINE")
    print("=" * 80)
    print("\nThis will run the complete ML pipeline:")
    print("  1. Data preparation (cleaning, filtering, splitting)")
    print("  2. Embedding generation (using GPU if available)")
    print("  3. Model training (LogisticRegression classifiers)")
    print("  4. Model evaluation (metrics and analysis)")
    print("  5. Inference demo (example predictions)")
    
    input("\nPress ENTER to start or Ctrl+C to cancel...")
    
    # Run each step
    steps = [
        ("prepare_data.py", "Clean data, filter rare subcategories, create train/val split"),
        ("embed_dataset.py", "Generate embeddings using BAAI/bge-small-en-v1.5"),
        ("train_models.py", "Train category and subcategory classifiers"),
        ("evaluate.py", "Evaluate models with detailed metrics"),
        ("inference.py", "Run inference demo on sample transactions"),
    ]
    
    for script, description in steps:
        run_script(script, description)
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nAll models have been trained and evaluated.")
    print("\nArtifacts saved in ./artifacts/:")
    print("  • category_classifier_logreg.joblib")
    print("  • subcategory_classifier_logreg.joblib")
    print("  • category_label_encoder.joblib")
    print("  • subcategory_label_encoder.joblib")
    print("  • cat_to_sub_mapping.joblib")
    print("\nYou can now use inference.py to classify new transactions!")

if __name__ == "__main__":
    main()
