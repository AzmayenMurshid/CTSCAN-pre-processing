"""
Quick script to run preprocessing with progress tracking
"""

import sys
from pathlib import Path
from preprocess_ct_scans import CTScanPreprocessor

def main():
    print("=" * 70)
    print("CT SCAN PREPROCESSING PIPELINE")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Load all CT scan images from 'Lung Cancer Dataset'")
    print("  2. Preprocess images (resize, normalize, convert formats)")
    print("  3. Split into train (70%), validation (15%), and test (15%) sets")
    print("  4. Save processed images to 'processed_data' directory")
    print("\n" + "=" * 70)
    
    # Check if input directory exists
    input_dir = Path("Lung Cancer Dataset")
    if not input_dir.exists():
        print(f"\n❌ Error: '{input_dir}' directory not found!")
        print("Please make sure the dataset folder exists.")
        sys.exit(1)
    
    # Check if processed data already exists
    output_dir = Path("processed_data")
    if output_dir.exists():
        existing_files = sum(1 for f in output_dir.rglob('*.png') if f.is_file())
        if existing_files > 0:
            print(f"\n⚠️  Found existing processed data: {existing_files} images")
            print("   The script will skip existing files (no duplicates will be created).")
            print("   To reprocess everything, delete the 'processed_data' folder first.")
            print()
    
    # Initialize preprocessor
    print("Initializing preprocessor...")
    preprocessor = CTScanPreprocessor(
        input_dir="Lung Cancer Dataset",
        output_dir="processed_data",
        img_size=(224, 224),
        normalize=True,
        grayscale=False,
        skip_existing=True  # Skip files that already exist (prevents duplicates)
    )
    
    # Run preprocessing
    print("\nStarting preprocessing...")
    try:
        preprocessor.preprocess_dataset(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        print("\n" + "=" * 70)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Verify preprocessing: python verify_preprocessing.py")
        print("  2. Start training: python example_training.py")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        print("\nPlease check the error message above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()

