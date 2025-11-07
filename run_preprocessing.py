"""
Quick script to run preprocessing with progress tracking
"""

import sys  # Import sys module for interacting with the interpreter (exit)
from pathlib import Path  # Import Path for convenient file/folder path manipulations
from preprocess_ct_scans import CTScanPreprocessor  # Import the preprocessor class

def main():
    # Print a decorative title/header for the user
    print("=" * 70)
    print("CT SCAN PREPROCESSING PIPELINE")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Load all CT scan images from 'Lung Cancer Dataset'")
    print("  2. Preprocess images (resize, normalize, convert formats)")
    print("  3. Split into train (70%), validation (15%), and test (15%) sets")
    print("  4. Save processed images to 'processed_data' directory")
    print("\n" + "=" * 70)
    
    # Define the input dataset directory as a Path object
    input_dir = Path("Lung Cancer Dataset")
    # Check if the input directory actually exists
    if not input_dir.exists():
        # If not found, print error and exit the script
        print(f"\n❌ Error: '{input_dir}' directory not found!")
        print("Please make sure the dataset folder exists.")
        sys.exit(1)
    
    # Define the output directory for processed data
    output_dir = Path("processed_data")
    # If the output directory already exists, check for existing processed images
    if output_dir.exists():
        # Count the number of PNG files that already exist in the processed_data folder recursively
        existing_files = sum(1 for f in output_dir.rglob('*.png') if f.is_file())
        # If any processed images are found, warn the user about existing data
        if existing_files > 0:
            print(f"\n⚠️  Found existing processed data: {existing_files} images")
            print("   The script will skip existing files (no duplicates will be created).")
            print("   To reprocess everything, delete the 'processed_data' folder first.")
            print()
    
    # Inform user that preprocessor is being initialized
    print("Initializing preprocessor...")
    # Create an instance of CTScanPreprocessor with desired options
    preprocessor = CTScanPreprocessor(
        input_dir="Lung Cancer Dataset",   # Use the input directory defined above
        output_dir="processed_data",       # Processed images will be written here
        img_size=(224, 224),               # All images will be resized to 224x224
        normalize=True,                    # Enable normalization to [0, 1]
        grayscale=False,                   # Keep images in RGB (not grayscale)
        skip_existing=True  # Skip files that already exist (prevents duplicates)
    )
    
    # Start preprocessing
    print("\nStarting preprocessing...")
    try:
        # Call the preprocess_dataset method to start processing images and splitting the dataset
        preprocessor.preprocess_dataset(
            train_ratio=0.7,      # 70% of data for training
            val_ratio=0.15,       # 15% for validation
            test_ratio=0.15,      # 15% for testing
            random_seed=42        # Seed for reproducible splitting
        )
        
        # Print completion banner and next instructions
        print("\n" + "=" * 70)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Verify preprocessing: python verify_preprocessing.py")
        print("  2. Start training: python example_training.py")
        print("\n" + "=" * 70)
        
    except Exception as e:
        # Print any errors that occurred during preprocessing and exit the script
        print(f"\n❌ Error during preprocessing: {e}")
        print("\nPlease check the error message above and try again.")
        sys.exit(1)

# If this script is run as the main program (not imported), execute main()
if __name__ == "__main__":
    main()

