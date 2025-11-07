"""
Verification script to check preprocessing results and dataset distribution
"""

import json  # For loading metadata in JSON format
from pathlib import Path  # For handling filesystem paths
from collections import Counter  # Not strictly needed in this script, but imported for possible use


def verify_preprocessing(data_dir="processed_data"):
    """
    Verify that preprocessing was completed successfully and show statistics.
    """
    data_path = Path(data_dir)  # Create a Path object for the processed data directory
    
    # Check if the processed data directory exists
    if not data_path.exists():
        print(f"❌ Error: {data_dir} directory not found!")  # Inform user if not found
        print("Please run preprocessing first: python preprocess_ct_scans.py")  # Suggest next step
        return False  # Exit the function with failure state
    
    print("=" * 70)
    print("PREPROCESSING VERIFICATION REPORT")  # Decorative header for the report
    print("=" * 70)
    
    # Check preprocessing metadata
    metadata_path = data_path / "metadata.json"  # Path to metadata file
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)  # Load metadata from JSON
            
        print("\n📋 PREPROCESSING SETTINGS:")  # Section header for preprocessing parameters
        print("-" * 70)
        print(f"  Image size: {metadata.get('img_size', 'N/A')}")  # Print image size used
        print(f"  Normalized: {metadata.get('normalize', 'N/A')}")  # Print whether normalization was applied
        print(f"  Grayscale: {metadata.get('grayscale', 'N/A')}")  # Print grayscale option
        print(f"  Random seed: {metadata.get('random_seed', 'N/A')}")  # Print random seed if available
        
        print("\n📊 ORIGINAL DATASET STATISTICS:")  # Section header for original stats
        print("-" * 70)
        if 'class_counts' in metadata:
            # Print image count for each original class
            for class_name, count in metadata['class_counts'].items():
                print(f"  {class_name}: {count} images")
            total = sum(metadata['class_counts'].values())  # Compute total image count
            print(f"  Total: {total} images")
    else:
        # Warn if metadata is missing
        print("⚠️  Warning: metadata.json not found")
    
    # Prepare to check splits (train/val/test)
    splits = ['train', 'val', 'test']  # Standard dataset splits
    all_splits_exist = True  # Flag to keep track if all splits are present
    
    print("\n📁 DATASET SPLITS:")  # Section header for split overview
    print("-" * 70)
    
    # Loop over each dataset split (train, val, test)
    for split in splits:
        split_dir = data_path / split  # Path to the current split directory
        if not split_dir.exists():
            print(f"❌ {split.upper()} directory not found!")  # Error if directory is missing
            all_splits_exist = False  # Set flag that something is missing
            continue  # Move to next split
        
        print(f"\n{split.upper()} SET:")  # Header for this split
        class_counts = {}  # Store class counts within the split
        total_images = 0  # Running total images for split
        
        # Iterate over each class subfolder in this split
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                # Collect all PNG and JPG images in this class folder
                images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                count = len(images)  # Number of images found
                class_counts[class_dir.name] = count  # Save in class counts dict
                total_images += count  # Add to total count
                print(f"  {class_dir.name:30s}: {count:4d} images")  # Print class count output
        
        print(f"  {'TOTAL':30s}: {total_images:4d} images")  # Print total images in this split
        
        # If this split is empty (no images), warn the user
        if total_images == 0:
            print(f"  ⚠️  Warning: {split.upper()} set is empty!")
            all_splits_exist = False  # Set flag to indicate an issue
    
    # Class distribution analysis on the training set to check for imbalance
    print("\n⚖️  CLASS DISTRIBUTION ANALYSIS:")  # Section header
    print("-" * 70)
    
    train_dir = data_path / "train"  # Path to training data
    if train_dir.exists():
        all_class_counts = {}  # Dict to hold count of images for each class
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))  # List all images in class
                all_class_counts[class_dir.name] = len(images)
        
        # If there is at least one class, perform imbalance analysis
        if all_class_counts:
            max_count = max(all_class_counts.values())  # Largest class size
            min_count = min(all_class_counts.values())  # Smallest class size
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')  # Compute ratio
            
            print(f"  Largest class: {max_count} images")  # Report largest class
            print(f"  Smallest class: {min_count} images")  # Report smallest class
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")  # Display the imbalance ratio
            
            # Give feedback based on imbalance ratio threshold
            if imbalance_ratio > 3:
                print("  ⚠️  Significant class imbalance detected!")
                print("     Consider using class weights during training.")
            else:
                print("  ✓ Class distribution is relatively balanced")
    
    # Attempt to test the data loader functionality with the processed dataset
    print("\n🔍 TESTING DATA LOADER:")  # Section title
    print("-" * 70)
    try:
        from data_loader import create_data_loaders  # Attempt to import the data loader utility
        
        # Try creating data loaders using the processed data
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            data_dir=data_dir,      # Path to processed data
            batch_size=16,          # Small batch size for test
            num_workers=0,          # No multiprocessing; safe default
            augment=False           # No augmentation for verification
        )
        
        # Print data loader statistics and success notices
        print(f"  ✓ Data loaders created successfully")
        print(f"  ✓ Number of classes: {len(class_names)}")
        print(f"  ✓ Classes: {', '.join(class_names)}")
        print(f"  ✓ Train batches: {len(train_loader)}")
        print(f"  ✓ Validation batches: {len(val_loader)}")
        print(f"  ✓ Test batches: {len(test_loader)}")
        
        # Attempt to load a single batch from the train loader to check data format
        try:
            sample_batch = next(iter(train_loader))  # Fetch one batch
            images, labels = sample_batch  # Unpack batch contents
            print(f"  ✓ Sample batch loaded successfully")  # Indicate batch loading worked
            print(f"  ✓ Batch shape: {images.shape}")      # Print shape of image batch
            print(f"  ✓ Image dtype: {images.dtype}")      # Print image data type
            print(f"  ✓ Image range: [{images.min():.3f}, {images.max():.3f}]")  # Print min/max
        except Exception as e:
            print(f"  ❌ Error loading batch: {e}")  # Report if loading fails
            
    except ImportError as e:
        print(f"  ❌ Error importing data_loader: {e}")  # Could not import loader module
    except Exception as e:
        print(f"  ❌ Error creating data loaders: {e}")  # Data loader instantiation failed
    
    print("\n" + "=" * 70)  # Footer separator
    # Print final overall verification result
    if all_splits_exist:
        print("✅ PREPROCESSING VERIFICATION COMPLETE")
        print("   Your dataset is ready for training!")
    else:
        print("⚠️  PREPROCESSING VERIFICATION FOUND ISSUES")
        print("   Please check the errors above and rerun preprocessing if needed.")
    print("=" * 70)
    
    return all_splits_exist  # Return whether everything is okay


if __name__ == "__main__":
    # If this script is run directly, call the verification routine
    verify_preprocessing()

