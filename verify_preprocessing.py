"""
Verification script to check preprocessing results and dataset distribution
"""

import json
from pathlib import Path
from collections import Counter


def verify_preprocessing(data_dir="processed_data"):
    """
    Verify that preprocessing was completed successfully and show statistics.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Error: {data_dir} directory not found!")
        print("Please run preprocessing first: python preprocess_ct_scans.py")
        return False
    
    print("=" * 70)
    print("PREPROCESSING VERIFICATION REPORT")
    print("=" * 70)
    
    # Check metadata
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("\n📋 PREPROCESSING SETTINGS:")
        print("-" * 70)
        print(f"  Image size: {metadata.get('img_size', 'N/A')}")
        print(f"  Normalized: {metadata.get('normalize', 'N/A')}")
        print(f"  Grayscale: {metadata.get('grayscale', 'N/A')}")
        print(f"  Random seed: {metadata.get('random_seed', 'N/A')}")
        
        print("\n📊 ORIGINAL DATASET STATISTICS:")
        print("-" * 70)
        if 'class_counts' in metadata:
            for class_name, count in metadata['class_counts'].items():
                print(f"  {class_name}: {count} images")
            total = sum(metadata['class_counts'].values())
            print(f"  Total: {total} images")
    else:
        print("⚠️  Warning: metadata.json not found")
    
    # Check splits
    splits = ['train', 'val', 'test']
    all_splits_exist = True
    
    print("\n📁 DATASET SPLITS:")
    print("-" * 70)
    
    for split in splits:
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"❌ {split.upper()} directory not found!")
            all_splits_exist = False
            continue
        
        print(f"\n{split.upper()} SET:")
        class_counts = {}
        total_images = 0
        
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                # Count images
                images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                count = len(images)
                class_counts[class_dir.name] = count
                total_images += count
                print(f"  {class_dir.name:30s}: {count:4d} images")
        
        print(f"  {'TOTAL':30s}: {total_images:4d} images")
        
        # Check if split is empty
        if total_images == 0:
            print(f"  ⚠️  Warning: {split.upper()} set is empty!")
            all_splits_exist = False
    
    # Check class distribution balance
    print("\n⚖️  CLASS DISTRIBUTION ANALYSIS:")
    print("-" * 70)
    
    train_dir = data_path / "train"
    if train_dir.exists():
        all_class_counts = {}
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                all_class_counts[class_dir.name] = len(images)
        
        if all_class_counts:
            max_count = max(all_class_counts.values())
            min_count = min(all_class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"  Largest class: {max_count} images")
            print(f"  Smallest class: {min_count} images")
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
            
            if imbalance_ratio > 3:
                print("  ⚠️  Significant class imbalance detected!")
                print("     Consider using class weights during training.")
            else:
                print("  ✓ Class distribution is relatively balanced")
    
    # Test data loader
    print("\n🔍 TESTING DATA LOADER:")
    print("-" * 70)
    try:
        from data_loader import create_data_loaders
        
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            data_dir=data_dir,
            batch_size=16,
            num_workers=0,
            augment=False
        )
        
        print(f"  ✓ Data loaders created successfully")
        print(f"  ✓ Number of classes: {len(class_names)}")
        print(f"  ✓ Classes: {', '.join(class_names)}")
        print(f"  ✓ Train batches: {len(train_loader)}")
        print(f"  ✓ Validation batches: {len(val_loader)}")
        print(f"  ✓ Test batches: {len(test_loader)}")
        
        # Try to load a batch
        try:
            sample_batch = next(iter(train_loader))
            images, labels = sample_batch
            print(f"  ✓ Sample batch loaded successfully")
            print(f"  ✓ Batch shape: {images.shape}")
            print(f"  ✓ Image dtype: {images.dtype}")
            print(f"  ✓ Image range: [{images.min():.3f}, {images.max():.3f}]")
        except Exception as e:
            print(f"  ❌ Error loading batch: {e}")
            
    except ImportError as e:
        print(f"  ❌ Error importing data_loader: {e}")
    except Exception as e:
        print(f"  ❌ Error creating data loaders: {e}")
    
    print("\n" + "=" * 70)
    if all_splits_exist:
        print("✅ PREPROCESSING VERIFICATION COMPLETE")
        print("   Your dataset is ready for training!")
    else:
        print("⚠️  PREPROCESSING VERIFICATION FOUND ISSUES")
        print("   Please check the errors above and rerun preprocessing if needed.")
    print("=" * 70)
    
    return all_splits_exist


if __name__ == "__main__":
    verify_preprocessing()

