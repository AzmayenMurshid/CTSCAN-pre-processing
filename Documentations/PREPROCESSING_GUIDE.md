# Step-by-Step Guide: Preprocessing and Dataset Creation

This guide will walk you through creating your training, validation, and test datasets from the CT scan images.

## Overview

Your dataset contains:
- **adenocarcinoma**: 337 PNG images
- **Benign cases**: 120 JPG images  
- **large cell carcinoma**: 187 PNG images
- **Normal cases**: 631 images (428 JPG, 203 PNG)
- **squamous cell carcinoma**: 260 PNG images

**Total**: ~1,535 images across 5 classes

---

## Step 1: Run the Preprocessing Script

Execute the preprocessing script to process all images and create train/val/test splits:

```bash
python preprocess_ct_scans.py
```

### What this does:
1. ✅ Loads all images from each class folder
2. ✅ Converts RGBA images to RGB (with white background)
3. ✅ Resizes all images to 224x224 pixels
4. ✅ Normalizes pixel values to [0, 1] range
5. ✅ Splits data into:
   - **Training set**: 70% (~1,074 images)
   - **Validation set**: 15% (~230 images)
   - **Test set**: 15% (~230 images)
6. ✅ Saves processed images to `processed_data/` directory
7. ✅ Creates `metadata.json` with preprocessing information

### Expected Output Structure:
```
processed_data/
├── train/
│   ├── adenocarcinoma/
│   ├── Benign cases/
│   ├── large cell carcinoma/
│   ├── Normal cases/
│   └── squamous cell carcinoma/
├── val/
│   └── [same class structure]
├── test/
│   └── [same class structure]
└── metadata.json
```

---

## Step 2: Verify Preprocessing Results

After preprocessing completes, verify the output:

### Check the output directory:
```bash
# On Windows PowerShell
dir processed_data
dir processed_data\train
dir processed_data\val
dir processed_data\test
```

### Check metadata:
The `metadata.json` file contains:
- Image dimensions used
- Normalization settings
- Class mapping
- Number of images in each split
- Original class counts

You can view it with:
```python
import json
with open('processed_data/metadata.json', 'r') as f:
    metadata = json.load(f)
    print(json.dumps(metadata, indent=2))
```

---

## Step 3: Verify Data Loaders Work

Test that your data loaders can successfully load the preprocessed data:

```python
from data_loader import create_data_loaders

# Create data loaders
train_loader, val_loader, test_loader, class_names = create_data_loaders(
    data_dir="processed_data",
    batch_size=32,
    num_workers=0,
    augment=True
)

print(f"Classes: {class_names}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Get a sample batch
sample_batch = next(iter(train_loader))
images, labels = sample_batch
print(f"Sample batch shape: {images.shape}")
print(f"Sample labels: {labels[:5]}")
```

---

## Step 4: Analyze Dataset Distribution

Check the class distribution in each split to ensure balanced splits:

```python
from pathlib import Path
from collections import Counter

def analyze_dataset(data_dir="processed_data"):
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n{split.upper()} SET:")
        print("-" * 60)
        
        split_dir = Path(data_dir) / split
        class_counts = {}
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')))
                class_counts[class_dir.name] = count
                print(f"  {class_dir.name}: {count} images")
        
        print(f"  Total: {sum(class_counts.values())} images")

analyze_dataset()
```

---

## Step 5: (Optional) Customize Preprocessing

If you want to change preprocessing parameters, modify the script:

```python
from preprocess_ct_scans import CTScanPreprocessor

preprocessor = CTScanPreprocessor(
    input_dir="Lung Cancer Dataset",
    output_dir="processed_data",
    img_size=(256, 256),  # Change size if needed
    normalize=True,        # Keep normalization
    grayscale=False        # Keep RGB
)

preprocessor.preprocess_dataset(
    train_ratio=0.7,   # 70% training
    val_ratio=0.15,    # 15% validation
    test_ratio=0.15,   # 15% testing
    random_seed=42     # For reproducibility
)
```

---

## Step 6: Handle Class Imbalance (Optional)

If you notice significant class imbalance, you can calculate class weights:

```python
from data_loader import CTScanDataset

train_dataset = CTScanDataset("processed_data", split='train')
class_weights = train_dataset.get_class_weights()

print("Class weights for handling imbalance:")
for label, weight in class_weights.items():
    print(f"  Class {label}: {weight:.4f}")
```

Use these weights in your loss function during training.

---

## Step 7: Ready for Training!

Once preprocessing is complete and verified, you can:

1. **Start training** with the example script:
   ```bash
   python example_training.py
   ```

2. **Or create your own training script** using the data loaders:
   ```python
   from data_loader import create_data_loaders
   import torch
   import torch.nn as nn
   
   # Create data loaders
   train_loader, val_loader, test_loader, class_names = create_data_loaders(
       data_dir="processed_data",
       batch_size=32,
       num_workers=0,
       augment=True
   )
   
   # Your training code here
   ```

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
- **Solution**: Make sure all packages are installed: `pip install -r requirements.txt`

### Issue: "No images found"
- **Solution**: Check that the folder name is exactly "Lung Cancer Dataset" (case-sensitive)

### Issue: "Out of memory" during preprocessing
- **Solution**: Process images in smaller batches or reduce image size

### Issue: "File access error"
- **Solution**: Close any programs that might be using the images, then retry

---

## Quick Commands Summary

```bash
# 1. Preprocess the dataset
python preprocess_ct_scans.py

# 2. Verify data loaders
python -c "from data_loader import create_data_loaders; train_loader, val_loader, test_loader, class_names = create_data_loaders('processed_data'); print(f'Train: {len(train_loader)} batches, Classes: {class_names}')"

# 3. Start training (optional)
python example_training.py
```

---

## Next Steps After Preprocessing

1. ✅ **Preprocessing done** - Images are resized, normalized, and split
2. 🔄 **Train your model** - Use `example_training.py` or create your own
3. 📊 **Evaluate performance** - Test on the test set
4. 🎯 **Fine-tune** - Adjust hyperparameters based on validation results
5. 💾 **Save your model** - Export the best model for inference


