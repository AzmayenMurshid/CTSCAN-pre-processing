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

The `preprocess_ct_scans.py` script carries out the following preprocessing steps for your CT scan lung cancer dataset:

1. **Loading Images:**  
   The script iterates through each of the five class folders (e.g., `adenocarcinoma`, `Benign cases`), reading in all the images. It can recognize both PNG and JPG file formats.

2. **Image Format Handling:**  
   Image format handling is performed as follows:
   - The script checks if an image is in RGBA format, which means it has a transparency (alpha) channel.
   - If RGBA is detected, the image is converted to standard RGB format.
   - During this conversion, any transparent areas are replaced with a white background.
   - This process ensures all images have consistent color channels and appearance, making them suitable for model training.

3. **Resizing:**  
   To standardize input for neural networks, all images are resized to 224x224 pixels. This is a common practice for models like ResNet, VGG, etc.
   - Images are resized using the OpenCV library's `cv2.resize` function.
   - Each image is scaled to a target size of 224x224 pixels (width x height).
   - All images, no matter their original dimensions or aspect ratio, are adjusted to this fixed size.
   - If an image's aspect ratio differs, it will be stretched to fit 224x224, following common computer vision practices for training deep learning models.


4. **Normalization:**  
   Pixel values, initially in the 0-255 range, are scaled to [0, 1] by dividing by 255. This normalization improves model training stability.
   
   - After loading and resizing each image, the script converts the pixel data to a NumPy array.
   - The array is divided by 255.0 (`img_array = img_array / 255.0`), converting all values to floats between 0 and 1.
   - This normalization ensures all images, regardless of format or source, have consistent pixel intensity scaling.
   - The result is images that are easier for neural networks to process, leading to more stable and efficient training.

5. **Splitting the Dataset:**  
   The dataset splitting is performed as follows:
   - All loaded images are grouped according to their class labels.
   - For each class, the images are shuffled randomly to ensure a fair split.
   - The dataset is then divided into three parts using stratified sampling, so that each split maintains the same class proportions as the original data:
     - 70% of the images go into the training set.
     - 15% go into the validation set.
     - 15% are allocated to the test set.
   - This stratified split helps ensure balanced representation of all classes in each subset, improving model reliability and evaluation.

6. **Saving Processed Data:**  
   Processed images are saved to a new, clean directory structure (`processed_data/`), with subfolders for `train`, `val`, and `test`, and further grouped by class. This makes them easy to use with PyTorch, TensorFlow, and other ML/DL frameworks.

7. **Metadata Generation:**  
   The script creates a `metadata.json` file in the `processed_data/` directory, recording:
   - How images were processed (resize, normalization)
   - Mapping of class names to indices
   - Number of images in each split and per class
   - Original image counts and split proportions

**In summary:**  
The preprocessing code ensures all images are in a consistent, model-ready format — with uniform size, color channels, and scaling — and builds splits for training, validation, and testing. The script also documents the transformations in a metadata file for transparency and reproducibility.

---

## Step 2: Verify Preprocessing Results

After preprocessing completes, verify the output:

- The output directory structure is checked to confirm that images have been sorted into `train`, `val`, and `test` splits, each containing their respective class folders.
- The presence of all expected class subfolders in each split ensures that data was partitioned for every class.
- The `metadata.json` file is inspected to confirm details about image processing steps, class mapping, and per-split image counts.
- You can programmatically check image shapes and normalization by loading a few samples and verifying they have the intended size (e.g. 224x224) and value range ([0, 1]).
- Comparing the class distribution in each split can help ensure that the stratified splitting correctly preserves class balance.
- Success messages and directory listings can confirm that files were saved without errors.

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

The provided data loader utility (`create_data_loaders`) is designed to:
- Organize the preprocessed images into PyTorch `Dataset` and `DataLoader` objects, which efficiently handle batching, shuffling, and parallel data loading.
- Automatically infer class labels from the directory structure (each class has its own subfolder) and map them to integer indices.
- Load images from the `processed_data/train`, `processed_data/val`, and `processed_data/test` folders, ensuring each split is properly separated.
- (Optionally) Apply data augmentation to the training set, such as random flips or rotations, if `augment=True` (helpful for model generalization).
- Return three data loaders (for training, validation, and testing) along with the list of class names.
- Allow easy iteration over image batches, so you can verify the shape and content before proceeding to model training.

This test step ensures that your dataset was properly structured and that batches can be drawn for model training, validation, and evaluation without errors.

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

The script below provides a simple utility to summarize and print the number of images per class in each of the `train`, `val`, and `test` folders. By iterating through each split, it prints the number of images contained in each class subfolder, as well as the total number of images per split. This helps you verify that the class distributions remain balanced after splitting and that no files are missing for any class.

How it works:
- For each split (`train`, `val`, `test`), the function locates the corresponding folder.
- It then iterates over each class subfolder, counts all `.jpg` and `.png` files, and prints the count by class.
- Finally, it sums up and prints the total image count for that split.
- This makes it easy to visually check for significant imbalances or missing data, and to ensure that all images have been processed and organized correctly.

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

If you notice significant class imbalance—that is, when some classes have far more images than others—your model may become biased towards the majority classes and perform poorly on minority classes. This problem is common in medical datasets.

**Class weights** are a way to correct for this: they tell the loss function during training to "pay more attention" to underrepresented classes. Usually, a higher weight is assigned to smaller classes so that errors on those examples have more influence on the model update.

Class imbalance can be corrected or mitigated in several ways:
- **Class weighting:** Supply calculated class weights to the loss function during training, so the model penalizes mistakes on rare classes more.
- **Resampling:** You can oversample minority classes (duplicate or augment rare class images), or undersample majority classes (remove some examples), to balance the dataset.
- **Data augmentation:** Apply more augmentations to the minority classes to synthetically grow them.

The most common approach is to use class weights with your loss function, especially if you want to fully utilize the available data without discarding or duplicating images. Most frameworks (like PyTorch and TensorFlow) allow you to supply a `weight` argument when defining your loss, and the weights can be calculated directly from your data distribution (see below).

To get started, you can calculate and print class weights as shown:

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


