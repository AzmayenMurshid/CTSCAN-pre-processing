# CT Scan Classification Pipeline - Complete Description

## Overview

This pipeline processes CT scan images and trains a 2D CNN model to classify lung cancer into 5 categories:
- **adenocarcinoma** (Class 0)
- **Benign cases** (Class 1)
- **large cell carcinoma** (Class 2)
- **Normal cases** (Class 3)
- **squamous cell carcinoma** (Class 4)

---

## Pipeline Architecture

The pipeline consists of three main stages:

```
Raw CT Scans → Preprocessing → Data Loading → Model Training → Evaluation
```

---

## Stage 1: Data Preprocessing (`preprocess_ct_scans.py`)

### Input
- Raw CT scan images in `Lung Cancer Dataset/`
- Mixed formats: PNG and JPG images
- Variable sizes and formats (including RGBA)

### Process Flow

#### 1.1 **Image Loading & Format Normalization**
```python
# Handles mixed formats (JPG/PNG)
img = Image.open(image_path)

# Convert RGBA to RGB (if needed)
if img.mode == 'RGBA':
    # Composited on white background
    background = Image.new('RGB', img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background
elif img.mode != 'RGB':
    img = img.convert('RGB')
```

**Purpose**: Ensures all images are in RGB format for consistent processing

#### 1.2 **Image Resizing**
```python
img_array = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)
```

**Purpose**: Standardizes all images to 224×224 pixels (standard CNN input size)

#### 1.3 **Pixel Normalization**
```python
img_array = img_array.astype(np.float32) / 255.0
```

**Purpose**: Normalizes pixel values from [0, 255] to [0, 1] for better training stability

#### 1.4 **Data Splitting**
```python
# Stratified split to maintain class distribution
train_files, temp_files = train_test_split(
    all_files, 
    test_size=0.3, 
    stratify=all_labels
)
val_files, test_files = train_test_split(
    temp_files,
    test_size=0.5,
    stratify=temp_labels
)
```

**Split Ratio**: 70% Train / 15% Validation / 15% Test

**Purpose**: 
- Stratified splitting ensures each set has proportional class representation
- Prevents class imbalance across splits

#### 1.5 **Duplicate Detection**
```python
if self.skip_existing and output_path.exists():
    skipped_count += 1
    continue
```

**Purpose**: Prevents reprocessing existing files, enables resumable preprocessing

### Output
- **Processed images** in `processed_data/`:
  - `train/` - Training set (1,074 images)
  - `val/` - Validation set (230 images)
  - `test/` - Test set (231 images)
- **Metadata file** (`metadata.json`):
  - Preprocessing settings
  - Class mappings
  - Split statistics

---

## Stage 2: Data Loading (`data_loader.py`)

### Purpose
Creates PyTorch-compatible data loaders for training with batching and augmentation support.

### Components

#### 2.1 **CTScanDataset Class**
Custom PyTorch Dataset that:
- Loads images from processed directories
- Applies normalization (if needed)
- Handles data augmentation (training only)
- Converts images to PyTorch tensors

#### 2.2 **Data Augmentation** (Training Only)
```python
def _augment_image(self, image):
    # Random horizontal flip
    if np.random.random() > 0.5:
        image = np.fliplr(image)
    
    # Random rotation (±15 degrees)
    if np.random.random() > 0.5:
        angle = np.random.uniform(-15, 15)
        # ... rotation logic
    
    # Random brightness adjustment
    if np.random.random() > 0.5:
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 1)
    
    # Random contrast adjustment
    if np.random.random() > 0.5:
        contrast = np.random.uniform(0.8, 1.2)
        # ... contrast logic
```

**Purpose**: Increases data diversity and reduces overfitting

#### 2.3 **DataLoader Creation**
```python
train_loader, val_loader, test_loader, class_names = create_data_loaders(
    data_dir="processed_data",
    batch_size=32,
    num_workers=0,
    augment=True
)
```

**Features**:
- **Batching**: Groups images into batches for efficient GPU processing
- **Shuffling**: Randomizes training data order
- **Parallel loading**: Optional multi-worker data loading

---

## Stage 3: Model Training (`example_training.py`)

### Model Architecture: SimpleLungCancerCNN

#### 3.1 **Feature Extraction Layers**
```
Input (3, 224, 224)
  ↓
Conv Block 1: 3 → 64 channels
  - Conv2d(3, 64, 3×3)
  - BatchNorm2d
  - ReLU
  - MaxPool2d(2×2)
  - Dropout2d(0.25)
  ↓
Conv Block 2: 64 → 128 channels
  - Conv2d(64, 128, 3×3)
  - BatchNorm2d
  - ReLU
  - MaxPool2d(2×2)
  - Dropout2d(0.25)
  ↓
Conv Block 3: 128 → 256 channels
  - Conv2d(128, 256, 3×3)
  - BatchNorm2d
  - ReLU
  - MaxPool2d(2×2)
  - Dropout2d(0.25)
  ↓
Conv Block 4: 256 → 512 channels
  - Conv2d(256, 512, 3×3)
  - BatchNorm2d
  - ReLU
  - MaxPool2d(2×2)
  - Dropout2d(0.25)
  ↓
AdaptiveAvgPool2d(7×7)
  ↓
Flatten
```

#### 3.2 **Classifier Layers**
```
Flattened Features (512 × 7 × 7 = 25,088)
  ↓
Linear(25,088 → 512)
  - BatchNorm1d
  - ReLU
  - Dropout(0.5)
  ↓
Linear(512 → 256)
  - BatchNorm1d
  - ReLU
  - Dropout(0.5)
  ↓
Linear(256 → 5)  # 5 classes
  ↓
Output: Class probabilities
```

### Training Process

#### 3.3 **Training Loop**
```
For each epoch:
    1. Train Phase:
       - Set model to training mode
       - For each batch:
         a. Forward pass: Compute predictions
         b. Calculate loss: CrossEntropyLoss
         c. Backward pass: Compute gradients
         d. Update weights: Optimizer step
       - Calculate training loss and accuracy
     
    2. Validation Phase:
       - Set model to evaluation mode
       - Disable gradient computation
       - For each batch:
         a. Forward pass only
         b. Calculate loss
       - Calculate validation loss and accuracy
     
    3. Learning Rate Scheduling:
       - Reduce LR if validation loss plateaus
     
    4. Model Checkpointing:
       - Save model if validation accuracy improves
```

#### 3.4 **Training Configuration**
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning_rate=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Batch Size**: 32
- **Epochs**: 20
- **Device**: CUDA (GPU) if available, else CPU

#### 3.5 **Evaluation Metrics**
- **Training Loss**: Average loss per epoch
- **Training Accuracy**: Percentage of correct predictions
- **Validation Loss**: Average validation loss per epoch
- **Validation Accuracy**: Percentage of correct predictions on validation set
- **Test Accuracy**: Final evaluation on held-out test set

---

## Complete Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: PREPROCESSING                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Raw Images (JPG/PNG, various sizes)                         │
│         ↓                                                     │
│  [Format Normalization] → RGB conversion, RGBA handling      │
│         ↓                                                     │
│  [Resizing] → All images to 224×224                          │
│         ↓                                                     │
│  [Normalization] → Pixel values [0,1]                        │
│         ↓                                                     │
│  [Stratified Split] → 70/15/15 (Train/Val/Test)             │
│         ↓                                                     │
│  Save to processed_data/                                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 2: DATA LOADING                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Load processed images from disk                             │
│         ↓                                                     │
│  Apply augmentation (training only)                          │
│    - Random flips                                            │
│    - Random rotation                                         │
│    - Brightness/Contrast adjustment                          │
│         ↓                                                     │
│  Convert to PyTorch tensors                                  │
│         ↓                                                     │
│  Create DataLoaders with batching                            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 3: MODEL TRAINING                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Initialize CNN Model                                        │
│         ↓                                                     │
│  For each epoch:                                             │
│    ├─ Training:                                              │
│    │    - Forward pass                                       │
│    │    - Calculate loss                                     │
│    │    - Backward pass (gradients)                          │
│    │    - Update weights                                     │
│    │                                                         │
│    ├─ Validation:                                            │
│    │    - Forward pass only                                  │
│    │    - Calculate validation metrics                       │
│    │                                                         │
│    ├─ Learning Rate Scheduling                               │
│    │                                                         │
│    └─ Save best model (if improved)                          │
│         ↓                                                     │
│  Final Evaluation on Test Set                                │
│         ↓                                                     │
│  Generate Training History Plots                             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. **Robust Preprocessing**
- Handles mixed image formats (PNG/JPG)
- Automatic RGBA to RGB conversion
- Consistent image sizing
- Pixel value normalization

### 2. **Smart Data Management**
- Stratified splitting maintains class balance
- Duplicate detection prevents reprocessing
- Metadata tracking for reproducibility

### 3. **Data Augmentation**
- Reduces overfitting
- Increases model generalization
- Applied only during training (not validation/test)

### 4. **Training Optimization**
- Batch normalization for stable training
- Dropout for regularization
- Learning rate scheduling
- Early stopping via best model checkpointing

### 5. **Evaluation & Monitoring**
- Separate train/val/test sets
- Training history tracking
- Visualization of training curves
- Final test set evaluation

---

## File Structure

```
CTSCAN-pre-processing/
├── preprocess_ct_scans.py      # Stage 1: Preprocessing
├── data_loader.py              # Stage 2: Data loading
├── example_training.py         # Stage 3: Model training
├── verify_preprocessing.py     # Verification utility
├── run_preprocessing.py        # Helper script
├── requirements.txt            # Dependencies
├── Lung Cancer Dataset/        # Raw input data
│   ├── adenocarcinoma/
│   ├── Benign cases/
│   ├── large cell carcinoma/
│   ├── Normal cases/
│   └── squamous cell carcinoma/
└── processed_data/             # Preprocessed output
    ├── train/
    ├── val/
    ├── test/
    └── metadata.json
```

---

## Usage Workflow

### Step 1: Preprocess Data
```bash
python preprocess_ct_scans.py
```

### Step 2: Verify Preprocessing (Optional)
```bash
python verify_preprocessing.py
```

### Step 3: Train Model
```bash
python example_training.py
```

### Output Files
- `processed_data/` - Preprocessed images
- `best_model.pth` - Trained model weights
- `training_history.png` - Training curves visualization

---

## Pipeline Characteristics

### Data Flow
- **Input**: ~1,535 raw CT scan images
- **After Preprocessing**: 1,535 normalized, resized images
- **Split**: 1,074 train / 230 validation / 231 test
- **Output**: Trained CNN model + evaluation metrics

### Performance Features
- **GPU Support**: Automatic CUDA detection
- **Batch Processing**: Efficient data loading
- **Memory Efficient**: Processes images in batches
- **Resumable**: Can skip existing preprocessed files

### Quality Assurance
- **Stratified Splitting**: Maintains class distribution
- **Validation Set**: Prevents overfitting
- **Test Set**: Unbiased final evaluation
- **Model Checkpointing**: Saves best model automatically

---

## Summary

This pipeline provides an end-to-end solution for CT scan classification:
1. **Preprocessing** standardizes and prepares images
2. **Data Loading** provides efficient, augmented batches
3. **Training** builds and optimizes a CNN model

The pipeline is designed to be robust, efficient, and suitable for medical imaging applications with proper validation and evaluation practices.

