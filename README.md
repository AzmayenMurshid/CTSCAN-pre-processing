# CT Scan Preprocessing Pipeline for Lung Cancer Classification

This project provides a comprehensive preprocessing pipeline for CT scan images to prepare them for 2D CNN-based lung cancer classification.

## Dataset Structure

The dataset is organized into the following classes:
- **adenocarcinoma** (337 PNG images)
- **Benign cases** (120 JPG images)
- **large cell carcinoma** (187 PNG images)
- **Normal cases** (631 images: 428 JPG, 203 PNG)
- **squamous cell carcinoma** (260 PNG images)

## Features

- **Handles mixed image formats**: Automatically processes both JPG and PNG images
- **RGBA to RGB conversion**: Converts RGBA images to RGB with proper background handling
- **Image normalization**: Normalizes pixel values to [0, 1] range
- **Consistent image sizing**: Resizes all images to a standard size (default: 224x224)
- **Train/Val/Test split**: Automatically splits data with stratification
- **Data augmentation**: Built-in augmentation support for training
- **Class imbalance handling**: Utilities for calculating class weights

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Preprocess the Dataset

Run the preprocessing script to process all CT scan images:

```bash
python preprocess_ct_scans.py
```

This will:
- Analyze the dataset structure
- Load and preprocess all images (handle mixed formats, normalize, resize)
- Split data into train (70%), validation (15%), and test (15%) sets
- Save processed images to `processed_data/` directory
- Generate metadata JSON file with preprocessing information

### Step 2: Use Data Loaders for Training

In your training script, use the data loader utilities:

```python
from data_loader import create_data_loaders

# Create data loaders
train_loader, val_loader, test_loader, class_names = create_data_loaders(
    data_dir="processed_data",
    batch_size=32,
    num_workers=4,
    augment=True
)

# Use in training loop
for images, labels in train_loader:
    # Your training code here
    pass
```

### Customization

You can customize the preprocessing by modifying the `CTScanPreprocessor` initialization:

```python
from preprocess_ct_scans import CTScanPreprocessor

preprocessor = CTScanPreprocessor(
    input_dir="Lung Cancer Dataset",
    output_dir="processed_data",
    img_size=(224, 224),  # Change image size
    normalize=True,        # Normalize to [0, 1]
    grayscale=False        # Keep RGB (False) or convert to grayscale (True)
)

preprocessor.preprocess_dataset(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)
```

## Preprocessing Details

### Image Processing Steps

1. **Format Handling**: Automatically detects and handles JPG and PNG formats
2. **RGBA Conversion**: Converts RGBA images to RGB by compositing on white background
3. **Resizing**: Resizes all images to target size (default: 224x224) using area interpolation
4. **Normalization**: Normalizes pixel values to [0, 1] range (float32)
5. **Format Consistency**: Saves all processed images as PNG

### Data Augmentation

The pipeline includes the following augmentation techniques (applied during training):
- Random horizontal flip
- Random rotation (±15 degrees)
- Random brightness adjustment (0.8-1.2x)
- Random contrast adjustment (0.8-1.2x)

### Class Weights

To handle class imbalance, you can calculate class weights:

```python
from data_loader import CTScanDataset

train_dataset = CTScanDataset("processed_data", split='train')
class_weights = train_dataset.get_class_weights()
```

## Output Structure

After preprocessing, the directory structure will be:

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

## Metadata

The `metadata.json` file contains:
- Image size used
- Normalization settings
- Class mapping
- Split counts
- Original class counts

## Example CNN Model

Here's a simple example of how to use the preprocessed data with a CNN:

```python
import torch
import torch.nn as nn
from data_loader import create_data_loaders

# Create data loaders
train_loader, val_loader, test_loader, class_names = create_data_loaders(
    data_dir="processed_data",
    batch_size=32
)

# Simple CNN model
class LungCancerCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize model
model = LungCancerCNN(num_classes=len(class_names))

# Training loop example
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## Notes

- The preprocessing handles class imbalance by using stratified splits
- All images are normalized to [0, 1] range for better training stability
- Default image size is 224x224 (compatible with many pre-trained models)
- Data augmentation is only applied during training, not validation/test

## Troubleshooting

1. **Memory issues**: Reduce batch size or image size
2. **Slow preprocessing**: Reduce image size or process in batches
3. **Mixed formats**: The pipeline automatically handles this, no action needed

## License

This preprocessing pipeline is provided as-is for research and educational purposes.
