# Start of Selection
"""
Data Loader for CT Scan CNN Training
Provides utilities for loading and batching preprocessed CT scan images.
"""

# ---- Imports ----
import os                                        # Import os: Interface to interact with the operating system, not actually used below but often kept for extension or directory/filework.
import numpy as np                               # Import numpy: Fast multidimensional arrays, used for image manipulation and augmentations.
from pathlib import Path                         # Import Path from pathlib: Convenient and robust path manipulations and filesystem navigation.
from PIL import Image                            # Import Image from PIL: To open and manipulate images.
import json                                      # Import json: For reading metadata and configuration from .json files.
from torch.utils.data import Dataset, DataLoader  # Import PyTorch Dataset and DataLoader utilities for custom datasets and batch data loading.
from torchvision import transforms               # Import torchvision transforms: To use optional image transformations; some users may plug these in.
import torch                                     # Import torch: The main PyTorch package for all tensor and DL framework operations.


# ---- Custom Dataset for CT Scans ----
class CTScanDataset(Dataset):
    """
    PyTorch Dataset class for CT scan images.
    - Handles a single data split ('train', 'val', or 'test').
    - Loads images, normalizes, applies augmentation, and provides labels.
    """

    def __init__(self, data_dir, split='train', transform=None, augment=False):
        """
        Initialize the dataset by gathering all image paths and storing class mappings.
        
        Args:
            data_dir: String or Path. Root directory containing preprocessed data and metadata.json.
            split: String. Indicates which split to use: 'train', 'val', or 'test'.
            transform: Optional. A torchvision transform function or pipeline for images.
            augment: Boolean. If True, enables augmentation (only has effect on training split).
        """

        self.data_dir = Path(data_dir) / split                  # Compose complete path to the current split, e.g., 'processed_data/train'
        self.split = split                                      # Store the split name for debugging or reference.
        self.transform = transform                              # Store optional transform for use later (e.g., additional normalization).
        self.augment = augment and (split == 'train')           # Only allow augmentation if augment=True and split is train.

        # -- Read metadata, in particular the mapping from class names to indices --
        metadata_path = Path(data_dir) / 'metadata.json'        # Set location of metadata file (processed_data/metadata.json)
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)                    # Parse JSON metadata into a Python dictionary.
            self.class_mapping = self.metadata['class_mapping'] # Load class mapping: {class name: index} for all classes found during preprocessing.
        else:
            # If metadata file missing, use the canonical hardcoded class mapping.
            self.class_mapping = {
                'adenocarcinoma': 0,
                'Benign cases': 1,
                'large cell carcinoma': 2,
                'Normal cases': 3,
                'squamous cell carcinoma': 4
            }
            self.metadata = {}                                  # Provide an empty dict for later use, in case code expects it.

        # -- Create reverse mapping so that from a class index you can recover the class name --
        self.label_to_class = {v: k for k, v in self.class_mapping.items()}

        # -- Gather image file paths and assign labels (class indices) --
        self.images = []                                        # Will hold all pathlib.Path objects to image files for this split
        self.labels = []                                        # Will hold the corresponding integer label for each image
        for class_name, label in self.class_mapping.items():
            class_dir = self.data_dir / class_name              # Subdirectory for a particular class, e.g., 'processed_data/train/adenocarcinoma'
            if class_dir.exists():
                image_files = list(class_dir.glob('*.png'))     # Get a list of all PNG files in the class folder
                image_files += list(class_dir.glob('*.jpg'))    # Also get a list of all JPG files (extension-insensitive)
                for img_path in image_files:
                    self.images.append(img_path)                # Store the full file path to image
                    self.labels.append(label)                   # Store the integer class label, same index as in self.images

        print(f"Loaded {len(self.images)} images from {split} split")   # Feedback: number of images loaded for this split

    def __len__(self):
        """
        Return length (number of samples) in the dataset.
        
        Returns:
            int: total number of (image,label) pairs (length of self.images)
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Loads a sample from the dataset at the provided index.

        Args:
            idx: int. The index of the sample as used by DataLoader.
        
        Returns:
            (torch.Tensor, int): the preprocessed image (CHW, float32) and its class label
        """
        img_path = self.images[idx]           # Find the path of the requested sample.
        label = self.labels[idx]              # Get the label (class number) associated with this image.
        try:
            image = Image.open(img_path).convert('RGB')     # Open image, converting any format to RGB.
            image = np.array(image)                         # Convert PIL Image to a NumPy ndarray (H,W,3), dtype uint8 in [0,255].
            # -- Normalize pixel values to range [0, 1], based on metadata if present --
            if self.metadata.get('normalize', True):        # If 'normalize' key in metadata, and is True...
                if image.max() > 1.0:                       # If the maximum value is > 1, then rescale from [0,255] float.
                    image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32) / 255.0    # Legacy or fallback: always convert to [0,1] floats.
            # -- Perform augmentation if enabled (for training) --
            if self.augment:
                image = self._augment_image(image)           # Apply augmentation to image as a NumPy array.
            # -- Rearrange to PyTorch format (channels, height, width) and convert to Tensor --
            image = torch.from_numpy(image).permute(2, 0, 1).float()   # Change [H,W,C] to [C,H,W], and convert to float32 tensor
            # -- Apply user-provided transforms (e.g., additional normalization, color jitter, etc.), if any --
            if self.transform:
                image = self.transform(image)
            return image, label                             # Return the prepared image tensor and its class index.
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")   # Warn if opening, reading, or processing the image failed
            img_size = self.metadata.get('img_size', (224, 224))  # Try to get size from metadata, else default to 224x224
            # Return a black image of appropriate shape as a placeholder for missing/corrupt files
            image = torch.zeros((3, img_size[1], img_size[0]), dtype=torch.float32)
            return image, label

    def _augment_image(self, image):
        """
        Applies random augmentation operations (flip, rotation, brightness, contrast).
        Should be called only on the training split.

        Args:
            image: NumPy array with shape (H, W, C), values in [0,1] and type float32.

        Returns:
            Augmented image as numpy array, same shape/type.
        """
        import cv2                                             # Import cv2 (OpenCV) here, since it's only needed for augmentation operations
        # -- Horizontal flip with 50% probability --
        if np.random.random() > 0.5:
            image = np.fliplr(image)                            # Flip left-right
        # -- Small random rotation with 50% probability --
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)                  # Choose a random angle between -15° and +15°
            h, w = image.shape[:2]                              # Compute center (column, row)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # -- Random brightness change with 50% probability --
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 1)           # Scale all pixels, clip to [0,1]
        # -- Random contrast change with 50% probability --
        if np.random.random() > 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * contrast + mean, 0, 1)
        return image                                            # Return the possibly augmented image

    def get_class_weights(self):
        """
        Compute and return inverse-frequency weights for each class.
        This is useful for loss weighting in case of class imbalance.
        
        Returns:
            dict: {int label: float class_weight}
        """
        from collections import Counter                    # Import Counter to count occurrences
        label_counts = Counter(self.labels)                # Count how many samples are in each class
        total = len(self.labels)                           # Total number of images in this dataset split
        # Inverse-frequency weighting: each class gets (total/N_classes) / count
        class_weights = {}
        for label, count in label_counts.items():
            class_weights[label] = total / (len(label_counts) * count)
        return class_weights


# ---- Utility to Build Data Loaders for All Splits ----
def create_data_loaders(data_dir, batch_size=32, num_workers=0, augment=True):
    """
    Factory function to create PyTorch DataLoader objects for all split sets.

    Args:
        data_dir: Path to the processed data root directory.
        batch_size: Number of images to load per batch.
        num_workers: How many loader worker processes to use (0 = load in main thread).
        augment: If True, enables random cropping, flipping, and other augmentations for *training* data only.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
        - train_loader: DataLoader over CTScanDataset('train', augment=augment)
        - val_loader:   DataLoader over CTScanDataset('val',   augment=False)
        - test_loader:  DataLoader over CTScanDataset('test',  augment=False)
        - class_names:  List of class names (sorted by integer label)
    """
    # Instantiate dataset for each split
    train_dataset = CTScanDataset(data_dir, split='train', augment=augment)
    val_dataset   = CTScanDataset(data_dir, split='val',   augment=False)
    test_dataset  = CTScanDataset(data_dir, split='test',  augment=False)
    # Instantiate corresponding data loader objects for each split
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for better SGD convergence
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False  # Speed up data transfer to GPU if available
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation/testing: ensures reproducibility
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    # Compose list of class names (in order of integer label: 0, 1, 2, ...)
    class_names = [train_dataset.label_to_class[i] for i in range(len(train_dataset.class_mapping))]
    return train_loader, val_loader, test_loader, class_names


# ---- If Run Directly: Simple Example Use Case and Print Shapes/Counts ----
if __name__ == "__main__":
    # Specify folder containing all processed data (train/val/test and metadata.json)
    data_dir = "processed_data"
    # Build all three data loaders and get the class name list
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=data_dir,
        batch_size=32,    # Load 32 images at a time in each batch
        num_workers=0,    # Load with the main process only for simplicity
        augment=True      # Enable augmentation for the training split dataset
    )
    # Print summary of dataset splits and classes to verify loading works as expected
    print(f"\nClass names: {class_names}")               # Print list like ['adenocarcinoma', ..., ...]
    print(f"Train batches: {len(train_loader)}")         # Number of batches in train split
    print(f"Val batches: {len(val_loader)}")             # Number of batches in validation split
    print(f"Test batches: {len(test_loader)}")           # Number of batches in test split
    # Draw one batch from the training set and print basic properties
    sample_batch = next(iter(train_loader))      # Get first batch from the train loader iterator (images, labels)
    images, labels = sample_batch
    print(f"\nSample batch shape: {images.shape}")       # Should be [batch_size, 3, H, W], e.g., (32,3,224,224)
    print(f"Sample labels: {labels[:5]}")                # Print first 5 labels to check correctness

# End of Selection
