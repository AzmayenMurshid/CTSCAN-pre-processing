"""
Data Loader for CT Scan CNN Training
Provides utilities for loading and batching preprocessed CT scan images.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class CTScanDataset(Dataset):
    """
    PyTorch Dataset class for CT scan images.
    """
    
    def __init__(self, data_dir, split='train', transform=None, augment=False):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to processed data directory
            split: 'train', 'val', or 'test'
            transform: Optional torchvision transforms
            augment: Whether to apply augmentation during training
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform
        self.augment = augment and (split == 'train')
        
        # Load metadata
        metadata_path = Path(data_dir) / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.class_mapping = self.metadata['class_mapping']
        else:
            # Default class mapping
            self.class_mapping = {
                'adenocarcinoma': 0,
                'Benign cases': 1,
                'large cell carcinoma': 2,
                'Normal cases': 3,
                'squamous cell carcinoma': 4
            }
        
        # Reverse mapping for labels to class names
        self.label_to_class = {v: k for k, v in self.class_mapping.items()}
        
        # Load all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name, label in self.class_mapping.items():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                for img_path in image_files:
                    self.images.append(img_path)
                    self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images from {split} split")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a single image and label.
        
        Args:
            idx: Index of the image
            
        Returns:
            tuple: (image tensor, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Convert to numpy array
            image = np.array(image)
            
            # Normalize if metadata says so (or if image is in [0, 255] range)
            if self.metadata.get('normalize', True):
                if image.max() > 1.0:
                    image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32) / 255.0
            
            # Apply augmentation if training
            if self.augment:
                image = self._augment_image(image)
            
            # Convert to tensor and change to CHW format
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            img_size = self.metadata.get('img_size', (224, 224))
            image = torch.zeros((3, img_size[1], img_size[0]), dtype=torch.float32)
            return image, label
    
    def _augment_image(self, image):
        """
        Apply random augmentation to image.
        
        Args:
            image: Input image array (H, W, C) in [0, 1]
            
        Returns:
            Augmented image array
        """
        import cv2
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        
        # Random rotation (small angle)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 1)
        
        # Random contrast adjustment
        if np.random.random() > 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * contrast + mean, 0, 1)
        
        return image
    
    def get_class_weights(self):
        """
        Calculate class weights for handling class imbalance.
        
        Returns:
            Dictionary with class weights
        """
        from collections import Counter
        
        label_counts = Counter(self.labels)
        total = len(self.labels)
        
        # Calculate inverse frequency weights
        class_weights = {}
        for label, count in label_counts.items():
            class_weights[label] = total / (len(label_counts) * count)
        
        return class_weights


def create_data_loaders(data_dir, batch_size=32, num_workers=0, augment=True):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        augment: Whether to apply augmentation to training set
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    # Create datasets
    train_dataset = CTScanDataset(data_dir, split='train', augment=augment)
    val_dataset = CTScanDataset(data_dir, split='val', augment=False)
    test_dataset = CTScanDataset(data_dir, split='test', augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    
    # Get class names
    class_names = [train_dataset.label_to_class[i] for i in range(len(train_dataset.class_mapping))]
    
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Example usage
    data_dir = "processed_data"
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=0,
        augment=True
    )
    
    print(f"\nClass names: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a sample batch
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch
    print(f"\nSample batch shape: {images.shape}")
    print(f"Sample labels: {labels[:5]}")

