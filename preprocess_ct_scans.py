"""
CT Scan Preprocessing Pipeline for 2D CNN Classification
This script preprocesses CT scan images for lung cancer classification.
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
import json


class CTScanPreprocessor:
    """
    Preprocessor for CT scan images to prepare them for CNN training.
    """
    
    def __init__(self, 
                 input_dir="Lung Cancer Dataset",
                 output_dir="processed_data",
                 img_size=(224, 224),
                 normalize=True,
                 grayscale=False,
                 skip_existing=True,
                 overwrite=False):
        """
        Initialize the preprocessor.
        
        Args:
            input_dir: Path to the input dataset directory
            output_dir: Path to save processed images
            img_size: Target image size (width, height)
            normalize: Whether to normalize pixel values to [0, 1]
            grayscale: Whether to convert to grayscale (default: RGB)
            skip_existing: If True, skip images that already exist in output (default: True)
            overwrite: If True, overwrite existing files (takes precedence over skip_existing)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.normalize = normalize
        self.grayscale = grayscale
        self.skip_existing = skip_existing and not overwrite
        self.overwrite = overwrite
        
        # Class mapping
        self.class_mapping = {
            'adenocarcinoma': 0,
            'Benign cases': 1,
            'large cell carcinoma': 2,
            'Normal cases': 3,
            'squamous cell carcinoma': 4
        }
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(exist_ok=True)
            for class_name in self.class_mapping.keys():
                (self.output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    def load_image(self, image_path):
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image using PIL (handles both JPG and PNG)
            img = Image.open(image_path)
            
            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Convert to grayscale if requested
            if self.grayscale:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)  # Convert back to 3 channels
            
            # Resize image
            img_array = cv2.resize(img_array, self.img_size, interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values to [0, 1] if requested
            if self.normalize:
                img_array = img_array.astype(np.float32) / 255.0
            else:
                img_array = img_array.astype(np.uint8)
            
            return img_array
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def get_image_stats(self):
        """
        Analyze the dataset to get statistics about image sizes.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_images': 0,
            'class_counts': {},
            'image_formats': {},
            'image_sizes': []
        }
        
        for class_name in self.class_mapping.keys():
            class_dir = self.input_dir / class_name
            if not class_dir.exists():
                continue
            
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            stats['class_counts'][class_name] = len(images)
            stats['total_images'] += len(images)
            
            # Sample some images to get size statistics
            for img_path in images[:10]:  # Sample first 10
                try:
                    img = Image.open(img_path)
                    stats['image_formats'][img.mode] = stats['image_formats'].get(img.mode, 0) + 1
                    stats['image_sizes'].append(img.size)
                except:
                    pass
        
        return stats
    
    def preprocess_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """
        Preprocess the entire dataset and split into train/val/test sets.
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            random_seed: Random seed for reproducibility
        """
        # Verify ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        print("=" * 60)
        print("CT Scan Preprocessing Pipeline")
        print("=" * 60)
        
        # Get dataset statistics
        print("\nAnalyzing dataset...")
        stats = self.get_image_stats()
        print(f"Total images: {stats['total_images']}")
        for class_name, count in stats['class_counts'].items():
            print(f"  {class_name}: {count} images")
        
        # Process each class
        all_files = []
        all_labels = []
        
        for class_name in self.class_mapping.keys():
            class_dir = self.input_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist, skipping...")
                continue
            
            # Get all images (JPG and PNG)
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for img_path in images:
                all_files.append((img_path, class_name))
                all_labels.append(class_name)
        
        # Split data
        print(f"\nSplitting dataset (train={train_ratio}, val={val_ratio}, test={test_ratio})...")
        
        # First split: train vs (val + test)
        train_files, temp_files = train_test_split(
            all_files, 
            test_size=(val_ratio + test_ratio), 
            random_state=random_seed,
            stratify=all_labels
        )
        
        # Second split: val vs test
        temp_labels = [label for _, label in temp_files]
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            random_state=random_seed,
            stratify=temp_labels
        )
        
        print(f"Train: {len(train_files)} images")
        print(f"Validation: {len(val_files)} images")
        print(f"Test: {len(test_files)} images")
        
        # Process and save images
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        for split_name, files in splits.items():
            print(f"\nProcessing {split_name} set...")
            for img_path, class_name in tqdm(files, desc=f"Processing {split_name}"):
                # Check if output file already exists
                output_path = self.output_dir / split_name / class_name / img_path.name
                
                if self.skip_existing and output_path.exists():
                    skipped_count += 1
                    continue
                
                # Load and preprocess image
                processed_img = self.load_image(img_path)
                
                if processed_img is None:
                    failed_count += 1
                    continue
                
                if self.normalize:
                    # Convert back to uint8 for saving
                    img_to_save = (processed_img * 255).astype(np.uint8)
                else:
                    img_to_save = processed_img
                
                # Save as PNG (supports both normalized and non-normalized)
                Image.fromarray(img_to_save).save(output_path)
                processed_count += 1
        
        print(f"\n{'='*60}")
        print(f"Preprocessing complete!")
        print(f"Successfully processed: {processed_count} images")
        if skipped_count > 0:
            print(f"Skipped (already exist): {skipped_count} images")
        print(f"Failed: {failed_count} images")
        print(f"Processed images saved to: {self.output_dir}")
        print(f"{'='*60}")
        
        # Save metadata
        metadata = {
            'img_size': self.img_size,
            'normalize': self.normalize,
            'grayscale': self.grayscale,
            'class_mapping': self.class_mapping,
            'train_count': len(train_files),
            'val_count': len(val_files),
            'test_count': len(test_files),
            'class_counts': stats['class_counts']
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved to: {self.output_dir / 'metadata.json'}")
    
    def augment_image(self, image):
        """
        Apply data augmentation to an image.
        Can be used during training.
        
        Args:
            image: Input image array
            
        Returns:
            Augmented image array
        """
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
            image = np.clip(image * brightness, 0, 1 if self.normalize else 255)
        
        return image


if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = CTScanPreprocessor(
        input_dir="Lung Cancer Dataset",
        output_dir="processed_data",
        img_size=(224, 224),  # Standard size for many CNN architectures
        normalize=True,  # Normalize to [0, 1]
        grayscale=False  # Keep RGB for better feature learning
    )
    
    # Preprocess the dataset
    preprocessor.preprocess_dataset(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

