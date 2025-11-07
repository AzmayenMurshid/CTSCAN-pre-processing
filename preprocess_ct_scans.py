"""
CT Scan Preprocessing Pipeline for 2D CNN Classification
This script preprocesses CT scan images for lung cancer classification.
"""

# Import standard and third-party libraries
import os                                     # For operating system functionalities (file handling)
import cv2                                    # OpenCV for image processing (reading, resizing, augmentation)
import numpy as np                            # For numerical operations on arrays
from PIL import Image                         # PIL for handling image loading/saving, especially various formats
from pathlib import Path                      # For convenient and reliable path manipulations
from tqdm import tqdm                         # For displaying progress bars during processing
import shutil                                 # For file operations like copying/removing (not used in this core script)
from sklearn.model_selection import train_test_split # For splitting dataset into train/val/test
import json                                   # For saving/loading metadata as JSON


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
        Initialize the preprocessor with configurable options.

        Args:
            input_dir: Path to the input dataset directory
            output_dir: Path to save processed images
            img_size: Target image size (width, height)
            normalize: Whether to normalize pixel values to [0, 1]
            grayscale: Whether to convert to grayscale (default: RGB)
            skip_existing: If True, skip images that already exist in output (default: True)
            overwrite: If True, overwrite existing files (takes precedence over skip_existing)
        """
        self.input_dir = Path(input_dir)                        # Store input directory as Path object
        self.output_dir = Path(output_dir)                      # Store output directory as Path object
        self.img_size = img_size                                # Target size for image resizing
        self.normalize = normalize                              # Should pixel values be normalized to [0, 1]
        self.grayscale = grayscale                              # Should images be converted to grayscale
        self.skip_existing = skip_existing and not overwrite    # Skip existing images if not overwriting
        self.overwrite = overwrite                              # Whether to overwrite existing files

        # Map class names to integer indices for label consistency
        self.class_mapping = {
            'adenocarcinoma': 0,
            'Benign cases': 1,
            'large cell carcinoma': 2,
            'Normal cases': 3,
            'squamous cell carcinoma': 4
        }

        # Create output directories for each dataset split and class
        self.output_dir.mkdir(exist_ok=True)                    # Make sure main output dir exists
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(exist_ok=True)      # Make directory for each split
            for class_name in self.class_mapping.keys():
                (self.output_dir / split / class_name).mkdir(parents=True, exist_ok=True) # ...and for each class

    def load_image(self, image_path):
        """
        Load and preprocess a single image file.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image array OR None if an error occurs
        """
        try:
            # Load image using PIL which supports both JPG and PNG
            img = Image.open(image_path)

            # If image has an alpha channel (RGBA), convert it to RGB on a white background
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))      # Create white background
                background.paste(img, mask=img.split()[3])                    # Paste original using alpha for transparency
                img = background                                              # Replace with merged RGB image
            elif img.mode != 'RGB':
                img = img.convert('RGB')                                      # Convert other modes (e.g., L) to RGB

            # Convert PIL Image to NumPy array for OpenCV processing
            img_array = np.array(img)

            # If grayscale processing is requested
            if self.grayscale:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)       # Convert to single channel grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)       # Then back to 3 channels for consistency

            # Resize the image using OpenCV
            img_array = cv2.resize(img_array, self.img_size, interpolation=cv2.INTER_AREA)

            # Normalize image if specified (scale pixel values to [0, 1])
            if self.normalize:
                img_array = img_array.astype(np.float32) / 255.0
            else:
                img_array = img_array.astype(np.uint8)

            return img_array                                                  # Return the processed image

        except Exception as e:
            # Print error message if image fails to process
            print(f"Error loading image {image_path}: {e}")
            return None

    def get_image_stats(self):
        """
        Analyze input dataset: collect number of images, class distribution, formats, and image sizes.

        Returns:
            Dictionary with dataset statistics including count, formats, sizes.
        """
        stats = {
            'total_images': 0,        # Total image count
            'class_counts': {},       # Image count per class
            'image_formats': {},      # Counts for unique formats (e.g., L, RGB, RGBA)
            'image_sizes': []         # Store sampled image sizes
        }

        # For each class folder
        for class_name in self.class_mapping.keys():
            class_dir = self.input_dir / class_name            # Path to this class
            if not class_dir.exists():
                continue                                       # Skip if folder not found

            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))   # Find all .jpg/.png
            stats['class_counts'][class_name] = len(images)                           # Count for this class
            stats['total_images'] += len(images)                                      # Add to total

            # Sample up to first 10 images to get format/size info (avoid loading whole dataset)
            for img_path in images[:10]:
                try:
                    img = Image.open(img_path)
                    # Increment occurrence of this image mode (e.g., RGB, RGBA, etc.)
                    stats['image_formats'][img.mode] = stats['image_formats'].get(img.mode, 0) + 1
                    stats['image_sizes'].append(img.size)            # Store dimensions (width, height)
                except:
                    pass

        return stats                            # Return statistics dictionary

    def preprocess_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """
        Preprocess the entire dataset and stratified split into train/val/test sets.

        Args:
            train_ratio: Ratio of training data (default 0.7)
            val_ratio: Ratio of validation data (default 0.15)
            test_ratio: Ratio of test data (default 0.15)
            random_seed: Random seed for reproducibility in splitting
        """
        # Make sure split ratios sum up to 1.0 (floating point tolerance allowed)
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        # Print pipeline title/header
        print("=" * 60)
        print("CT Scan Preprocessing Pipeline")
        print("=" * 60)

        # Analyze and report input dataset stats
        print("\nAnalyzing dataset...")
        stats = self.get_image_stats()
        print(f"Total images: {stats['total_images']}")
        for class_name, count in stats['class_counts'].items():
            print(f"  {class_name}: {count} images")

        # Collect all image file paths and class names
        all_files = []                              # Will contain tuples: (img_path, class_name)
        all_labels = []                             # List of class names (for stratification)

        # Walk through each class
        for class_name in self.class_mapping.keys():
            class_dir = self.input_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist, skipping...")
                continue                            # If no directory, skip

            # Gather all .jpg and .png files in this class folder
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

            for img_path in images:
                all_files.append((img_path, class_name))     # Store file and its class
                all_labels.append(class_name)                # Store class for stratified split

        # Inform user and split into train/val/test
        print(f"\nSplitting dataset (train={train_ratio}, val={val_ratio}, test={test_ratio})...")

        # First, split files into train and (val+test) sets, stratified to preserve class distribution
        train_files, temp_files = train_test_split(
            all_files,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            stratify=all_labels
        )

        # Prepare labels for remaining temp set, for second stratified split
        temp_labels = [label for _, label in temp_files]
        # Now, split temp_files into validation and test sets accordingly
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            random_state=random_seed,
            stratify=temp_labels
        )

        # Print dataset sizes for each split
        print(f"Train: {len(train_files)} images")
        print(f"Validation: {len(val_files)} images")
        print(f"Test: {len(test_files)} images")

        # Organize file splits for easier processing
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        processed_count = 0         # Counter for successfully processed images
        failed_count = 0            # Count images that failed processing
        skipped_count = 0           # Count skipped due to pre-existing file

        # Loop through each split and its files
        for split_name, files in splits.items():
            print(f"\nProcessing {split_name} set...")
            # Progress bar for each split using tqdm
            for img_path, class_name in tqdm(files, desc=f"Processing {split_name}"):
                # Build target output path for processed image
                output_path = self.output_dir / split_name / class_name / img_path.name

                # If image already processed and skipping is enabled, skip it
                if self.skip_existing and output_path.exists():
                    skipped_count += 1
                    continue

                # Load and process the image
                processed_img = self.load_image(img_path)

                # If failed, move to next
                if processed_img is None:
                    failed_count += 1
                    continue

                # If normalization was performed, convert to uint8 (0-255) for saving
                if self.normalize:
                    img_to_save = (processed_img * 255).astype(np.uint8)
                else:
                    img_to_save = processed_img

                # Save processed image as PNG in output directory
                Image.fromarray(img_to_save).save(output_path)
                processed_count += 1

        # Print processing summary
        print(f"\n{'='*60}")
        print(f"Preprocessing complete!")
        print(f"Successfully processed: {processed_count} images")
        if skipped_count > 0:
            print(f"Skipped (already exist): {skipped_count} images")
        print(f"Failed: {failed_count} images")
        print(f"Processed images saved to: {self.output_dir}")
        print(f"{'='*60}")

        # Prepare metadata describing the processing and dataset,
        # and include split counts per class in metadata.json

        # Collect split class counts: count of each class in each split
        # Structure: {split: {class: count}} - e.g., {'train': {'adenocarcinoma': 50, ...}, ...}
        split_class_counts = {split: {k: 0 for k in self.class_mapping.keys()} for split in ['train', 'val', 'test']} # Dictionary to store split counts per class
        for split_name, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]): # For each split, count images per class
            for img_path, class_name in files:                                # For each image in the split
                if class_name in split_class_counts[split_name]:               # If the class name is in the split counts
                    split_class_counts[split_name][class_name] += 1            # Increment the count for this class in this split

        # Transform split_class_counts to per-class split distribution (transpose the structure)
        # Structure: {class: {split: count}} - e.g., {'adenocarcinoma': {'train': 50, 'val': 10, ...}, ...}
        per_class_split_distribution = {
            class_name: {                                         # For each class
                split_name: split_class_counts[split_name][class_name] # For each split, get the count for this class
                for split_name in ['train', 'val', 'test']            # For each split
            }                                                      # For each class, get the counts for each split
            for class_name in self.class_mapping.keys()              # For each class
        }

        # Now build metadata with all required info including both split_class_counts and per_class_split_distribution
        metadata = {
            'img_size': self.img_size,                              # Image size used for preprocessing (tuple, e.g. (224, 224))
            'normalize': self.normalize,                            # Whether normalization to [0, 1] was performed (bool)
            'grayscale': self.grayscale,                            # Whether images were converted to grayscale (bool)
            'class_mapping': self.class_mapping,                    # Mapping from class names to integer labels
            'train_count': len(train_files),                        # Total number of images in the training split
            'val_count': len(val_files),                            # Total number of images in the validation split
            'test_count': len(test_files),                          # Total number of images in the test split
            'class_counts': stats['class_counts'],                  # Number of images per class across the dataset (pre-split)
            'split_class_counts': split_class_counts,               # Counts of each class in each split
            'per_class_split_distribution': per_class_split_distribution  # For each class, distribution among train/val/test
        }

        # Save metadata to output directory as JSON
        with open(self.output_dir / 'metadata.json', 'w') as f:    # Open the metadata file for writing
            json.dump(metadata, f, indent=2)                       # Write the metadata to the file

        print(f"\nMetadata saved to: {self.output_dir / 'metadata.json'}") # Print the path to the metadata file

    def augment_image(self, image):             # Function to apply data augmentation to an image.
                                                # Augmentation is used to increase the diversity of the dataset and improve the performance of the model.
        """
        Apply data augmentation to an image.
        Can be used during training.

        Args:
            image: Input image array

        Returns:
            Augmented image array
        """
        # Randomly flip image horizontally with 50% probability
        if np.random.random() > 0.5:
            image = np.fliplr(image)

        # Randomly rotate image within [-15, +15] degrees, with 50% probability
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)                        # Random angle
            h, w = image.shape[:2]                                    # Image size
            center = (w // 2, h // 2)                                 # Image center
            M = cv2.getRotationMatrix2D(center, angle, 1.0)           # Rotation matrix
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)  # Rotate with border reflection

        # Randomly adjust brightness with 50% probability
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)                  # Brighten/darken
            image = np.clip(image * brightness, 0, 1 if self.normalize else 255) # Keep pixel values valid

        return image


if __name__ == "__main__":
    # If this script is run directly (not imported), start preprocessing

    # Initialize the preprocessor object with configuration options
    preprocessor = CTScanPreprocessor(
        input_dir="Lung Cancer Dataset",
        output_dir="processed_data",
        img_size=(224, 224),      # Standard size for many CNN architectures
        normalize=True,           # Normalize pixel values to [0, 1]
        grayscale=False,          # Use RGB (not grayscale) for better learning
        skip_existing=False       # Process all images, don't skip existing ones
    )

    # Launch the full preprocessing pipeline, with dataset split ratios and random seed
    preprocessor.preprocess_dataset(
        train_ratio=0.7,          # 70% train
        val_ratio=0.15,           # 15% val
        test_ratio=0.15,          # 15% test
        random_seed=42            # For reproducibility
    )

