"""
CT Scan Dataset Distribution Visualizer
Creates comprehensive visualizations for CT scan dataset analysis.
"""

import json                     # Import json to read dataset metadata if present
import matplotlib.pyplot as plt # Import matplotlib for plotting charts and figures
import seaborn as sns           # Import seaborn for pretty plotting styles and color palettes
from pathlib import Path        # Import Path for cross-platform directory and file handling
from collections import Counter # Import Counter for counting objects (not strictly needed)
import numpy as np              # Import numpy for numeric operations and arrays
from PIL import Image           # Import PIL for opening and handling image files
import os                      # Import os for interacting with the filesystem

# Set the plotting style to whitegrid using seaborn
sns.set_style("whitegrid")
# Set default figure size for plots
plt.rcParams['figure.figsize'] = (12, 6)
# Set default font size for plots
plt.rcParams['font.size'] = 10

class CTScanVisualizer:
    """Visualizer for CT scan dataset distributions."""

    def __init__(self, data_dir="Lung Cancer Dataset", processed_dir="processed_data"):
        """
        Initialize the visualizer.

        Args:
            data_dir: Path to original dataset directory
            processed_dir: Path to processed data directory
        """
        self.data_dir = Path(data_dir)                       # Store original data directory as Path object
        self.processed_dir = Path(processed_dir)             # Store processed data directory as Path object
        self.output_dir = Path("visualizations")             # Directory to save all generated visualizations
        self.output_dir.mkdir(exist_ok=True)                 # Create visualization dir if it doesn't exist

        # Load dataset metadata if metadata.json exists in processed data dir
        self.metadata = None                                 # Default to no metadata
        metadata_path = self.processed_dir / "metadata.json" # Path to metadata.json
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)                 # Read metadata into dictionary

    def get_class_distribution(self):
        """Get class distribution from original dataset."""
        class_counts = {}    # Dictionary: class name -> image count
        format_counts = {}   # Dictionary: class name -> {PNG count, JPG count}

        # Try to get class counts from metadata for processed data
        if self.metadata and 'class_counts' in self.metadata:
            class_counts = self.metadata['class_counts'].copy()   # Use counts from metadata if available
        else:
            # Otherwise, count manually from original data directory
            for class_dir in self.data_dir.iterdir():
                if class_dir.is_dir():
                    png_count = len(list(class_dir.glob('*.png')))    # Count PNG images in this class
                    jpg_count = len(list(class_dir.glob('*.jpg')))    # Count JPG images in this class
                    class_counts[class_dir.name] = png_count + jpg_count
                    format_counts[class_dir.name] = {'PNG': png_count, 'JPG': jpg_count}
        return class_counts, format_counts

    def get_split_distribution(self):
        """Get distribution across train/val/test splits."""
        splits = {}         # Dictionary: split name -> {class -> count}

        # Preferred: get split counts from metadata if available
        if self.metadata and 'split_class_counts' in self.metadata:
            splits = self.metadata['split_class_counts'].copy()
            return splits

        # Otherwise, count from split folders in processed data
        if not self.processed_dir.exists():
            return splits    # Return empty if processed dir missing

        # For each split (train, val, test)
        for split in ['train', 'val', 'test']:
            split_dir = self.processed_dir / split
            if not split_dir.exists():
                continue    # Skip split if not found

            class_counts = {}   # Dict: class -> count in this split
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    # Count images in this class for this split
                    count = len(list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')))
                    class_counts[class_dir.name] = count

            splits[split] = class_counts

        return splits

    def plot_class_distribution(self):
        """Plot class distribution using bar and pie charts."""
        class_counts, format_counts = self.get_class_distribution()       # Get class and format counts

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))            # Create two subplots (bar, pie)

        # Prepare data for bar chart
        classes = list(class_counts.keys())               # List of class names
        counts = list(class_counts.values())              # Image count for each class
        colors = sns.color_palette("husl", len(classes))  # Choose a distinct color for each class

        # Plot bar chart of class distribution
        bars = ax1.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax1.set_title('CT Scan Class Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Annotate bar chart bars with actual values
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')

        # Rotate x-axis labels for better readability
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Prepare data for pie chart
        total = sum(counts)
        percentages = [count/total*100 for count in counts]
        explode = [0.05] * len(classes)          # Slightly separate each class segment

        # Plot pie chart for class proportions
        wedges, texts, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%',
                                           colors=colors, explode=explode,
                                           startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

        # Format pie chart percentage labels
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir / 'class_distribution.png'}")    # Notify user of output

    def plot_format_distribution(self):
        """Plot a stacked bar chart showing PNG vs JPG distribution for each class."""
        _, format_counts = self.get_class_distribution()                # Only want format counts

        if not format_counts:
            print("⚠️  Format distribution data not available")         # Warn if no format info
            return

        fig, ax = plt.subplots(figsize=(14, 6))                        # Create plot area

        classes = list(format_counts.keys())                           # All class names
        png_counts = [format_counts[c]['PNG'] for c in classes]        # Count PNG per class
        jpg_counts = [format_counts[c]['JPG'] for c in classes]        # Count JPG per class

        x = np.arange(len(classes))                                    # Position for each class
        width = 0.35                                                   # Width of each bar

        # Plot PNG images bars
        bars1 = ax.bar(x - width/2, png_counts, width, label='PNG',
                       color='#3498db', edgecolor='black', linewidth=1)
        # Plot JPG images bars, next to PNGs
        bars2 = ax.bar(x + width/2, jpg_counts, width, label='JPG',
                       color='#e74c3c', edgecolor='black', linewidth=1)

        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_title('Image Format Distribution by Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Annotate the bars with counts
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'format_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir / 'format_distribution.png'}")    # Notify user

    def plot_split_distribution(self):
        """Plot distribution of data across train/val/test splits with stacked bar and bar chart."""
        splits = self.get_split_distribution()            # Get split information

        if not splits:
            print("⚠️  Split distribution data not available")           # Warn if missing
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Collect all unique class names from all splits
        all_classes = set()
        for split_data in splits.values():
            all_classes.update(split_data.keys())
        all_classes = sorted(all_classes)

        # Prepare split counts for plotting stacked bar
        split_names = list(splits.keys())      # List of 'train', 'val', 'test'
        split_counts = {split: [splits[split].get(cls, 0) for cls in all_classes]
                       for split in split_names}

        x = np.arange(len(all_classes))        # Bar positions
        width = 0.6                            # Width of bars
        bottom = np.zeros(len(all_classes))    # Bottom for stacking bars

        colors = {'train': '#2ecc71', 'val': '#f39c12', 'test': '#9b59b6'}    # Colors

        # Plot one bar per split (stacked)
        for split in split_names:
            counts = split_counts[split]
            ax1.bar(x, counts, width, label=split.capitalize(),
                   bottom=bottom, color=colors[split],
                   edgecolor='black', linewidth=1)
            bottom += counts                     # Stack bars

        ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax1.set_title('Train/Validation/Test Split Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_classes, rotation=45, ha='right')
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Plot bar chart with total counts per split
        total_counts = [sum(splits[split].values()) for split in split_names]
        bars = ax2.bar(split_names, total_counts, color=[colors[s] for s in split_names],
                       edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Split', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Images', fontsize=12, fontweight='bold')
        ax2.set_title('Total Images per Split', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Annotate bars with counts
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'split_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir / 'split_distribution.png'}")      # Print output

    def plot_class_balance_analysis(self):
        """Analyze and visualize class balance (imbalance) with statistical charts."""
        class_counts, _ = self.get_class_distribution()    # Get class counts

        if not class_counts:
            return    # Early exit if count info is missing

        counts = list(class_counts.values())               # Image count for each class
        classes = list(class_counts.keys())                # All class names

        # Compute statistics
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12)) # 4-plot grid

        # 1. Bar chart, indicate mean and std deviation
        bars = ax1.bar(classes, counts, color=sns.color_palette("husl", len(classes)),
                       edgecolor='black', linewidth=1.5)
        ax1.axhline(y=mean_count, color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_count:.1f}')
        ax1.axhline(y=mean_count + std_count, color='orange', linestyle='--',
                   linewidth=1, alpha=0.7, label=f'±1 Std: {std_count:.1f}')
        ax1.axhline(y=mean_count - std_count, color='orange', linestyle='--',
                   linewidth=1, alpha=0.7)
        ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax1.set_title('Class Distribution with Statistical Measures', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Annotate bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')

        # 2. Deviation from mean chart
        # Calculate the deviation of each class count from the overall mean count
        deviations = [count - mean_count for count in counts]

        # Assign colors: green for above/equal to mean, red for below mean (visual distinction for imbalance)
        colors_dev = ['green' if d >= 0 else 'red' for d in deviations]

        # Create the bar plot: x-axis is classes, y-axis is deviation value; color encodes positive/negative deviation
        bars2 = ax2.bar(
            classes, deviations, color=colors_dev,
            edgecolor='black', linewidth=1.5
        )

        # Draw a horizontal line at y=0 to represent the mean (i.e., no deviation)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

        # Label axes and chart, and style visual aspects
        ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Deviation from Mean', fontsize=12, fontweight='bold')
        ax2.set_title('Deviation from Mean Count', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Rotate x-tick labels for better readability
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Annotate deviation bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):+d}',
                    ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

        # 3. Box plot for numerical summary
        ax3.boxplot(counts, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax3.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax3.set_title('Distribution Statistics', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        # 4. Text summary panel
        ax4.axis('off')
        stats_text = f"""
        CLASS BALANCE ANALYSIS

        Total Classes: {len(classes)}
        Total Images: {sum(counts)}

        Statistics:
        • Maximum: {max_count} images
        • Minimum: {min_count} images
        • Mean: {mean_count:.1f} images
        • Median: {np.median(counts):.1f} images
        • Std Dev: {std_count:.1f} images

        Imbalance Metrics:
        • Imbalance Ratio: {imbalance_ratio:.2f}x
        • Coefficient of Variation: {(std_count/mean_count)*100:.1f}%

        Assessment:
        {'✓ Well Balanced' if imbalance_ratio < 2 else '⚠️  Moderate Imbalance' if imbalance_ratio < 5 else '✗ Severe Imbalance'}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_balance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir / 'class_balance_analysis.png'}")     # Print output

    def plot_sample_images(self, samples_per_class=3):
        """Display a grid of sample images from each class with labels."""

        if not self.processed_dir.exists():
            print("⚠️  Processed data directory not found")   # Warn if processed_dir missing
            return

        train_dir = self.processed_dir / "train"            # Use the training set for display
        if not train_dir.exists():
            return

        # Try to get class names from metadata mapping first
        if self.metadata and 'class_mapping' in self.metadata:
            classes = sorted(self.metadata['class_mapping'].keys(),
                          key=lambda x: self.metadata['class_mapping'][x])
        else:
            classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

        num_classes = len(classes)
        # Create a grid of subplots: one row per class, specified number of samples per row
        fig, axes = plt.subplots(num_classes, samples_per_class,
                                figsize=(samples_per_class * 4, num_classes * 3.5))

        # Handle 1D/2D shape edge cases
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        if samples_per_class == 1:
            axes = axes.reshape(-1, 1)

        # For each class we want to display
        for class_idx, class_name in enumerate(classes):
            class_dir = train_dir / class_name

            # If the class dir doesn't exist, fill row with placeholder panels
            if not class_dir.exists():
                for img_idx in range(samples_per_class):
                    ax = axes[class_idx, img_idx] if num_classes > 1 else axes[img_idx]
                    ax.axis('off')
                    ax.text(0.5, 0.5, f'No images\nfor {class_name}',
                           ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                continue

            images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))

            if len(images) == 0:
                # Class folder exists but empty - fill with placeholders
                for img_idx in range(samples_per_class):
                    ax = axes[class_idx, img_idx] if num_classes > 1 else axes[img_idx]
                    ax.axis('off')
                    ax.text(0.5, 0.5, f'No images\nfor {class_name}',
                           ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                continue

            # Randomly sample images to display for this class
            num_samples = min(samples_per_class, len(images))
            sample_indices = np.random.choice(len(images), num_samples, replace=False)

            for img_idx, sample_idx in enumerate(sample_indices):
                img_path = images[sample_idx]
                img = Image.open(img_path)  # Load the image

                # Find correct axis for the grid
                if num_classes > 1:
                    ax = axes[class_idx, img_idx]
                else:
                    ax = axes[img_idx]

                # Show image in subplot
                ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
                ax.axis('off')

                # Add class label at the start of each row
                if img_idx == 0:
                    ax.text(-0.15, 0.5, class_name,
                           transform=ax.transAxes,
                           fontsize=12, fontweight='bold',
                           rotation=0, ha='right', va='center',
                           bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor='white', edgecolor='black', linewidth=2))

                # Add label to indicate sample number on the image
                ax.text(0.02, 0.98, f'Sample {img_idx + 1}',
                       transform=ax.transAxes,
                       fontsize=9, fontweight='bold',
                       ha='left', va='top',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='yellow', alpha=0.7, edgecolor='black'))

        # Add a main title for the grid display
        plt.suptitle(f'Sample CT Scan Images by Class ({num_classes} classes, {samples_per_class} samples each)',
                    fontsize=16, fontweight='bold', y=0.995)
        # Tight layout to prevent overlap and clipping
        plt.tight_layout(rect=[0.03, 0, 1, 0.98])
        plt.savefig(self.output_dir / 'sample_images.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved: {self.output_dir / 'sample_images.png'}")                  # Notify user

    def create_summary_dashboard(self): # dashboard
        """Create a comprehensive summary dashboard with improved spacing."""

        # Create a large figure for dashboard layout
        fig = plt.figure(figsize=(24, 14))
        # Retrieve all statistics and counts needed for dashboard
        class_counts, format_counts = self.get_class_distribution()
        splits = self.get_split_distribution()

        # Initialize summary values in case some info missing
        total_images = 0
        train_count = 'N/A'
        val_count = 'N/A'
        test_count = 'N/A'
        img_size = 'N/A'
        normalize = 'N/A'
        grayscale = 'N/A'
        num_classes = 0

        # Fill summary info for dataset
        if class_counts:
            total_images = sum(class_counts.values())
            num_classes = len(class_counts)

        if self.metadata:
            train_count = self.metadata.get('train_count', 'N/A')
            val_count = self.metadata.get('val_count', 'N/A')
            test_count = self.metadata.get('test_count', 'N/A')
            img_size = self.metadata.get('img_size', 'N/A')
            if isinstance(img_size, list):
                img_size = f"{img_size[0]}x{img_size[1]}"
            normalize = 'Yes' if self.metadata.get('normalize', False) else 'No'
            grayscale = 'Yes' if self.metadata.get('grayscale', False) else 'No'

        # Single-line statistics summary for dashboard footer
        stats_text = f"Total Images: {total_images:,} | Classes: {num_classes} | Train: {train_count:,} | Val: {val_count:,} | Test: {test_count:,} | Size: {img_size} | Normalized: {normalize} | Grayscale: {grayscale}"

        # Compose a grid with 3 rows (last row minimal just for summary stats)
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.15],
                             hspace=0.55, wspace=0.4,
                             left=0.08, right=0.95, top=0.90, bottom=0.08)

        # ========== Top Row ==========

        # (1) Top left: Class distribution bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            colors = sns.color_palette("husl", len(classes))
            bars = ax1.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
            ax1.set_title('Class Distribution', fontweight='bold', fontsize=13, pad=15, color='#2c3e50')
            ax1.set_ylabel('Number of Images', fontweight='bold', fontsize=11, color='#34495e')
            ax1.tick_params(axis='both', colors='#34495e', labelsize=9)
            ax1.grid(axis='y', alpha=0.3, linestyle='--', color='#bdc3c7')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color('#95a5a6')
            ax1.spines['bottom'].set_color('#95a5a6')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')
        else:
            ax1.text(0.5, 0.5, 'No class data available', ha='center', va='center', fontsize=12, color='#7f8c8d')
            ax1.set_title('Class Distribution', fontweight='bold', fontsize=13, pad=15, color='#2c3e50')

        # (2) Top middle: Format distribution grouped bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        if format_counts and len(format_counts) > 0:
            classes = list(format_counts.keys())
            png_counts = [format_counts[c].get('PNG', 0) for c in classes]
            jpg_counts = [format_counts[c].get('JPG', 0) for c in classes]
            x = np.arange(len(classes))
            width = 0.35
            bars1 = ax2.bar(x - width/2, png_counts, width, label='PNG',
                           color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.85)
            bars2 = ax2.bar(x + width/2, jpg_counts, width, label='JPG',
                           color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.85)
            ax2.set_title('Format Distribution by Class', fontweight='bold', fontsize=13, pad=15, color='#2c3e50')
            ax2.set_ylabel('Number of Images', fontweight='bold', fontsize=11, color='#34495e')
            ax2.tick_params(axis='both', colors='#34495e', labelsize=9)
            ax2.set_xticks(x)
            ax2.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
            ax2.legend(fontsize=10, loc='upper right', framealpha=0.9, edgecolor='#bdc3c7')
            ax2.grid(axis='y', alpha=0.3, linestyle='--', color='#bdc3c7')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_color('#95a5a6')
            ax2.spines['bottom'].set_color('#95a5a6')
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}',
                               ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2c3e50')
        else:
            ax2.text(0.5, 0.5, 'No format data available', ha='center', va='center', fontsize=12, color='#7f8c8d')
            ax2.set_title('Format Distribution by Class', fontweight='bold', fontsize=13, pad=15, color='#2c3e50')

        # (3) Top right: Split bar chart (train/val/test total images)
        ax3 = fig.add_subplot(gs[0, 2])
        if splits and len(splits) > 0:
            split_names = list(splits.keys())
            total_counts = [sum(splits[split].values()) for split in split_names]
            colors = {'train': '#2ecc71', 'val': '#f39c12', 'test': '#9b59b6'}
            bars = ax3.bar(split_names, total_counts,
                   color=[colors.get(s, 'gray') for s in split_names],
                   edgecolor='black', linewidth=1.5, alpha=0.85)
            ax3.set_title('Total Images per Split', fontweight='bold', fontsize=13, pad=15, color='#2c3e50')
            ax3.set_ylabel('Total Images', fontweight='bold', fontsize=11, color='#34495e')
            ax3.set_xlabel('Split', fontweight='bold', fontsize=11, color='#34495e')
            ax3.tick_params(axis='both', colors='#34495e', labelsize=9)
            ax3.grid(axis='y', alpha=0.3, linestyle='--', color='#bdc3c7')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_color('#95a5a6')
            ax3.spines['bottom'].set_color('#95a5a6')
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')
        else:
            ax3.text(0.5, 0.5, 'No split data available', ha='center', va='center', fontsize=12, color='#7f8c8d')
            ax3.set_title('Total Images per Split', fontweight='bold', fontsize=13, pad=15, color='#2c3e50')

        # ========== Second Row ==========

        # (4) Left: Pie chart for class percent
        ax4 = fig.add_subplot(gs[1, 0])
        if class_counts and len(class_counts) > 0:
            counts = list(class_counts.values())
            classes = list(class_counts.keys())
            colors = sns.color_palette("husl", len(classes))
            wedges, texts, autotexts = ax4.pie(counts, labels=classes, autopct='%1.1f%%',
                                               startangle=90, colors=colors,
                                               textprops={'fontsize': 9, 'color': '#2c3e50'},
                                               wedgeprops=dict(edgecolor='white', linewidth=2))
            ax4.set_title('Class Proportions (Percentage)', fontweight='bold', fontsize=13, pad=15, color='#2c3e50')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            for text in texts:
                text.set_color('#2c3e50')
                text.set_fontweight('bold')
        else:
            ax4.text(0.5, 0.5, 'No class data available', ha='center', va='center', fontsize=12, color='#7f8c8d')
            ax4.set_title('Class Proportions (Percentage)', fontweight='bold', fontsize=13, pad=15, color='#2c3e50')

        # (5) Center+Right: Grouped split-by-class chart
        ax5 = fig.add_subplot(gs[1, 1:])
        if splits and class_counts and len(splits) > 0 and len(class_counts) > 0:
            all_classes = sorted(class_counts.keys())
            split_names = list(splits.keys())
            x = np.arange(len(all_classes))
            width = 0.25
            colors = {'train': '#2ecc71', 'val': '#f39c12', 'test': '#9b59b6'}

            for i, split in enumerate(split_names):
                counts = [splits[split].get(cls, 0) for cls in all_classes]
                bars = ax5.bar(x + i*width, counts, width, label=split.capitalize(),
                       color=colors.get(split, 'gray'), edgecolor='black', linewidth=1.5)
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax5.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}',
                               ha='center', va='bottom', fontsize=8, fontweight='bold')

            ax5.set_xlabel('Class', fontweight='bold', fontsize=12, labelpad=10, color='#34495e')
            ax5.set_ylabel('Number of Images', fontweight='bold', fontsize=12, color='#34495e')
            ax5.set_title('Split Distribution by Class', fontweight='bold', fontsize=13, pad=15, color='#2c3e50')
            ax5.tick_params(axis='both', colors='#34495e', labelsize=10)
            ax5.set_xticks(x + width)
            ax5.set_xticklabels(all_classes, rotation=45, ha='right', fontsize=10)
            ax5.legend(fontsize=11, loc='upper right', framealpha=0.9, edgecolor='#bdc3c7')
            ax5.grid(axis='y', alpha=0.3, linestyle='--', color='#bdc3c7')
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.spines['left'].set_color('#95a5a6')
            ax5.spines['bottom'].set_color('#95a5a6')
            ax5.margins(y=0.05)
        else:
            ax5.text(0.5, 0.5, 'No split data available', ha='center', va='center', fontsize=12, color='#7f8c8d')
            ax5.set_title('Split Distribution by Class', fontweight='bold', fontsize=13, pad=15, color='#2c3e50')

        # ========== Third Row (Summary) ==========

        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        ax6.text(0.5, 0.5, stats_text, fontsize=9, family='monospace',
                ha='center', va='center', color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='#ecf0f1', alpha=0.9,
                         edgecolor='#bdc3c7', linewidth=1.5))

        # Add main dashboard title
        fig.suptitle('CT Scan Dataset Analysis Dashboard',
                    fontsize=24, fontweight='bold', y=0.992,
                    color='#2c3e50', family='sans-serif')

        # Add a dashboard subtitle with key info
        fig.text(0.85, 0.962, f'Lung Cancer Classification - {num_classes} Classes | {total_images:,} Total Images',
                ha='center', fontsize=11, style='italic', color='#7f8c8d',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', alpha=0.8, edgecolor='#bdc3c7', linewidth=1))

        plt.savefig(self.output_dir / 'dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved: {self.output_dir / 'dashboard.png'}")             # Print output

    def generate_all_visualizations(self):
        """Generate all supported visualizations and save them to disk."""
        print("=" * 60)
        print("CT Scan Dataset Visualization Pipeline")
        print("=" * 60)
        print("\nGenerating visualizations...")
        print("-" * 60)

        self.plot_class_distribution()        # Bar+pie for class distribution
        self.plot_format_distribution()       # Format distribution by class
        self.plot_split_distribution()        # Data split stats
        self.plot_class_balance_analysis()    # Class balance/imbalance
        self.plot_sample_images(samples_per_class=3)    # Sample images grid, 3 per class
        self.create_summary_dashboard()       # Overall dashboard

        print("-" * 60)
        print(f"\n✓ All visualizations saved to: {self.output_dir}/")
        print("=" * 60)

# Main execution entrypoint
if __name__ == "__main__":
    visualizer = CTScanVisualizer()                 # Create a visualizer using default dirs
    visualizer.generate_all_visualizations()        # Generate all the charts and dashboard
