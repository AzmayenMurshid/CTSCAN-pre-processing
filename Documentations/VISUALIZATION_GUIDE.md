# CT Scan Distribution Visualization Guide

This guide explains how to use the visualization tools to analyze and visualize CT scan dataset distributions.

## Overview

The `visualize_ctscan_distribution.py` script creates comprehensive visualizations for:
- Class distribution analysis
- Format distribution (PNG vs JPG)
- Train/Validation/Test split distribution
- Class balance analysis
- Sample image displays
- Summary dashboard

## Quick Start

### Run All Visualizations

```bash
python visualize_ctscan_distribution.py
```

This will generate all visualizations and save them to the `visualizations/` directory.

## Generated Visualizations

### 1. Class Distribution (`class_distribution.png`)
- **Bar Chart**: Shows number of images per class
- **Pie Chart**: Shows percentage distribution across classes
- Helps identify class imbalance issues

### 2. Format Distribution (`format_distribution.png`)
- **Stacked Bar Chart**: Shows PNG vs JPG distribution per class
- Useful for identifying format-class correlations
- Helps ensure format consistency

### 3. Split Distribution (`split_distribution.png`)
- **Stacked Bar Chart**: Shows train/val/test distribution per class
- **Bar Chart**: Shows total images per split
- Verifies stratified splitting worked correctly

### 4. Class Balance Analysis (`class_balance_analysis.png`)
- **4-panel analysis**:
  - Distribution with statistical measures (mean, std dev)
  - Deviation from mean count
  - Box plot showing distribution statistics
  - Summary statistics and imbalance assessment
- Calculates imbalance ratio and coefficient of variation
- Provides assessment of class balance quality

### 5. Sample Images (`sample_images.png`)
- **Grid Display**: Shows sample images from each class
- 3 samples per class (configurable)
- Helps visually verify image quality and class characteristics

### 6. Summary Dashboard (`dashboard.png`)
- **Comprehensive overview**: Single-page dashboard with all key metrics
- Multiple charts in one view
- Dataset statistics summary
- Quick reference for dataset overview

## Usage Examples

### Basic Usage
```python
from visualize_ctscan_distribution import CTScanVisualizer

# Initialize visualizer
visualizer = CTScanVisualizer()

# Generate all visualizations
visualizer.generate_all_visualizations()
```

### Custom Directory Paths
```python
visualizer = CTScanVisualizer(
    data_dir="path/to/raw/data",
    processed_dir="path/to/processed/data"
)
```

### Generate Specific Visualizations
```python
visualizer = CTScanVisualizer()

# Generate only specific visualizations
visualizer.plot_class_distribution()
visualizer.plot_format_distribution()
visualizer.plot_split_distribution()
visualizer.plot_class_balance_analysis()
visualizer.plot_sample_images(samples_per_class=5)  # More samples
visualizer.create_summary_dashboard()
```

## Output Location

All visualizations are saved to:
```
visualizations/
├── class_distribution.png
├── format_distribution.png
├── split_distribution.png
├── class_balance_analysis.png
├── sample_images.png
└── dashboard.png
```

## Visualization Features

### Color Schemes
- **Classes**: Distinct colors using HSL palette
- **Splits**: Train (green), Validation (orange), Test (purple)
- **Formats**: PNG (blue), JPG (red)

### Statistical Insights
- Mean, median, standard deviation
- Imbalance ratio calculation
- Coefficient of variation
- Balance assessment (Well Balanced / Moderate Imbalance / Severe Imbalance)

### Professional Formatting
- High-resolution (300 DPI) PNG outputs
- Clear labels and legends
- Grid lines for readability
- Value labels on bars for easy reading

## Interpreting Results

### Class Distribution
- **Balanced**: Similar counts across classes (imbalance ratio < 2)
- **Moderate Imbalance**: Some variation (imbalance ratio 2-5)
- **Severe Imbalance**: Large differences (imbalance ratio > 5)

### Format Distribution
- Check for format-class correlation (e.g., all of one class in one format)
- Should be mixed across classes for better generalization

### Split Distribution
- Verify stratified splitting maintained class proportions
- Each split should have similar class ratios
- Check minimum samples per class in test set (should be ≥ 10-20)

## Customization

### Change Number of Sample Images
```python
visualizer.plot_sample_images(samples_per_class=5)  # Show 5 samples per class
```

### Custom Output Directory
```python
visualizer = CTScanVisualizer()
visualizer.output_dir = Path("custom/output/path")
visualizer.generate_all_visualizations()
```

### Adjust Figure Sizes
Modify the `figsize` parameters in the script for different output sizes.

## Troubleshooting

### Issue: "Format distribution data not available"
- **Solution**: Ensure metadata.json exists in processed_data directory
- Or run preprocessing first to generate metadata

### Issue: "Processed data directory not found"
- **Solution**: Run preprocessing script first: `python preprocess_ct_scans.py`

### Issue: Images not displaying correctly
- **Solution**: Ensure PIL/Pillow is installed: `pip install Pillow`

## Integration with Pipeline

The visualization script integrates seamlessly with the preprocessing pipeline:

1. **After Preprocessing**: Run visualization to verify preprocessing results
2. **Before Training**: Use visualizations to understand dataset characteristics
3. **After Training**: Compare with training results to analyze model performance

## Best Practices

1. **Run after preprocessing**: Verify data splits and distributions
2. **Review before training**: Understand class imbalance and plan accordingly
3. **Document findings**: Use visualizations in reports and documentation
4. **Regular updates**: Re-run visualizations if dataset changes

## Example Workflow

```bash
# 1. Preprocess data
python preprocess_ct_scans.py

# 2. Generate visualizations
python visualize_ctscan_distribution.py

# 3. Review visualizations
# Open visualizations/ directory and review all PNG files

# 4. Start training (if visualizations look good)
python example_training.py
```

## Additional Notes

- All visualizations use professional styling with seaborn
- Outputs are high-resolution (300 DPI) suitable for publications
- Colors are colorblind-friendly where possible
- Statistical summaries provide actionable insights

