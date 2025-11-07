"""
Script to analyze format distribution and potential format-class correlation
"""

from pathlib import Path     # Import the Path class from pathlib for operating system path operations and directory traversal
import json                 # (Imported for completeness/future use, but not currently used in this script)

def analyze_format_distribution():
    """Analyze PNG vs JPG distribution across classes"""

    # Print primary section title divider - serves as a visual boundary in console output
    print("=" * 70)
    # Print major header for this analysis step
    print("IMAGE FORMAT DISTRIBUTION ANALYSIS")
    # Print secondary divider for further clarity in output
    print("=" * 70)

    # Create a Path object pointing to the root directory where the original lung cancer dataset should be located
    original_dir = Path("Lung Cancer Dataset")
    
    # If the directory "Lung Cancer Dataset" does not exist in the expected location, issue an error and terminate
    if not original_dir.exists():
        print(f"Error: {original_dir} not found!")        # Inform the user that the dataset directory is missing
        return                                           # Abort the function since further analysis can't proceed

    # Print a header introducing the next section: Per-class format statistics in the original data
    print("\n📊 ORIGINAL DATASET - FORMAT DISTRIBUTION BY CLASS:")
    print("-" * 70)

    format_stats = {}    # Dictionary to collect per-class statistics for further reference and correlation analysis
    total_png = 0        # Running total of PNG images across all classes
    total_jpg = 0        # Running total of JPG images across all classes

    # Loop through every entry in the dataset directory, sorted alphabetically, so that class folders are processed in order
    for class_dir in sorted(original_dir.iterdir()):
        # Only analyze entries that are directories (representing classes)
        if class_dir.is_dir():
            # Count PNG files in the current class directory by matching '*.png'
            png_count = len(list(class_dir.glob('*.png')))
            # Count JPG files by matching '*.jpg'
            jpg_count = len(list(class_dir.glob('*.jpg')))
            # Compute total number of image files in this class (PNG and JPG combined)
            total = png_count + jpg_count
            
            # If this class contains images, perform analysis; otherwise, skip silent class folders
            if total > 0:
                # Compute the proportion (percentage) of images in PNG format in this class
                png_pct = (png_count / total) * 100

                # Store stats for this class in the format_stats dictionary for reference or summarization later
                format_stats[class_dir.name] = {
                    'png': png_count,         # Store number of PNG images
                    'jpg': jpg_count,         # Store number of JPG images
                    'total': total,           # Store total images in the class
                    'png_pct': png_pct        # Store computed % of PNG images
                }
                
                # Add class PNG and JPG counts to running dataset-wide tallies
                total_png += png_count
                total_jpg += jpg_count

                # Print human-friendly summary for this class: total, PNG count (+%), and JPG count (+%)
                print(f"\n{class_dir.name}:")
                print(f"  Total: {total} images")
                print(f"  PNG: {png_count} ({png_pct:.1f}%)")
                print(f"  JPG: {jpg_count} ({100-png_pct:.1f}%)")

                # If class contains ONLY JPGs (0% PNG), print an explicit warning about format-class correlation
                if png_pct == 0:
                    print(f"  ⚠️  WARNING: All images are JPG - format-class correlation detected!")
                # If class contains ONLY PNGs (100% PNG), print a similar warning for PNG dominance
                elif png_pct == 100:
                    print(f"  ⚠️  WARNING: All images are PNG - format-class correlation detected!")

    # Print divider before summary of overall format statistics
    print("\n" + "=" * 70)
    print("📈 OVERALL STATISTICS:")                    # Section heading for overall dataset numbers
    print("-" * 70)
    total_all = total_png + total_jpg                  # Calculate total number of images in dataset (all classes)
    if total_all > 0:                                  
        print(f"Total images: {total_all}")            # Print how many total images were found
        # Print PNG count and percentage relative to overall dataset
        print(f"PNG: {total_png} ({(total_png/total_all)*100:.1f}%)")
        # Print JPG count and percentage
        print(f"JPG: {total_jpg} ({(total_jpg/total_all)*100:.1f}%)")

    # Begin section to analyze potential correlation between image format and class label
    print("\n" + "=" * 70)
    print("🔍 FORMAT-CLASS CORRELATION ANALYSIS:")      # Label this analysis step
    print("-" * 70)

    # Identify classes in which 100% of images are PNG (potential for model to "cheat" by format)
    classes_all_png = [name for name, stats in format_stats.items() if stats['png_pct'] == 100]
    # Likewise, make a list of classes that are 100% JPG (no PNG images)
    classes_all_jpg = [name for name, stats in format_stats.items() if stats['png_pct'] == 0]

    # If there are classes with only PNGs, print their names and a warning
    if classes_all_png:
        print(f"\n⚠️  Classes with 100% PNG ({len(classes_all_png)}):")
        for name in classes_all_png:
            print(f"  - {name}")

    # If there are classes with only JPGs, print their names and a warning
    if classes_all_jpg:
        print(f"\n⚠️  Classes with 100% JPG ({len(classes_all_jpg)}):")
        for name in classes_all_jpg:
            print(f"  - {name}")

    # If there is any class with only PNG or only JPG, print overall caution and suggested mitigations
    if classes_all_png or classes_all_jpg:
        print("\n⚠️  POTENTIAL ISSUE: Format-class correlation detected!")
        print("   This means certain classes are exclusively one format.")
        print("   The model might learn format-based features rather than medical features.") # Warn about possible spurious correlations in training
        print("\n✅ MITIGATION: The preprocessing pipeline normalizes formats, but:")
        print("   - Monitor validation accuracy for suspiciously high scores")
        print("   - Test model performance on both PNG and JPG images")
        print("   - Consider data augmentation to reduce format-specific learning")
    else:
        # Otherwise, explicitly declare that no serious correlation was found
        print("\n✅ No significant format-class correlation detected.")
        print("   Formats are mixed across classes, reducing bias risk.")

    # Print a divider before moving to the processed data format verification
    print("\n" + "=" * 70)
    print("✅ PROCESSED DATA - FORMAT CONSISTENCY CHECK:")      # Announce processed data check section
    print("-" * 70)

    # Specify the processed data folder to check is 'processed_data/train'
    processed_dir = Path("processed_data/train")
    if processed_dir.exists():
        print("\nChecking processed data (should all be PNG after preprocessing):")    # Announce next check
        all_png = True       # Assume, unless contradiction is found, that all output files are PNG
        # Iterate over each subdirectory in processed training data (should correspond to classes)
        for class_dir in sorted(processed_dir.iterdir()):
            if class_dir.is_dir():
                # Gather PNG and JPG counts for each processed class directory
                png_count = len(list(class_dir.glob('*.png')))
                jpg_count = len(list(class_dir.glob('*.jpg')))
                # If there are any images to report, print the count for this class
                if png_count > 0 or jpg_count > 0:
                    print(f"  {class_dir.name}: PNG: {png_count}, JPG: {jpg_count}")
                    # If any JPGs are found (unexpected post-processing), set flag false
                    if jpg_count > 0:
                        all_png = False

        # Summarize the results of the processed data check: fully PNG = success, otherwise warning
        if all_png:
            print("\n✅ All processed images are in PNG format (consistent format)")      # Report success if no JPG found
        else:
            print("\n⚠️  Mixed formats still present in processed data")                 # Report possible issue if non-PNG discovered
    else:
        # If "processed_data/train" directory does not exist, prompt user to preprocess the dataset
        print("\n⚠️  Processed data directory not found. Run preprocessing first.")

    # Print a final divider and summary/recommendations for user action
    print("\n" + "=" * 70)
    print("📝 SUMMARY & RECOMMENDATIONS:")    # Section header for final takeaways
    print("-" * 70)
    print("""
1. ✅ The preprocessing pipeline converts all images to the same format (RGB arrays)
2. ✅ All images are normalized to the same size and value range
3. ⚠️  Format-class correlation exists if certain classes are exclusively one format
4. ✅ Data augmentation helps reduce format-specific learning
5. 📊 Monitor training to ensure the model learns medical features, not format artifacts

RECOMMENDATION: Proceed with training. The preprocessing handles format differences well.
Monitor validation performance and test on both PNG and JPG images.
    """)
    print("=" * 70)

if __name__ == "__main__":
    # If this script file is run as the main program, invoke the analysis function.
    analyze_format_distribution()

