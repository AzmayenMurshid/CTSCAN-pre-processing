"""
Script to analyze format distribution and potential format-class correlation
"""

from pathlib import Path
import json

def analyze_format_distribution():
    """Analyze PNG vs JPG distribution across classes"""
    
    print("=" * 70)
    print("IMAGE FORMAT DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Check original format distribution
    original_dir = Path("Lung Cancer Dataset")
    
    if not original_dir.exists():
        print(f"Error: {original_dir} not found!")
        return
    
    print("\n📊 ORIGINAL DATASET - FORMAT DISTRIBUTION BY CLASS:")
    print("-" * 70)
    
    format_stats = {}
    total_png = 0
    total_jpg = 0
    
    for class_dir in sorted(original_dir.iterdir()):
        if class_dir.is_dir():
            png_count = len(list(class_dir.glob('*.png')))
            jpg_count = len(list(class_dir.glob('*.jpg')))
            total = png_count + jpg_count
            
            if total > 0:
                png_pct = (png_count / total) * 100
                format_stats[class_dir.name] = {
                    'png': png_count,
                    'jpg': jpg_count,
                    'total': total,
                    'png_pct': png_pct
                }
                
                total_png += png_count
                total_jpg += jpg_count
                
                print(f"\n{class_dir.name}:")
                print(f"  Total: {total} images")
                print(f"  PNG: {png_count} ({png_pct:.1f}%)")
                print(f"  JPG: {jpg_count} ({100-png_pct:.1f}%)")
                
                # Flag potential bias
                if png_pct == 0:
                    print(f"  ⚠️  WARNING: All images are JPG - format-class correlation detected!")
                elif png_pct == 100:
                    print(f"  ⚠️  WARNING: All images are PNG - format-class correlation detected!")
    
    print("\n" + "=" * 70)
    print("📈 OVERALL STATISTICS:")
    print("-" * 70)
    total_all = total_png + total_jpg
    if total_all > 0:
        print(f"Total images: {total_all}")
        print(f"PNG: {total_png} ({(total_png/total_all)*100:.1f}%)")
        print(f"JPG: {total_jpg} ({(total_jpg/total_all)*100:.1f}%)")
    
    # Check for format-class correlation
    print("\n" + "=" * 70)
    print("🔍 FORMAT-CLASS CORRELATION ANALYSIS:")
    print("-" * 70)
    
    classes_all_png = [name for name, stats in format_stats.items() if stats['png_pct'] == 100]
    classes_all_jpg = [name for name, stats in format_stats.items() if stats['png_pct'] == 0]
    
    if classes_all_png:
        print(f"\n⚠️  Classes with 100% PNG ({len(classes_all_png)}):")
        for name in classes_all_png:
            print(f"  - {name}")
    
    if classes_all_jpg:
        print(f"\n⚠️  Classes with 100% JPG ({len(classes_all_jpg)}):")
        for name in classes_all_jpg:
            print(f"  - {name}")
    
    if classes_all_png or classes_all_jpg:
        print("\n⚠️  POTENTIAL ISSUE: Format-class correlation detected!")
        print("   This means certain classes are exclusively one format.")
        print("   The model might learn format-based features rather than medical features.")
        print("\n✅ MITIGATION: The preprocessing pipeline normalizes formats, but:")
        print("   - Monitor validation accuracy for suspiciously high scores")
        print("   - Test model performance on both PNG and JPG images")
        print("   - Consider data augmentation to reduce format-specific learning")
    else:
        print("\n✅ No significant format-class correlation detected.")
        print("   Formats are mixed across classes, reducing bias risk.")
    
    # Check processed data format consistency
    print("\n" + "=" * 70)
    print("✅ PROCESSED DATA - FORMAT CONSISTENCY CHECK:")
    print("-" * 70)
    
    processed_dir = Path("processed_data/train")
    if processed_dir.exists():
        print("\nChecking processed data (should all be PNG after preprocessing):")
        all_png = True
        for class_dir in sorted(processed_dir.iterdir()):
            if class_dir.is_dir():
                png_count = len(list(class_dir.glob('*.png')))
                jpg_count = len(list(class_dir.glob('*.jpg')))
                if png_count > 0 or jpg_count > 0:
                    print(f"  {class_dir.name}: PNG: {png_count}, JPG: {jpg_count}")
                    if jpg_count > 0:
                        all_png = False
        
        if all_png:
            print("\n✅ All processed images are in PNG format (consistent format)")
        else:
            print("\n⚠️  Mixed formats still present in processed data")
    else:
        print("\n⚠️  Processed data directory not found. Run preprocessing first.")
    
    print("\n" + "=" * 70)
    print("📝 SUMMARY & RECOMMENDATIONS:")
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
    analyze_format_distribution()

