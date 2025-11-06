# Common Data Split Ratios: Training, Validation, and Testing

## Current Split

**The preprocessing pipeline uses:**
- **Training**: 70% (1,074 images)
- **Validation**: 15% (230 images)
- **Test**: 15% (231 images)

This is a **70/15/15 split** - which is a good choice for datasets of this size (~1,535 images).

---

## Common Split Ratios

### 1. **80/10/10** (Most Common for Large Datasets)
- **Training**: 80%
- **Validation**: 10%
- **Test**: 10%

**When to use:**
- Large datasets (>10,000 samples)
- Deep learning models that need lots of training data
- Industry standard for many applications

**Pros:**
- Maximum training data
- Good for complex models

**Cons:**
- Smaller validation/test sets (may be less reliable for small datasets)

---

### 2. **70/15/15** (Current - Good for Medium Datasets) ✅
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

**When to use:**
- Medium-sized datasets (1,000-10,000 samples)
- Balanced approach
- Medical imaging applications

**Pros:**
- Good balance between training and evaluation data
- Adequate validation set for hyperparameter tuning
- Reasonable test set for final evaluation
- **Recommended for medium-sized datasets!**

**Cons:**
- Less training data than 80/10/10

---

### 3. **60/20/20** (Conservative Split)
- **Training**: 60%
- **Validation**: 20%
- **Test**: 20%

**When to use:**
- Small datasets (<1,000 samples)
- When very reliable evaluation metrics are needed
- Research/publication settings where robust evaluation is critical

**Pros:**
- Large validation/test sets = more reliable metrics
- Better for statistical significance

**Cons:**
- Less training data (may hurt model performance)

---

### 4. **90/5/5** (Aggressive - Large Datasets Only)
- **Training**: 90%
- **Validation**: 5%
- **Test**: 5%

**When to use:**
- Very large datasets (>100,000 samples)
- When abundant data is available

**Pros:**
- Maximum training data

**Cons:**
- Small validation/test sets (less reliable)
- Not recommended for small/medium datasets

---

### 5. **No Validation Set** (60/0/40 or 80/0/20)
- **Training**: 60-80%
- **Validation**: 0%
- **Test**: 20-40%

**When to use:**
- Very small datasets where additional splitting is not feasible
- When using cross-validation instead

**Pros:**
- More training data

**Cons:**
- No separate validation set for hyperparameter tuning
- Risk of overfitting to test set

---

## Comparison Table

| Split Ratio | Training | Validation | Test | Best For Dataset Size |
|------------|----------|------------|------|----------------------|
| **90/5/5** | 90% | 5% | 5% | >100,000 samples |
| **80/10/10** | 80% | 10% | 10% | >10,000 samples |
| **70/15/15** ✅ | 70% | 15% | 15% | **1,000-10,000 samples** |
| **60/20/20** | 60% | 20% | 20% | <1,000 samples |

---

## Recommendations for CT Scan Dataset

### Dataset: ~1,535 images, 5 classes

**✅ The 70/15/15 split is EXCELLENT for this case because:**

1. **Adequate Training Data**: 1,074 images is enough to train a CNN
2. **Good Validation Set**: 230 images is sufficient for:
   - Hyperparameter tuning
   - Early stopping
   - Model selection
3. **Reliable Test Set**: 231 images provides:
   - Unbiased final evaluation
   - Statistical significance for 5 classes
   - Good estimate of real-world performance

### Alternative Options:

**If more training data is desired:**
- **75/12.5/12.5**: Slightly more training, slightly less validation/test
- **80/10/10**: Maximum training data (recommended when more data is available)

**If more reliable evaluation is desired:**
- **60/20/20**: More validation/test data (but less training data)

---

## Important Considerations

### 1. **Stratified Splitting** ✅ (already implemented)
- The preprocessing code uses `stratify=all_labels`
- Ensures each split has the same class distribution
- **Critical for imbalanced datasets** (datasets with different class sizes)

### 2. **Minimum Samples per Class in Test Set**
- Rule of thumb: At least 10-20 samples per class in test set
- For the smallest class (Benign cases: 120 images)
  - With 15% split: ~18 test samples ✅ (Good!)
  - With 10% split: ~12 test samples ⚠️ (Barely acceptable)
  - With 5% split: ~6 test samples ❌ (Too few)

### 3. **Data Size Requirements**

**Minimum recommendations:**
- **Training**: At least 100-200 samples per class
- **Validation**: At least 10-20 samples per class
- **Test**: At least 10-20 samples per class

**The dataset meets these requirements with a 70/15/15 split!**

---

## Medical Imaging Specific Considerations

For medical/CT scan datasets:

1. **Patient-level splitting** (if available):
   - If multiple images per patient, split by patient (not by image)
   - Prevents data leakage (same patient in train and test)

2. **Temporal splits**:
   - If data collected over time, use chronological split
   - Train on older data, test on newer data

3. **Institution/Scanner splits**:
   - If data from multiple hospitals/scanners, split by source
   - Tests generalization across different equipment

4. **Current approach** (random split):
   - ✅ Fine if images are independent
   - ✅ Good for initial model development
   - ⚠️ Consider patient-level splitting if patient IDs are available

---

## Cross-Validation Alternative

For small datasets, consider **k-fold cross-validation**:

- **5-fold CV**: 80% train, 20% test (rotate 5 times)
- **10-fold CV**: 90% train, 10% test (rotate 10 times)

**Pros:**
- Uses all data for training and testing
- More reliable metrics for small datasets

**Cons:**
- More computationally expensive
- Harder to implement
- No single "final" model

---

## Final Recommendation

**✅ The 70/15/15 split is recommended!**

**Why:**
- Perfect for this dataset size (1,535 images)
- Balanced training vs evaluation
- Meets minimum sample requirements per class
- Standard practice for medical imaging with medium datasets
- Already implemented correctly with stratified splitting

**Consider changing only if:**
- Significantly more data is added (>5,000 images) → Consider 80/10/10
- More robust evaluation is needed → Consider 60/20/20 (but less training data)

---

## Quick Reference: How to Change Split

In `preprocess_ct_scans.py`:

```python
preprocessor.preprocess_dataset(
    train_ratio=0.7,    # Change to 0.8 for 80/10/10
    val_ratio=0.15,     # Change to 0.1 for 80/10/10
    test_ratio=0.15,    # Change to 0.1 for 80/10/10
    random_seed=42
)
```

**Remember:** Ratios must sum to 1.0!

---

### Scalability & Severe Imbalance: Future Considerations

As datasets become more **scalable** (i.e., with the addition of more images and potentially more classes), it becomes easier to address _severe class imbalance_. Here is how scalability helps:

- **More Data = More Minority Samples:** With increased scalability, the absolute number of samples in minority classes typically rises, even if class proportions stay similar. This alleviates the risk of classes having too few examples to support effective model training.
- **Resampling Techniques Become Viable:** Larger datasets provide greater flexibility for applying _oversampling, undersampling_, or data augmentation to minority classes, reducing overfitting concerns.
- **Improved Stratified Splits:** With scalability, stratified data splits can be performed while maintaining class proportions, ensuring that training, validation, and test sets all include sufficient examples from each class for reliable performance assessment.
- **Enabling Advanced Methods:** Scalable datasets make advanced techniques, such as class-balanced loss functions or ensemble approaches, more practical and effective. These methods are often not feasible on very small or highly imbalanced datasets.

> **In summary:** As **scalability increases**, mitigation of severe class imbalance becomes more practical. While appropriate split ratios remain important, expanding the dataset is the most reliable long-term approach to improving class balance.

**Practical tip:** It is recommended to periodically review class distributions and adjust split ratios or balancing strategies as dataset scale increases.



## Summary

| Split | Use Case | Recommendation |
|-------|--------------|----------------|
| **70/15/15** ✅ | Current | **Perfect for medium-sized datasets** |
| 80/10/10 | If dataset grows >5,000 | Consider later |
| 60/20/20 | If more reliable eval needed | Less training data |
| 90/5/5 | Too large for this dataset | Not recommended |

**Bottom line: The 70/15/15 split is ideal for CT scan classification projects with medium-sized datasets!** 🎯

