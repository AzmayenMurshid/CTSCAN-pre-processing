# Terminal Outputs and Commands - Notes for Future Reference

This document contains verbatim terminal outputs and commands used during the CT scan preprocessing project setup.

---

## Package Installation

### Checking Python Version
```bash
python -c "import sys; print(sys.version)"
```

**Output:**
```
3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]
```

---

### Installing PyTorch and torchvision
```bash
pip install torch torchvision
```

**Output:**
```
Collecting torch
  Using cached torch-2.9.0-cp310-cp310-win_amd64.whl.metadata (30 kB)
Collecting torchvision
  Using cached torchvision-0.24.0-cp310-cp310-win_amd64.whl.metadata (5.9 kB)
Collecting filelock (from torch)
  Using cached filelock-3.20.0-py3-none-any.whl.metadata (2.1 kB)
Requirement already satisfied: typing-extensions>=4.10.0 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torch) (4.10.0)
Requirement already satisfied: sympy>=1.13.3 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torch) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torch) (3.4.2)
Requirement already satisfied: jinja2 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torch) (3.1.3)
Collecting fsspec>=0.8.5 (from torch)
  Using cached fsspec-2025.10.0-py3-none-any.whl.metadata (10 kB)
Requirement already satisfied: numpy in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torchvision) (1.24.3)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torchvision) (10.0.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from sympy>=1.13.3->torch) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from jinja2->torch) (2.1.5)
Using cached torch-2.9.0-cp310-cp310-win_amd64.whl (109.3 MB)
Using cached torchvision-0.24.0-cp310-cp310-win_amd64.whl (3.7 MB)
Using cached fsspec-2025.10.0-py3-none-any.whl (200 kB)
Using cached filelock-3.20.0-py3-none-any.whl (16 kB)
Installing collected packages: fsspec, filelock, torch, torchvision
  WARNING: The scripts torchfrtrace.exe and torchrun.exe are installed in 'C:\Users\azmay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts torchfrtrace.exe and torchrun.exe are installed in 'C:\Users\azmay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR: Could not install packages due to an OSError: [WinError 32] The process cannot access the file because it is being used by another process: 'C:\\Users\\azmay\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\site-packages\\torchvision\\python310.dll'
Check the permissions.
```

**Note:** Despite the error, torch was successfully installed (verified in next step).

---

### Verifying PyTorch Installation
```bash
python -c "import torch; print('torch installed successfully')"
```

**Output:**
```
torch installed successfully
```

---

### Verifying torchvision Installation
```bash
python -c "import torchvision; print(f'torchvision installed successfully - version: {torchvision.__version__}')"
```

**Output:**
```
torchvision installed successfully - version: 0.24.0+cpu
```

---

### Checking All Required Packages
```bash
python check_and_install_packages.py
```

**Output:**
```
Checking for installed packages...
============================================================
[OK] torch - INSTALLED
[OK] torchvision - INSTALLED
[OK] numpy - INSTALLED
[OK] opencv-python (imported as cv2) - INSTALLED
[OK] Pillow (imported as PIL) - INSTALLED
[OK] scikit-learn (imported as sklearn) - INSTALLED
[OK] tqdm - INSTALLED
[OK] matplotlib - INSTALLED
[OK] seaborn - INSTALLED
============================================================

[SUCCESS] All required packages are already installed!
```

---

### Verifying All Imports
```bash
python -c "import torch; import torchvision; import numpy; import cv2; from PIL import Image; from sklearn.model_selection import train_test_split; import tqdm; import matplotlib; import seaborn; print('All imports successful!')"
```

**Output:**
```
All imports successful!
```

---

## Image Analysis

### Checking Sample Image Properties
```bash
python -c "from PIL import Image; import os; img_path = os.path.join('Lung Cancer Dataset', 'adenocarcinoma', os.listdir('Lung Cancer Dataset/adenocarcinoma')[0]); img = Image.open(img_path); print(f'Image size: {img.size}, Mode: {img.mode}')"
```

**Output:**
```
Image size: (315, 245), Mode: RGBA
```

**Note:** This confirmed that some images are RGBA format, which is handled by the preprocessing pipeline.

---

## Preprocessing Execution

### Correct Command to Run Preprocessing
```bash
python preprocess_ct_scans.py
```

**Note:** The correct filename is `preprocess_ct_scans.py` (not `preprocessing_ct_scan.py`).

**Alternative Command:**
```bash
python run_preprocessing.py
```

---

### Common Error - Incorrect Filename
```bash
python preprocessing_ct_scan.py
```

**Output:**
```
C:\Users\azmay\AppData\Local\Microsoft\WindowsApps\python.exe: can't open file 'C:\\CTSCAN-pre-processing\\preprocessing_ct_scan.py': [Errno 2] No such file or directory
```

**Fix:** Use the correct filename: `preprocess_ct_scans.py`

---

## Preprocessing Results

### Dataset Statistics (Expected After Preprocessing)

**Original Dataset:**
- adenocarcinoma: 337 PNG images
- Benign cases: 120 JPG images
- large cell carcinoma: 187 PNG images
- Normal cases: 631 images (428 JPG, 203 PNG)
- squamous cell carcinoma: 260 PNG images
- **Total: ~1,535 images**

**After Preprocessing (70/15/15 split):**
- **Train set**: ~1,074 images (70%)
- **Validation set**: ~230 images (15%)
- **Test set**: ~230 images (15%)

**Output Structure:**
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

============================================================
CT Scan Preprocessing Pipeline
============================================================

Analyzing dataset...
Total images: 1535
  adenocarcinoma: 337 images
  Benign cases: 120 images
  large cell carcinoma: 187 images
  Normal cases: 631 images
  squamous cell carcinoma: 260 images

Splitting dataset (train=0.7, val=0.15, test=0.15)...
Train: 1074 images
Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Processing val set...
Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s] 

Processing test set...
Processing test: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 231/231 [00:06<00:00, 33.55it/s] 

============================================================
Preprocessing complete!
Successfully processed: 1535 images
Failed: 0 images
Processed images saved to: processed_data
============================================================
Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Processing val set...
Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s] 

Processing test set...
Processing test: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 231/231 [00:06<00:00, 33.55it/s] 

============================================================
Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Processing val set...
Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s] 

Processing test set...
Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Processing val set...
Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s] 
Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Validation: 230 images
Test: 231 images

Processing train set...
Validation: 230 images
Test: 231 images

Validation: 230 images
Test: 231 images
Validation: 230 images
Validation: 230 images
Validation: 230 images
Test: 231 images

Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]


Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]


Processing val set...
Processing val set...
Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s]

Processing test set...
Processing test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s]

Processing test set...
Processing test: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 231/231 [00:06<00:00, 33.55it/s]

============================================================
Preprocessing complete!
Successfully processed: 1535 images
Failed: 0 images
Processed images saved to: processed_data
============================================================


---

## Verification Commands

### Verify Preprocessing Results
```bash
python verify_preprocessing.py
```

**Expected Output Format:**
```
======================================================================
PREPROCESSING VERIFICATION REPORT
======================================================================

📋 PREPROCESSING SETTINGS:
----------------------------------------------------------------------
  Image size: [224, 224]
  Normalized: True
  Grayscale: False

📊 ORIGINAL DATASET STATISTICS:
----------------------------------------------------------------------
  [Class counts from metadata]

📁 DATASET SPLITS:
----------------------------------------------------------------------
TRAIN SET:
  [Class distribution in train set]

VAL SET:
  [Class distribution in val set]

TEST SET:
  [Class distribution in test set]
----------------------------------------------------------------------

TRAIN SET:
  adenocarcinoma                :  236 images
  Benign cases                  :   84 images
  large cell carcinoma          :  131 images
  Normal cases                  :  441 images
  squamous cell carcinoma       :  182 images
  TOTAL                         : 1074 images

VAL SET:
  adenocarcinoma                :   50 images
  Benign cases                  :   18 images
  large cell carcinoma          :   28 images
  Normal cases                  :   95 images
  squamous cell carcinoma       :   39 images
  TOTAL                         :  230 images

TEST SET:
  adenocarcinoma                :   51 images
  Benign cases                  :   18 images
  large cell carcinoma          :   28 images
  Normal cases                  :   95 images
  squamous cell carcinoma       :   39 images
  TOTAL                         :  231 images

⚖️  CLASS DISTRIBUTION ANALYSIS:
----------------------------------------------------------------------
  [Imbalance analysis]

🔍 TESTING DATA LOADER:
----------------------------------------------------------------------
  ✓ Data loaders created successfully
  ✓ Number of classes: 5
  ✓ Classes: [class names]
  ✓ Train batches: [number]
  ✓ Validation batches: [number]
  ✓ Test batches: [number]

======================================================================
✅ PREPROCESSING VERIFICATION COMPLETE
   Your dataset is ready for training!
======================================================================
```

======================================================================
IMAGE FORMAT DISTRIBUTION ANALYSIS
======================================================================

📊 ORIGINAL DATASET - FORMAT DISTRIBUTION BY CLASS:
----------------------------------------------------------------------

adenocarcinoma:
  Total: 337 images
  PNG: 337 (100.0%)
  JPG: 0 (0.0%)
  ⚠️  WARNING: All images are PNG - format-class correlation detected!

Benign cases:
  Total: 120 images
  PNG: 0 (0.0%)
  JPG: 120 (100.0%)
  ⚠️  WARNING: All images are JPG - format-class correlation detected!

large cell carcinoma:
  Total: 187 images
  PNG: 187 (100.0%)
  JPG: 0 (0.0%)
  ⚠️  WARNING: All images are PNG - format-class correlation detected!

Normal cases:
  Total: 631 images
  PNG: 203 (32.2%)
  JPG: 428 (67.8%)

squamous cell carcinoma:
  Total: 260 images
  PNG: 260 (100.0%)
  JPG: 0 (0.0%)
  ⚠️  WARNING: All images are PNG - format-class correlation detected!

======================================================================
📈 OVERALL STATISTICS:
----------------------------------------------------------------------
Total images: 1535
PNG: 987 (64.3%)
JPG: 548 (35.7%)
---

## Troubleshooting Notes

### Issue: File Access Error During Installation
**Error:**
```
ERROR: Could not install packages due to an OSError: [WinError 32] The process cannot access the file because it is being used by another process
```

**Solution:** Close any Python processes or IDEs that might be using the packages, then retry installation.

---

### Issue: ModuleNotFoundError
**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:** Install packages using `pip install -r requirements.txt`

---

### Issue: UnicodeEncodeError in Windows Console
**Error:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 0
```

**Solution:** Use ASCII characters instead of Unicode symbols in print statements for Windows compatibility.

---

## Important File Names

- Preprocessing script: `preprocess_ct_scans.py`
- Helper script: `run_preprocessing.py`
- Verification script: `verify_preprocessing.py`
- Data loader: `data_loader.py`
- Training example: `example_training.py`
- Requirements: `requirements.txt`

---

## Package Versions Installed

- torch: 2.9.0
- torchvision: 0.24.0+cpu
- numpy: 1.24.3
- opencv-python: 4.8.1.78
- Pillow: 10.0.0
- scikit-learn: 1.4.1.post1
- tqdm: 4.65.0
- matplotlib: 3.8.0
- seaborn: 0.13.2

---

## Next Steps After Preprocessing

1. Verify preprocessing: `python verify_preprocessing.py`
2. Start training: `python example_training.py`
3. Monitor training progress and adjust hyperparameters as needed

---

## Date Created
Document created for future reference to maintain consistency in preprocessing pipeline setup and troubleshooting.

