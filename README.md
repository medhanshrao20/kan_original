# KANInformer Leakage Comparison

This repository contains two independent experiment folders for comparing a paper-style data-leaky pipeline against a chronological no-data-leak pipeline.

## Folder Structure

```text
original_dataleak/
no_dataleak/
results.py
```

## 1. Original Data-Leak Pipeline

This follows the paper-style order where full-season cleaning, feature selection, decomposition, and scaling happen before the train/validation/test split.

```powershell
cd original_dataleak
pip install -r requirements.txt
python run.py
```

## 2. No-Data-Leak Pipeline

This follows the same core architecture, but splits chronologically first and fits preprocessing decisions using only the training part.

```powershell
cd ..\no_dataleak
pip install -r requirements.txt
python run.py
```

## 3. Combined Comparison

After running both folders:

```powershell
cd ..
python results.py
```

Combined outputs are written to:

```text
combined_results/
```

Each experiment also writes its own outputs to:

```text
original_dataleak/results/
no_dataleak/results/
```
