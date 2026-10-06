# KANInformer Decomposition Leakage Comparison

This repository contains two independent experiment folders for comparing the same paper-replica KANInformer pipeline under one controlled difference:

```text
VMD-CA-EWT decomposition before train/test split
vs
VMD-CA-EWT decomposition after train/test split
```

## Folder Structure

```text
original_dataleak/
no_dataleak/
results.py
```

## 1. Original Data-Leak Paper Replica

This follows the suspected leakage order where VMD-CA-EWT decomposition is applied to the full seasonal wind-speed sequence before chronological splitting.

```powershell
cd original_dataleak
pip install -r requirements.txt
python run.py
```

## 2. No-Data-Leak Paper Replica

This follows the same paper-replica architecture, but performs chronological splitting before VMD-CA-EWT decomposition.

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
