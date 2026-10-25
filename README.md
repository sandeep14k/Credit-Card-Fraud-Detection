# Credit Card Fraud Detection

Leakage-safe fraud detection on the ULB credit card dataset (284,807 transactions, 492 frauds = 0.172%).

## Project structure
```
fraud-detection/
├── data/creditcard.csv      <- you download this (not committed to git)
├── outputs/                 <- generated: metrics, plots, saved model
├── download_data.py         <- fetches the dataset from Kaggle
├── train.py                 <- full pipeline (split, CV, tuning, test, plots)
├── predict.py               <- score new transactions with the saved model
├── requirements.txt
└── README.md
```

## Step-by-step

1. **Set up environment**
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows   |   source venv/bin/activate  (Mac/Linux)
   pip install -r requirements.txt
   ```
2. **Get the data** (pick one)
   - Manual: go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud , download the zip, extract `creditcard.csv` into `data/`.
   - Automatic: create a Kaggle API token (Kaggle > Settings > API > Create New Token), place `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`), then `python download_data.py`.
3. **Quick test** on 20% of data: `python train.py --sample 0.2`
4. **Full run**: `python train.py` (Random Forest CV takes several minutes)
5. **Look in `outputs/`**: `cv_results.csv`, `val_results.csv`, `test_metrics.json`, `pr_curves.png`, `threshold_tradeoff.png`, `confusion_matrices.png`, `feature_importance.png`, `fraud_model.joblib`.
6. **Tune the tradeoff**: `python train.py --min-precision 0.90` raises the precision floor (recall drops); lower it for more recall.

## How each resume bullet maps to code
| Resume bullet | Where |
|---|---|
| Leakage-safe pipeline, stratified CV, imbalance handling | Split once; scaler + `RandomUnderSampler` + `class_weight` live inside `Pipeline` in `build_models()`; `StratifiedKFold` in `cross_validate` on train only |
| Compare LR, DT, KNN, RF on PR-AUC, ROC-AUC, P/R/F1 | `cv_results.csv`, `val_results.csv` |
| Threshold tuning on validation | `tune_threshold()` - max recall subject to a precision floor |
| Confusion matrices and feature importance | `confusion_matrices.png`, `feature_importance.png`, FP/FN counts in `test_metrics.json` |

## Important
Your results will not match any number exactly. Run it, then put **your own** test-set precision/recall on the resume (the 87% / 74% in the bullet is a target to expect, not a guarantee). Be ready to explain: why PR-AUC beats accuracy on imbalanced data, why scaling must be inside the CV pipeline, why the threshold is tuned on validation and not test, and what a false positive vs false negative costs.


