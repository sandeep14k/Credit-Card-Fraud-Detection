# Credit Card Fraud Detection

Leakage-safe fraud detection on the ULB credit card dataset (284,807 transactions, 492 frauds = 0.172%).

## Project structure
```
fraud-detection/
├── data/creditcard.csv      
├── outputs/                 
├── download_data.py         
├── train.py                
├── predict.py               
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
