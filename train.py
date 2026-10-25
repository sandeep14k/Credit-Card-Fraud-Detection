
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
OUT = Path("outputs")


def build_models():
    """Each model handles class imbalance its own way (all inside the pipeline)."""
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                           class_weight="balanced", random_state=SEED)),
        ]),
        # KNN has no class_weight -> undersample the majority class (train folds only)
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("under", RandomUnderSampler(sampling_strategy=0.1, random_state=SEED)),
            ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                           class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=SEED)),
        ]),
    }


def scores(model, X):
    return model.predict_proba(X)[:, 1]


def metrics_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def tune_threshold(y_val, proba_val, min_precision):
    """Highest recall subject to precision >= min_precision; fall back to max F1."""
    p, r, t = precision_recall_curve(y_val, proba_val)
    p, r = p[:-1], r[:-1]  # align with thresholds
    ok = p >= min_precision
    if ok.any():
        idx = np.where(ok)[0][np.argmax(r[ok])]
        how = f"max recall with precision >= {min_precision}"
    else:
        f1 = 2 * p * r / (p + r + 1e-12)
        idx = int(np.argmax(f1))
        how = "max F1 (precision target not reachable)"
    return float(t[idx]), how


def plot_confusions(y_val, pv, y_te, pt, thr, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (y, p, title) in zip(axes, [(y_te, pt, "Default threshold 0.5"),
                                         (y_te, pt, f"Tuned threshold {thr:.3f}")]):
        t = 0.5 if "Default" in title else thr
        cm = confusion_matrix(y, (p >= t).astype(int))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Legit", "Fraud"])
        ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
    fig.suptitle(f"{name} - test set confusion matrices")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of data (for quick tests)")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--min-precision", type=float, default=0.85,
                    help="precision floor used when tuning the threshold")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # ---- 1. load + split once -------------------------------------------------
    df = pd.read_csv(args.data)
    if args.sample < 1.0:
        df = df.groupby("Class", group_keys=False).sample(frac=args.sample, random_state=SEED)
    X, y = df.drop(columns="Class"), df["Class"]
    print(f"Rows: {len(df):,} | frauds: {y.sum():,} ({y.mean():.3%})")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp,
                                                random_state=SEED)
    print(f"Train {len(X_tr):,} | Val {len(X_val):,} | Test {len(X_te):,}")

    # ---- 2. cross-validation on TRAIN only ------------------------------------
    models = build_models()
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "precision": "precision",
               "recall": "recall", "f1": "f1"}
    cv_rows = []
    for name, m in models.items():
        print(f"CV: {name} ...")
        res = cross_validate(m, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        cv_rows.append({"model": name, **{k: res[f"test_{k}"].mean() for k in scoring},
                        **{f"{k}_std": res[f"test_{k}"].std() for k in scoring}})
    cv_df = pd.DataFrame(cv_rows).round(4)
    cv_df.to_csv(OUT / "cv_results.csv", index=False)
    print(cv_df[["model", "pr_auc", "roc_auc", "precision", "recall", "f1"]].to_string(index=False))

    # ---- 3. fit on train, score on validation ---------------------------------
    val_rows, val_proba = [], {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = scores(m, X_val)
        val_proba[name] = p
        val_rows.append({"model": name,
                         "pr_auc": average_precision_score(y_val, p),
                         "roc_auc": roc_auc_score(y_val, p)})
    val_df = pd.DataFrame(val_rows).round(4)
    val_df.to_csv(OUT / "val_results.csv", index=False)
    print("\nValidation:\n", val_df.to_string(index=False))

    best = val_df.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    print(f"\nBest model by validation PR-AUC: {best}")

    # PR curves for all models (validation)
    plt.figure(figsize=(6, 5))
    for name, p in val_proba.items():
        pr, rc, _ = precision_recall_curve(y_val, p)
        plt.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_val, p):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves (validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "pr_curves.png", dpi=150)
    plt.close()

    # ---- 4. threshold tuning on VALIDATION ------------------------------------
    thr, how = tune_threshold(y_val, val_proba[best], args.min_precision)
    print(f"Tuned threshold = {thr:.4f} ({how})")

    pr, rc, th = precision_recall_curve(y_val, val_proba[best])
    plt.figure(figsize=(6, 4))
    plt.plot(th, pr[:-1], label="Precision")
    plt.plot(th, rc[:-1], label="Recall")
    plt.axvline(thr, color="k", ls="--", label=f"chosen = {thr:.3f}")
    plt.xlabel("Threshold")
    plt.title(f"{best}: precision/recall vs threshold (validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "threshold_tradeoff.png", dpi=150)
    plt.close()

    # ---- 5. final test evaluation (touched once) ------------------------------
    final = models[best]
    p_te = scores(final, X_te)
    results = {
        "best_model": best,
        "test_pr_auc": float(average_precision_score(y_te, p_te)),
        "test_roc_auc": float(roc_auc_score(y_te, p_te)),
        "default_threshold_0.5": metrics_at(y_te, p_te, 0.5),
        "tuned_threshold": metrics_at(y_te, p_te, thr),
    }
    (OUT / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print("\nTEST RESULTS\n", json.dumps(results, indent=2))
    plot_confusions(y_val, val_proba[best], y_te, p_te, thr, best)

    # ---- 6. explainability ----------------------------------------------------
    clf = final.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=X.columns)
    elif hasattr(clf, "coef_"):
        imp = pd.Series(np.abs(clf.coef_[0]), index=X.columns)  # features are scaled
    else:
        imp = None
    if imp is not None:
        top = imp.sort_values(ascending=True).tail(15)
        plt.figure(figsize=(6, 5))
        top.plot.barh()
        plt.title(f"{best}: top 15 features")
        plt.tight_layout()
        plt.savefig(OUT / "feature_importance.png", dpi=150)
        plt.close()
        imp.sort_values(ascending=False).to_csv(OUT / "feature_importance.csv")

    joblib.dump({"model": final, "threshold": thr}, OUT / "fraud_model.joblib")
    print("\nAll outputs saved in outputs/")


if __name__ == "__main__":
    main()
