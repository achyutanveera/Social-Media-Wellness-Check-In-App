"""Train the Psyber Mentalytics MLP from the Kaggle CSV and export serving artifacts.

Usage (Colab or local):  python train_export.py path/to/smmh.csv [out_dir]

Prints three holdout accuracies so you can see what the model actually adds:
  1. rule        - thresholding impact_sum (how the risk label is defined)
  2. mlp_full    - the notebook's model (impact_sum is one of its inputs)
  3. mlp_no_sum  - same MLP with impact_sum removed from the inputs
"""
import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import preprocess as P


def make_mlp():
    # Same settings as the notebook
    return MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=1000, random_state=42,
                         activation="relu", solver="lbfgs")


def main(csv_path, out_dir="artifacts"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    df = P.from_kaggle_raw(pd.read_csv(csv_path))
    impact = df[P.ITEMS].sum(axis=1)
    y = impact.apply(P.risk_from_impact)
    print("rows:", len(df), "| class counts:", y.value_counts().to_dict())

    X_full = P.build_features(df, with_impact=True)
    X_nosum = P.build_features(df, with_impact=False)
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)

    rule_acc = accuracy_score(y[idx_test], impact[idx_test].apply(P.risk_from_impact))
    m_full = make_mlp().fit(X_full.loc[idx_train], y[idx_train])
    m_nosum = make_mlp().fit(X_nosum.loc[idx_train], y[idx_train])
    acc_full = accuracy_score(y[idx_test], m_full.predict(X_full.loc[idx_test]))
    acc_nosum = accuracy_score(y[idx_test], m_nosum.predict(X_nosum.loc[idx_test]))
    print(f"holdout accuracy  rule={rule_acc:.3f}  mlp_full={acc_full:.3f}  mlp_no_sum={acc_nosum:.3f}")
    print("\nmlp_no_sum report:\n", classification_report(y[idx_test], m_nosum.predict(X_nosum.loc[idx_test])))

    # Refit on all rows for deployment
    m_full = make_mlp().fit(X_full, y)
    m_nosum = make_mlp().fit(X_nosum, y)
    joblib.dump(m_full, f"{out_dir}/model_full.joblib")
    joblib.dump(m_nosum, f"{out_dir}/model_no_sum.joblib")
    meta = {"columns_full": list(X_full.columns), "columns_no_sum": list(X_nosum.columns),
            "classes": list(m_full.classes_), "high_cut": P.HIGH_CUT, "med_cut": P.MED_CUT,
            "holdout_accuracy": {"rule": rule_acc, "mlp_full": acc_full, "mlp_no_sum": acc_nosum},
            "n_rows": len(df)}
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=2)
    print("saved to", out_dir)


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
