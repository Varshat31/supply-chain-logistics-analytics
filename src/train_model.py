 
import argparse
import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "dataset", "DataCoSupplyChainDataset.csv")
OUT = os.path.join(ROOT, "outputs")
MODELS = os.path.join(ROOT, "models")

TARGET_CANDIDATES = ["Late_delivery_risk", "late_delivery_risk"]
DROP_EXACT = {
    "late_delivery", "delivery_status", "days_for_shipping_real",
    "actual_shipping_days", "actual_delivery_date", "date_of_delivery",
    "order_status", "shipping_date", "delivery_date"
}

def read_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\nDownload DataCo Smart Supply Chain Dataset "
            "and place its CSV at dataset/DataCoSupplyChainDataset.csv."
        )
    # DataCo files are commonly tab-separated; delimiter inference handles CSV/TSV.
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="latin1")
    except Exception:
        df = pd.read_csv(path, encoding="latin1")
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df

def find_target(df):
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(
        "Could not find target column 'Late_delivery_risk'. "
        f"Columns found: {list(df.columns)}"
    )

def prepare_features(df, target):
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    # Keep only binary target labels.
    binary = y.isin([0, 1])
    df, y = df.loc[binary].copy(), y.loc[binary]
    if y.nunique() != 2:
        raise ValueError("Target must contain both classes 0 and 1.")

    # Remove identifiers and post-outcome fields to reduce leakage.
    drop_cols = []
    for col in df.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
        if col == target:
            continue
        if normalized in DROP_EXACT:
            drop_cols.append(col)
        if normalized in {"order_id", "customer_id", "product_card_id", "order_item_id"}:
            drop_cols.append(col)

    X = df.drop(columns=list(set(drop_cols + [target])), errors="ignore").copy()

    # Remove columns that are entirely empty or constant.
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]

    # Parse date/time-like columns into useful numeric components.
    for col in list(X.columns):
        if any(token in col.lower() for token in ["date", "time"]):
            parsed = pd.to_datetime(X[col], errors="coerce")
            if parsed.notna().mean() > 0.5:
                X[col + "_year"] = parsed.dt.year
                X[col + "_month"] = parsed.dt.month
                X[col + "_day"] = parsed.dt.day
                X[col + "_weekday"] = parsed.dt.dayofweek
                X.drop(columns=[col], inplace=True)

    # Convert numeric-looking text columns where safe.
    for col in X.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().mean() > 0.9:
            X[col] = converted

    # Drop extremely high-cardinality text columns (usually names/free text).
    high_card = [
        c for c in X.select_dtypes(include=["object"]).columns
        if X[c].nunique(dropna=True) > min(500, max(100, int(0.5 * len(X))))
    ]
    X.drop(columns=high_card, inplace=True, errors="ignore")

    return X, y, drop_cols, high_card

def main():
    parser = argparse.ArgumentParser(description="Train shipment delay risk model.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to dataset CSV/TSV")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    raw = read_dataset(args.data)
    target = find_target(raw)
    X, y, dropped, high_card = prepare_features(raw, target)

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])

    model = RandomForestClassifier(
    n_estimators=80,
    max_depth=18,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        "Value": [
            accuracy_score(y_test, pred),
            precision_score(y_test, pred, zero_division=0),
            recall_score(y_test, pred, zero_division=0),
            f1_score(y_test, pred, zero_division=0),
            roc_auc_score(y_test, proba) if y_test.nunique() == 2 else np.nan
        ]
    }
    pd.DataFrame(metrics).to_csv(os.path.join(OUT, "model_metrics.csv"), index=False)

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    pd.DataFrame(cm, index=["Actual_0", "Actual_1"],
                 columns=["Predicted_0", "Predicted_1"]).to_csv(
        os.path.join(OUT, "confusion_matrix.csv")
    )

    pred_out = pd.DataFrame({
        "Record_ID": idx_test,
        "Actual_Late_Risk": y_test.values,
        "Predicted_Late_Risk": pred,
        "Late_Risk_Probability": proba,
        "Risk_Band": np.where(proba >= 0.70, "High",
                     np.where(proba >= 0.40, "Medium", "Low"))
    })
    pred_out.to_csv(os.path.join(OUT, "predictions.csv"), index=False)

    # Export the full cleaned feature table and attach out-of-sample predictions.
    all_pred = pipeline.predict_proba(X)[:, 1]
    powerbi = X.copy()
    powerbi.insert(0, "Record_ID", X.index)
    powerbi["Actual_Late_Risk"] = y.values
    powerbi["Predicted_Late_Risk_Probability"] = all_pred
    powerbi["Predicted_Late_Risk"] = (all_pred >= 0.5).astype(int)
    powerbi["Risk_Band"] = np.where(all_pred >= 0.70, "High",
                           np.where(all_pred >= 0.40, "Medium", "Low"))
    powerbi.to_csv(os.path.join(OUT, "powerbi_orders.csv"), index=False)

    summary = pd.DataFrame({
        "KPI": ["Total Records", "Late Risk Orders", "Not Late Risk Orders",
                "Late Risk Rate", "High Risk Predictions"],
        "Value": [len(y), int((y == 1).sum()), int((y == 0).sum()),
                  float((y == 1).mean()), int((all_pred >= 0.70).sum())]
    })
    summary.to_csv(os.path.join(OUT, "summary_metrics.csv"), index=False)

    # Feature importance, when the transformed feature names are available.
    try:
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        fi = pd.DataFrame({"Feature": names, "Importance": importances})
        fi.sort_values("Importance", ascending=False).head(30).to_csv(
            os.path.join(OUT, "feature_importance.csv"), index=False
        )
    except Exception as exc:
        print("Feature importance export skipped:", exc)

    joblib.dump(pipeline, os.path.join(MODELS, "delivery_delay_pipeline.joblib"))

    print("\nMODEL TRAINING COMPLETE")
    print(pd.DataFrame(metrics).to_string(index=False))
    print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))
    print(f"\nOutputs saved in: {OUT}")
    print("Power BI file: outputs/powerbi_orders.csv")

if __name__ == "__main__":
    main()
