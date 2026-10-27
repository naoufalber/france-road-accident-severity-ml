"""Reproducible person-level injury severity benchmark on BAAC 2021-2023."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, accuracy_score,
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
CAT = ["catv", "choc", "manv", "motor", "catu", "place", "trajet",
       "secu1", "secu2", "secu3", "lum", "atm", "agg", "int", "col",
       "surf", "catr", "mois", "weekday"]
NUM = ["hour", "vma", "lat", "long"]
LOCATION = ["surf", "catr", "vma"]


def collapse_locations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per accident; ambiguous road attributes become missing."""
    groups = df.groupby("Num_Acc", sort=True, dropna=False)[LOCATION]
    distinct = groups.nunique(dropna=True)
    result = groups.first().mask(distinct.gt(1)).reset_index()
    audit = {
        "location_rows": len(df),
        "unique_accidents": len(result),
        "accidents_with_multiple_location_rows": int(df.Num_Acc.value_counts().gt(1).sum()),
        "conflicting_location_attributes": {c: int(distinct[c].gt(1).sum()) for c in LOCATION},
    }
    return result, audit


def parse_hour(values: pd.Series) -> pd.Series:
    """Accept HH:MM and legacy HHMM, reject invalid hours/minutes."""
    compact = values.astype("string").str.strip().str.replace(":", "", regex=False).str.zfill(4)
    valid = compact.str.fullmatch(r"\d{4}", na=False)
    hours = pd.to_numeric(compact.str[:2], errors="coerce")
    minutes = pd.to_numeric(compact.str[2:], errors="coerce")
    return hours.where(valid & hours.between(0, 23) & minutes.between(0, 59)).astype(float)


def target_from_grav(values: pd.Series) -> pd.Series:
    """Official BAAC: 1 unhurt, 2 killed, 3 hospitalized, 4 slight injury."""
    if not values.isin([1, 2, 3, 4]).all():
        raise ValueError("Filter unknown severity codes before creating the target.")
    return values.isin([2, 3]).astype(int)


def load_year(root: Path, year: int) -> tuple[pd.DataFrame, dict, dict]:
    aliases = {"caracteristiques", "caractéristiques", "carcteristiques", "caract"}
    tables, hashes = {}, {}
    for kind in ["carac", "lieux", "usagers", "vehicules"]:
        candidates = [p for p in (root / str(year)).glob("*.csv")
                      if p.stem.lower().rsplit("-", 1)[0] in (aliases if kind == "carac" else {kind})]
        if len(candidates) != 1:
            raise ValueError(f"Expected one {kind}-{year}.csv in {root / str(year)}; found {len(candidates)}")
        path = candidates[0]
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        table = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
        table.columns = table.columns.str.strip()
        table = table.rename(columns={"Accident_Id": "Num_Acc", "Num_Accident": "Num_Acc"})
        for col in ["Num_Acc", "id_vehicule", "id_usager"]:
            if col in table:
                table[col] = table[col].str.replace(r"\s+", "", regex=True)
                if table[col].isna().any() or table[col].eq("").any():
                    raise ValueError(f"Missing key {col} in {path.name}")
        tables[kind] = table
    users = tables["usagers"]
    if users.duplicated(["Num_Acc", "id_usager"]).any():
        raise ValueError(f"Duplicate user records in {year}")
    locations, audit = collapse_locations(tables["lieux"])
    features = {
        "vehicules": (["Num_Acc", "id_vehicule"], ["catv", "choc", "manv", "motor"]),
        "carac": (["Num_Acc"], ["jour", "mois", "hrmn", "lum", "atm", "agg", "int", "col", "lat", "long"]),
    }
    for kind, (keys, columns) in features.items():
        users = users.merge(tables[kind][keys + columns], on=keys, how="left", validate="many_to_one", indicator=True)
        if users["_merge"].ne("both").any():
            raise ValueError(f"Unmatched {kind} references in {year}")
        users = users.drop(columns="_merge")
    users = users.merge(locations, on="Num_Acc", how="left", validate="many_to_one", indicator=True)
    if users["_merge"].ne("both").any():
        raise ValueError(f"Unmatched locations in {year}")
    users = users.drop(columns="_merge")
    if len(users) != len(tables["usagers"]):
        raise ValueError("Join changed the number of people")
    users["grav"] = pd.to_numeric(users["grav"], errors="coerce")
    audit.update(year=year, input_users=len(users), unknown_severity=int((~users.grav.isin([1, 2, 3, 4])).sum()))
    users = users.loc[users.grav.isin([1, 2, 3, 4])].copy()
    users["target"] = target_from_grav(users.grav)
    users["year"] = year
    users["accident_key"] = str(year) + ":" + users.Num_Acc
    users["hour"] = parse_hour(users.hrmn)
    date = pd.to_datetime({"year": year, "month": pd.to_numeric(users.mois), "day": pd.to_numeric(users.jour)}, errors="coerce")
    users["weekday"] = date.dt.dayofweek
    for col in NUM:
        users[col] = pd.to_numeric(users[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    users["vma"] = users.vma.where(users.vma.between(1, 130))
    users["lat"] = users.lat.where(users.lat.between(-90, 90))
    users["long"] = users.long.where(users.long.between(-180, 180))
    users.loc[users.lat.eq(0) & users.long.eq(0), ["lat", "long"]] = np.nan
    for col in CAT:
        codes = pd.to_numeric(users[col], errors="coerce").replace(-1, np.nan)
        users[col] = codes.astype("Int64").astype("string").fillna("unknown").astype(object)
    audit.update(retained_users=len(users), positive_users=int(users.target.sum()), accidents=int(users.accident_key.nunique()))
    return users, audit, hashes


def evaluate(y, probability):
    prediction = (probability >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, prediction),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "precision": precision_score(y, prediction, zero_division=0),
            "recall": recall_score(y, prediction, zero_division=0),
            "f1": f1_score(y, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y, probability),
            "average_precision": average_precision_score(y, probability)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/validated")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    frames, audits, hashes = [], [], {}
    for year in (2021, 2022, 2023):
        frame, audit, digest = load_year(args.data_dir, year)
        frames.append(frame); audits.append(audit); hashes.update(digest)
        print(json.dumps(audit), flush=True)
    data = pd.concat(frames, ignore_index=True)
    train, test = data.loc[data.year < 2023], data.loc[data.year == 2023]
    overlap = set(train.accident_key) & set(test.accident_key)
    if overlap:
        raise ValueError("An accident occurs in both train and test")
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), CAT),
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), NUM),
    ])
    parameters = dict(n_estimators=120, max_depth=18, min_samples_leaf=5,
                      max_features="sqrt", class_weight="balanced_subsample", random_state=42)
    model = Pipeline([("preprocess", preprocess),
                      ("model", RandomForestClassifier(**parameters, n_jobs=args.jobs))])
    columns = CAT + NUM
    print("Training fixed Random Forest on 2021-2022...", flush=True)
    model.fit(train[columns], train.target)
    probability = model.predict_proba(test[columns])[:, 1]
    baseline = DummyClassifier(strategy="prior").fit(train[columns], train.target)
    baseline_probability = baseline.predict_proba(test[columns])[:, 1]
    metrics = pd.DataFrame([
        {"model": "Majority / training-prior baseline", **evaluate(test.target, baseline_probability)},
        {"model": "Random Forest (fixed configuration)", **evaluate(test.target, probability)},
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame([{k:v for k,v in a.items() if not isinstance(v,dict)} for a in audits]).to_csv(args.output_dir / "data_summary.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ConfusionMatrixDisplay.from_predictions(test.target, probability >= .5, display_labels=["Unhurt / slight", "Killed / hospitalized"], cmap="Blues", ax=axes[0], colorbar=False, values_format=",d")
    axes[0].set_title("2023 holdout | threshold 0.50")
    PrecisionRecallDisplay.from_predictions(test.target, probability, ax=axes[1], name="Random Forest", color="#175D6B")
    axes[1].axhline(test.target.mean(), color="#D99035", linestyle="--", label=f"Positive rate: {test.target.mean():.1%}")
    axes[1].set_title("Precision-recall trade-off"); axes[1].legend(loc="upper right")
    fig.suptitle("French road accidents · person-level severity", fontsize=16, weight="bold")
    fig.savefig(args.output_dir / "evaluation.png", dpi=170); plt.close(fig)
    counts = data.groupby(["year", "grav"]).size().unstack(fill_value=0)
    counts.columns = ["Unhurt", "Killed", "Hospitalized", "Slight injury"]
    ax = counts.plot.bar(stacked=True, color=["#B8D7D3", "#9E2A2B", "#E49C45", "#3D7B94"], figsize=(9, 4.8), rot=0)
    ax.set(title="Recorded road users by injury category", xlabel="BAAC year", ylabel="People (one row per user)")
    ax.legend(title="Official BAAC category", bbox_to_anchor=(1.02,1), loc="upper left")
    ax.figure.tight_layout(); ax.figure.savefig(args.output_dir / "severity_distribution.png", dpi=170); plt.close(ax.figure)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
                "train_years": [2021, 2022], "test_year": 2023, "train_rows": len(train), "test_rows": len(test),
                "train_positive_rate": float(train.target.mean()), "test_positive_rate": float(test.target.mean()),
                "accident_overlap": len(overlap), "target_positive_codes": [2, 3], "threshold": .5,
                "features": columns, "parameters": parameters, "source_sha256": hashes,
                "audit": audits, "confusion_matrix": confusion_matrix(test.target, probability >= .5).tolist(),
                "limitations": ["Retrospective classification of people already involved in a recorded injury accident.",
                                "Features include information recorded after the collision; no claim of pre-crash prediction.",
                                "No tuning, probability calibration, SHAP rerun or subgroup validation in this benchmark."]}
    (args.output_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
