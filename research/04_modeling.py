import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

DATA_INTERIM = "../data_interim"
INPUT_FILE = os.path.join(DATA_INTERIM, "accidents_FE_2021_2023.csv")


def preprocess(df):

    # 🔹 Extract hour from "hrmn" (e.g. '1325' -> 13)
    df["hour"] = df["hrmn"].astype(str).str.zfill(4).str[:2].astype(int)

    # 🗺️ Drop hrmn (raw version)
    df = df.drop(columns=["hrmn"])

    # 🎯 Target
    y = df["grav_bin"]

    # 🔧 Features (remove original grav + target)
    X = df.drop(columns=["grav", "grav_bin"])

    # ⚠️ Convert object columns to numeric (lat/long)
    for col in ["lat", "long"]:
        X[col] = X[col].str.replace(",", ".").astype(float)

    return X, y


def main():

    df = pd.read_csv(INPUT_FILE)
    print("Data Loaded :", df.shape)

    # =============================
    # 1️⃣ Preprocess Data
    # =============================
    X, y = preprocess(df)

    # =============================
    # 2️⃣ Train/Test Split
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # =============================
    # 3️⃣ Model : Random Forest
    # =============================
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # =============================
    # 4️⃣ Evaluate Model
    # =============================
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Performance ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))

    # =============================
    # 5️⃣ Confusion Matrix
    # =============================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    # =============================
    # 6️⃣ Feature Importance
    # =============================
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n=== Top Features ===")
    print(importances.head(10))

    plt.figure(figsize=(8,5))
    sns.barplot(data=importances.head(10), x="importance", y="feature")
    plt.title("Top 10 Feature Importances")
    plt.show()


if __name__ == "__main__":
    main()
