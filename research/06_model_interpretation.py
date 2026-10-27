import os
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ========= Chemins =========
DATA_INTERIM = "../data_interim"
INPUT_FILE = os.path.join(DATA_INTERIM, "accidents_FE_2021_2023.csv")


def preprocess(df: pd.DataFrame):
    # Extraire l'heure à partir de hrmn (ex: '1325' -> 13)
    df["hour"] = df["hrmn"].astype(str).str.zfill(4).str[:2].astype(int)
    df = df.drop(columns=["hrmn"])

    # Cible
    y = df["grav_bin"]

    # Features : toutes les colonnes sauf grav et grav_bin
    X = df.drop(columns=["grav", "grav_bin"])

    # lat/long en float
    for col in ["lat", "long"]:
        X[col] = X[col].astype(str).str.replace(",", ".").astype(float)

    return X, y


def main():
    print("🔹 Chargement du dataset…")
    df = pd.read_csv(INPUT_FILE)
    X, y = preprocess(df)

    print("Shape total :", X.shape)

    # 1) Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("Train shape :", X_train.shape)
    print("Test shape  :", X_test.shape)

    # 2) Modèle final (hyperparamètres Optuna)
    print("\n🔹 Entraînement du modèle final…")

    model = RandomForestClassifier(
        n_estimators=206,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="log2",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # 3) Préparation SHAP (échantillon léger)
    print("\n🔹 Préparation SHAP…")

    sample_size = 300
    if len(X_test) > sample_size:
        X_shap = X_test.sample(sample_size, random_state=42)
    else:
        X_shap = X_test

    print("Taille utilisée pour SHAP :", X_shap.shape)

    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(X_shap, check_additivity=False)

    # Cas SHAP multi-output → on récupère un seul “canal”
    if isinstance(shap_values_all, list):
        shap_values_class1 = shap_values_all[1]
    else:
        # si c'est (n_samples, n_features, 2), on prend la dernière dimension moyenne
        if shap_values_all.ndim == 3:
            shap_values_class1 = shap_values_all[..., 1]
        else:
            shap_values_class1 = shap_values_all

    print("Shape shap_values_class1 :", shap_values_class1.shape)

    # 4) SHAP Summary Plot
    print("\n🔹 SHAP Summary Plot (global)…")
    shap.summary_plot(shap_values_class1, X_shap, plot_type="dot", show=False)
    plt.title("SHAP Summary - Gravité sévère")
    plt.tight_layout()
    plt.show()

    # 5) SHAP Bar Plot
    shap.summary_plot(shap_values_class1, X_shap, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar) - Gravité sévère")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
