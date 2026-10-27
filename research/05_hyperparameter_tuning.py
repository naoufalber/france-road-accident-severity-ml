import os
import optuna
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

DATA_INTERIM = "../data_interim"
INPUT_FILE = os.path.join(DATA_INTERIM, "accidents_FE_2021_2023.csv")


# =========================
#  Prétraitement (même logique que 04_modeling.py)
# =========================
def preprocess(df: pd.DataFrame):
    # Extraire l'heure à partir de hrmn (ex: '1325' -> 13)
    df["hour"] = df["hrmn"].astype(str).str.zfill(4).str[:2].astype(int)
    df = df.drop(columns=["hrmn"])

    # Cible
    y = df["grav_bin"]

    # Features = toutes les colonnes sauf grav & grav_bin
    X = df.drop(columns=["grav", "grav_bin"])

    # Conversion lat/long en float
    for col in ["lat", "long"]:
        X[col] = X[col].astype(str).str.replace(",", ".").astype(float)

    return X, y


# =========================
#  Fonction objectif pour Optuna
# =========================
def objective(trial):

    # On recharge à chaque essai pour être sûr
    df = pd.read_csv(INPUT_FILE)
    X, y = preprocess(df)

    # Split train / valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Espace de recherche des hyperparamètres
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    f1 = f1_score(y_valid, y_pred)

    # On optimise le F1-score (gravité sévère vs non sévère)
    return f1


def main():

    print("Chargement des données...")
    df = pd.read_csv(INPUT_FILE)
    X, y = preprocess(df)

    print("Shape total :", X.shape, "| Positives :", y.sum(), "| Negatives :", (y == 0).sum())

    # =========================
    #  Lancement d'Optuna
    # =========================
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)  # tu peux augmenter à 50+ plus tard

    print("\n===== Résultats Optuna =====")
    print("Best F1-score :", study.best_value)
    print("Best params  :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # =========================
    #  Ré-entraînement avec les meilleurs paramètres
    # =========================
    print("\nRé-entraînement du modèle avec les meilleurs hyperparamètres...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    best_model = RandomForestClassifier(
        n_estimators=study.best_params["n_estimators"],
        max_depth=study.best_params["max_depth"],
        min_samples_split=study.best_params["min_samples_split"],
        min_samples_leaf=study.best_params["min_samples_leaf"],
        max_features=study.best_params["max_features"],
        n_jobs=-1,
        random_state=42
    )

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n===== Performance modèle optimisé =====")
    print(f"Accuracy : {acc:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nClassification report :")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Random Forest (Optuna tuned)")
    plt.show()

    # Importance des variables
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nTop 10 features (modèle optimisé) :")
    print(importances.head(10))

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importances.head(10), x="importance", y="feature")
    plt.title("Top 10 Feature Importances (Optimised RF)")
    plt.show()


if __name__ == "__main__":
    main()
