import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_INTERIM = "../data_interim"
INPUT_FILE = os.path.join(DATA_INTERIM, "accidents_usagers_2021_2023.csv")
OUTPUT_FILE = os.path.join(DATA_INTERIM, "accidents_FE_2021_2023.csv")


def main():
    df = pd.read_csv(INPUT_FILE)
    print("Shape initiale :", df.shape)

    # 🔥 Supprimer les grav = -1 (inconnus)
    df = df[df["grav"] != -1]
    print("Après suppression grav = -1 :", df.shape)

    # 🎯 Créer la cible binaire
    df["grav_bin"] = df["grav"].apply(lambda x: 1 if x >= 3 else 0)

    print("\nRépartition grav_bin :")
    print(df["grav_bin"].value_counts())

    # 📊 Visualisation de la répartition
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="grav_bin")
    plt.title("Répartition de la gravité binaire (2023)")
    plt.xticks([0,1], ["Gravité faible", "Gravité sévère"])
    plt.show()

    # 💡 Option : on ne garde que les colonnes utiles
    cols_to_keep = [
        "grav_bin", "grav", "catv", "choc", "manv", "motor", "catu",
        "place", "trajet", "secu1", "secu2", "secu3",
        "lum", "atm", "agg", "int", "col", "surf",
        "mois", "jour", "hrmn",
        "lat", "long",
        "vma", "year"
    ]

    df_reduced = df[cols_to_keep]
    print("Shape après sélection colonnes utiles :", df_reduced.shape)

    # 💾 Sauvegarde
    df_reduced.to_csv(OUTPUT_FILE, index=False)
    print("Fichier sauvegardé :", OUTPUT_FILE)


if __name__ == "__main__":
    main()
