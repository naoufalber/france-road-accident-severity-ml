import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_INTERIM = "../data_interim"
FILE = os.path.join(DATA_INTERIM, "accidents_usagers_2021_2023.csv")

def main():
    df = pd.read_csv(FILE)
    print("\n=== Aperçu des données ===")
    print(df.head())

    print("\n=== Infos sur les colonnes ===")
    print(df.info())

    print("\n=== Statistiques ===")
    print(df.describe())

    # Analyse de la cible grav
    if "grav" in df.columns:
        print("\n=== Distribution de la gravité ===")
        print(df["grav"].value_counts())

        plt.figure(figsize=(6,4))
        sns.countplot(data=df, x="grav")
        plt.title("Distribution de la gravité des usagers (2023)")
        plt.show()
    else:
        print("\n⚠️ Colonne 'grav' non trouvée, on regardera le nom exact ensemble.")

if __name__ == "__main__":
    main()
