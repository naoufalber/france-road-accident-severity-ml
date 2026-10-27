import os
import pandas as pd

DATA_RAW = "../data_raw"
DATA_INTERIM = "../data_interim"

# On garde 3 années
YEARS = [2021, 2022, 2023]


def harmonize_keys(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Harmonise les noms de colonnes clés :
    - Accident_Id, Num_Accident, num_acc  --> Num_Acc
    - Vehicle_Id, id_veh, ...            --> id_vehicule (pour usag/veh)
    """

    # mapping lower → original
    lower_to_orig = {c.lower(): c for c in df.columns}

    # ---- clé d'accident : Num_Acc ----
    accident_key_candidates = ["num_acc", "num_accident", "accident_id"]
    if "Num_Acc" not in df.columns:
        for cand in accident_key_candidates:
            if cand in lower_to_orig:
                orig = lower_to_orig[cand]
                df = df.rename(columns={orig: "Num_Acc"})
                break

    # ---- clé de véhicule : id_vehicule (seulement pour usagers / vehicules) ----
    if table_name in ["usag", "veh"]:
        if "id_vehicule" not in df.columns:
            veh_key_candidates = ["id_vehicule", "id_veh", "vehicle_id", "vehicule_id"]
            for cand in veh_key_candidates:
                if cand in lower_to_orig:
                    orig = lower_to_orig[cand]
                    df = df.rename(columns={orig: "id_vehicule"})
                    break

    return df


def load_year(year: int) -> pd.DataFrame:
    year_dir = os.path.join(DATA_RAW, str(year))
    print(f"\n========== Année {year} ==========")

    file_carac = os.path.join(year_dir, f"carcteristiques-{year}.csv")
    file_lieux = os.path.join(year_dir, f"lieux-{year}.csv")
    file_usag  = os.path.join(year_dir, f"usagers-{year}.csv")
    file_veh   = os.path.join(year_dir, f"vehicules-{year}.csv")

    print("Chargement des fichiers...")
    carac = pd.read_csv(file_carac, sep=";", encoding="latin-1", low_memory=False)
    lieux = pd.read_csv(file_lieux, sep=";", encoding="latin-1", low_memory=False)
    usag  = pd.read_csv(file_usag,  sep=";", encoding="latin-1", low_memory=False)
    veh   = pd.read_csv(file_veh,   sep=";", encoding="latin-1", low_memory=False)

    # Harmonisation des colonnes clés
    carac = harmonize_keys(carac, "carac")
    lieux = harmonize_keys(lieux, "lieux")
    usag  = harmonize_keys(usag,  "usag")
    veh   = harmonize_keys(veh,   "veh")

    # Debug si jamais une clé manque encore
    for name, df in [("carac", carac), ("lieux", lieux), ("usag", usag), ("veh", veh)]:
        if "Num_Acc" not in df.columns:
            print(f"\n⚠️ 'Num_Acc' absent dans la table {name} pour {year}")
            print("Colonnes disponibles :", list(df.columns))
        if name in ["usag", "veh"] and "id_vehicule" not in df.columns:
            print(f"\n⚠️ 'id_vehicule' absent dans la table {name} pour {year}")
            print("Colonnes disponibles :", list(df.columns))

    # Vérifications fortes (si ça casse ici, on voit où)
    assert "Num_Acc" in carac.columns
    assert "Num_Acc" in lieux.columns
    assert "Num_Acc" in usag.columns and "id_vehicule" in usag.columns
    assert "Num_Acc" in veh.columns and "id_vehicule" in veh.columns

    # Fusion des tables
    accidents = carac.merge(lieux, on="Num_Acc", how="left")
    veh_usag = veh.merge(usag, on=["Num_Acc", "id_vehicule"], how="right")
    full = veh_usag.merge(accidents, on="Num_Acc", how="left")

    full["year"] = year
    print(f"Shape finale pour {year} :", full.shape)

    return full


def main():
    os.makedirs(DATA_INTERIM, exist_ok=True)

    dfs = []
    for year in YEARS:
        df = load_year(year)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print("\nShape finale toutes années :", df_all.shape)

    output = os.path.join(DATA_INTERIM, "accidents_usagers_2021_2023.csv")
    df_all.to_csv(output, index=False)
    print("\n🚀 Fichier sauvegardé dans :", output)


if __name__ == "__main__":
    main()
