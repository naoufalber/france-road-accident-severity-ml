# Data setup

Source: **French Ministry of the Interior / ONISR**, annual BAAC injury-accident open data, years **2021, 2022 and 2023**.

- [Official dataset and downloads](https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024)
- [ONISR open-data guidance](https://www.onisr.securite-routiere.gouv.fr/outils-statistiques/open-data)
- [Official BAAC coding guide](https://www.onisr.securite-routiere.gouv.fr/sites/default/files/2019-02/Guide%20BAAC%202017.pdf), printed page 78: injury severity.

The dataset page lists the **Licence Ouverte / Open Licence**. Retain the producer and source when reusing it. Raw person-level data and intermediate tables are deliberately excluded from Git; only aggregate outputs are committed.

## Download and arrange

For each year, download the four CSV resources named **caractéristiques**, **lieux**, **véhicules** and **usagers**. Use the regular vehicle table, not the separate registered-vehicle (`immatricules`) resource. Store them as:

| Directory | Required CSV names |
|---|---|
| `data/raw/2021/` | `caracteristiques-2021.csv`, `lieux-2021.csv`, `vehicules-2021.csv`, `usagers-2021.csv` |
| `data/raw/2022/` | `caracteristiques-2022.csv`, `lieux-2022.csv`, `vehicules-2022.csv`, `usagers-2022.csv` |
| `data/raw/2023/` | `caracteristiques-2023.csv`, `lieux-2023.csv`, `vehicules-2023.csv`, `usagers-2023.csv` |

The loader also accepts the original archive's `carcteristiques-YYYY.csv` typo, `caractéristiques-YYYY.csv`, or `caract-YYYY.csv`. Keep exactly one characteristics file per year. CSV files must be semicolon-separated UTF-8, with the official column names. The 2022 `Accident_Id` key is normalized automatically.

You can use a different directory with `python src/train.py --data-dir /path/to/data_raw`.

## Snapshot provenance

The committed run uses the 12 raw CSVs in the supplied academic archive, not a newly downloaded dataset. File SHA-256 hashes are recorded in [run.json](../reports/validated/run.json). Upstream corrections may change future results. In this snapshot, the 2023 locations file has multiple road rows for some accidents; the loader preserves a single person-level observation and marks conflicting selected road attributes as missing.

The official publisher cautions that hospitalisation recording has changed since 2018. The target below refers to recorded BAAC categories; it is not a clinical serious-injury scale such as MAIS3+.

| `grav` | Meaning | Binary target |
|---:|---|---:|
| 1 | Unhurt | 0 |
| 2 | Killed | 1 |
| 3 | Hospitalized for more than 24 hours | 1 |
| 4 | Slight injury | 0 |
| Other / missing | Unknown | Excluded |
