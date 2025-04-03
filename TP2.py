
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# INFO834 - TP2 : Parquet
# 
# Pour le bon fonctionnement du programme il faut :
# - installer les packages suivants : pyarrow, pandas, matplotlib, seaborn
# - créer un dossier 'data' dans le même répertoire que ce script et y placer les fichiers 'academies_virgule.csv' et 'villes_virgule.csv'

# Conversion csv -> dataframe
def csv_to_df(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=",")
    df = nettoyage(df)
    return df

# Conversion dataframe -> csv
def df_to_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(filename, sep=",", index=False)

# Nettoyage des données dans le dataframe
def nettoyage(df: pd.DataFrame) -> pd.DataFrame:
    # Ajout d'un zero devant les noms de départements à un chiffre
    df["dep"] = df["dep"].apply(lambda x: f"0{x}" if len(str(x)) == 1 else str(x))
    # Suppression de la dernière lettre pour le département de Lyon
    df["dep"] = df["dep"].apply(lambda x: x[:-1] if x.endswith("M") else x)
    df["dep"] = df["dep"].apply(lambda x: x[:-1] if x.endswith("D") else x)
    return df

# Conversion dataframe -> table
def df_to_table(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df)

# Conversion table -> dataframe
def table_to_df(table: pa.Table) -> pd.DataFrame:
    return table.to_pandas()

# Conversion table -> parquet
def table_to_parquet(table: pa.Table, filename: str) -> None:
    pq.write_table(table, filename)

# Conversion parquet -> table
def parquet_to_table(filename: str) -> pa.Table:
    return pq.read_table(filename)

# Schéma d'une table
def afficher_schema(table: pa.Table) -> None:
    print(table.schema)

# Récupération d'une colonne
def recuperer_colonne(table: pa.Table, col: str) -> pa.Array:
    return table.column(col)

# Ajout d'une colonne
def ajouter_colonne(table: pa.Table, col: str, valeurs: list) -> pa.Table:
    if len(valeurs) != table.num_rows:
        raise ValueError("la longueur de la liste de valeurs ne correspond pas au nombre de lignes de la table.")
    return table.append_column(col, pa.array(valeurs))

# Statistiques sur une colonne
def compute(table: pa.Table, col: str, operation: str) -> float:
    colonne = recuperer_colonne(table, col)
    if operation == "count":
        return pc.count(colonne).as_py()
    elif operation == "count_distinct":
        return pc.count_distinct(colonne).as_py()
    elif operation == "sum":
        return pc.sum(colonne).as_py()
    elif operation == "min":
        return pc.min(colonne).as_py()
    elif operation == "max":
        return pc.max(colonne).as_py()
    elif operation == "mean":
        return pc.mean(colonne).as_py()
    else: 
        raise ValueError(f"Operation '{operation}' non reconnue")
    
# Filtrage des données (texte ou numérique)
def filtre(table: pa.Table, col: str, operation: str, valeur: any) -> pa.Table:
    if operation == "==":
        return table.filter(pc.equal(table.column(col), valeur))
    elif operation == "!=":
        return table.filter(pc.not_equal(table.column(col), valeur))
    elif operation == "<":
        return table.filter(pc.less(table.column(col), valeur))
    elif operation == "<=":
        return table.filter(pc.less_equal(table.column(col), valeur))
    elif operation == ">":
        return table.filter(pc.greater(table.column(col), valeur))
    elif operation == ">=":
        return table.filter(pc.greater_equal(table.column(col), valeur))
    elif operation == "in":
        return table.filter(pc.is_in(table.column(col), pa.array(valeur)))
    elif operation == "not in":
        return table.filter(pc.is_not_in(table.column(col), pa.array(valeur)))
    else:
        raise ValueError(f"Operation '{operation}' non reconnue")
    
# Tri des données
def trier(table: pa.Table, col: str, ordre: str) -> pa.Table:
    if ordre == "asc":
        return table.sort_by([(col, "ascending")])
    elif ordre == "desc":
        return table.sort_by([(col, "descending")])
    else:
        raise ValueError(f"L'ordre '{ordre}' n'existe pas... faut faire un effort enfin ('asc' ou 'desc' svp)")
    
# Grouper les données
def grouper(table: pa.Table, col: str, operation: str, group_by: str) -> pa.Table:
    return table.group_by(group_by).aggregate([(col, operation)])

# Jointure
def jointure(table1: pa.Table, table2: pa.Table, col1: str, col2: str) -> pa.Table:
    return table1.join(table2, keys=[col1], right_keys=[col2])

# Partitionnement des données dans un dataset partitionné
def table_to_dataset(table: pa.Table, partition_cols: list, root_path: str, max_partitions: int = 1024) -> None:
    if os.path.exists(root_path):
        try:
            os.rmdir(root_path)
        except OSError:
            raise OSError(f"Le répertoire {root_path} n'a pas pu être supprimé. Veuillez le supprimer manuellement.")
    pq.write_to_dataset(table, root_path=root_path, partition_cols=partition_cols, max_partitions=max_partitions)

# Lecture des données partitionnées
def dataset_to_table(root_path: str) -> pa.Table:
    return pq.ParquetDataset(root_path).read()

if __name__ == "__main__":
    # Création du dossier de sortie
    if not os.path.exists("output"):
        os.makedirs("output")

    # Chargement des données
    donnees_academies = csv_to_df("data/academies_virgule.csv")
    donnees_villes = csv_to_df("data/villes_virgule.csv")

    # Test des conversions
    table_academies = df_to_table(donnees_academies)
    table_to_parquet(table_academies, "output/academies.parquet")
    table_villes = df_to_table(donnees_villes)
    table_to_parquet(table_villes, "output/villes.parquet")

    table_academies_loaded = parquet_to_table("output/academies.parquet")
    donnees_academies_loaded = table_to_df(table_academies_loaded)
    table_villes_loaded = parquet_to_table("output/villes.parquet")
    donnees_villes_loaded = table_to_df(table_villes_loaded)
    print(donnees_academies_loaded.head())
    print("\n")
    print(donnees_villes_loaded.head())
    print("\n")

    # Schéma des tables
    afficher_schema(table_academies_loaded)
    print("\n")
    afficher_schema(table_villes_loaded)
    print("\n")

    # Récupération d'une colonne
    colonne_academies = recuperer_colonne(table_academies_loaded, "academie")
    print(colonne_academies)
    print("\n")

    # Calcul des statistiques
    count_departements = compute(table_academies_loaded, "dep", "count")
    count_distinct_zones = compute(table_academies_loaded, "vacances", "count_distinct")
    sum_population = compute(table_villes_loaded, "nb_hab_2012", "sum")
    min_surface = compute(table_villes_loaded, "surf", "min")
    max_surface = compute(table_villes_loaded, "surf", "max")
    mean_density = compute(table_villes_loaded, "dens", "mean")
    print(f"Nombre total de départements : {count_departements}")
    print(f"Nombre de zones distinctes : {count_distinct_zones}")
    print(f"Population totale : {sum_population}")
    print(f"Surface minimale : {min_surface}")
    print(f"Surface maximale : {max_surface}")
    print(f"Densité moyenne : {mean_density}")

    # Fitlre et tri 
    table_villes_filtree = filtre(table_villes_loaded, "nb_hab_2012", ">", 100000)
    table_villes_filtree_triee = trier(table_villes_filtree, "nb_hab_2012", "desc")
    print(table_to_df(table_villes_filtree_triee)[["nom", "nb_hab_2012"]])
    print("\n")
    print(table_to_df(filtre(table_villes_loaded, "nom", "==", "Annecy"))[["nom", "nb_hab_2012"]])

    # Calculs sur plusieurs colonnes et agregats
    mean_nb_hab_2012 = compute(table_villes_loaded, "nb_hab_2012", "mean")
    print(f"Nombre moyen d'habitants en 2012 : {mean_nb_hab_2012}")
    print("\n")
    table_mean_nb_hab_2012_par_dep = grouper(table_villes_loaded, "nb_hab_2012", "mean", "dep")
    print(table_to_df(table_mean_nb_hab_2012_par_dep))
    print("\n")
    print(f"Nombre moyen d'habitants pour le département 74 : {table_to_df(filtre(table_mean_nb_hab_2012_par_dep, 'dep', '==', '74'))['nb_hab_2012_mean'].values[0]}")

    # Opérations ensemblistes jointures
        # Zones de vacances des villes
    table_academies_villes = jointure(table_academies_loaded, table_villes_loaded, "dep", "dep")
    print(table_to_df(table_academies_villes)[["nom", "dep", "vacances"]])
    df_to_csv(table_to_df(table_academies_villes)[["nom", "dep", "vacances"]], "output/academies_villes.csv")
    print("\n")
        # Villes de la zone A
    table_academies_villes_zone_A = filtre(table_academies_villes, "vacances", "==", "Zone A")
    print(table_to_df(table_academies_villes_zone_A)[["nom", "dep"]])
    df_to_csv(table_to_df(table_academies_villes_zone_A)[["nom", "dep"]], "output/academies_villes_zone_A.csv")
    print("\n")
        # Départements des zones A et B
    table_academies_zone_A_B = filtre(table_academies_loaded, "vacances", "in", ["Zone A", "Zone B"])
    print(table_to_df(table_academies_zone_A_B)[["dep", "vacances"]])
    df_to_csv(table_to_df(table_academies_zone_A_B)[["dep", "departement", "vacances"]], "output/academies_zone_A_B.csv")
    print("\n")
        # Nombre de villes par academie (+ histogramme)
    table_nb_villes_academie = grouper(table_academies_villes, "nom", "count", "academie")
    table_nb_villes_academie = trier(table_nb_villes_academie, "nom_count", "desc")
    print(table_to_df(table_nb_villes_academie)[["academie", "nom_count"]])
    df_to_csv(table_to_df(table_nb_villes_academie)[["academie", "nom_count"]], "output/nb_villes_academie.csv")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="academie", y="nom_count", data=table_to_df(table_nb_villes_academie), color="lightgreen", edgecolor="green")
    plt.xticks(rotation=90)
    plt.title("Nombre de villes par académie")
    plt.xlabel("Académie")
    plt.ylabel("Nombre de villes")
    plt.tight_layout()
    plt.savefig("output/nb_villes_academie.png")
    print("\n")

    # Pour aller plus loin : partitionnement des données
    df_to_csv(table_to_df(table_academies_villes), "output/academies_villes.csv")
    print(f"Taille du fichier csv d'origine : {os.path.getsize('output/academies_villes.csv') / 1024:.2f} Ko")
    # print(table_academies_villes.schema)
    table_to_dataset(table_academies_villes, ["academie", "departement"], "output/academies_villes_partitionnees")
    # On voit que les caractères spéciaux (espace, accents, ...) sont transformés en format URL (%20, %C3%A9, ...) dans le nom des fichiers de partitionnement (voir capture d'écran jointe)

    # taille totale du dataset partitionné 'output/academies_villes_partitionnees'
    total_size = 0
    for dirpath, dirnames, filenames in os.walk("output/academies_villes_partitionnees"):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    print(f"Taille du dataset partionné : {total_size / 1024:.2f} Ko ({total_size / os.path.getsize('output/academies_villes.csv'):.2%} du fichier d'origine)")
    
    # Compression des données 
    algos = [
        'snappy', # par défaut, 
        'brotli',
        'gzip',
        'zstd',
        'lz4',
        'none'
    ]
    for algo in algos:
        pq.write_table(table_academies_villes, f"output/academies_villes_{algo}.parquet", compression=algo)
        table_academies_villes_compressed = pq.read_table(f"output/academies_villes_{algo}.parquet")
        print(f"Compression {algo} : {os.path.getsize(f'output/academies_villes_{algo}.parquet') / 1024:.2f} Ko ({os.path.getsize(f'output/academies_villes_{algo}.parquet') / os.path.getsize('output/academies_villes.csv'):.2%} du fichier d'origine)")

    # Résultats de la compression des données :
    #     
    # Taille du fichier csv d'origine : 5922.97 Ko
    # Taille du dataset partionné : 1961.03 Ko (33.11% du fichier d'origine)
    # Compression snappy : 1119.89 Ko (18.91% du fichier d'origine)
    # Compression brotli : 865.67 Ko (14.62% du fichier d'origine)
    # Compression gzip : 902.62 Ko (15.24% du fichier d'origine)
    # Compression zstd : 945.41 Ko (15.96% du fichier d'origine)
    # Compression lz4 : 1122.02 Ko (18.94% du fichier d'origine)
    # Compression none : 1451.47 Ko (24.51% du fichier d'origine)