import pandas as pd

ADULT_COLUNAS = ["age", "workclass", "fnlwgt", "education", "education-num",
              "marital-status", "occupation", "relationship", "race", "sex",
              "capital-gain", "capital-loss", "hours-per-week",
              "native-country", "income"]

def carregar_adult(caminho="data/adult.data"):
    df = pd.read_csv(caminho, names=ADULT_COLUNAS, skipinitialspace=True,
                     na_values="?")
    return df

def carregar_drybean(caminho="data/Dry_Bean_Dataset.csv"):
    df = pd.read_csv(caminho, sep=";", decimal=",")
    return df


'''drybean = carregar_drybean()


print(drybean.head())   


print(drybean.dtypes)'''