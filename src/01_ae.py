import matplotlib.pyplot as plt
import seaborn as sns
import os
from contextlib import redirect_stdout
from utils import carregar_adult, carregar_drybean


#Faz um resumo da base e salva em um arquivo de texto
def resumo(df, alvo, nome):
    pasta = "resultados"
    os.makedirs(pasta, exist_ok=True)

    caminho = os.path.join(pasta, f"resumo_{nome}.txt")

    with open(caminho, "w", encoding="utf-8") as arquivo:
        with redirect_stdout(arquivo):

            print(f"\n===== {nome} =====")
            print("Formato:", df.shape)
            print("\nTipos:\n", df.dtypes)
            print("\nAusentes por coluna:\n", df.isna().sum()[df.isna().sum() > 0])
            print("\nLinhas duplicadas:", df.duplicated().sum())
            print("\nDistribuição das classes (%):\n",
                  (df[alvo].value_counts(normalize=True) * 100).round(2))
            print("\nEstatísticas numéricas:\n", df.describe().T)

    print(f"Resumo salvo em: {caminho}")


#Faz um grafico com os valores encontrados na base
def grafico_classes(df, alvo, nome):
    plt.figure(figsize=(7, 4))
    ordem = df[alvo].value_counts().index
    sns.countplot(data=df, x=alvo, order=ordem)
    plt.title(f"Distribuição das classes - {nome}")
    plt.xlabel("Classe"); plt.ylabel("Quantidade")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"resultados/figuras/classes_{nome}.png", dpi=150)
    plt.close()

#Faz um grafico com as correlacoes encontradas na base
def grafico_correlacao(df, nome):
    corr = df.select_dtypes("number").corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=len(corr) <= 10,
                fmt=".2f")
    plt.title(f"Correlação entre atributos numéricos - {nome}")
    plt.tight_layout()
    plt.savefig(f"resultados/figuras/corr_{nome}.png", dpi=150)
    plt.close()

#roda
if __name__ == "__main__":
    adult = carregar_adult()
    bean = carregar_drybean()

    resumo(adult, "income", "Adult")
    resumo(bean, "Class", "DryBean")

    for df, alvo, nome in [(adult, "income", "adult"), (bean, "Class", "drybean")]:
        grafico_classes(df, alvo, nome)
        grafico_correlacao(df, nome)