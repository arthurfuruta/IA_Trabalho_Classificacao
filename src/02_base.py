import warnings
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils import carregar_adult, carregar_drybean

SEED = 42

#Preparar dados
def montar_preprocessador(X):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

#Dropa as variaveis que queremos prever das variaveis alvo
def carregar(nome):
    if nome == "Adult":
        df = carregar_adult()
        return df.drop(columns="income"), df["income"]
    df = carregar_drybean()
    return df.drop(columns="Class"), df["Class"]


def checar_dados(X, y, X_tr, X_te, y_tr, y_te, nome):
    print(f"\n--- Checagem dos dados: {nome} ---")
    print("Formato X:", X.shape, "| classes:", y.nunique())
    print("Classes:", sorted(y.unique()))

    #Checa se o alvo não pode estar dentro das features
    assert y.name not in X.columns, "O alvo vazou para as features!"

    #Checa se algum registro aparece simultaneamente no conjunto de treinamento e no conjunto de teste
    assert len(set(X_tr.index) & set(X_te.index)) == 0, "Treino e teste se sobrepõem!"

    #Checa a estratificação
    prop = pd.DataFrame({
        "treino": y_tr.value_counts(normalize=True),
        "teste": y_te.value_counts(normalize=True),
    }).round(3)
    print("Proporção das classes:\n", prop)
    assert (prop["treino"] - prop["teste"]).abs().max() < 0.01, "Estratificação falhou"

    #Checa se ainda existem valores NaN 
    prep = montar_preprocessador(X_tr)
    Xt = prep.fit_transform(X_tr)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert not np.isnan(Xt).any(), "Ainda há NaN após o pré-processamento"
    print("Shape após pré-processamento:", Xt.shape)

    #Checa a padronizacao
    n_num = X_tr.select_dtypes("number").shape[1]
    print("Média dos numéricos (treino):", Xt[:, :n_num].mean().round(3),
          "| desvio:", Xt[:, :n_num].std().round(3))

#Executa as etapas de carregamento preparacao treinamento e avaliacao
def rodar(nome):
    X, y = carregar(nome)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED) #Divide os dados de treinamento e teste
    checar_dados(X, y, X_tr, X_te, y_tr, y_te, nome)

    modelos = {
        "Dummy": DummyClassifier(strategy="most_frequent"),  #Classificador de referencia
        "MLP": MLPClassifier(max_iter=300, random_state=SEED), #Rede Neural
        "RandomForest": RandomForestClassifier(random_state=SEED), #Arvore de selecao
    }

    linhas = []
    for nome_m, clf in modelos.items():
        pipe = Pipeline([("prep", montar_preprocessador(X_tr)), ("clf", clf)])
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            pipe.fit(X_tr, y_tr)
        convergiu = not any(issubclass(a.category, ConvergenceWarning) for a in avisos)

        p_tr, p_te = pipe.predict(X_tr), pipe.predict(X_te)
        linhas.append({
            "modelo": nome_m,
            "acc_treino": accuracy_score(y_tr, p_tr),
            "acc_teste": accuracy_score(y_te, p_te),
            "f1macro_treino": f1_score(y_tr, p_tr, average="macro"),
            "f1macro_teste": f1_score(y_te, p_te, average="macro"),
            "convergiu": convergiu,
        })

        #Teste de reprodutibilidade
        if nome_m != "Dummy":
            pipe2 = Pipeline([("prep", montar_preprocessador(X_tr)),
                              ("clf", clf.__class__(**clf.get_params()))])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe2.fit(X_tr, y_tr)
            igual = np.array_equal(p_te, pipe2.predict(X_te))
            print(f"[{nome_m}] reprodutível com a mesma semente: {igual}")

    res = pd.DataFrame(linhas).set_index("modelo").round(4) #Criacao da tabela de resultados
    print(f"\n=== Resultados baseline: {nome} ===")
    print(res)

    # Sanidade do Dummy: acurácia deve ser igual à proporção da classe majoritária
    maj = y_te.value_counts(normalize=True).max()
    assert abs(res.loc["Dummy", "acc_teste"] - maj) < 0.01, "Dummy fora do esperado"
    # Os modelos reais precisam bater o Dummy no F1 macro
    for m in ["MLP", "RandomForest"]:
        assert res.loc[m, "f1macro_teste"] > res.loc["Dummy", "f1macro_teste"] + 0.1, \
            f"{m} não supera o Dummy: há algo errado"

    # Matriz de confusão da melhor opção, só para inspeção visual rápida
    pipe = Pipeline([("prep", montar_preprocessador(X_tr)),
                     ("clf", RandomForestClassifier(random_state=SEED))]).fit(X_tr, y_tr)
    print("\nMatriz de confusão (RandomForest, teste):\n",
          confusion_matrix(y_te, pipe.predict(X_te)))
    return res


if __name__ == "__main__":
    for base in ["Adult", "DryBean"]:
        rodar(base)