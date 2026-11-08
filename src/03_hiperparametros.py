
'''Estudo de hiperparâmetros (MLP + Random Forest) nas bases Adult e Dry Bean.

Saídas (pasta resultados/):
    isolado_<base>.csv / figuras/isolado_<base>.png   -> variação de 1 hiperparâmetro por vez (MLP)
    grid_<base>_<modelo>.csv                          -> todas as combinações do GridSearch
    classification_report_<base>_<modelo>.txt         -> métricas por classe no teste
    figuras/cm_<base>_<modelo>.png                    -> matriz de confusão normalizada
    figuras/loss_<base>.png                           -> curva de perda da melhor MLP
    resumo_final.csv                                  -> tabela final base x modelo x métricas
'''

SEED = 42                 
TEST_SIZE = 0.20          #Tamanho do teste
CV_FOLDS = 5              #Numero de folds da validação cruzada estratificada
SCORING = "f1_macro"      #Metrica usada para escolher os hiperparametros
N_JOBS = -1               # -1 = todos os núcleos use 1 se der problema de memoria


BASES_ATIVAS = ["Adult", "DryBean"]   
EXECUTAR_ESTUDO_ISOLADO = True        #Variação de 1 hiperparâmetro por vez (graficos)
EXECUTAR_GRID_SEARCH = True           #Busca combinada + avaliacao no teste


COLUNAS_REMOVIDAS = {
    "Adult": ["fnlwgt"],      
    "DryBean": [],
}
REMOVER_DUPLICATAS = {
    "Adult": False,
    "DryBean": True,
}

#Cuidado ao mexer nos parametros, pode acabar demorando muito tempo

#Hiperparametros base do MLP
MLP_BASE = {
    "hidden_layer_sizes": (64,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "batch_size": 64,
    "max_iter": 300,
    "early_stopping": False,
}

#Hiperparametros para estudo isolado do mlp
MLP_ISOLADO = {
    "hidden_layer_sizes": [(8,), (32,), (64,), (128,), (32, 16), (64, 32), (64, 32, 16)],
    "activation": ["relu", "tanh", "logistic"],
    "learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "solver": ["adam", "sgd"],
    "batch_size": [16, 32, 64, 128],
    "max_iter": [100, 300, 500],
    "early_stopping": [True, False],
}

#Hiperparametros busca combinada do mlp
MLP_GRID = {
    "hidden_layer_sizes": [(32,), (64,), (64, 32)],
    "activation": ["relu", "tanh"],
    "alpha": [1e-4, 1e-2],
    "learning_rate_init": [1e-3, 1e-2, 0.01, 0.2],
}

#Hiperparametros randomforest
ALGORITMO_2_NOME = "RandomForest"
GRID_2 = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 3],
}

PASTA_RESULTADOS = "resultados/hp"
PASTA_FIGURAS = "resultados/hp/23figuras"

#FIM DA CONFIGURAÇÃO

import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, cross_validate)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             ConfusionMatrixDisplay)

from utils import carregar_adult, carregar_drybean

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ALVOS = {"Adult": "income", "DryBean": "Class"}



#Dados e pre processamento

def carregar(nome):
    df = carregar_adult() if nome == "Adult" else carregar_drybean()
    if REMOVER_DUPLICATAS.get(nome, False):
        df = df.drop_duplicates().reset_index(drop=True)
    alvo = ALVOS[nome]
    X = df.drop(columns=[alvo] + COLUNAS_REMOVIDAS.get(nome, []))
    return X, df[alvo]


def montar_preprocessador(X):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def montar_pipeline(X_tr, clf):
    return Pipeline([("prep", montar_preprocessador(X_tr)), ("clf", clf)])


def novo_mlp(**params):
    return MLPClassifier(random_state=SEED, **{**MLP_BASE, **params})


def novo_algoritmo_2():
    return RandomForestClassifier(random_state=SEED, n_jobs=1)


#Estudo isolado varia 1 hiperparametro do MLP por vez
def estudo_isolado(X_tr, y_tr, cv, nome):
    linhas = []
    for param, valores in MLP_ISOLADO.items():
        for v in valores:
            params = {**MLP_BASE, param: v}
            pipe = montar_pipeline(X_tr, MLPClassifier(random_state=SEED, **params))
            t0 = time.time()
            r = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=SCORING,
                               n_jobs=N_JOBS, return_train_score=True,
                               return_estimator=True)
            n_iter = [e.named_steps["clf"].n_iter_ for e in r["estimator"]]
            linhas.append({
                "hiperparametro": param,
                "valor": str(v),
                "val_media": r["test_score"].mean(),
                "val_desvio": r["test_score"].std(),
                "treino_media": r["train_score"].mean(),
                "tempo_s": time.time() - t0,
                "epocas_medias": np.mean(n_iter),
                "convergiu": all(n < params["max_iter"] for n in n_iter),
            })
            print(f"  [{nome}] {param}={v}: {SCORING}={linhas[-1]['val_media']:.4f} "
                  f"± {linhas[-1]['val_desvio']:.4f} ({linhas[-1]['tempo_s']:.1f}s)")

    df = pd.DataFrame(linhas)
    df.round(4).to_csv(f"{PASTA_RESULTADOS}/isolado_{nome}.csv", index=False)
    plotar_isolado(df, nome)
    return df


def plotar_isolado(df, nome):
    params = list(df["hiperparametro"].unique())
    if not params:
        return
    ncols = 3
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows),
                             squeeze=False)
    for ax, p in zip(axes.ravel(), params):
        d = df[df["hiperparametro"] == p]
        x = range(len(d))
        ax.errorbar(x, d["val_media"], yerr=d["val_desvio"], marker="o",
                    capsize=3, label="Validação (média ± dp)")
        ax.plot(x, d["treino_media"], marker="s", linestyle="--", label="Treino")
        ax.set_xticks(list(x))
        ax.set_xticklabels(d["valor"], rotation=35, ha="right", fontsize=8)
        ax.set_title(p)
        ax.set_ylabel(SCORING)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(params):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Impacto dos hiperparâmetros da MLP - {nome}", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{PASTA_FIGURAS}/isolado_{nome}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)



#Busca combinada e avaliacao final no teste

def grid_e_avaliacao(X_tr, X_te, y_tr, y_te, cv, nome):
    modelos = {
        "MLP": (novo_mlp(), {f"clf__{k}": v for k, v in MLP_GRID.items()}),
        ALGORITMO_2_NOME: (novo_algoritmo_2(),
                           {f"clf__{k}": v for k, v in GRID_2.items()}),
    }
    resultados = []
    for nome_m, (clf, grade) in modelos.items():
        n_comb = int(np.prod([len(v) for v in grade.values()]))
        print(f"  [{nome}] GridSearch {nome_m}: {n_comb} combinações x {CV_FOLDS} folds")

        gs = GridSearchCV(montar_pipeline(X_tr, clf), grade, cv=cv, scoring=SCORING,
                          n_jobs=N_JOBS, return_train_score=True, refit=True)
        t0 = time.time()
        gs.fit(X_tr, y_tr)
        t_busca = time.time() - t0

        pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").round(4).to_csv(
            f"{PASTA_RESULTADOS}/grid_{nome}_{nome_m}.csv", index=False)

        t0 = time.time()
        pred = gs.predict(X_te)
        t_inf = time.time() - t0
        melhor = gs.best_estimator_.named_steps["clf"]

        resultados.append({
            "base": nome,
            "modelo": nome_m,
            "melhores_params": {k.replace("clf__", ""): v
                                for k, v in gs.best_params_.items()},
            f"cv_{SCORING}": gs.best_score_,
            "acc_teste": accuracy_score(y_te, pred),
            "prec_macro": precision_score(y_te, pred, average="macro", zero_division=0),
            "rec_macro": recall_score(y_te, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_te, pred, average="macro"),
            "f1_weighted": f1_score(y_te, pred, average="weighted"),
            "tempo_busca_s": t_busca,
            "tempo_inferencia_s": t_inf,
        })
        print(f"  [{nome}] {nome_m}: melhores={resultados[-1]['melhores_params']}")
        print(f"           CV={gs.best_score_:.4f} | teste acc={resultados[-1]['acc_teste']:.4f}"
              f" f1_macro={resultados[-1]['f1_macro']:.4f}")

        with open(f"{PASTA_RESULTADOS}/classification_report_{nome}_{nome_m}.txt", "w") as f:
            f.write(classification_report(y_te, pred, zero_division=0))

        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_te, pred, normalize="true", ax=ax, xticks_rotation=45,
            values_format=".2f", colorbar=False)
        ax.set_title(f"Matriz de confusão (normalizada) - {nome} - {nome_m}")
        fig.tight_layout()
        fig.savefig(f"{PASTA_FIGURAS}/cm_{nome}_{nome_m}.png", dpi=150)
        plt.close(fig)

        if nome_m == "MLP":
            plotar_loss(melhor, nome)
            resultados[-1]["epocas"] = melhor.n_iter_
            resultados[-1]["convergiu"] = melhor.n_iter_ < melhor.max_iter

    return resultados


def plotar_loss(mlp, nome):
    if not hasattr(mlp, "loss_curve_"):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mlp.loss_curve_, label="Loss de treino")
    if getattr(mlp, "validation_scores_", None) is not None:
        ax2 = ax.twinx()
        ax2.plot(mlp.validation_scores_, color="tab:orange", linestyle="--",
                 label="Score de validação interna")
        ax2.set_ylabel("Score de validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title(f"Curva de perda da melhor MLP - {nome}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{PASTA_FIGURAS}/loss_{nome}.png", dpi=150)
    plt.close(fig)



#Execucao

def main():
    Path(PASTA_RESULTADOS).mkdir(parents=True, exist_ok=True)
    Path(PASTA_FIGURAS).mkdir(parents=True, exist_ok=True)

    todos = []
    for nome in BASES_ATIVAS:
        print(f"\n================ {nome} ================")
        X, y = carregar(nome)
        print(f"Instâncias: {X.shape[0]} | atributos: {X.shape[1]} | classes: {y.nunique()}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

        if EXECUTAR_ESTUDO_ISOLADO and MLP_ISOLADO:
            print("\n-- Estudo isolado de hiperparâmetros (MLP) --")
            estudo_isolado(X_tr, y_tr, cv, nome)

        if EXECUTAR_GRID_SEARCH:
            print("\n-- Busca combinada e avaliação no teste --")
            todos += grid_e_avaliacao(X_tr, X_te, y_tr, y_te, cv, nome)

    if todos:
        resumo = pd.DataFrame(todos)
        resumo["melhores_params"] = resumo["melhores_params"].astype(str)
        resumo.round(4).to_csv(f"{PASTA_RESULTADOS}/resumo_final.csv", index=False)
        print("\n=========== RESUMO FINAL ===========")
        cols = ["base", "modelo", "acc_teste", "prec_macro", "rec_macro", "f1_macro"]
        print(resumo[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()