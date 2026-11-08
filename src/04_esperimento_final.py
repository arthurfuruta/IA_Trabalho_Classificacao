'''
Rodada final do experimento (Adult + Dry Bean; MLP vs Random Forest).

Diferenças
1. Grades refinadas, uma por base (no 03, vários melhores valores ficaram na
   borda da grade, então o ótimo podia estar além dela).
2. Avaliação final repetida com N_SEMENTES sementes: os hiperparâmetros são
   escolhidos UMA vez por validação cruzada (só com o treino) e o melhor modelo
   é retreinado com cada semente. Reporta-se média ± desvio no teste.
3. Comparação pareada MLP x outro algoritmo com intervalo de confiança
   bootstrap, para saber se a diferença é real ou ruído.
4. Tempo de treino da MELHOR configuração (comparável entre modelos),
   além do tempo total da busca.

O split treino/teste é idêntico ao do 03 (mesma SEED, TEST_SIZE e pré-
processamento de linhas), então os resultados são comparáveis com os anteriores

O estudo isolado de hiperparâmetros continua sendo o do 03
'''

# =============================================================================
# BLOCO DE CONFIGURAÇÃO  (edite apenas aqui)
# =============================================================================

# ---------- Geral (mantenha igual ao 03 para o split ser o mesmo) ----------
SEED = 42                 # semente do split, da CV e da 1ª semente final
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = "f1_macro"      # métrica usada para escolher hiperparâmetros
N_JOBS = -1               # -1 = todos os núcleos; use 1 se faltar memória

#Múltiplas sementes ----------
N_SEMENTES = 5                 #Sementes usadas: SEED, SEED+1, ..., SEED+N-1
BOOTSTRAP_POR_SEMENTE = 200    #Reamostragens por semente para o IC da diferença


BASES_ATIVAS = ["Adult", "DryBean"]

#Pre processamento por base
COLUNAS_REMOVIDAS = {
    "Adult": ["fnlwgt"],
    "DryBean": [],
}
REMOVER_DUPLICATAS = {
    "Adult": False,
    "DryBean": True,
}


CLASSE_POSITIVA = {
    "Adult": ">50K",
}


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


GRID_MLP = {
   
    "Adult": {
        "hidden_layer_sizes": [(32,), (64, 32)],
        "alpha": [0.03, 0.1, 0.3],
        "learning_rate_init": [1e-3, 1e-2],
        "early_stopping": [True, False],
    },
    
    "DryBean": {
        "hidden_layer_sizes": [(16,), (32,), (64,)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 1e-2],
    },
}


ALGORITMO_2_NOME = "RandomForest"
GRID_2 = {

    "Adult": {
        "n_estimators": [300, 500],
        "max_depth": [20, None],
        "min_samples_leaf": [3, 5, 10],
    },

    "DryBean": {
        "n_estimators": [100, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 3, 5],
    },
}


PASTA_RESULTADOS = "resultados_final"
PASTA_FIGURAS = "resultados_final/figuras"


#FIM DA CONFIGURACAO


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

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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


#Modelos

def novo_mlp(seed=SEED, **params):
    return MLPClassifier(random_state=seed, **{**MLP_BASE, **params})


def novo_algoritmo_2(seed=SEED, **params):

    return RandomForestClassifier(random_state=seed, n_jobs=1, **params)


def criar_modelo(nome_m, params, seed):
    if nome_m == "MLP":
        return novo_mlp(seed, **params)
    return novo_algoritmo_2(seed, **params)



#Metricas

def metricas(y_te, pred, nome):
    m = {
        "acc": accuracy_score(y_te, pred),
        "prec_macro": precision_score(y_te, pred, average="macro", zero_division=0),
        "rec_macro": recall_score(y_te, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_te, pred, average="macro"),
    }
    pos = CLASSE_POSITIVA.get(nome)
    if pos is not None:
        kw = dict(pos_label=pos, average="binary", zero_division=0)
        m["prec_pos"] = precision_score(y_te, pred, **kw)
        m["rec_pos"] = recall_score(y_te, pred, **kw)
        m["f1_pos"] = f1_score(y_te, pred, **kw)
    return m


def f1_macro_np(y, p, k):
    """F1 macro rápido (numpy) para rótulos inteiros 0..k-1; usado no bootstrap."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))


def ic_bootstrap_diferenca(y_te, preds_a, preds_b):
  
    classes = np.unique(np.asarray(y_te))
    k = len(classes)
    y = np.searchsorted(classes, np.asarray(y_te))
    n = len(y)
    rng = np.random.default_rng(SEED)
    difs = []
    for s in preds_a:
        pa = np.searchsorted(classes, preds_a[s])
        pb = np.searchsorted(classes, preds_b[s])
        for _ in range(BOOTSTRAP_POR_SEMENTE):
            idx = rng.integers(0, n, n)
            difs.append(f1_macro_np(y[idx], pa[idx], k) - f1_macro_np(y[idx], pb[idx], k))
    difs = np.array(difs)
    return difs.mean(), np.percentile(difs, [2.5, 97.5])



#Busca + avaliacao multi semente de um modelo

def avisar_borda(grade, melhores, nome_m):
    for k, valores in grade.items():
        if len(valores) >= 3 and melhores[k] in (valores[0], valores[-1]):
            print(f"    AVISO [{nome_m}]: '{k}' = {melhores[k]} está na borda da grade "
                  f"{valores}; considere estendê-la.")


def buscar_e_avaliar(nome, nome_m, grade, X_tr, X_te, y_tr, y_te, cv):
    base_clf = novo_mlp() if nome_m == "MLP" else novo_algoritmo_2()
    grade_clf = {f"clf__{k}": v for k, v in grade.items()}
    n_comb = int(np.prod([len(v) for v in grade.values()]))
    print(f"  [{nome}] {nome_m}: busca com {n_comb} combinações x {CV_FOLDS} folds")

    #refit=False so queremos os melhores hiperparametros o retreino e feito

    gs = GridSearchCV(montar_pipeline(X_tr, base_clf), grade_clf, cv=cv,
                      scoring=SCORING, n_jobs=N_JOBS, return_train_score=True,
                      refit=False)
    t0 = time.time()
    gs.fit(X_tr, y_tr)
    t_busca = time.time() - t0

    pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").round(4).to_csv(
        f"{PASTA_RESULTADOS}/grid_{nome}_{nome_m}.csv", index=False)

    melhores = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    cv_dp = gs.cv_results_["std_test_score"][gs.best_index_]
    print(f"    melhores: {melhores}")
    print(f"    CV {SCORING} = {gs.best_score_:.4f} ± {cv_dp:.4f} (busca: {t_busca:.0f}s)")
    avisar_borda(grade, melhores, nome_m)

    #Avaliação final com várias sementes
    sementes = [SEED + i for i in range(N_SEMENTES)]
    linhas, preds = [], {}
    for s in sementes:
        pipe = montar_pipeline(X_tr, criar_modelo(nome_m, melhores, s))
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        t_fit = time.time() - t0
        t0 = time.time()
        pred = pipe.predict(X_te)
        t_inf = time.time() - t0
        preds[s] = pred

        linha = {"base": nome, "modelo": nome_m, "semente": s,
                 **metricas(y_te, pred, nome),
                 "tempo_treino_s": t_fit, "tempo_inferencia_s": t_inf}
        if nome_m == "MLP":
            clf = pipe.named_steps["clf"]
            linha["epocas"] = clf.n_iter_
            linha["convergiu"] = clf.n_iter_ < clf.max_iter
        linhas.append(linha)

        if s == sementes[0]:
            guardar_artefatos(nome, nome_m, y_te, pred, pipe)
        print(f"    semente {s}: acc={linha['acc']:.4f}  f1_macro={linha['f1_macro']:.4f}")

    df = pd.DataFrame(linhas)
    cols = [c for c in df.columns
            if c in ("acc", "prec_macro", "rec_macro", "f1_macro",
                     "prec_pos", "rec_pos", "f1_pos",
                     "tempo_treino_s", "tempo_inferencia_s")]
    resumo = {"base": nome, "modelo": nome_m, "melhores_params": str(melhores),
              f"cv_{SCORING}": gs.best_score_, "cv_dp": cv_dp,
              "n_combinacoes": n_comb, "tempo_busca_s": t_busca,
              "n_sementes": N_SEMENTES}
    for c in cols:
        resumo[f"{c}_media"] = df[c].mean()
        resumo[f"{c}_dp"] = df[c].std()         
    if nome_m == "MLP":
        resumo["epocas_media"] = df["epocas"].mean()
        resumo["convergiu_todas"] = bool(df["convergiu"].all())
    return resumo, linhas, preds


def guardar_artefatos(nome, nome_m, y_te, pred, pipe):
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
        plotar_loss(pipe.named_steps["clf"], nome)


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



#Comparacao pareada MLP vs algoritmo 2

def comparar(nome, y_te, linhas_a, linhas_b, preds_a, preds_b):
    d_seed = np.array([a["f1_macro"] - b["f1_macro"]
                       for a, b in zip(linhas_a, linhas_b)])
    media, (lo, hi) = ic_bootstrap_diferenca(y_te, preds_a, preds_b)
    if lo > 0:
        concl = "MLP melhor (IC 95% acima de 0)"
    elif hi < 0:
        concl = f"{ALGORITMO_2_NOME} melhor (IC 95% abaixo de 0)"
    else:
        concl = "diferença não distinguível de ruído (IC 95% inclui 0)"
    print(f"  [{nome}] F1 macro, MLP - {ALGORITMO_2_NOME}: "
          f"{d_seed.mean():+.4f} ± {d_seed.std(ddof=1):.4f} entre sementes | "
          f"IC95% bootstrap = [{lo:+.4f}, {hi:+.4f}] -> {concl}")
    return {"base": nome,
            "dif_f1_macro_media_sementes": d_seed.mean(),
            "dif_f1_macro_dp_sementes": d_seed.std(ddof=1),
            "ic95_inferior": lo, "ic95_superior": hi, "conclusao": concl}



#Execucao

def fmt(r, c, casas=4):
    if f"{c}_media" not in r:
        return "-"
    return f"{r[f'{c}_media']:.{casas}f} ± {r[f'{c}_dp']:.{casas}f}"


def main():
    Path(PASTA_RESULTADOS).mkdir(parents=True, exist_ok=True)
    Path(PASTA_FIGURAS).mkdir(parents=True, exist_ok=True)

    resumos, por_semente, comparacoes = [], [], []
    for nome in BASES_ATIVAS:
        print(f"\n================ {nome} ================")
        X, y = carregar(nome)
        print(f"Instâncias: {X.shape[0]} | atributos: {X.shape[1]} | classes: {y.nunique()}")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

        res = {}
        for nome_m, grade in [("MLP", GRID_MLP[nome]), (ALGORITMO_2_NOME, GRID_2[nome])]:
            resumo, linhas, preds = buscar_e_avaliar(
                nome, nome_m, grade, X_tr, X_te, y_tr, y_te, cv)
            resumos.append(resumo)
            por_semente += linhas
            res[nome_m] = (linhas, preds)

        print("\n-- Comparação pareada --")
        (la, pa), (lb, pb) = res["MLP"], res[ALGORITMO_2_NOME]
        comparacoes.append(comparar(nome, y_te, la, lb, pa, pb))

    pd.DataFrame(por_semente).round(5).to_csv(f"{PASTA_RESULTADOS}/por_semente.csv", index=False)
    pd.DataFrame(resumos).round(5).to_csv(f"{PASTA_RESULTADOS}/resumo_final.csv", index=False)
    pd.DataFrame(comparacoes).round(5).to_csv(f"{PASTA_RESULTADOS}/comparacao_pareada.csv", index=False)

    tabela = pd.DataFrame([{
        "Base": r["base"], "Modelo": r["modelo"],
        "Acurácia": fmt(r, "acc"), "Precisão (macro)": fmt(r, "prec_macro"),
        "Recall (macro)": fmt(r, "rec_macro"), "F1 (macro)": fmt(r, "f1_macro"),
        "Precisão (pos.)": fmt(r, "prec_pos"), "Recall (pos.)": fmt(r, "rec_pos"),
        "F1 (pos.)": fmt(r, "f1_pos"),
        "Treino da melhor config. (s)": fmt(r, "tempo_treino_s", 1),
        "Inferência (s)": fmt(r, "tempo_inferencia_s", 3),
    } for r in resumos])
    tabela.to_csv(f"{PASTA_RESULTADOS}/tabela_relatorio.csv", index=False)

    print(f"\n=========== RESUMO FINAL ({N_SEMENTES} sementes, média ± dp) ===========")
    print(tabela[["Base", "Modelo", "Acurácia", "F1 (macro)", "F1 (pos.)",
                  "Treino da melhor config. (s)"]].to_string(index=False))
    print(f"\nArquivos salvos em '{PASTA_RESULTADOS}/'.")


if __name__ == "__main__":
    main()