# Classificação com MLP e Random Forest: Adult e Dry Bean

Trabalho prático de Aprendizado de Máquina Supervisionado (disciplina de Inteligência Artificial).
Compara uma Rede Neural Artificial (MLP) e uma Random Forest em duas bases públicas do
UCI Machine Learning Repository, com estudo do impacto dos hiperparâmetros da MLP.



## Bases de dados

## Bases não incluídas 

| Base | Tarefa | Instâncias | Fonte |
|---|---|---|---|
| Adult | Binária (renda ≤50K ou >50K) | 32.561 | https://archive.ics.uci.edu/dataset/2/adult |
| Dry Bean | Multiclasse (7 variedades de feijão) | 13.611 | https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |

- **Adult:** use o arquivo `adult.data` (sem cabeçalho; valores ausentes marcados com `?`).
  O `adult.test` não é usado: a divisão treino/teste é feita pelo próprio código.
- **Dry Bean:** o arquivo original é `.xlsx`. Como o `openpyxl` não está entre as bibliotecas
  permitidas, abra-o no Excel, LibreOffice ou Google Sheets e salve como
  `data/dry_bean.csv` (a coluna do alvo se chama `Class`).
  Se o CSV sair com separador `;` ou decimal com vírgula, ajuste `carregar_drybean` em
  `src/utils.py` (`sep=";"`, `decimal=","`).

## Estrutura do projeto

```
projeto/
├── data/
│   ├── adult.data
│   └── dry_bean.csv
├── src/
│   ├── utils.py                  # funções de leitura das duas bases
│   ├── 01_ae.py                  # análise exploratória
|   ├── 02_base.py                # baseline
│   ├── 03_hiperparametros.py     # estudo isolado da MLP + 1ª rodada da busca em grade
│   └── 04_experimento_final.py   # busca refinada, 5 sementes e comparação pareada
├── resultados/  
    ├── hp/                       # saídas de 01 e 03
│   └── figuras/                  # saídas de 01 e 03
├── resultados_final/             # saídas de 04 (resultados reportados no relatório)
│   └── figuras/
├── requirements.txt
└── README.md
```

## Requisitos


```
numpy
pandas
scikit-learn
matplotlib
seaborn
```

Instalação:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para registrar as versões exatas usadas, rode `pip freeze` e anote-as aqui.

Não foram usados frameworks de aprendizado profundo nem bibliotecas fora da lista
(TensorFlow, Keras, PyTorch, XGBoost, LightGBM, CatBoost etc.). Além das bibliotecas acima,
os scripts usam apenas a biblioteca padrão do Python (`os`, `time`, `warnings`, `pathlib`).

## Como reproduzir

**Execute sempre a partir da raiz do projeto**, nesta ordem:

```bash
python src/01_ae.py                  
python src/02_base.py
python src/03_hiperparametros.py      
python src/04_experimento_final.py    
```



Principais arquivos de `resultados_final/`:

- `resumo_final.csv`: média e desvio de todas as métricas, hiperparâmetros escolhidos e tempos
- `tabela_relatorio.csv`: a mesma tabela formatada como "média ± dp"
- `por_semente.csv`: uma linha por modelo e por semente
- `comparacao_pareada.csv`: diferença de F1 macro (MLP − RF) com IC de 95%
- `grid_<base>_<modelo>.csv`: todas as combinações testadas na busca em grade
- `classification_report_<base>_<modelo>.txt` e `figuras/cm_*.png`, `figuras/loss_*.png`

### Configuração

Todos os hiperparâmetros, grades de busca e opções de pré-processamento ficam no **bloco de
configuração no topo** de `03_hiperparametros.py` e `04_experimento_final.py`. Não é preciso
editar o restante do código. Para rodar só uma base, altere `BASES_ATIVAS`. Para um teste
rápido, reduza as listas das grades e use `CV_FOLDS = 3`.

Tempo aproximado na máquina usada no trabalho (varia conforme o hardware):
cerca de 45 a 50 minutos para `03_hiperparametros.py` e 13 a 15 minutos para `04_experimento_final.py`.

## Metodologia (resumo)

- **Pré-processamento:** remoção de `fnlwgt` (Adult) e de duplicatas (Dry Bean); mediana +
  padronização nos numéricos; moda + one-hot nos categóricos. Tudo em um `Pipeline`
  ajustado só no treino, refeito em cada partição da validação cruzada (sem vazamento).
- **Divisão:** 80% treino e 20% teste, estratificada, semente 42. O teste só é usado na avaliação final.
- **Escolha de hiperparâmetros:** `GridSearchCV` com validação cruzada estratificada de 5
  partições, critério F1 macro.
- **Avaliação final:** melhor configuração retreinada com 5 sementes (42 a 46). Métricas:
  acurácia, precisão, revocação, F1 (macro), matriz de confusão e, no Adult, métricas da classe >50K.
  Comparação entre modelos por bootstrap pareado (IC de 95% da diferença de F1 macro).

## Resultados principais

F1 macro no teste (média ± desvio-padrão, 5 sementes):

| Base | MLP | Random Forest | Conclusão |
|---|---|---|---|
| Adult | 0,805 ± 0,006 | 0,810 ± 0,001 | Sem diferença distinguível (IC 95% inclui 0) |
| Dry Bean | 0,937 ± 0,002 | 0,928 ± 0,001 | MLP melhor, diferença pequena (+0,009) |

A análise completa está no relatório técnico (PDF).

## Reprodutibilidade

Todas as sementes são fixas (`SEED = 42` para o split e a validação cruzada; sementes 42 a 46
para a avaliação final). Mantenha `SEED`, `TEST_SIZE` e as opções de pré-processamento
iguais nos scripts `03` e `04`, para que o split seja o mesmo. Os tempos de execução
dependem da máquina; compare-os apenas de forma relativa.
