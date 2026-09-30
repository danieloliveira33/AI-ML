"""Acesso ao modelo de Machine Learning (Decision Tree) fora das views.

Usado pelas views do Django e pelo servidor MCP, para que os dois usem
exatamente o mesmo modelo e a mesma lógica de explicação.
"""

import json
import os

import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_PATH = os.path.join(BASE_DIR, "modelo", "modelo_iris.joblib")
METRICAS_PATH = os.path.join(BASE_DIR, "modelo", "metricas.json")

ESPECIES = ("setosa", "versicolor", "virginica")

NOMES_PT = {
    "sepal length (cm)": "comprimento da sépala",
    "sepal width (cm)": "largura da sépala",
    "petal length (cm)": "comprimento da pétala",
    "petal width (cm)": "largura da pétala",
}

_artefato = None


def _carregar():
    global _artefato
    if _artefato is None:
        _artefato = joblib.load(MODELO_PATH)
    return _artefato


def _num(valor, casas=2):
    """Formata número no padrão brasileiro, sem zeros à direita."""
    texto = f"{valor:.{casas}f}".rstrip("0").rstrip(".")
    return texto.replace(".", ",")


def _pct(fracao):
    return f"{fracao * 100:.1f}".replace(".", ",") + "%"


def especie_no_texto(texto):
    """Primeira espécie citada em um texto livre (ex.: resposta do Qwen)."""
    texto = (texto or "").lower()
    achadas = [(texto.find(e), e) for e in ESPECIES if e in texto]
    return min(achadas)[1] if achadas else None


# ==========================================
# PREVISÃO COM EXPLICAÇÃO
# ==========================================

def prever_com_explicacao(comp_sepala, larg_sepala, comp_petala, larg_petala):
    """Classifica a flor e devolve o caminho exato percorrido na árvore."""
    artefato = _carregar()
    modelo = artefato["modelo"]
    classes = artefato["classes"]
    nomes = artefato["features"]

    entrada = [[comp_sepala, larg_sepala, comp_petala, larg_petala]]
    classe = int(modelo.predict(entrada)[0])
    probas = modelo.predict_proba(entrada)[0]

    arvore = modelo.tree_
    nos = modelo.decision_path(entrada).indices

    passos = []
    for no in nos:
        feature = arvore.feature[no]
        if feature < 0:  # nó folha
            continue
        valor = entrada[0][feature]
        limiar = round(float(arvore.threshold[no]), 2)
        passos.append({
            "caracteristica": NOMES_PT.get(nomes[feature], nomes[feature]),
            "valor": valor,
            "valor_txt": _num(valor, 2),
            "sinal": "<=" if valor <= limiar else ">",
            "limiar": limiar,
            "limiar_txt": _num(limiar, 2),
        })

    regras = [
        f"{p['caracteristica']} = {p['valor_txt']} {p['sinal']} {p['limiar_txt']}"
        for p in passos
    ]

    return {
        "especie": classes[classe],
        "classe": classe,
        "probabilidades": {c: round(float(p), 4) for c, p in zip(classes, probas)},
        "passos": passos,
        "regras_percorridas": regras,
        "amostras_de_treino_na_folha": int(arvore.n_node_samples[nos[-1]]),
    }


# ==========================================
# MÉTRICAS DO MODELO
# ==========================================

def carregar_metricas():
    """Lê o metricas.json gerado por treinamento/metricas.py."""
    if not os.path.exists(METRICAS_PATH):
        return None
    with open(METRICAS_PATH, encoding="utf-8") as arquivo:
        return json.load(arquivo)


def painel_modelo():
    """Dados prontos para o painel 'Sobre o modelo' do template."""
    m = carregar_metricas()
    if not m:
        return None

    dt = m["decision_tree"]
    classes = _carregar()["classes"]
    matriz = dt["matriz_confusao"]
    maior = max(max(linha) for linha in matriz) or 1

    linhas = []
    for i, linha in enumerate(matriz):
        celulas = []
        for j, valor in enumerate(linha):
            if valor == 0:
                cor = "rgba(0,0,0,0.04)"
            else:
                opacidade = round(0.2 + 0.8 * valor / maior, 2)
                base = "16,185,129" if i == j else "239,68,68"
                cor = f"rgba({base},{opacidade})"
            celulas.append({"valor": valor, "cor": cor, "escura": valor / maior > 0.6})
        linhas.append({"classe": classes[i], "celulas": celulas})

    importancias = sorted(
        (
            {"nome": NOMES_PT.get(n, n), "pct": round(v * 100, 1), "txt": _pct(v)}
            for n, v in dt["importancia_features"].items()
        ),
        key=lambda x: x["pct"],
        reverse=True,
    )

    rotulos = {
        "decision_tree": "Decision Tree",
        "decision_tree_profundidade_3": "Decision Tree (profundidade 3)",
        "random_forest": "Random Forest",
        "knn": "k-NN",
        "regressao_logistica": "Regressão Logística",
    }
    comparacao = [
        {
            "nome": rotulos.get(chave, chave),
            "pct": round(v["media"] * 100, 1),
            "txt": _pct(v["media"]),
            "desvio": _pct(v["desvio"]),
            "destaque": chave == "decision_tree",
        }
        for chave, v in m["validacao_cruzada_5_folds"].items()
    ]

    cv_dt = m["validacao_cruzada_5_folds"]["decision_tree"]["media"]

    return {
        "classes": classes,
        "matriz": linhas,
        "importancias": importancias,
        "comparacao": comparacao,
        "resumo": {
            "acuracia_teste": _pct(dt["acuracia_teste"]),
            "acuracia_treino": _pct(dt["acuracia_treino"]),
            "cv_dt": _pct(cv_dt),
            "profundidade": dt["profundidade"],
            "folhas": dt["folhas"],
            "amostras_teste": m["dataset"]["teste"],
            "amostras_total": m["dataset"]["amostras"],
        },
    }