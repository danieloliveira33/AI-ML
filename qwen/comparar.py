# ==========================================
# PROJETO IA - COMPARAÇÃO
# Decision Tree (treinada) vs Qwen (pré-treinada)
# ==========================================

import os
import sys
import requests
import joblib

OLLAMA_URL = "http://localhost:11434"
MODELO_QWEN = "qwen2.5"

# Caminho para o modelo salvo pela Pessoa 1
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_PATH = os.path.join(RAIZ, "modelo", "modelo_iris.joblib")

FLORES_TESTE = [
    {"medidas": [5.1, 3.5, 1.4, 0.2], "esperado": "setosa"},
    {"medidas": [6.0, 2.9, 4.5, 1.5], "esperado": "versicolor"},
    {"medidas": [6.5, 3.0, 5.5, 2.0], "esperado": "virginica"},
    {"medidas": [4.9, 3.0, 1.4, 0.2], "esperado": "setosa"},
    {"medidas": [5.8, 2.7, 5.1, 1.9], "esperado": "virginica"},
]


# ==========================================
# 1. PREVISÃO VIA DECISION TREE
# ==========================================

def prever_decision_tree(medidas):
    artefato = joblib.load(MODELO_PATH)
    modelo = artefato["modelo"]
    classes = artefato["classes"]
    resultado = modelo.predict([medidas])[0]
    return classes[resultado]


# ==========================================
# 2. PREVISÃO VIA QWEN
# Prompt simplificado: resposta de uma palavra
# ==========================================

def prever_qwen(sepal_length, sepal_width, petal_length, petal_width):
    prompt = f"""Classify this Iris flower. Respond with ONLY ONE WORD: setosa, versicolor, or virginica.

Measurements:
- Sepal length: {sepal_length} cm
- Sepal width:  {sepal_width} cm
- Petal length: {petal_length} cm
- Petal width:  {petal_width} cm

Classification rules:
- setosa:     petal length < 2.5 cm
- versicolor: petal length between 3.0 and 5.1 cm
- virginica:  petal length > 4.9 cm and petal width > 1.7 cm

Answer (one word only):"""

    payload = {
        "model": MODELO_QWEN,
        "prompt": prompt,
        "stream": False
    }

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60
        )
        r.raise_for_status()
        resposta = r.json().get("response", "").strip().lower()

        for especie in ["setosa", "versicolor", "virginica"]:
            if especie in resposta:
                return especie

        return resposta or "indefinido"

    except requests.exceptions.ConnectionError:
        return "OFFLINE"
    except Exception as e:
        return "ERRO"


# ==========================================
# 3. TABELA COMPARATIVA
# ==========================================

def imprimir_tabela(resultados):
    col1 = 24  # medidas
    col2 = 12  # esperado
    col3 = 16  # decision tree
    col4 = 14  # qwen
    total = col1 + col2 + col3 + col4 + 3

    print("=" * total)
    print("COMPARAÇÃO: DECISION TREE vs QWEN")
    print("=" * total)
    print(
        f"{'Medidas':<{col1}}"
        f"{'Esperado':<{col2}}"
        f"{'Decision Tree':<{col3}}"
        f"{'Qwen':<{col4}}"
    )
    print("-" * total)

    acertos_dt = 0
    acertos_qwen = 0

    for r in resultados:
        esperado = r["esperado"]
        dt = r["decision_tree"]
        qwen = r["qwen"]

        ok_dt = "✓" if esperado == dt else "✗"
        ok_qwen = "✓" if esperado == qwen else "✗"

        if ok_dt == "✓":
            acertos_dt += 1
        if ok_qwen == "✓":
            acertos_qwen += 1

        medidas_str = str(r["medidas"])
        print(
            f"{medidas_str:<{col1}}"
            f"{esperado:<{col2}}"
            f"{dt+' '+ok_dt:<{col3}}"
            f"{qwen+' '+ok_qwen:<{col4}}"
        )

    total_flores = len(resultados)
    print("-" * total)
    print(f"Acurácia Decision Tree : {acertos_dt}/{total_flores}")
    print(f"Acurácia Qwen          : {acertos_qwen}/{total_flores}")
    print("=" * total)


# ==========================================
# 4. DIFERENÇAS CONCEITUAIS
# ==========================================

def imprimir_diferenca():
    print()
    print("DIFERENÇA CONCEITUAL ENTRE AS ABORDAGENS")
    print("-" * 67)
    print()
    print("Decision Tree (treinada pela equipe):")
    print("  - Aprendeu padrões a partir dos dados do Iris dataset")
    print("  - Divide os dados em regras matemáticas (nós da árvore)")
    print("  - Resultado determinístico: mesma entrada → mesma saída")
    print("  - Limitada ao domínio em que foi treinada")
    print()
    print("Qwen (IA pré-treinada, executada localmente):")
    print("  - Treinada em bilhões de textos gerais (não no Iris)")
    print("  - Interpreta linguagem natural e raciocina sobre os dados")
    print("  - Usa o prompt como instrução para classificar")
    print("  - Pode explicar a decisão em texto livre")
    print()


# ==========================================
# EXECUÇÃO
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(MODELO_PATH):
        print(f"ERRO: modelo não encontrado em {MODELO_PATH}")
        print("Execute primeiro: python treinamento/treinar.py")
        sys.exit(1)

    print(f"Carregando Decision Tree de: {MODELO_PATH}")
    print(f"Consultando Qwen via Ollama em: {OLLAMA_URL}")
    print()

    resultados = []

    for flor in FLORES_TESTE:
        medidas = flor["medidas"]
        esperado = flor["esperado"]

        print(f"Testando {medidas}...", end=" ", flush=True)

        dt = prever_decision_tree(medidas)
        qwen = prever_qwen(*medidas)

        print(f"DT={dt} | Qwen={qwen}")

        resultados.append({
            "medidas": medidas,
            "esperado": esperado,
            "decision_tree": dt,
            "qwen": qwen,
        })

    print()
    imprimir_tabela(resultados)
    imprimir_diferenca()