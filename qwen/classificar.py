# ==========================================
# PROJETO IA - CLASSIFICAÇÃO VIA QWEN
# IA local pré-treinada via Ollama
# ==========================================

import os
import time
import requests

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://localhost:11434"
)

MODELO = "qwen2.5:3b"


# ==========================================
# 1. MONTAR PROMPT COM AS MEDIDAS DA FLOR
# ==========================================

def montar_prompt(sepal_length, sepal_width, petal_length, petal_width):
    return f"""You are a botanist specialized in Iris flower classification.

Given the measurements below, identify which Iris species this flower belongs to.
The three possible species are: Iris setosa, Iris versicolor, and Iris virginica.

Flower measurements:
- Sepal length: {sepal_length} cm
- Sepal width:  {sepal_width} cm
- Petal length: {petal_length} cm
- Petal width:  {petal_width} cm

Classification reference:
- Iris setosa:     petal length typically < 2.5 cm (very small petals)
- Iris versicolor: petal length typically between 3.0 and 5.1 cm (medium petals)
- Iris virginica:  petal length typically > 4.9 cm and petal width > 1.7 cm (large petals)

Respond in Portuguese using exactly this format:
Espécie: [species name]
Justificativa: [one sentence explaining the classification based on the measurements above]
"""


# ==========================================
# 2. ENVIAR PROMPT AO QWEN COM DETALHES
# ==========================================

def classificar_flor_com_detalhes(
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
):
    """
    Executa a classificação no Qwen e retorna a resposta
    junto com os dados necessários para persistência.
    """

    prompt = montar_prompt(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    payload = {
        "model": MODELO,
        "prompt": prompt,
        "stream": False
    }

    inicio = time.perf_counter()

    try:
        print(
            f"[Qwen] Conectando em: "
            f"{OLLAMA_URL}/api/generate "
            f"(modelo: {MODELO})"
        )

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        dados = response.json()

        resposta = dados.get("response", "").strip()

        tempo_ms = (
            time.perf_counter() - inicio
        ) * 1000

        return {
            "response": resposta,
            "prompt": prompt,
            "model": MODELO,
            "ollama_url": OLLAMA_URL,
            "status": "ok",
            "tempo_ms": round(tempo_ms, 2),
            "ollama_metadata": {
                chave: valor
                for chave, valor in dados.items()
                if chave != "response"
            }
        }

    except requests.exceptions.ConnectionError as e:
        tempo_ms = (
            time.perf_counter() - inicio
        ) * 1000

        print(f"[Qwen] ERRO de conexão: {e}")

        return {
            "response": (
                f"ERRO: Ollama não está acessível em "
                f"{OLLAMA_URL}. Verifique se está rodando "
                f"com: OLLAMA_HOST=0.0.0.0:11434 ollama serve"
            ),
            "prompt": prompt,
            "model": MODELO,
            "ollama_url": OLLAMA_URL,
            "status": "connection_error",
            "tempo_ms": round(tempo_ms, 2),
            "ollama_metadata": {}
        }

    except requests.exceptions.Timeout:
        tempo_ms = (
            time.perf_counter() - inicio
        ) * 1000

        print("[Qwen] ERRO: Timeout após 60s")

        return {
            "response": "ERRO: Timeout ao aguardar resposta do Qwen.",
            "prompt": prompt,
            "model": MODELO,
            "ollama_url": OLLAMA_URL,
            "status": "timeout",
            "tempo_ms": round(tempo_ms, 2),
            "ollama_metadata": {}
        }

    except Exception as e:
        tempo_ms = (
            time.perf_counter() - inicio
        ) * 1000

        print(f"[Qwen] ERRO inesperado: {e}")

        return {
            "response": f"ERRO: {e}",
            "prompt": prompt,
            "model": MODELO,
            "ollama_url": OLLAMA_URL,
            "status": "error",
            "tempo_ms": round(tempo_ms, 2),
            "ollama_metadata": {}
        }


# ==========================================
# 3. FUNÇÃO ORIGINAL
# ==========================================

def classificar_flor(
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
):
    """
    Mantém a interface original utilizada pelo projeto.
    Retorna somente o texto da resposta do Qwen.
    """

    detalhes = classificar_flor_com_detalhes(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    return detalhes["response"]


# ==========================================
# 4. EXIBIR RESULTADO FORMATADO
# ==========================================

def exibir_resultado(i, medidas, esperado, resposta):
    sl, sw, pl, pw = medidas

    print(f"\nFlor {i}:")
    print(f"  Comprimento da sépala: {sl} cm")
    print(f"  Largura da sépala:     {sw} cm")
    print(f"  Comprimento da pétala: {pl} cm")
    print(f"  Largura da pétala:     {pw} cm")
    print(f"  Espécie esperada:      {esperado}")
    print()
    print("  Resposta do Qwen:")

    for linha in resposta.split("\n"):
        linha = linha.strip()

        if linha:
            print(f"    {linha}")

    print("-" * 55)


# ==========================================
# EXECUÇÃO — FLORES DE TESTE
# ==========================================

FLORES_TESTE = [
    {
        "medidas": [5.1, 3.5, 1.4, 0.2],
        "esperado": "setosa"
    },
    {
        "medidas": [6.0, 2.9, 4.5, 1.5],
        "esperado": "versicolor"
    },
    {
        "medidas": [6.5, 3.0, 5.5, 2.0],
        "esperado": "virginica"
    },
]


if __name__ == "__main__":
    print("=" * 55)
    print("CLASSIFICAÇÃO DE FLORES — QWEN (IA LOCAL)")
    print(f"Modelo: {MODELO} via Ollama")
    print("=" * 55)

    for i, flor in enumerate(FLORES_TESTE, start=1):
        sl, sw, pl, pw = flor["medidas"]

        resposta = classificar_flor(
            sl,
            sw,
            pl,
            pw
        )

        exibir_resultado(
            i,
            flor["medidas"],
            flor["esperado"],
            resposta
        )

    print("\nClassificação concluída.")
