# ==========================================
# PROJETO IA - CONFIGURAÇÃO DO QWEN LOCAL
# Verifica Ollama e baixa o modelo qwen2.5
# ==========================================

import sys
import json
import requests

OLLAMA_URL = "http://localhost:11434"
MODELO = "qwen2.5"


# ==========================================
# 1. VERIFICAR SE OLLAMA ESTÁ RODANDO
# ==========================================

def verificar_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            print("Ollama está rodando.")
            return True
    except requests.exceptions.ConnectionError:
        pass

    print("Ollama não está rodando.")
    print("Inicie com o comando: ollama serve")
    return False


# ==========================================
# 2. VERIFICAR SE O MODELO JÁ EXISTE
# ==========================================

def verificar_modelo():
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    modelos = [m["name"] for m in r.json().get("models", [])]

    for nome in modelos:
        if MODELO in nome:
            print(f"Modelo '{nome}' já está disponível.")
            return True

    print(f"Modelo '{MODELO}' não encontrado localmente.")
    return False


# ==========================================
# 3. BAIXAR O MODELO
# ==========================================

def baixar_modelo():
    print(f"Baixando '{MODELO}'... (pode demorar na primeira vez)")
    print("Aguarde...\n")

    r = requests.post(
        f"{OLLAMA_URL}/api/pull",
        json={"name": MODELO},
        stream=True,
        timeout=600
    )

    for linha in r.iter_lines():
        if linha:
            data = json.loads(linha)
            status = data.get("status", "")
            total = data.get("total")
            completed = data.get("completed")

            if total and completed:
                pct = int((completed / total) * 100)
                print(f"\r{status}: {pct}%", end="", flush=True)
            elif status:
                print(status)

    print("\nModelo baixado com sucesso.")


# ==========================================
# 4. TESTAR O MODELO COM PROMPT SIMPLES
# ==========================================

def testar_modelo():
    print("\nTestando o modelo...")

    payload = {
        "model": MODELO,
        "prompt": "Respond with only the word: OK",
        "stream": False
    }

    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=60
    )

    resposta = r.json().get("response", "").strip()
    print(f"Resposta de teste: {resposta}")
    print("Modelo funcionando corretamente.")


# ==========================================
# EXECUÇÃO
# ==========================================

if __name__ == "__main__":
    print("=" * 50)
    print("CONFIGURAÇÃO - QWEN LOCAL VIA OLLAMA")
    print("=" * 50)

    if not verificar_ollama():
        sys.exit(1)

    if not verificar_modelo():
        baixar_modelo()

    testar_modelo()

    print("\nConfiguração concluída.")
    print(f"Modelo pronto: {MODELO}")