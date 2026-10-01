"""Agente MCP local usando Qwen2.5 via Ollama."""

import asyncio
import os
import sys
import time

# Antes de importar mcp_use.
os.environ["MCP_USE_ANONYMIZED_TELEMETRY"] = "false"

from langchain_ollama import ChatOllama  # noqa: E402
from mcp_use import MCPAgent, MCPClient  # noqa: E402


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# 1.5B erra a escolha de ferramentas com frequência. Prefira 7b (ou 3b).
QWEN_MODEL = os.environ.get("QWEN_AGENT_MODEL", "qwen2.5:3b")

INSTRUCOES = """
Você é um assistente que consulta o histórico de análises de classificação Iris.
Responda SEMPRE em português do Brasil, em no máximo 3 frases.

Escolha a ferramenta pelo tipo de pergunta:
- Quantas / total / média / por espécie -> resumo_historico
- Últimas análises / histórico recente -> listar_analises
- Análise com um ID específico -> obter_analise
- Filtrar por espécie ou status -> buscar_analises
- Classificar uma flor com medidas fornecidas -> classificar_flor
- Acurácia, profundidade, importância, qualidade do modelo -> informacoes_modelo
- Decision Tree vs Qwen, concordância, divergências -> comparar_dt_qwen

Regras:
- Chame uma única ferramenta e responda com base no resultado dela.
- Se o resultado tiver o campo "texto", use-o como base da resposta.
- Ignore campos técnicos de tempo e metadados, a menos que perguntem por eles.
- Nunca invente dados que não estejam no resultado da ferramenta.
""".strip()


def criar_agente():
    server_path = os.path.join(BASE_DIR, "mcp_interface", "server.py")

    config = {
        "mcpServers": {
            "historico_ml": {
                "command": sys.executable,
                "args": [server_path],
                "env": {
                    "MCP_USE_ANONYMIZED_TELEMETRY": "false",
                    "PYTHONUNBUFFERED": "1",
                },
            }
        }
    }

    client = MCPClient.from_dict(config)

    llm = ChatOllama(
        model=QWEN_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        num_ctx=4096,      # contexto menor = prefill mais rápido
        num_predict=400,   # limita o tamanho da resposta
        keep_alive="30m",  # mantém o modelo carregado entre execuções
    )

    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=5,
        verbose=False,
        additional_instructions=INSTRUCOES,
    )

    return agent, client


async def consultar(pergunta: str):
    """Envia uma pergunta ao agente; o Qwen decide qual ferramenta MCP usar."""
    agent, client = criar_agente()

    try:
        return await agent.run(pergunta, max_steps=5)
    finally:
        await client.close_all_sessions()


async def main():
    if len(sys.argv) > 1:
        pergunta = " ".join(sys.argv[1:])
    else:
        pergunta = input("Pergunta sobre o histórico de ML: ").strip()

    if not pergunta:
        print("Informe uma pergunta.")
        return

    inicio = time.perf_counter()

    try:
        resposta = await consultar(pergunta)
        print("\nResposta:")
        print(resposta)
        print(f"\n(tempo: {time.perf_counter() - inicio:.1f}s)")
    except Exception as exc:
        print(f"\nErro no agente MCP: {exc}")
        print(
            "Verifique se o Ollama está ativo e se o modelo "
            f"'{QWEN_MODEL}' está disponível (ollama pull {QWEN_MODEL})."
        )


if __name__ == "__main__":
    asyncio.run(main())
