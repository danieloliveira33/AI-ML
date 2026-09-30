"""MCP Server local para consulta do histórico de análises de ML."""

import logging
import os
import sys

# Precisa vir antes de importar mcp_use: a telemetria escreve no stdout
# e corrompe o protocolo JSON-RPC do transporte stdio.
os.environ["MCP_USE_ANONYMIZED_TELEMETRY"] = "false"

import django  # noqa: E402
from asgiref.sync import sync_to_async  # noqa: E402
from django.db.models import Avg, Count  # noqa: E402
from mcp_use import MCPServer  # noqa: E402

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

django.setup()

from classificador.models import AnaliseML  # noqa: E402
from classificador import ml

server = MCPServer(
    name="iris-ml-history",
    version="1.0.0",
)

MAX_TEXTO = 1500


def _cortar(texto, limite=MAX_TEXTO):
    if texto is None:
        return None
    texto = str(texto)
    return texto if len(texto) <= limite else texto[:limite] + "…"


def _resumir(analise):
    """Versão compacta: usada em listagens e buscas."""
    return {
        "id": analise.id,
        "criado_em": analise.criado_em.isoformat(timespec="seconds"),
        "entrada": {
            "comprimento_sepala": analise.comprimento_sepala,
            "largura_sepala": analise.largura_sepala,
            "comprimento_petala": analise.comprimento_petala,
            "largura_petala": analise.largura_petala,
        },
        "especie_decision_tree": analise.resultado_decision_tree,
        "qwen_status": analise.qwen_status,
        "qwen_tempo_ms": analise.qwen_tempo_ms,
    }


def _detalhar(analise):
    """Versão completa: usada apenas em obter_analise."""
    dados = _resumir(analise)
    dados["decision_tree"] = {
        "resultado": analise.resultado_decision_tree,
        "classe": analise.classe_decision_tree,
        "probabilidades": analise.probabilidades_decision_tree,
    }
    dados["qwen"] = {
        "modelo": analise.qwen_modelo,
        "status": analise.qwen_status,
        "tempo_ms": analise.qwen_tempo_ms,
        "resposta": _cortar(analise.resultado_qwen),
        "prompt": _cortar(analise.qwen_prompt),
    }
    return dados


@sync_to_async
def _listar_analises_sync(limit):
    queryset = AnaliseML.objects.order_by("-criado_em")[:limit]
    return [_resumir(a) for a in queryset]


@sync_to_async
def _obter_analise_sync(analise_id):
    try:
        analise = AnaliseML.objects.get(pk=analise_id)
    except AnaliseML.DoesNotExist:
        return {"erro": f"Análise {analise_id} não encontrada."}

    return _detalhar(analise)


@sync_to_async
def _buscar_analises_sync(especie, status_qwen, limit):
    queryset = AnaliseML.objects.all()

    if especie.strip():
        queryset = queryset.filter(
            resultado_decision_tree__icontains=especie.strip()
        )

    if status_qwen.strip():
        queryset = queryset.filter(qwen_status__icontains=status_qwen.strip())

    total = queryset.count()
    itens = [_resumir(a) for a in queryset.order_by("-criado_em")[:limit]]

    return {"total_encontrado": total, "retornadas": len(itens), "analises": itens}


@sync_to_async
def _resumo_historico_sync():
    total = AnaliseML.objects.count()

    por_especie = {
        (especie or "desconhecida"): qtd
        for especie, qtd in (
            AnaliseML.objects.order_by()
            .values_list("resultado_decision_tree")
            .annotate(qtd=Count("id"))
        )
    }

    por_status_qwen = {
        (status or "desconhecido"): qtd
        for status, qtd in (
            AnaliseML.objects.order_by()
            .values_list("qwen_status")
            .annotate(qtd=Count("id"))
        )
    }

    tempo_medio = AnaliseML.objects.aggregate(m=Avg("qwen_tempo_ms"))["m"]

    ultima = AnaliseML.objects.order_by("-criado_em").first()

    return {
        "texto": f"Existem {total} análises no histórico.",
        "total_analises": total,
        "por_especie_decision_tree": por_especie,
        "por_status_qwen": por_status_qwen,
        "tempo_medio_qwen_ms": round(tempo_medio, 1) if tempo_medio else None,
        "ultima_analise_id": ultima.id if ultima else None,
    }


@server.tool()
async def resumo_historico() -> dict:
    """USE PARA CONTAGENS E ESTATÍSTICAS. Responde perguntas como 'quantas
    análises existem', 'quantas por espécie', 'qual o tempo médio do Qwen'.
    Não precisa de parâmetros e retorna totais agregados."""

    return await _resumo_historico_sync()


@server.tool()
async def listar_analises(limit: int = 5) -> list[dict]:
    """Lista as análises mais recentes (versão compacta, máx. 20).
    Use quando pedirem 'as últimas análises' ou 'o histórico recente'.
    NÃO use para contar análises: para isso use resumo_historico."""

    limit = max(1, min(limit, 20))

    return await _listar_analises_sync(limit)


@server.tool()
async def obter_analise(analise_id: int) -> dict:
    """Retorna os detalhes completos de UMA análise, dado o ID numérico.
    Use quando o usuário citar um ID específico (ex.: 'análise 7')."""

    return await _obter_analise_sync(analise_id)


@server.tool()
async def buscar_analises(
    especie: str = "",
    status_qwen: str = "",
    limit: int = 10,
) -> dict:
    """Filtra análises por espécie (ex.: 'setosa', 'versicolor', 'virginica')
    e/ou por status do Qwen. Deixe em branco o filtro que não for pedido.
    Retorna o total encontrado e uma amostra compacta."""

    limit = max(1, min(limit, 20))

    return await _buscar_analises_sync(especie, status_qwen, limit)


ESPECIES = ("setosa", "versicolor", "virginica")


def _especie_no_texto(texto):
    """Primeira espécie citada no texto do Qwen, ou None."""
    texto = (texto or "").lower()
    achadas = [(texto.find(e), e) for e in ESPECIES if e in texto]
    return min(achadas)[1] if achadas else None


@sync_to_async
def _comparar_dt_qwen_sync():
    concordam = 0
    sem_resposta = 0
    divergentes = []

    for a in AnaliseML.objects.order_by("-criado_em"):
        esp_dt = (a.resultado_decision_tree or "").lower()
        esp_qwen = _especie_no_texto(a.resultado_qwen)

        if esp_qwen is None:
            sem_resposta += 1
        elif esp_qwen == esp_dt:
            concordam += 1
        else:
            divergentes.append({
                "id": a.id,
                "decision_tree": esp_dt,
                "qwen": esp_qwen,
                "comprimento_petala": a.comprimento_petala,
                "largura_petala": a.largura_petala,
            })

    comparadas = concordam + len(divergentes)

    return {
        "texto": (
            f"Decision Tree e Qwen concordaram em {concordam} de "
            f"{comparadas} análises comparáveis."
        ),
        "concordancia": round(concordam / comparadas, 4) if comparadas else None,
        "sem_resposta_do_qwen": sem_resposta,
        "divergencias_recentes": divergentes[:5],
    }


@server.tool()
async def classificar_flor(
    comprimento_sepala: float,
    largura_sepala: float,
    comprimento_petala: float,
    largura_petala: float,
) -> dict:
    """USE PARA CLASSIFICAR UMA FLOR NOVA com o modelo de Machine Learning
    (Decision Tree). Recebe as 4 medidas em cm e retorna a espécie prevista,
    as regras da árvore que levaram à decisão e as probabilidades.
    Não salva no histórico."""

    r = ml.prever_com_explicacao(
        comprimento_sepala, largura_sepala, comprimento_petala, largura_petala
    )
    r["texto"] = (
        f"A Decision Tree classificou como {r['especie']}. "
        f"Regras percorridas: {'; '.join(r['regras_percorridas'])}."
    )
    return r


@server.tool()
async def informacoes_modelo() -> dict:
    """USE PARA PERGUNTAS SOBRE O MODELO DE ML: acurácia, validação cruzada,
    profundidade da árvore, importância das características, comparação com
    outros algoritmos. Não recebe parâmetros."""

    m = ml.carregar_metricas()
    if m is None:
        return {"erro": "metricas.json não encontrado. Rode treinamento/metricas.py."}

    dt = m["decision_tree"]
    cv = m["validacao_cruzada_5_folds"]["decision_tree"]["media"]
    principal = max(dt["importancia_features"], key=dt["importancia_features"].get)

    m["texto"] = (
        f"Decision Tree com profundidade {dt['profundidade']} e {dt['folhas']} folhas. "
        f"Acurácia de teste {dt['acuracia_teste']:.2%}, validação cruzada média "
        f"{cv:.2%}. Característica mais importante: {principal}."
    )
    return m


@server.tool()
async def comparar_dt_qwen() -> dict:
    """USE PARA COMPARAR a Decision Tree com o Qwen no histórico: taxa de
    concordância e casos em que discordaram. Não recebe parâmetros."""

    return await _comparar_dt_qwen_sync()


if __name__ == "__main__":
    server.run(transport="stdio")