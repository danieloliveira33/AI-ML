import asyncio
import json
import time

from django.http import JsonResponse
from django.shortcuts import render

from classificador import ml
from classificador.models import AnaliseML
from qwen.classificar import classificar_flor_com_detalhes

CORES_ESPECIES = {
    "Setosa": "#0ea5e9",
    "Versicolor": "#f59e0b",
    "Virginica": "#8b5cf6",
}


def _formatar_tempo_ms(ms):
    """0,05 ms / 812 ms / 24,3 s."""
    if ms is None:
        return "—"
    if ms < 1:
        return f"{ms:.2f} ms".replace(".", ",")
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.1f} s".replace(".", ",")


def _barras_probabilidade(probabilidades):
    """Lista pronta para o template, da mais provável para a menos provável."""
    barras = [
        {
            "nome": nome,
            "pct": round(valor * 100, 1),
            "txt": f"{valor * 100:.1f}%".replace(".", ","),
            "cor": CORES_ESPECIES.get(nome, "#64748b"),
        }
        for nome, valor in probabilidades.items()
    ]
    return sorted(barras, key=lambda b: b["pct"], reverse=True)


# ==========================================
# VIEW PRINCIPAL
# ==========================================

def index(request):
    contexto = {}

    if request.method == "POST":
        comp_sepala = float(request.POST.get("comprimento_sepala"))
        larg_sepala = float(request.POST.get("largura_sepala"))
        comp_petala = float(request.POST.get("comprimento_petala"))
        larg_petala = float(request.POST.get("largura_petala"))

        # ==========================================
        # MACHINE LEARNING
        # ==========================================

        resultado_dt = "Erro ao ler modelo"
        classe_dt = None
        probabilidades_dt = {}
        explicacao = None

        inicio_dt = time.perf_counter()

        try:
            explicacao = ml.prever_com_explicacao(
                comp_sepala, larg_sepala, comp_petala, larg_petala
            )
            resultado_dt = explicacao["especie"].capitalize()
            classe_dt = explicacao["classe"]
            probabilidades_dt = {
                nome.capitalize(): valor
                for nome, valor in explicacao["probabilidades"].items()
            }
        except Exception as e:
            print(f"Erro no Scikit-learn: {e}")

        tempo_dt_ms = (time.perf_counter() - inicio_dt) * 1000

        # ==========================================
        # IA GENERATIVA — QWEN
        # ==========================================

        detalhes_qwen = classificar_flor_com_detalhes(
            comp_sepala, larg_sepala, comp_petala, larg_petala
        )

        resultado_qwen = detalhes_qwen["response"]
        tempo_qwen_ms = detalhes_qwen["tempo_ms"]

        # ==========================================
        # PERSISTÊNCIA DA ANÁLISE
        # ==========================================

        AnaliseML.objects.create(
            comprimento_sepala=comp_sepala,
            largura_sepala=larg_sepala,
            comprimento_petala=comp_petala,
            largura_petala=larg_petala,
            resultado_decision_tree=resultado_dt,
            classe_decision_tree=classe_dt,
            probabilidades_decision_tree=probabilidades_dt,
            resultado_qwen=resultado_qwen,
            qwen_modelo=detalhes_qwen["model"],
            qwen_prompt=detalhes_qwen["prompt"],
            qwen_status=detalhes_qwen["status"],
            qwen_tempo_ms=tempo_qwen_ms,
            metadados={
                "ollama_url": detalhes_qwen["ollama_url"],
                "ollama_metadata": detalhes_qwen["ollama_metadata"],
                "decision_tree_tempo_ms": round(tempo_dt_ms, 3),
                "caminho_arvore": explicacao["regras_percorridas"] if explicacao else [],
                "entrada_original": {
                    "comprimento_sepala": comp_sepala,
                    "largura_sepala": larg_sepala,
                    "comprimento_petala": comp_petala,
                    "largura_petala": larg_petala,
                },
            },
        )

        # ==========================================
        # COMPARATIVO DECISION TREE x QWEN
        # ==========================================

        especie_qwen = ml.especie_no_texto(resultado_qwen)
        concordancia = None
        if especie_qwen and classe_dt is not None:
            concordancia = especie_qwen == resultado_dt.lower()

        razao = None
        if (
            detalhes_qwen["status"] == "ok"
            and tempo_qwen_ms
            and tempo_dt_ms > 0
        ):
            razao = f"{int(tempo_qwen_ms / tempo_dt_ms):,}".replace(",", ".")

        contexto = {
            "resultado_dt": resultado_dt,
            "resultado_qwen": resultado_qwen,
            "dados": request.POST,
            "explicacao": explicacao,
            "barras": _barras_probabilidade(probabilidades_dt),
            "tempo_dt": _formatar_tempo_ms(tempo_dt_ms),
            "tempo_qwen": _formatar_tempo_ms(tempo_qwen_ms),
            "razao_tempo": razao,
            "modelo_qwen": detalhes_qwen["model"],
            "qwen_status": detalhes_qwen["status"],
            "especie_qwen": especie_qwen,
            "concordancia": concordancia,
        }

    # O painel "Sobre o modelo" aparece com ou sem análise
    contexto["painel"] = ml.painel_modelo()

    return render(request, "classificador/index.html", contexto)


# ==========================================
# CHAT MCP
# ==========================================

def mcp_chat(request):
    if request.method != "POST":
        return JsonResponse({"erro": "Método não permitido."}, status=405)

    try:
        dados = json.loads(request.body)
        pergunta = dados.get("pergunta", "").strip()

        if not pergunta:
            return JsonResponse({"erro": "Pergunta não informada."}, status=400)

        from mcp_interface.agent import consultar

        resposta = asyncio.run(consultar(pergunta))

        return JsonResponse({"resposta": str(resposta)})

    except Exception as e:
        print(f"[MCP] Erro: {e}")

        return JsonResponse(
            {"erro": f"Erro ao consultar o MCP: {str(e)}"},
            status=500,
        )