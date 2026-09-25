import os
import joblib
from django.shortcuts import render
from django.conf import settings
from qwen.classificar import classificar_flor

# 1. Carrega o arquivo joblib
caminho_modelo = os.path.join(settings.BASE_DIR, 'modelo', 'modelo_iris.joblib')
arquivo_carregado = joblib.load(caminho_modelo)

# 2. Verifica se o colega salvou como dicionário e extrai apenas o modelo
if isinstance(arquivo_carregado, dict):
    # Pega o primeiro item dentro do dicionário (que será a Decision Tree)
    modelo_arvore = list(arquivo_carregado.values())[0]
else:
    modelo_arvore = arquivo_carregado

def index(request):
    contexto = {}

    if request.method == 'POST':
        comp_sepala = float(request.POST.get('comprimento_sepala'))
        larg_sepala = float(request.POST.get('largura_sepala'))
        comp_petala = float(request.POST.get('comprimento_petala'))
        larg_petala = float(request.POST.get('largura_petala'))

        # --- MACHINE LEARNING (Pessoa 1) ---
        features = [[comp_sepala, larg_sepala, comp_petala, larg_petala]]
        
        try:
            previsao_dt = modelo_arvore.predict(features)[0]
            especies = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
            resultado_dt = especies.get(previsao_dt, previsao_dt)
        except Exception as e:
            resultado_dt = "Erro ao ler modelo"
            print(f"Erro no Scikit-learn: {e}")

        # --- IA GENERATIVA QWEN (Pessoa 2) ---
        resultado_qwen = classificar_flor(
            comp_sepala, larg_sepala, comp_petala, larg_petala
        )

        contexto = {
            'resultado_dt': resultado_dt,
            'resultado_qwen': resultado_qwen,
            'dados': request.POST
        }

    return render(request, 'classificador/index.html', contexto)