# ==========================================
# PROJETO IA - CLASSIFICAÇÃO DE FLORES
# Árvore de Decisão + Iris Dataset
# ==========================================


# ==========================================
# 1. IMPORTAÇÕES
# ==========================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import matplotlib.pyplot as plt
import joblib


# ==========================================
# 2. CARREGAR DATASET
# ==========================================

iris = load_iris()

print("Características:")
print(iris.feature_names)

print("\nClasses:")
print(iris.target_names)

print("\nQuantidade de exemplos:")
print(len(iris.data))


# ==========================================
# 3. SEPARAR CARACTERÍSTICAS E ALVOS
# ==========================================

X = iris.data
y = iris.target

# X = características das flores
# y = espécie da flor


# ==========================================
# 4. DIVIDIR DADOS EM TREINO E TESTE
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nQuantidade de dados para treinamento:")
print(len(X_train))

print("\nQuantidade de dados para teste:")
print(len(X_test))


# ==========================================
# 5. CRIAR O MODELO
# ==========================================

modelo = DecisionTreeClassifier(
    random_state=42
)


# ==========================================
# 6. TREINAR O MODELO
# ==========================================

modelo.fit(X_train, y_train)

print("\nModelo treinado com sucesso!")


# ==========================================
# 7. FAZER PREVISÕES NOS DADOS DE TESTE
# ==========================================

previsoes = modelo.predict(X_test)


# ==========================================
# 8. CALCULAR ACURÁCIA
# ==========================================

acuracia = accuracy_score(
    y_test,
    previsoes
)

print("\nAcurácia:", acuracia)


# ==========================================
# 9. MATRIZ DE CONFUSÃO
# ==========================================

matriz = confusion_matrix(
    y_test,
    previsoes
)

print("\nMatriz de confusão:")
print(matriz)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=iris.target_names
)

disp.plot()

plt.title("Matriz de Confusão")

plt.savefig(
    "modelo/matriz_confusao.png",
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 10. RELATÓRIO DE CLASSIFICAÇÃO
# ==========================================

print("\nRelatório de classificação:")

print(
    classification_report(
        y_test,
        previsoes,
        target_names=iris.target_names
    )
)


# ==========================================
# 11. IMPORTÂNCIA DAS CARACTERÍSTICAS
# ==========================================

print("\nImportância das características:")

for nome, importancia in zip(
        iris.feature_names,
        modelo.feature_importances_
):
    print(
        f"{nome}: {importancia:.4f}"
    )


# ==========================================
# 12. TESTAR VÁRIAS FLORES MANUALMENTE
# ==========================================

flores = [
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.5, 3.0, 5.5, 2.0]
]

resultados = modelo.predict(flores)

print("\nPrevisões:")

for flor, resultado in zip(
        flores,
        resultados
):
    especie = iris.target_names[resultado]

    print(
        f"{flor} -> {especie}"
    )


# ==========================================
# 13. TESTE MANUAL DE UMA NOVA FLOR
# ==========================================

flor = [
    [5.1, 3.5, 1.4, 0.2]
]

resultado = modelo.predict(flor)

especie = iris.target_names[resultado[0]]

print("\nPrevisão para a nova flor:")

print(
    "Características:",
    flor[0]
)

print(
    "Classe:",
    resultado[0]
)

print(
    "Espécie:",
    especie
)


# ==========================================
# 14. VISUALIZAR E SALVAR A ÁRVORE
# ==========================================

plt.figure(
    figsize=(15, 10)
)

plot_tree(
    modelo,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title(
    "Árvore de Decisão - Iris"
)

plt.savefig(
    "modelo/arvore_iris.png",
    bbox_inches="tight"
)

plt.show()

print(
    "\nÁrvore salva com sucesso!"
)

print(
    "Arquivo: modelo/arvore_iris.png"
)


# ==========================================
# 15. GRÁFICO DE IMPORTÂNCIA
# ==========================================

nomes = iris.feature_names
importancias = modelo.feature_importances_

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    nomes,
    importancias
)

plt.title(
    "Importância das Características"
)

plt.xlabel(
    "Características"
)

plt.ylabel(
    "Importância"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    "modelo/importancia_caracteristicas.png",
    bbox_inches="tight"
)

plt.show()

print(
    "\nGráfico de importância salvo com sucesso!"
)

print(
    "Arquivo: modelo/importancia_caracteristicas.png"
)


# ==========================================
# 16. SALVAR MODELO
# ==========================================

artefato = {
    "modelo": modelo,
    "classes": iris.target_names.tolist(),
    "features": iris.feature_names
}

joblib.dump(
    artefato,
    "modelo/modelo_iris.joblib"
)

print(
    "\nModelo salvo com sucesso!"
)

print(
    "Arquivo: modelo/modelo_iris.joblib"
)