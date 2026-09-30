"""Gera modelo/metricas.json com a avaliação do modelo treinado.

Uso (na raiz do projeto): python treinamento/metricas.py
Usa a mesma divisão treino/teste do treinar.py (test_size=0.2, random_state=42).
"""

import json
import os

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

artefato = joblib.load(os.path.join(RAIZ, "modelo", "modelo_iris.joblib"))
modelo = artefato["modelo"]

previsoes = modelo.predict(X_test)


def cv(estimador):
    notas = cross_val_score(estimador, X, y, cv=5)
    return {"media": round(float(notas.mean()), 4), "desvio": round(float(notas.std()), 4)}


metricas = {
    "dataset": {"amostras": len(X), "treino": len(X_train), "teste": len(X_test)},
    "decision_tree": {
        "profundidade": int(modelo.get_depth()),
        "folhas": int(modelo.get_n_leaves()),
        "acuracia_treino": round(float(accuracy_score(y_train, modelo.predict(X_train))), 4),
        "acuracia_teste": round(float(accuracy_score(y_test, previsoes)), 4),
        "matriz_confusao": confusion_matrix(y_test, previsoes).tolist(),
        "importancia_features": {
            nome: round(float(v), 4)
            for nome, v in zip(iris.feature_names, modelo.feature_importances_)
        },
    },
    "validacao_cruzada_5_folds": {
        "decision_tree": cv(DecisionTreeClassifier(random_state=42)),
        "decision_tree_profundidade_3": cv(DecisionTreeClassifier(max_depth=3, random_state=42)),
        "random_forest": cv(RandomForestClassifier(random_state=42)),
        "knn": cv(make_pipeline(StandardScaler(), KNeighborsClassifier())),
        "regressao_logistica": cv(make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))),
    },
}

with open(os.path.join(RAIZ, "modelo", "metricas.json"), "w", encoding="utf-8") as f:
    json.dump(metricas, f, ensure_ascii=False, indent=2)

print(json.dumps(metricas, ensure_ascii=False, indent=2))