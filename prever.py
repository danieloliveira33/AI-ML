import joblib


# ==========================================
# 1. CARREGAR O MODELO
# ==========================================

artefato = joblib.load(
    "modelo/modelo_iris.joblib"
)

modelo = artefato["modelo"]
classes = artefato["classes"]
features = artefato["features"]


# ==========================================
# 2. MOSTRAR INFORMAÇÕES DO MODELO
# ==========================================

print("Características utilizadas pelo modelo:")

for feature in features:
    print("-", feature)


print("\nClasses possíveis:")

for classe in classes:
    print("-", classe)


# ==========================================
# 3. INFORMAR UMA NOVA FLOR
# ==========================================

flor = [
    [5.1, 3.5, 1.4, 0.2]
]


# ==========================================
# 4. FAZER A PREVISÃO
# ==========================================

resultado = modelo.predict(flor)[0]

especie = classes[resultado]


# ==========================================
# 5. MOSTRAR O RESULTADO
# ==========================================

print("\nCaracterísticas da nova flor:")
print(flor[0])

print("\nClasse prevista:")
print(resultado)

print("\nEspécie prevista:")
print(especie)