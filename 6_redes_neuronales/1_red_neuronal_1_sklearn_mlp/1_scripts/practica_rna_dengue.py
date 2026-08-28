"""
PRACTICA: Redes Neuronales Artificiales aplicadas a la predicción de
casos de dengue a partir de variables meteorológicas y rezagos (lags)
=====================================================================

Este dataset es una serie de tiempo epidemiológica: cada fila es una
semana epidemiológica con variables climáticas (temperatura, humedad,
precipitación, viento, índices oceánicos SOI/SST) y el número de casos
de dengue. Además incluye 12 rezagos (lags) de cada variable, es decir,
el valor de esa variable hace 1, 2, 3... hasta 12 semanas.

La tarea: predecir `casos_dengue` usando variables meteorológicas
(con y sin rezago) y los propios rezagos de `casos_dengue`.

IMPORTANTE SOBRE SERIES DE TIEMPO:
A diferencia de un dataset "normal", aquí NO se puede mezclar
aleatoriamente el train/test (shuffle), porque eso filtraría
información del futuro hacia el pasado (data leakage). El split debe
respetar el orden cronológico.

Cómo usar este script:
1. Si tienes tu archivo real, ponlo en la misma carpeta y cambia
   RUTA_CSV más abajo (debe tener exactamente las columnas del
   dataset descrito).
2. Si no tienes el archivo a mano, el script genera datos sintéticos
   con la MISMA estructura para que puedas practicar y entender el
   flujo completo; luego solo reemplazas la fuente de datos.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

RUTA_CSV = "dengue_dataset.csv"   # <-- cambia esto por tu archivo real
N_LAGS = 12


# ============================================================
# 0. CARGA DE DATOS (real o sintética para practicar)
# ============================================================

VARIABLES_BASE = [
    "temp", "temp_max", "temp_min", "hum_esp", "hum_rel",
    "prec", "dias_lluvia", "vel_vi", "vel_vi_max", "vel_vi_min",
    "soi", "sst",
]


def generar_datos_sinteticos(n_semanas=520, semilla=42):
    """
    Genera una serie de tiempo semanal ficticia con la misma estructura
    del dataset real (variables climáticas + casos_dengue + lags),
    SOLO para poder ejecutar y entender la práctica sin el archivo real.
    Sustituye esta función por pd.read_csv(RUTA_CSV) cuando tengas tus datos.
    """
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2014-01-05", periods=n_semanas, freq="W")

    t = np.arange(n_semanas)
    estacionalidad = np.sin(2 * np.pi * t / 52)

    temp = 24 + 3 * estacionalidad + rng.normal(0, 0.7, n_semanas)
    temp_max = temp + rng.uniform(2, 5, n_semanas)
    temp_min = temp - rng.uniform(2, 5, n_semanas)
    hum_rel = 70 + 10 * estacionalidad + rng.normal(0, 3, n_semanas)
    hum_esp = 12 + 2 * estacionalidad + rng.normal(0, 0.5, n_semanas)
    prec = np.clip(40 + 30 * estacionalidad + rng.normal(0, 15, n_semanas), 0, None)
    dias_lluvia = np.clip((prec / 10 + rng.normal(0, 1, n_semanas)).round(), 0, 7)
    vel_vi = np.clip(8 + rng.normal(0, 2, n_semanas), 0, None)
    vel_vi_max = vel_vi + rng.uniform(2, 6, n_semanas)
    vel_vi_min = np.clip(vel_vi - rng.uniform(1, 4, n_semanas), 0, None)
    soi = rng.normal(0, 1, n_semanas).cumsum() * 0.05
    sst = 27 + 0.5 * np.sin(2 * np.pi * t / 208) + rng.normal(0, 0.3, n_semanas)

    # casos_dengue depende (con rezago) de temperatura y precipitación,
    # simulando el ciclo de vida del mosquito (efecto retardado real)
    base = np.zeros(n_semanas)
    for i in range(n_semanas):
        efecto_temp = temp[max(0, i - 4):i + 1].mean() if i > 0 else temp[0]
        efecto_prec = prec[max(0, i - 6):i + 1].mean() if i > 0 else prec[0]
        base[i] = max(0, 5 + 3 * (efecto_temp - 24) + 0.3 * efecto_prec)
    ruido = rng.normal(0, 4, n_semanas)
    casos_dengue = np.clip((base + ruido).round(), 0, None)

    df = pd.DataFrame({
        "fecha": fechas,
        "año": fechas.year,
        "semana_epi": fechas.isocalendar().week.astype(int),
        "temp": temp, "temp_max": temp_max, "temp_min": temp_min,
        "hum_esp": hum_esp, "hum_rel": hum_rel,
        "prec": prec, "dias_lluvia": dias_lluvia,
        "vel_vi": vel_vi, "vel_vi_max": vel_vi_max, "vel_vi_min": vel_vi_min,
        "soi": soi, "sst": sst,
        "casos_dengue": casos_dengue,
    })

    # Generar los 12 rezagos de cada variable (incluyendo casos_dengue)
    for var in VARIABLES_BASE + ["casos_dengue"]:
        for lag in range(1, N_LAGS + 1):
            df[f"{var}_lag_{lag}"] = df[var].shift(lag)

    return df.dropna().reset_index(drop=True)


def cargar_datos():
    try:
        df = pd.read_csv(RUTA_CSV, parse_dates=["fecha"])
        print(f"Datos reales cargados desde {RUTA_CSV} ({len(df)} filas).")
    except FileNotFoundError:
        print(f"No se encontró '{RUTA_CSV}'. Generando datos sintéticos "
              f"para practicar (reemplaza esto por tu archivo real).")
        df = generar_datos_sinteticos()
    return df.sort_values("fecha").reset_index(drop=True)


df = cargar_datos()
print(df[["fecha", "temp", "prec", "casos_dengue"]].head(), "\n")


# ============================================================
# EJERCICIO 1: Split temporal (train/test) sin mezclar el tiempo
# ============================================================
# A diferencia de un split aleatorio (train_test_split con shuffle),
# en series de tiempo el conjunto de prueba debe ser SIEMPRE posterior
# en el tiempo al de entrenamiento.

def split_temporal(df, proporcion_train=0.8):
    n = len(df)
    corte = int(n * proporcion_train)
    return df.iloc[:corte].copy(), df.iloc[corte:].copy()


train_df, test_df = split_temporal(df)
print(f"=== Ejercicio 1: Split temporal ===")
print(f"Entrenamiento: {train_df['fecha'].min().date()} a {train_df['fecha'].max().date()} "
      f"({len(train_df)} semanas)")
print(f"Prueba:        {test_df['fecha'].min().date()} a {test_df['fecha'].max().date()} "
      f"({len(test_df)} semanas)\n")

# TODO 1: ¿Qué pasaría con las métricas del modelo si usaras
# train_test_split(df, shuffle=True) en vez de un corte temporal?
# Pruébalo y compara el error: notarás que el modelo "hace trampa"
# porque ve semanas futuras cercanas a las de prueba durante el
# entrenamiento (fuga de datos / data leakage).


# ============================================================
# EJERCICIO 2: Tres conjuntos de variables predictoras
# ============================================================
# Comparamos 3 escenarios, como sugiere el enunciado del dataset:
#   A) Solo clima actual (sin rezagos)
#   B) Clima actual + rezagos climáticos (sin lags de casos_dengue)
#   C) Clima + rezagos climáticos + rezagos de casos_dengue (autoregresivo)

cols_clima_actual = VARIABLES_BASE
cols_clima_lags = VARIABLES_BASE + [
    f"{v}_lag_{i}" for v in VARIABLES_BASE for i in range(1, N_LAGS + 1)
]
cols_clima_lags_autoreg = cols_clima_lags + [
    f"casos_dengue_lag_{i}" for i in range(1, N_LAGS + 1)
]

escenarios = {
    "A) Solo clima actual": cols_clima_actual,
    "B) Clima + rezagos climáticos": cols_clima_lags,
    "C) Clima + rezagos + autoregresivo": cols_clima_lags_autoreg,
}


def entrenar_evaluar_mlp(features, semilla=0, hidden_layer_sizes=(32, 16)):
    X_train = train_df[features].values
    y_train = train_df["casos_dengue"].values
    X_test = test_df[features].values
    y_test = test_df["casos_dengue"].values

    escalador_X = StandardScaler().fit(X_train)
    X_train_s = escalador_X.transform(X_train)
    X_test_s = escalador_X.transform(X_test)

    modelo = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=2000,
        early_stopping=True,
        random_state=semilla,
    )
    modelo.fit(X_train_s, y_train)

    pred_test = modelo.predict(X_test_s)
    rmse = mean_squared_error(y_test, pred_test) ** 0.5
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)

    return modelo, escalador_X, {"RMSE": rmse, "MAE": mae, "R2": r2}, pred_test


print("=== Ejercicio 2: Comparación de conjuntos de variables ===")
resultados_escenarios = {}
for nombre, features in escenarios.items():
    _, _, metricas, _ = entrenar_evaluar_mlp(features)
    resultados_escenarios[nombre] = metricas
    print(f"{nombre:38s} -> RMSE={metricas['RMSE']:.2f}  "
          f"MAE={metricas['MAE']:.2f}  R2={metricas['R2']:.3f}")
print()

# TODO 2: ¿Cuál de los tres escenarios da mejor R2? Reflexiona: si el
# escenario C (autoregresivo) es mucho mejor que A y B, ¿significa que
# el modelo aprendió del clima, o que simplemente está "copiando" el
# valor de casos_dengue de semanas recientes? ¿Cómo lo comprobarías?


# ============================================================
# EJERCICIO 3: Efecto de la arquitectura (número de neuronas/capas)
# ============================================================
# Igual que el texto menciona sobre el escalamiento de arquitecturas,
# probemos cómo cambia el desempeño según el tamaño de la red.

arquitecturas = {
    "Pequeña (8,)": (8,),
    "Mediana (32,16)": (32, 16),
    "Grande (64,32,16)": (64, 32, 16),
}

print("=== Ejercicio 3: Efecto del tamaño de la arquitectura ===")
mejores_features = cols_clima_lags_autoreg
for nombre_arq, arq in arquitecturas.items():
    _, _, metricas, _ = entrenar_evaluar_mlp(mejores_features, hidden_layer_sizes=arq)
    print(f"{nombre_arq:20s} -> RMSE={metricas['RMSE']:.2f}  R2={metricas['R2']:.3f}")
print()

# TODO 3: Agrega una arquitectura muy grande, por ejemplo (128, 64, 32),
# y observa si el desempeño en TEST mejora o empeora respecto al de
# entrenamiento. Si el error de train baja pero el de test sube,
# es sobreajuste (overfitting): la red "memorizó" en vez de generalizar.


# ============================================================
# EJERCICIO 4: Óptimos locales (retomando el texto original)
# ============================================================
# El texto decía que el entrenamiento puede quedar atascado en
# óptimos locales, pero que en la práctica no suele ser grave.
# Entrenemos el MISMO modelo con distintas semillas (distintos pesos
# iniciales) y comparemos qué tanto varía el resultado final.

print("=== Ejercicio 4: Variabilidad por óptimos locales (distintas semillas) ===")
r2_por_semilla = []
for semilla in range(5):
    _, _, metricas, _ = entrenar_evaluar_mlp(cols_clima_lags_autoreg, semilla=semilla)
    r2_por_semilla.append(metricas["R2"])
    print(f"Semilla {semilla} -> R2={metricas['R2']:.3f}")

print(f"\nDesviación estándar del R2 entre semillas: {np.std(r2_por_semilla):.4f}")
print("Si esta desviación es pequeña, confirma lo que dice el texto:")
print("distintos óptimos locales llevan a resultados casi tan buenos "
      "como el mejor encontrado.\n")


# ============================================================
# EJERCICIO 5: ¿Qué variables pesan más? (importancia por permutación)
# ============================================================
# Este método mide cuánto empeora el error si "revolvemos" (permutamos)
# los valores de una variable, dejando las demás intactas: si el error
# sube mucho, esa variable era importante para el modelo.

modelo, escalador_X, metricas_finales, pred_test = entrenar_evaluar_mlp(
    cols_clima_lags_autoreg, hidden_layer_sizes=(32, 16)
)

X_test_s = escalador_X.transform(test_df[cols_clima_lags_autoreg].values)
y_test = test_df["casos_dengue"].values

importancia = permutation_importance(
    modelo, X_test_s, y_test, n_repeats=10, random_state=0, scoring="r2"
)

df_importancia = pd.DataFrame({
    "variable": cols_clima_lags_autoreg,
    "importancia_media": importancia.importances_mean,
}).sort_values("importancia_media", ascending=False)

print("=== Ejercicio 5: Top 10 variables más importantes ===")
print(df_importancia.head(10).to_string(index=False), "\n")

# TODO 4: Observa si predominan lags cortos (1-3 semanas) o largos
# (10-12 semanas) de casos_dengue, y si aparecen lags climáticos entre
# los más importantes. En epidemiología del dengue se espera que el
# efecto del clima sobre los casos tenga un retardo de varias semanas
# (tiempo de cría del mosquito + incubación); ¿tu modelo lo refleja?


# ============================================================
# GRÁFICA: Real vs. predicho en el periodo de prueba
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(test_df["fecha"], y_test, label="Casos reales", linewidth=1.5)
plt.plot(test_df["fecha"], pred_test, label="Casos predichos (RNA)", linewidth=1.5)
plt.title("Casos de dengue: real vs. predicho (conjunto de prueba)")
plt.xlabel("Fecha")
plt.ylabel("Casos de dengue")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\marco\Downloads\dengue_real_vs_predicho.png", dpi=120)
print("Gráfica guardada como 'dengue_real_vs_predicho.png'")


# ============================================================
# PREGUNTAS DE REFLEXIÓN FINAL
# ============================================================
# 1. ¿Por qué es incorrecto evaluar un modelo de series de tiempo con
#    un split aleatorio (shuffle=True) en vez de uno cronológico?
#
# 2. Comparando los escenarios A, B y C del Ejercicio 2: ¿los rezagos
#    climáticos (sin usar casos_dengue pasado) ya aportan buena
#    capacidad predictiva, o el modelo depende casi todo del propio
#    historial de casos?
#
# 3. En el Ejercicio 3, ¿en qué punto empieza el sobreajuste? ¿Qué
#    técnicas del propio MLPRegressor (early_stopping, alpha de
#    regularización) podrías ajustar para controlarlo?
#
# 4. El texto original menciona el Transformer y la atención como el
#    mecanismo que permite a un modelo "decidir" qué parte de la
#    secuencia es más relevante. La importancia por permutación del
#    Ejercicio 5 es una forma mucho más simple de responder una
#    pregunta parecida ("¿qué rezago importa más?"). ¿Qué ventajas
#    tendría un modelo tipo Transformer sobre un MLP para este dataset?
