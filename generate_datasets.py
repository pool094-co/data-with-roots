# Importación de librerías para manipulación de datos y generación de números aleatorios
import pandas as pd
import numpy as np

# Configuración de semilla para garantizar la reproducibilidad de los datos sintéticos
np.random.seed(42)

# ==============================================================================
# 1. DATASET PARA REGRESIÓN LOGÍSTICA (E-COMMERCE)
# ==============================================================================
n_samples = 400
tiempo_navegacion = np.random.uniform(1, 60, n_samples)

logit = -3.5 + 0.12 * tiempo_navegacion
probabilidad_compra = 1 / (1 + np.exp(-logit))
compra_realizada = np.random.binomial(1, probabilidad_compra)

df_logistic = pd.DataFrame({
    'Tiempo_Navegacion_Minutos': np.round(tiempo_navegacion, 1),
    'Compra_Realizada': compra_realizada
})
df_logistic.to_csv('dataset_logistic.csv', index=False)
print("✓ Archivo 'dataset_logistic.csv' generado exitosamente con", len(df_logistic), "registros.")

# ==============================================================================
# 2. DATASET PARA K-NEAREST NEIGHBORS (RECURSOS HUMANOS)
# ==============================================================================
n_samples_knn = 500
experiencia_anios = np.random.uniform(0.5, 15.0, n_samples_knn)
puntaje_tecnico = np.random.uniform(40.0, 100.0, n_samples_knn)
participacion_proyectos = np.random.randint(1, 11, n_samples_knn)

score_promocion = (
    0.35 * experiencia_anios + 
    0.08 * (puntaje_tecnico - 40) + 
    0.45 * participacion_proyectos
)

ruido = np.random.normal(0, 1.2, n_samples_knn)
promovido = np.where((score_promocion + ruido) > 8.5, 1, 0)

df_knn = pd.DataFrame({
    'Experiencia_Anios': np.round(experiencia_anios, 1),
    'Puntaje_Tecnico': np.round(puntaje_tecnico, 1),
    'Participacion_Proyectos': participacion_proyectos,
    'Promovido': promovido
})
df_knn.to_csv('dataset_knn.csv', index=False)
print("✓ Archivo 'dataset_knn.csv' generado exitosamente con", len(df_knn), "registros.")