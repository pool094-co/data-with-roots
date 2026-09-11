# Importación de librerías para análisis de datos, modelado estadístico y visualización
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configuración del backend sin interfaz gráfica para renderizado web
import matplotlib.pyplot as plt
import io
import base64

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

# ==============================================================================
# CARGA Y PREPROCESAMIENTO DE DATOS DE REGRESIÓN LOGÍSTICA
# ==============================================================================
# Carga del conjunto de datos desde el archivo CSV mediante Pandas
df_logistic = pd.read_csv('dataset_logistic.csv')

# Definición de la variable independiente (X) y la variable objetivo binaria (y)
X_logistic = df_logistic[['Tiempo_Navegacion_Minutos']]
y_logistic = df_logistic['Compra_Realizada']

# División formal de datos: 80% Entrenamiento y 20% Evaluación / Pruebas
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_logistic, y_logistic, test_size=0.20, random_state=42
)

# Inicialización y entrenamiento del algoritmo de Regresión Logística
model_logistic = LogisticRegression()
model_logistic.fit(X_train_log, y_train_log)


def get_logistic_summary():
    """Retorna un resumen estadístico del dataset de Regresión Logística."""
    return {
        'total_records': len(df_logistic),
        'train_records': len(X_train_log),
        'test_records': len(X_test_log),
        'feature_name': 'Tiempo de Navegación (Minutos)',
        'target_name': 'Compra Realizada',
        'class_0': 'Abandona (0)',
        'class_1': 'Compra (1)'
    }


def predict_logistic(tiempo_minutos):
    """Genera la predicción de clase binaria y la probabilidad para un nuevo valor de entrada."""
    input_data = np.array([[float(tiempo_minutos)]])
    prediction = int(model_logistic.predict(input_data)[0])
    probabilities = model_logistic.predict_proba(input_data)[0]
    
    return {
        'prediction': prediction,
        'prob_class_0': round(float(probabilities[0]) * 100, 2),
        'prob_class_1': round(float(probabilities[1]) * 100, 2)
    }


def generate_logistic_plot():
    """Genera la gráfica de dispersión y la curva sigmoide ajustada en formato Base64."""
    plt.figure(figsize=(8, 4.5))
    
    # Graficar puntos reales de entrenamiento
    plt.scatter(
        X_train_log['Tiempo_Navegacion_Minutos'], y_train_log, 
        c=y_train_log, cmap='coolwarm', alpha=0.6, edgecolors='k', label='Datos de Entrenamiento'
    )
    
    # Generar curva de probabilidad sigmoide suave
    X_curve = np.linspace(X_logistic.min().values[0], X_logistic.max().values[0], 300).reshape(-1, 1)
    y_prob_curve = model_logistic.predict_proba(X_curve)[:, 1]
    
    plt.plot(X_curve, y_prob_curve, color='green', linewidth=2.5, label='Curva Sigmoide (Probabilidad de Compra)')
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Umbral de Decisión (0.5)')
    
    plt.title('Regresión Logística: Tiempo de Navegación vs Decisión de Compra', fontsize=12, fontweight='bold')
    plt.xlabel('Tiempo de Navegación en el Sitio Web (Minutos)', fontsize=10)
    plt.ylabel('Probabilidad de Compra / Clase Binaria', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()

    # Conversión del gráfico a buffer de memoria en codificación Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64


def get_logistic_metrics():
    """Calcula las métricas de evaluación basándose exclusivamente en el 20% del dataset de prueba."""
    y_pred_test = model_logistic.predict(X_test_log)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test_log, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test_log, y_pred_test)
    prec = precision_score(y_test_log, y_pred_test, zero_division=0)
    
    return {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'accuracy': round(float(acc) * 100, 2),
        'precision': round(float(prec) * 100, 2)
    }