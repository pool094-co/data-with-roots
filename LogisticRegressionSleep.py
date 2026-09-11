import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica para evitar errores de renderizado en servidores web como Render
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# =========================================================
# 1. GENERACIÓN Y PREPROCESAMIENTO DEL DATASET
# =========================================================

# Fijamos la semilla aleatoria para garantizar la reproducibilidad de las muestras en cada ejecución
np.random.seed(42)

# Generamos 500 registros para la variable independiente X (horas de sueño por noche, entre 3 y 10 horas)
horas_sueno = np.round(np.random.uniform(3, 10, size=500), 1)

# Modelamos la probabilidad de deserción mediante una función logística: a menos horas de sueño,
# mayor probabilidad de deserción. z es la combinación lineal que alimenta la sigmoide.
z = -1.3 * (horas_sueno - 6.2) + np.random.normal(0, 1.0, size=500)
prob_desercion = 1 / (1 + np.exp(-z))

# La variable objetivo binaria se genera mediante un muestreo Bernoulli sobre la probabilidad simulada
desercion = np.random.binomial(1, prob_desercion)

# Estructuramos el conjunto de datos en un DataFrame de Pandas y lo exportamos a CSV para persistencia
df = pd.DataFrame({
    'horas_sueno': horas_sueno,
    'desercion': desercion
})
df.to_csv('logistic_dataset.csv', index=False)

# =========================================================
# 2. DIVISIÓN TRAIN/TEST Y ENTRENAMIENTO DEL MODELO
# =========================================================

# Extraemos la matriz de características (X) y el vector objetivo (y)
x = df[["horas_sueno"]].values
y = df["desercion"].values

# Dividimos el dataset en 80% entrenamiento y 20% prueba, usando random_state fijo para reproducibilidad
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Instanciamos el algoritmo de Regresión Logística y ajustamos el modelo sobre el conjunto de entrenamiento
model = LogisticRegression()
model.fit(x_train, y_train)

# Generamos las predicciones sobre el conjunto de prueba, necesarias para las métricas de evaluación
y_pred_test = model.predict(x_test)

# =========================================================
# 3. FUNCIONES DE PREDICCIÓN Y VISUALIZACIÓN
# =========================================================

def predictDesercion(horas):
    """
    Recibe el número de horas de sueño y retorna la clase predicha (0 o 1)
    junto con la probabilidad estimada de pertenecer a la clase 1 (Deserta).
    """
    pred_class = int(model.predict([[horas]])[0])
    pred_proba = model.predict_proba([[horas]])[0][1]  # Probabilidad de la clase 1
    return pred_class, round(pred_proba * 100, 2)

def generatePlot():
    """
    Genera el gráfico de dispersión de los datos reales, coloreados según su clase (0/1),
    junto con la curva sigmoide de probabilidad ajustada por el modelo.
    Convierte la imagen a un búfer de memoria codificado en Base64 para inyectarlo en el HTML.
    """
    plt.figure(figsize=(8, 4.5))

    # Separamos los puntos por clase para poder graficarlos con colores y leyenda distintos
    mask_continua = (y == 0)
    mask_deserta = (y == 1)

    plt.scatter(x[mask_continua], y[mask_continua], color='#198754', alpha=0.5, label='Continúa (0)')
    plt.scatter(x[mask_deserta], y[mask_deserta], color='#dc3545', alpha=0.5, label='Deserta (1)')

    # Generamos un vector denso de puntos dentro del rango de X para trazar la curva sigmoide continua
    x_line = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_line = model.predict_proba(x_line)[:, 1]
    plt.plot(x_line, y_line, color='#0d6efd', linewidth=2, label='Probabilidad estimada (Sigmoide)')

    # Configuración de etiquetas del gráfico
    plt.title('Student Dropout Probability vs Hours of Sleep', fontsize=12)
    plt.xlabel('Hours of Sleep per Night', fontsize=10)
    plt.ylabel('Dropout (0 = Continues, 1 = Drops out)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Guardamos el renderizado de Matplotlib en un búfer de bytes en memoria (BytesIO)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)

    # Codificamos a string Base64 para que la etiqueta <img> de HTML pueda interpretarlo directamente
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return image_base64

def getEvaluationMetrics():
    """
    Calcula la matriz de confusión y las métricas de clasificación (Accuracy, Precision,
    Recall, F1-score) sobre el conjunto de prueba (20%) que el modelo nunca vio en entrenamiento.
    """
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'accuracy': round(accuracy_score(y_test, y_pred_test) * 100, 2),
        'precision': round(precision_score(y_test, y_pred_test) * 100, 2),
        'recall': round(recall_score(y_test, y_pred_test) * 100, 2),
        'f1': round(f1_score(y_test, y_pred_test) * 100, 2),
        'n_test': len(y_test)
    }
    return metrics