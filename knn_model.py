# Importación de librerías para el algoritmo K-Nearest Neighbors y normalización
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

# ==============================================================================
# CARGA Y PREPROCESAMIENTO DE DATOS PARA K-NEAREST NEIGHBORS (KNN)
# ==============================================================================
df_knn = pd.read_csv('dataset_knn.csv')

# Definición de 3 variables independientes (X) y la variable objetivo (y)
feature_cols = ['Experiencia_Anios', 'Puntaje_Tecnico', 'Participacion_Proyectos']
X_knn = df_knn[feature_cols]
y_knn = df_knn['Promovido']

# División de datos: 80% Entrenamiento / 20% Prueba
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn, y_knn, test_size=0.20, random_state=42
)

# Escalamiento estándar de características (StandardScaler indispensable para KNN por distancia euclidiana)
scaler_knn = StandardScaler()
X_train_scaled = scaler_knn.fit_transform(X_train_knn)
X_test_scaled = scaler_knn.transform(X_test_knn)

# Entrenamiento del clasificador K-Nearest Neighbors con K=5
k_neighbors = 5
model_knn = KNeighborsClassifier(n_neighbors=k_neighbors)
model_knn.fit(X_train_scaled, y_train_knn)


def get_knn_summary():
    """Retorna un resumen del conjunto de datos y la configuración del modelo KNN."""
    return {
        'total_records': len(df_knn),
        'train_records': len(X_train_knn),
        'test_records': len(X_test_knn),
        'k_value': k_neighbors,
        'features': ['Años de Experiencia', 'Puntaje en Test Técnico', 'Participación en Proyectos'],
        'target_name': 'Promoción Laboral',
        'class_0': 'No Promovido (0)',
        'class_1': 'Promovido (1)'
    }


def predict_knn(experiencia, puntaje, participacion):
    """Efectúa la clasificación de un nuevo empleado enviando los datos escalados al modelo KNN."""
    raw_input = np.array([[float(experiencia), float(puntaje), float(participacion)]])
    scaled_input = scaler_knn.transform(raw_input)
    
    prediction = int(model_knn.predict(scaled_input)[0])
    probabilities = model_knn.predict_proba(scaled_input)[0]
    
    return {
        'prediction': prediction,
        'prob_class_0': round(float(probabilities[0]) * 100, 2),
        'prob_class_1': round(float(probabilities[1]) * 100, 2)
    }


def generate_knn_plot():
    """Genera una representación gráfica de dispersión 2D de las variables más representativas."""
    plt.figure(figsize=(8, 4.5))
    
    # Graficar Experiencia vs Puntaje Técnico coloreado por la clase de Promoción
    scatter = plt.scatter(
        df_knn['Experiencia_Anios'], 
        df_knn['Puntaje_Tecnico'], 
        c=df_knn['Promovido'], 
        cmap='viridis', 
        alpha=0.7, 
        edgecolors='k',
        s=df_knn['Participacion_Proyectos'] * 15
    )
    
    plt.title('Modelo KNN: Agrupación por Experiencia y Puntaje Técnico', fontsize=12, fontweight='bold')
    plt.xlabel('Años de Experiencia Laboral', fontsize=10)
    plt.ylabel('Puntaje en Test Técnico (0-100)', fontsize=10)
    
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.ax.set_yticklabels(['No Promovido (0)', 'Promovido (1)'])
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64


def get_knn_metrics():
    """Calcula las métricas de desempeño sobre el 20% del dataset de prueba."""
    y_pred_test = model_knn.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test_knn, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test_knn, y_pred_test)
    prec = precision_score(y_test_knn, y_pred_test, zero_division=0)
    
    return {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'accuracy': round(float(acc) * 100, 2),
        'precision': round(float(prec) * 100, 2)
    }