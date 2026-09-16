import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def ejecutar_entrenamiento_completo(ruta_csv='dataset_appointments.csv'):
    df = pd.read_csv(ruta_csv)
    columnas_analisis = ['Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']
    
    X = df[columnas_analisis].copy()
    X['No_show_numeric'] = df['No-show'].apply(lambda x: 1 if x == 'Yes' else 0)
    X['Gender_numeric'] = df['Gender'].apply(lambda x: 1 if x == 'F' else 0)
    
    # Estandarización crucial para distancias homogéneas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Forzamos la identificación estricta de 3 Clústeres
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster_Asignado'] = kmeans.fit_predict(X_scaled)
    
    # --- NUEVA LÓGICA: TABLA DE DISTANCIAS A CENTROIDES ---
    # kmeans.transform() calcula automáticamente la distancia euclidiana de cada dato a los 3 centroides
    distancias = kmeans.transform(X_scaled)
    
    df['Distancia_Centroide_0'] = distancias[:, 0]
    df['Distancia_Centroide_1'] = distancias[:, 1]
    df['Distancia_Centroide_2'] = distancias[:, 2]
    
    # Guardamos los resultados con las nuevas columnas de distancias
    df.to_csv('dataset_kmeans_resultados.csv', index=False)
    return df

def generate_kmeans_plot():
    """Genera la gráfica de clústeres forzando el entrenamiento si el archivo es viejo."""
    try:
        df = pd.read_csv('dataset_kmeans_resultados.csv')
        if 'Distancia_Centroide_0' not in df.columns:
            raise KeyError()
    except (FileNotFoundError, KeyError):
        df = ejecutar_entrenamiento_completo()

    plt.figure(figsize=(8, 5))
    jitter_y = df['Hipertension'] + np.random.normal(0, 0.04, size=len(df))
    scatter = plt.scatter(df['Age'], jitter_y, c=df['Cluster_Asignado'], cmap='viridis', alpha=0.6, edgecolors='k')
    
    plt.title('Segmentación de Pacientes (K-Means Clustering)')
    plt.xlabel('Edad del Paciente')
    plt.ylabel('Hipertensión (Con dispersión aleatoria)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.yticks([0, 1], ['No', 'Sí'])
    plt.legend(*scatter.legend_elements(), title="Clústeres")
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def get_kmeans_summary():
    """Retorna las estadísticas descriptivas y la matriz de distancias a centroides."""
    try:
        df = pd.read_csv('dataset_kmeans_resultados.csv')
        # Si el CSV es antiguo y no tiene las distancias, forzamos un re-entrenamiento limpio
        if 'Distancia_Centroide_0' not in df.columns:
            raise KeyError()
    except (FileNotFoundError, KeyError):
        df = ejecutar_entrenamiento_completo()
        
    resumen_grupos = {}
    for i in range(3):
        cluster_data = df[df['Cluster_Asignado'] == i]
        if not cluster_data.empty:
            resumen_grupos[f'Grupo_{i}'] = {
                'total': int(len(cluster_data)),
                'edad_promedio': round(float(cluster_data['Age'].mean()), 1),
                'hipertension_pct': round(float(cluster_data['Hipertension'].mean() * 100), 1),
                'inasistencia_pct': round(float(cluster_data['No-show'].apply(lambda x: 1 if x=='Yes' else 0).mean() * 100), 1)
            }
            
    # Mapeo controlado y seguro de los primeros 15 registros para la tabla visual
    tabla_distancias = []
    for idx, row in df.head(15).iterrows():
        tabla_distancias.append({
            'id_paciente': int(row['PatientId']),
            'edad': int(row['Age']),
            'cluster_real': int(row['Cluster_Asignado']),
            'd0': round(float(row['Distancia_Centroide_0']), 3),
            'd1': round(float(row['Distancia_Centroide_1']), 3),
            'd2': round(float(row['Distancia_Centroide_2']), 3),
        })
            
    return {
        'total_records': len(df),
        'grupos': resumen_grupos,
        'tabla_distancias': tabla_distancias
    }
