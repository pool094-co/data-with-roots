# ==============================================================================
# MÓDULO: LinearRegressionPhone.py
# DESCRIPCIÓN: Módulo de backend que entrena el modelo de Regresión Lineal Simple
#              para estimar precios de celulares usados en función de su antigüedad.
#              Incluye resolución dinámica de columnas para prevenir KeyError.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Desactivación de interfaz gráfica interactiva para uso en servidor web
import matplotlib.pyplot as plt
import io
import base64

from sklearn.linear_model import LinearRegression

# ==============================================================================
# CARGA DEL DATASET Y DETECCIÓN AUTOMÁTICA DE COLUMNAS
# ==============================================================================
# Carga del archivo CSV original de la Actividad 1 mediante Pandas
df_phone = pd.read_csv('dataset.csv')

# Identificación dinámica de la columna X (Antigüedad / Meses)
cols = list(df_phone.columns)
col_x = next((c for c in cols if 'age' in c.lower() or 'mes' in c.lower() or 'antig' in c.lower()), cols[0])

# Identificación dinámica de la columna Y (Precio)
col_y = next((c for c in cols if 'price' in c.lower() or 'prec' in c.lower() or 'val' in c.lower()), cols[1] if len(cols) > 1 else cols[0])

# Extracción de variables de entrada (X) y objetivo (y) usando los nombres detectados
X_phone = df_phone[[col_x]]
y_phone = df_phone[col_y]

# Entrenamiento del algoritmo de Regresión Lineal mediante mínimos cuadrados
model_phone = LinearRegression()
model_phone.fit(X_phone, y_phone)


def predict_price(age_months):
    """
    Recibe la antigüedad en meses (float/int) y retorna la predicción 
    del precio estimado en COP utilizando el modelo entrenado.
    """
    input_data = np.array([[float(age_months)]])
    predicted_val = model_phone.predict(input_data)[0]
    return round(float(predicted_val), 2)


def generate_plot():
    """
    Genera el gráfico de dispersión con la recta de tendencia ajustada 
    y lo retorna codificado en formato Base64 para inyección directa en HTML.
    """
    plt.figure(figsize=(8, 4.5))
    
    # Renderizado de los puntos observados del dataset
    plt.scatter(
        df_phone[col_x], 
        df_phone[col_y], 
        color='#198754', 
        alpha=0.5, 
        edgecolors='k', 
        label='Teléfonos del Dataset'
    )
    
    # Generación de la línea de regresión continua
    x_vals = np.linspace(df_phone[col_x].min(), df_phone[col_x].max(), 100).reshape(-1, 1)
    y_vals = model_phone.predict(x_vals)
    plt.plot(x_vals, y_vals, color='#dc3545', linewidth=2.5, label='Recta de Regresión Ajustada')
    
    plt.title('Regresión Lineal: Antigüedad vs Precio de Celulares Usados', fontsize=12, fontweight='bold')
    plt.xlabel(f'Antigüedad del Celular ({col_x})', fontsize=10)
    plt.ylabel(f'Precio Comercial Estimado ({col_y})', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    # Conversión del gráfico a buffer de memoria en string Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64


def get_dataset_summary():
    """
    Retorna un diccionario con metadatos descriptivos del dataset 
    de teléfonos móviles para alimentar las tarjetas resumidas en Jinja2.
    """
    return {
        'total_records': len(df_phone),
        'min_age': int(df_phone[col_x].min()),
        'max_age': int(df_phone[col_x].max()),
        'min_price': float(df_phone[col_y].min()),
        'max_price': float(df_phone[col_y].max())
    }