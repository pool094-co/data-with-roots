import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica para evitar errores de renderizado en servidores web como Render
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

# =========================================================
# 1. GENERACIÓN Y PREPROCESAMIENTO DEL DATASET
# =========================================================

# Fijamos la semilla aleatoria para garantizar la reproducibilidad de las muestras en cada ejecución
np.random.seed(42)

# Generamos 500 registros para la variable independiente X (antigüedad en meses de 1 a 60)
antiguedad_meses = np.random.randint(1, 61, size=500)

# Modelamos la variable dependiente Y (precio en COP) mediante una relación lineal decreciente
# con un componente de ruido blanco Gaussiano para simular la dispersión del mercado real
precio_cop = 3800000 - (45000 * antiguedad_meses) + np.random.normal(0, 150000, size=500)
precio_cop = np.clip(precio_cop, 200000, 5000000)  # Acotamos los valores para evitar precios negativos o ilógicos

# Estructuramos el conjunto de datos en un DataFrame de Pandas y lo exportamos a CSV para persistencia
df = pd.DataFrame({
    'antiguedad_meses': antiguedad_meses,
    'precio_cop': np.round(precio_cop, 0)
})
df.to_csv('dataset.csv', index=False)

# =========================================================
# 2. ENTRENAMIENTO DEL MODELO DE MACHINE LEARNING
# =========================================================

# Extraemos la matriz de características (X) y el vector objetivo (y)
# Scikit-Learn requiere que X sea una matriz 2D con forma [n_muestras, n_características]
x = df[["antiguedad_meses"]].values
y = df["precio_cop"].values

# Instanciamos el algoritmo de Regresión Lineal y ajustamos la recta mediante Mínimos Cuadrados Ordinarios (OLS)
model = LinearRegression()
model.fit(x, y)

# =========================================================
# 3. FUNCIONES DE PREDICCIÓN Y VISUALIZACIÓN
# =========================================================

def calculatePrice(antiguedad):
    """
    Recibe el valor de la variable independiente y retorna la estimación del modelo.
    Se pasa como una estructura bidimensional [[antiguedad]] según lo exige la librería.
    """
    prediction = model.predict([[antiguedad]])[0]
    return round(max(prediction, 0), 0)  # Garantizamos que la estimación no entregue valores inferiores a cero

def generatePlot():
    """
    Genera el gráfico de dispersión y la línea de regresión ajustada.
    Convierte la imagen a un búfer de memoria codificado en Base64 para inyectarlo en el HTML sin guardar archivos en disco.
    """
    plt.figure(figsize=(8, 4.5))
    
    # Graficamos el diagrama de dispersión dividiendo entre 1M para simplificar la escala visual del eje Y
    plt.scatter(x, y / 1e6, color='#0d6efd', alpha=0.4, label='500 Phones (Real Data)')
    
    # Generamos un vector denso de puntos dentro del rango de X para trazar la recta predictiva continua
    x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line / 1e6, color='#dc3545', linewidth=2, label='Regression Line')
    
    # Configuración de etiquetas del gráfico
    plt.title('Depreciation: Age vs Price of Used Mobile Phones', fontsize=12)
    plt.xlabel('Age (Months)', fontsize=10)
    plt.ylabel('Estimated Price (Millions COP)', fontsize=10)
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