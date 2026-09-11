# ==============================================================================
# MÓDULO PRINCIPAL: app.py
# DESCRIPCIÓN: Servidor web Flask que orquesta el enrutamiento de las vistas,
#              la recepción de datos de formularios mediante metodos seguros (.get)
#              y el llamado a los modelos de Regresión Lineal, Logística y KNN.
# ==============================================================================

from flask import Flask, render_template, request

# Importación de los módulos locales de Machine Learning
from LinearRegressionPhone import predict_price, generate_plot, get_dataset_summary
import logistic_model
import knn_model

# Inicialización de la aplicación Flask
app = Flask(__name__)

# ==============================================================================
# RUTAS DE ACTIVIDAD 1
# ==============================================================================
@app.route('/')
def home():
    """Ruta principal / Inicio de la aplicación."""
    return render_template('home.html')

@app.route('/ml/concepts')
def ml_concepts():
    """Vista teórica de conceptos fundamentales de ML."""
    return render_template('ml_concepts.html')

@app.route('/ml/types')
def ml_types():
    """Vista explicativa de tipos de Machine Learning."""
    return render_template('ml_types.html')

@app.route('/ml/use-case-1')
def use_case_1():
    """Caso de uso 1: Sistema Anti-Cheat."""
    return render_template('use_case_1.html')

@app.route('/ml/use-case-2')
def use_case_2():
    """Caso de uso 2: Mantenimiento Predictivo."""
    return render_template('use_case_2.html')

@app.route('/ml/use-case-3')
def use_case_3():
    """Caso de uso 3: Selección de Personal."""
    return render_template('use_case_3.html')

@app.route('/ml/use-case-4')
def use_case_4():
    """Caso de uso 4: Clustering / Diagnóstico."""
    return render_template('use_case_4.html')

@app.route('/regression/concepts')
def regression_concepts():
    """Vista teórica de Regresión Lineal."""
    return render_template('regression_concepts.html')

@app.route('/LinearRegression/', methods=['GET', 'POST'])
def linear_regression_app():
    """
    Controlador de la vista práctica de Regresión Lineal.
    Mantiene compatibilidad exacta con la plantilla original utilizando las
    variables 'antiguedad', 'result', 'error' y 'num_records'.
    """
    result = None
    antiguedad = None
    error = None
    
    if request.method == 'POST':
        # Captura el campo 'antiguedad' enviado por el formulario HTML de tu plantilla
        val = request.form.get('antiguedad')
        if val:
            try:
                antiguedad = float(val)
                # Ejecuta la función de predicción en el modelo de Regresión Lineal
                result = predict_price(antiguedad)
            except (ValueError, TypeError):
                error = "Por favor ingrese un valor numérico válido."

    # Generación del gráfico dinámico y obtención de metadatos del CSV
    plot_url = generate_plot()
    summary = get_dataset_summary()
    num_records = summary.get('total_records', 0)
    
    # Retorno de la plantilla original con todos sus argumentos requeridos
    return render_template('regression_application.html', 
                           result=result, 
                           antiguedad=antiguedad, 
                           error=error,
                           plot_url=plot_url, 
                           num_records=num_records)

# ==============================================================================
# RUTAS DE ACTIVIDAD 2 — REGRESIÓN LOGÍSTICA
# ==============================================================================
@app.route('/logistic/concepts')
def logistic_concepts():
    """Vista teórica de Regresión Logística Binaria."""
    return render_template('logistic_concepts.html')

@app.route('/logistic/application', methods=['GET', 'POST'])
def logistic_application():
    """
    Aplicación interactiva de Regresión Logística.
    Captura de forma segura el tiempo de navegación y retorna la clasificación binaria.
    """
    prediction_result = None
    time_input = None
    
    if request.method == 'POST':
        val = request.form.get('tiempo_navegacion')
        if val:
            try:
                time_input = float(val)
                prediction_result = logistic_model.predict_logistic(time_input)
            except (ValueError, TypeError):
                prediction_result = None

    plot_url = logistic_model.generate_logistic_plot()
    summary = logistic_model.get_logistic_summary()
    
    return render_template('logistic_application.html',
                           prediction=prediction_result,
                           time_input=time_input,
                           plot_url=plot_url,
                           summary=summary)

@app.route('/logistic/metrics')
def logistic_metrics():
    """Vista de métricas de evaluación calculadas sobre el 20% del dataset de test."""
    metrics = logistic_model.get_logistic_metrics()
    return render_template('logistic_metrics.html', metrics=metrics)

# ==============================================================================
# RUTAS DE ACTIVIDAD 2 — K-NEAREST NEIGHBORS (KNN)
# ==============================================================================
@app.route('/knn/concepts')
def knn_concepts():
    """Vista teórica del algoritmo K-Nearest Neighbors."""
    return render_template('knn_concepts.html')

@app.route('/knn/application', methods=['GET', 'POST'])
def knn_application():
    """
    Aplicación interactiva multivariable de KNN (3 variables de entrada).
    """
    prediction_result = None
    inputs = {}
    
    if request.method == 'POST':
        exp_val = request.form.get('experiencia')
        pun_val = request.form.get('puntaje')
        par_val = request.form.get('participacion')
        
        if exp_val and pun_val and par_val:
            try:
                inputs['experiencia'] = float(exp_val)
                inputs['puntaje'] = float(pun_val)
                inputs['participacion'] = float(par_val)
                
                prediction_result = knn_model.predict_knn(
                    inputs['experiencia'], inputs['puntaje'], inputs['participacion']
                )
            except (ValueError, TypeError):
                prediction_result = None

    plot_url = knn_model.generate_knn_plot()
    summary = knn_model.get_knn_summary()
    
    return render_template('knn_application.html',
                           prediction=prediction_result,
                           inputs=inputs,
                           plot_url=plot_url,
                           summary=summary)

@app.route('/knn/metrics')
def knn_metrics():
    """Vista de métricas de evaluación calculadas sobre el 20% del dataset de test para KNN."""
    metrics = knn_model.get_knn_metrics()
    return render_template('knn_metrics.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)