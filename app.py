from flask import Flask, render_template, request
import LinearRegressionPhone  # Importación del módulo local que contiene la lógica del modelo y dataset
import LogisticRegressionSleep  # Importación del módulo local de Regresión Logística
import KnnRegression # Importación del módulo local de KNN
modelo_knn, escalador_knn = KnnRegression.entrenar_modelo_knn()


# Instanciación de la aplicación web en Flask
app = Flask(__name__)

# =========================================================
# RUTAS PARA EL MENÚ DE NAVEGACIÓN Y PÁGINAS ESTÁTICAS
# =========================================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/ml/concepts')
def ml_concepts():
    return render_template('ml_concepts.html')

@app.route('/ml/types')
def ml_types():
    return render_template('ml_types.html')

@app.route('/ml/use-case-1')
def use_case_1():
    return render_template('use_case_1.html')

@app.route('/ml/use-case-2')
def use_case_2():
    return render_template('use_case_2.html')

@app.route('/ml/use-case-3')
def use_case_3():
    return render_template('use_case_3.html')

@app.route('/ml/use-case-4')
def use_case_4():
    return render_template('use_case_4.html')

@app.route('/regression/concepts')
def regression_concepts():
    return render_template('regression_concepts.html')

# =========================================================
# RUTA CONTROLADORA PARA LA APLICACIÓN PRÁCTICA DE REGRESIÓN
# =========================================================

# Habilitamos métodos GET (carga inicial) y POST (procesamiento de datos del formulario)
@app.route('/LinearRegression/', methods=['GET', 'POST'])
def LRegressionPhone():
    calculatePriceResult = None
    antiguedad_ingresada = None
    error = None
    
    # Verificamos si la petición proviene del envío del formulario mediante el método POST
    if request.method == 'POST':
        try:
            # Extracción del parámetro enviado por el usuario y casteo a número de punto flotante
            val = float(request.form['antiguedad'])
            
            # Validamos que el rango de entrada sea coherente con el dominio del problema
            if val < 0 or val > 120:
                error = "Please enter a valid age between 0 and 120 months."
            else:
                antiguedad_ingresada = val
                # Invocamos la función de inferencia del modelo entrenado
                raw_price = LinearRegressionPhone.calculatePrice(val)
                # Formateamos el resultado agregando puntos como separadores de miles
                calculatePriceResult = f"{raw_price:,.0f}".replace(",", ".")
        except (ValueError, KeyError):
            error = "Please enter a valid numeric value."
        
    # Invocamos la generación de la imagen Base64 de la gráfica
    plot_url = LinearRegressionPhone.generatePlot()
    
    # Inyectamos variables dinámicas en la plantilla Jinja2 para su renderizado en el cliente
    return render_template(
        'regression_application.html', 
        result=calculatePriceResult,
        antiguedad=antiguedad_ingresada,
        error=error,
        plot_url=plot_url,
        num_records=len(LinearRegressionPhone.df)
    )


@app.route('/logistic/concepts')
def logistic_concepts():
    return render_template('logistic_concepts.html')

@app.route('/logistic/', methods=['GET', 'POST'])
def logisticApplication():
    predicted_class = None
    predicted_label = None
    predicted_proba = None
    horas_ingresadas = None
    error = None

    if request.method == 'POST':
        try:
            val = float(request.form['horas_sueno'])
            if val < 0 or val > 24:
                error = "Please enter a valid number of hours between 0 and 24."
            else:
                horas_ingresadas = val
                predicted_class, predicted_proba = LogisticRegressionSleep.predictDesercion(val)
                predicted_label = "Drops out" if predicted_class == 1 else "Continues"
        except (ValueError, KeyError):
            error = "Please enter a valid numeric value."

    plot_url = LogisticRegressionSleep.generatePlot()

    return render_template(
        'logistic_application.html',
        result_class=predicted_class,
        result_label=predicted_label,
        result_proba=predicted_proba,
        horas=horas_ingresadas,
        error=error,
        plot_url=plot_url,
        num_records=len(LogisticRegressionSleep.df)
    )

@app.route('/logistic/metrics')
def logistic_metrics():
    metrics = LogisticRegressionSleep.getEvaluationMetrics()
    return render_template('logistic_metrics.html', m=metrics)

# Punto de entrada para la ejecución del servidor de desarrollo local
if __name__ == '__main__':
    app.run(debug=True)


# =========================================================
# KNN
# =========================================================
@app.route('/predict_phone', methods=['POST']) # Ajusta el nombre de tu ruta si es diferente
def predict_phone():
    ram = float(request.form['ram']) # Capturar los datos numéricos que ingresó el usuario en el formulario web
    almacenamiento = float(request.form['almacenamiento'])
    bateria = float(request.form['bateria'])
    
    caracteristicas_usuario = [[ram, almacenamiento, bateria]] 

    datos_escalados = escalador_knn.transform(caracteristicas_usuario) # Escalar los datos con el escalador entrenado de KNN

    precio_predicho = modelo_knn.predict(datos_escalados)[0]  # Realiza la predicción del precio con KNN

    return render_template('regression_application.html', prediction=precio_predicho) # 4. Envia el resultado de vuelta a la plantilla HTML


