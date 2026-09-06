from flask import Flask, render_template, request
import LinearRegressionPhone  # Importación del módulo local que contiene la lógica del modelo y dataset

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

# Punto de entrada para la ejecución del servidor de desarrollo local
if __name__ == '__main__':
    app.run(debug=True)