import logging
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for
from sklearn import __version__ as sklearn_version


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "crop-yield-dev-secret")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RUTA_MODELO = BASE_DIR / "pipeline_crop_yield_final.joblib"
VERSION_SKLEARN_MODELO = "1.7.1"

COLUMNAS_MODELO = [
    "Fertilizer_Used_kg",
    "Rainfall_mm",
    "Temperature_C",
    "Previous_Crop",
    "Humidity_pct",
    "Region",
]

CATEGORIAS_VALIDAS = {
    "Region": {"Region_A", "Region_B", "Region_C", "Region_D"},
    "Previous_Crop": {
        "Rice",
        "Barley",
        "Wheat",
        "Maize",
        "Sin_cultivo_previo",
    },
}

ETIQUETAS = {
    "Region": "región",
    "Rainfall_mm": "lluvia",
    "Temperature_C": "temperatura",
    "Humidity_pct": "humedad",
    "Fertilizer_Used_kg": "fertilizante usado",
    "Previous_Crop": "cultivo anterior",
}


def cargar_modelo():
    """Carga una sola vez el pipeline entrenado; nunca entrena ni lo modifica."""
    if not RUTA_MODELO.is_file():
        return None, (
            f"No se encontró el pipeline requerido: {RUTA_MODELO.name}. "
            "Colóquelo en la raíz del proyecto."
        )

    if sklearn_version != VERSION_SKLEARN_MODELO:
        return None, (
            "Versión de scikit-learn incompatible con el pipeline guardado. "
            f"Se requiere {VERSION_SKLEARN_MODELO} y está instalada "
            f"{sklearn_version}. Ejecute: pip install -r requirements.txt"
        )

    try:
        pipeline = joblib.load(RUTA_MODELO)
    except Exception as exc:  # El servidor debe iniciar para poder informar el error.
        logger.exception("No fue posible cargar el pipeline")
        return None, f"No fue posible cargar el pipeline guardado: {exc}"

    if not callable(getattr(pipeline, "predict", None)):
        return None, "El archivo cargado no contiene un pipeline con método predict."

    return pipeline, None


modelo, error_carga_modelo = cargar_modelo()


def leer_categoria(nombre):
    valor = request.form.get(nombre, "").strip()
    if not valor:
        return float("nan")
    if valor not in CATEGORIAS_VALIDAS[nombre]:
        raise ValueError(
            f"El valor de {ETIQUETAS[nombre]} no pertenece a las categorías admitidas."
        )
    return valor


def leer_numero(nombre):
    texto = request.form.get(nombre, "").strip()
    if not texto:
        return float("nan")
    try:
        valor = float(texto)
    except ValueError as exc:
        raise ValueError(f"{ETIQUETAS[nombre].capitalize()} debe ser numérico.") from exc
    if not math.isfinite(valor):
        raise ValueError(f"{ETIQUETAS[nombre].capitalize()} debe ser un número finito.")
    return valor


def validar_rangos(datos):
    if not pd.isna(datos["Rainfall_mm"]) and datos["Rainfall_mm"] < 0:
        raise ValueError("La lluvia no puede ser negativa.")
    if (
        not pd.isna(datos["Temperature_C"])
        and not -50 <= datos["Temperature_C"] <= 60
    ):
        raise ValueError("La temperatura debe estar entre -50 y 60 °C.")
    if (
        not pd.isna(datos["Humidity_pct"])
        and not 0 <= datos["Humidity_pct"] <= 100
    ):
        raise ValueError("La humedad debe estar entre 0 y 100 %.")
    if (
        not pd.isna(datos["Fertilizer_Used_kg"])
        and datos["Fertilizer_Used_kg"] < 0
    ):
        raise ValueError("El fertilizante no puede ser negativo.")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        resultado = session.pop("resultado", None)
        error_sesion = session.pop("error", None)
        error = error_sesion if error_sesion is not None else error_carga_modelo

        return render_template(
            "index.html",
            resultado=resultado,
            error=error,
            valores={},
        )

    if modelo is None:
        session["error"] = error_carga_modelo
        return redirect(url_for("index"))

    try:
        datos = {
            "Fertilizer_Used_kg": leer_numero("Fertilizer_Used_kg"),
            "Rainfall_mm": leer_numero("Rainfall_mm"),
            "Temperature_C": leer_numero("Temperature_C"),
            "Previous_Crop": leer_categoria("Previous_Crop"),
            "Humidity_pct": leer_numero("Humidity_pct"),
            "Region": leer_categoria("Region"),
        }
        validar_rangos(datos)

        # El orden coincide con las seis características seleccionadas.
        datos_usuario = pd.DataFrame(
            [[datos[columna] for columna in COLUMNAS_MODELO]],
            columns=COLUMNAS_MODELO,
        )

        prediccion = float(modelo.predict(datos_usuario)[0])
        if not math.isfinite(prediccion):
            raise ValueError("El pipeline produjo una predicción no válida.")
        session["resultado"] = round(prediccion, 2)
    except ValueError as exc:
        session["error"] = str(exc)
    except AttributeError as exc:
        if "_fill_dtype" in str(exc):
            session["error"] = (
                "El pipeline fue cargado con una versión incompatible de "
                "scikit-learn. Este archivo requiere scikit-learn==1.7.1."
            )
        else:
            logger.exception("Error de atributos durante la predicción")
            session["error"] = f"No se pudo generar la predicción: {exc}"
    except Exception as exc:
        logger.exception("No se pudo generar la predicción")
        session["error"] = f"No se pudo generar la predicción: {exc}"

    # Redirige después de predecir para recargar la página, limpiar los inputs
    # y evitar reenviar el formulario al actualizar el navegador.
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
