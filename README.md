# Predictor de rendimiento agrícola

Aplicación Flask que carga `pipeline_crop_yield_final.joblib` y predice
`Yield_ton_per_ha` a partir de las seis características seleccionadas por el
modelo final: fertilizante, lluvia, temperatura, cultivo anterior, humedad y
región. Flask no entrena ni reconstruye el pipeline.

## Compatibilidad del modelo

El artefacto guardado declara internamente `scikit-learn 1.7.1`; por eso
`requirements.txt` fija exactamente `scikit-learn==1.7.1`. El entorno original de
la libreta usa Python 3.10.20, joblib 1.5.3, pandas 2.3.3, NumPy 2.2.5 y SciPy
1.15.3. Usar otra versión de scikit-learn puede producir errores como
`'SimpleImputer' object has no attribute '_fill_dtype'`.

La libreta convierte el valor `None` de `Previous_Crop` en la categoría explícita
`Sin_cultivo_previo`. Los campos también pueden dejarse vacíos porque el pipeline
incluye imputación antes de `OrdinalEncoder` y `StandardScaler`.

## Ejecución local

Se recomienda Python 3.10.20 y un entorno virtual:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Abra `http://127.0.0.1:5000`. El archivo
`pipeline_crop_yield_final.joblib` debe permanecer en la raíz del proyecto.

## Despliegue en Render

1. Publique el proyecto en un repositorio Git.
2. En Render, cree un **Web Service** conectado al repositorio.
3. Use la configuración incluida en `render.yaml`, o configure manualmente:

   - Runtime: `Python`
   - Python Version: `3.10.20`
   - Build Command: `python -m pip install --upgrade pip && pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`

4. Verifique que `pipeline_crop_yield_final.joblib` esté incluido en el
   repositorio y despliegue.

El archivo `requirements.txt` usa `--only-binary=:all:` para evitar que Render
intente compilar paquetes científicos como `scipy` o `scikit-learn` desde fuente.
Si Render ignora la versión de Python y usa una versión más nueva sin wheels
compatibles, la instalación fallará temprano en lugar de intentar compilar con
Fortran.

Si se necesita regenerar el artefacto, debe hacerse ejecutando las celdas de
entrenamiento y `joblib.dump` en `crop_yield_new.ipynb` con el entorno compatible.
Nunca se regenera desde `app.py`.
