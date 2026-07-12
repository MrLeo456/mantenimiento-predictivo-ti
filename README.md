# Predicción de fallas en infraestructura TI mediante aprendizaje supervisado

Proyecto final del curso **Inteligencia Artificial Aplicada** (2026-1) — Universidad de Lima, Facultad de Ingeniería.
Asociado al **ODS 9: Industria, innovación e infraestructura**.

🌐 **Página del proyecto:** https://mrleo456.github.io/mantenimiento-predictivo-ti/

## Descripción

Sistema de mantenimiento predictivo que anticipa fallas en infraestructura TI mediante
clasificación supervisada binaria (falla / no falla). Se entrenaron y compararon cuatro
modelos —Random Forest, SVM, Naïve Bayes y Red Neuronal— sobre el AI4I 2020 Predictive
Maintenance Dataset (UCI), implementando el flujo en Orange Data Mining con una réplica
de validación en Python (scikit-learn).

**Resultado principal:** Random Forest obtuvo el mejor desempeño global (AUC 0,982;
F1 0,660; MCC 0,681), aunque su recall de 51,5 % evidencia que la exactitud es una
métrica engañosa bajo desbalance de clases (3,4 % de fallas).

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `ai4i2020.csv` | Dataset AI4I 2020 (UCI, DOI: 10.24432/C5HS5C, licencia CC BY 4.0) |
| `mantenimiento_predictivo.ows` | Flujo completo de Orange Data Mining |
| `pipeline_mantenimiento.py` | Réplica del experimento en Python (scikit-learn) |
| `app_demo.py` | Demo interactiva de predicción (Streamlit) |
| `index.html` | Página web del proyecto (GitHub Pages) |
| `requirements.txt` | Dependencias de Python |

## Instrucciones de ejecución

### Requisitos previos
- Python 3.10 o superior
- Orange Data Mining 3.36+ (solo para abrir el archivo `.ows`)

### 1. Clonar el repositorio e instalar dependencias

```bash
git clone https://github.com/MrLeo456/mantenimiento-predictivo-ti.git
cd mantenimiento-predictivo-ti
pip install -r requirements.txt
```

### 2. Ejecutar el pipeline en Python

```bash
python pipeline_mantenimiento.py
```

Entrena los cuatro modelos con partición 80/20 estratificada y muestra las métricas
(AUC, F1, precisión, recall, MCC) y la matriz de confusión de cada uno.

### 3. Ejecutar la demo Streamlit

```bash
streamlit run app_demo.py
```

Se abre en `http://localhost:8501`. Permite ingresar los parámetros operativos del
equipo y obtener la predicción de falla del modelo.

### 4. Abrir el flujo de Orange

Abrir Orange Data Mining → File → Open → `mantenimiento_predictivo.ows`.
En el widget **File**, apuntar al `ai4i2020.csv` local. Las banderas de subtipo de
falla (TWF, HDF, PWF, OSF, RNF) están configuradas como *skip* para evitar fuga de
datos, y la evaluación usa el modo *Test on test data*.

## Autores

Matías Sebastián Calvo Muñoz · Leonardo Fernandez Ocaña · Mariano Ganoza Faroun ·
Paolo Gabriel Medina Ruiz · Diego Alejandro Vértiz Manrique

Profesor: Javier More Sánchez — Lima, Perú, 2026
