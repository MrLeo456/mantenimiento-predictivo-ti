# -*- coding: utf-8 -*-
"""
================================================================================
APP DE DEMOSTRACIÓN — Predicción de fallas en infraestructura TI
Prototipo (Anexo 01) — Inteligencia Artificial Aplicada 2026-1
================================================================================

Interfaz web que carga el modelo entrenado (modelo_final.pkl) y predice, a
partir de los valores operativos de un equipo, si ocurrirá una falla.

--------------------------------------------------------------------------------
INSTRUCCIONES DE EJECUCIÓN
--------------------------------------------------------------------------------
1. Instalar dependencias:
       pip install streamlit scikit-learn pandas numpy joblib

2. Ejecutar PRIMERO el notebook 'pipeline_mantenimiento.py' para generar
   los archivos: modelo_final.pkl, scaler.pkl y columnas.pkl

3. Lanzar la app:
       streamlit run app_demo.py

4. Se abrirá en el navegador (http://localhost:8501). Ajustar los controles
   y pulsar "Predecir".
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------
st.set_page_config(page_title="Predicción de fallas TI", page_icon="⚙️",
                   layout="centered")

st.title("⚙️ Predicción de fallas en infraestructura TI")
st.markdown(
    "Prototipo basado en un modelo de **clasificación supervisada** entrenado "
    "sobre el *AI4I 2020 Predictive Maintenance Dataset*. Ingrese los valores "
    "operativos del equipo y obtenga la predicción de falla.")

# ---------------------------------------------------------------
# Carga del modelo entrenado (con manejo de errores)
# ---------------------------------------------------------------
@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load('modelo_final.pkl')
    scaler = joblib.load('scaler.pkl')
    columnas = joblib.load('columnas.pkl')
    return modelo, scaler, columnas

try:
    modelo, scaler, columnas = cargar_artefactos()
except FileNotFoundError:
    st.error("No se encontró el modelo. Ejecute primero 'pipeline_mantenimiento.py' "
             "para generar modelo_final.pkl, scaler.pkl y columnas.pkl.")
    st.stop()

# ---------------------------------------------------------------
# Panel de entrada de datos (sidebar)
# ---------------------------------------------------------------
st.sidebar.header("Parámetros del equipo")

tipo = st.sidebar.selectbox("Tipo de producto (calidad)", ['L', 'M', 'H'],
                            help="L = baja, M = media, H = alta")
air_temp = st.sidebar.slider("Temperatura del aire [K]", 295.0, 305.0, 300.0, 0.1)
proc_temp = st.sidebar.slider("Temperatura de proceso [K]", 305.0, 314.0, 310.0, 0.1)
rot_speed = st.sidebar.slider("Velocidad de rotación [rpm]", 1168, 2886, 1500, 1)
torque = st.sidebar.slider("Torque [Nm]", 3.5, 76.6, 40.0, 0.1)
tool_wear = st.sidebar.slider("Desgaste de herramienta [min]", 0, 253, 100, 1)

# ---------------------------------------------------------------
# Construcción del vector de entrada (respetando el orden de columnas)
# ---------------------------------------------------------------
tipo_num = {'L': 0, 'M': 1, 'H': 2}[tipo]
entrada = pd.DataFrame([{
    'Type': tipo_num,
    'Air temperature [K]': air_temp,
    'Process temperature [K]': proc_temp,
    'Rotational speed [rpm]': rot_speed,
    'Torque [Nm]': torque,
    'Tool wear [min]': tool_wear,
}])
# Reordenar según las columnas con que se entrenó el modelo
entrada = entrada[[c for c in columnas if c in entrada.columns]]

# ---------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------
if st.button("🔍 Predecir"):
    entrada_s = scaler.transform(entrada)
    pred = modelo.predict(entrada_s)[0]
    proba = modelo.predict_proba(entrada_s)[0][1]

    st.subheader("Resultado")
    if pred == 1:
        st.error(f"⚠️ FALLA probable — probabilidad estimada: {proba:.1%}")
    else:
        st.success(f"✅ Sin falla — probabilidad de falla: {proba:.1%}")

    st.progress(float(proba))
    st.caption("La probabilidad corresponde a la clase 'falla' según el modelo. "
               "En contextos de mantenimiento se prioriza la sensibilidad (recall) "
               "para no dejar pasar fallas reales.")

# ---------------------------------------------------------------
# Pie de página
# ---------------------------------------------------------------
st.markdown("---")
st.caption("Proyecto académico — Universidad de Lima — IA Aplicada 2026-1. "
           "Modelo entrenado con datos públicos (AI4I 2020, UCI).")
