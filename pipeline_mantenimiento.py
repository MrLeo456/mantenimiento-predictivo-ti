# -*- coding: utf-8 -*-
"""
================================================================================
PREDICCIÓN DE FALLAS EN INFRAESTRUCTURA TI MEDIANTE APRENDIZAJE SUPERVISADO
Notebook de documentación (Anexo 01) — Inteligencia Artificial Aplicada 2026-1
================================================================================

Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI, DOI: 10.24432/C5HS5C)
Objetivo: clasificación binaria (Machine failure: 0 = no falla, 1 = falla)
Modelos comparados: Random Forest, SVM, Naïve Bayes, Red Neuronal (MLP)

--------------------------------------------------------------------------------
INSTRUCCIONES DE EJECUCIÓN
--------------------------------------------------------------------------------
1. Instalar dependencias (una sola vez):
       pip install pandas numpy scikit-learn matplotlib seaborn

2. Descargar el dataset real desde UCI:
       https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
   Descomprimir y colocar 'ai4i2020.csv' en la misma carpeta que este archivo.

3. Ejecutar:  python pipeline_mantenimiento.py
   (o pegar cada bloque en celdas de un Jupyter Notebook)
================================================================================
"""

# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)

RANDOM_STATE = 42  # semilla fija -> resultados reproducibles

# ============================================================
# 2. CARGA DEL DATASET
# ============================================================
# NOTA: usar el CSV real de UCI. El nombre del archivo suele ser 'ai4i2020.csv'.
df = pd.read_csv('ai4i2020.csv')
print("Dimensiones del dataset:", df.shape)
print("\nPrimeras filas:")
print(df.head())
print("\nInformación de columnas:")
print(df.info())

# ============================================================
# 3. ANÁLISIS EXPLORATORIO (EDA)
# ============================================================
# 3.1 Balance de la variable objetivo
print("\nDistribución de la clase objetivo (Machine failure):")
print(df['Machine failure'].value_counts())
tasa_falla = df['Machine failure'].mean() * 100
print(f"Tasa de fallas: {tasa_falla:.2f} %  ->  DESBALANCE DE CLASES")

plt.figure(figsize=(5, 4))
sns.countplot(x='Machine failure', data=df)
plt.title('Distribución de clases (0 = no falla, 1 = falla)')
plt.tight_layout()
plt.savefig('eda_balance_clases.png', dpi=150)
plt.close()

# 3.2 Estadísticas descriptivas de variables numéricas
print("\nEstadísticas descriptivas:")
print(df.describe())

# 3.3 Matriz de correlación (solo variables numéricas)
num_cols = ['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'Machine failure']
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de correlación')
plt.tight_layout()
plt.savefig('eda_correlacion.png', dpi=150)
plt.close()

# ============================================================
# 4. PREPROCESAMIENTO
# ============================================================
# 4.1 Eliminar identificadores y las 5 banderas de subtipo de falla.
#     IMPORTANTE: TWF, HDF, PWF, OSF, RNF describen el TIPO de falla y solo
#     existen cuando ya hubo falla -> si se dejan, el modelo "hace trampa"
#     (data leakage) y las métricas salen infladas e irreales.
cols_a_eliminar = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df = df.drop(columns=[c for c in cols_a_eliminar if c in df.columns])

# 4.2 Codificar la variable categórica 'Type' (L, M, H -> 0, 1, 2)
if 'Type' in df.columns and not pd.api.types.is_numeric_dtype(df['Type']):
    df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

# 4.3 Verificar valores faltantes
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# 4.4 Separar features (X) y target (y)
X = df.drop(columns=['Machine failure'])
y = df['Machine failure']
print("\nVariables predictoras utilizadas:", list(X.columns))

# ============================================================
# 5. PARTICIÓN ENTRENAMIENTO / PRUEBA (80/20 ESTRATIFICADO)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
print(f"\nEntrenamiento: {X_train.shape[0]} instancias")
print(f"Prueba:        {X_test.shape[0]} instancias")

# ============================================================
# 6. NORMALIZACIÓN (escala [0, 1])
# ============================================================
# Se ajusta el escalador SOLO con train y se aplica a ambos (evita fuga de datos)
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# 7. DEFINICIÓN Y ENTRENAMIENTO DE LOS 4 MODELOS
# ============================================================
# class_weight='balanced' compensa el desbalance en RF y SVM.
modelos = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced'),
    'SVM': SVC(
        probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
    'Naive Bayes': GaussianNB(),
    'Red Neuronal': MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE),
}

resultados = []
predicciones = {}
probabilidades = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train_s, y_train)
    y_pred = modelo.predict(X_test_s)
    y_proba = modelo.predict_proba(X_test_s)[:, 1]
    predicciones[nombre] = y_pred
    probabilidades[nombre] = y_proba
    resultados.append({
        'Modelo': nombre,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_proba),
    })

# ============================================================
# 8. TABLA COMPARATIVA DE MÉTRICAS
# ============================================================
tabla = pd.DataFrame(resultados).sort_values('AUC', ascending=False)
tabla_fmt = tabla.copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']:
    tabla_fmt[col] = tabla_fmt[col].round(4)
print("\n" + "=" * 60)
print("RESULTADOS COMPARATIVOS")
print("=" * 60)
print(tabla_fmt.to_string(index=False))
tabla_fmt.to_csv('resultados_metricas.csv', index=False)

# ============================================================
# 9. MATRIZ DE CONFUSIÓN DEL MEJOR MODELO (por AUC)
# ============================================================
mejor = tabla.iloc[0]['Modelo']
print(f"\nMejor modelo según AUC: {mejor}")
cm = confusion_matrix(y_test, predicciones[mejor])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['No falla', 'Falla'])
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Matriz de confusión — {mejor}')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=150)
plt.close()
print("Matriz de confusión:")
print(cm)

# ============================================================
# 10. CURVAS ROC DE LOS 4 MODELOS
# ============================================================
plt.figure(figsize=(7, 6))
for nombre in modelos:
    fpr, tpr, _ = roc_curve(y_test, probabilidades[nombre])
    plt.plot(fpr, tpr, label=f'{nombre} (AUC = {auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Azar')
plt.xlabel('Tasa de falsos positivos (1 - Especificidad)')
plt.ylabel('Tasa de verdaderos positivos (Sensibilidad)')
plt.title('Curvas ROC — comparación de modelos')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('curvas_roc.png', dpi=150)
plt.close()

# ============================================================
# 11. GUARDAR EL MEJOR MODELO Y EL ESCALADOR (para la app Streamlit)
# ============================================================
import joblib
joblib.dump(modelos[mejor], 'modelo_final.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'columnas.pkl')
print(f"\nModelo '{mejor}' guardado como modelo_final.pkl")
print("Gráficos generados: eda_balance_clases.png, eda_correlacion.png,")
print("                    matriz_confusion.png, curvas_roc.png")
print("\n✓ EJECUCIÓN COMPLETADA")
