import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import make_classification
from typing import Tuple, Dict, TypeAlias, Optional
import os

# ==========================================
# 1. DEFINICIÓN DE TIPOS (TypeAlias)
# ==========================================
Features: TypeAlias = np.ndarray
Target: TypeAlias = np.ndarray
Predictions: TypeAlias = np.ndarray
Model: TypeAlias = RandomForestClassifier
DataFrame: TypeAlias = pd.DataFrame

# ==========================================
# 2. FUNCIONES DEL PIPELINE
# ==========================================

def cargar_datos(ruta: str) -> Tuple[Features, Target]:
    """
    Carga el CSV, valida integridad y separa Features (X) de Target (y).
    
    Args:
        ruta (str): Ubicación del archivo CSV.
    
    Returns:
        Tuple[Features, Target]: Arrays de numpy X, y.
    
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo está vacío o con formato incorrecto.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Archivo no encontrado en: {ruta}")

    try:
        df: DataFrame = pd.read_csv(ruta)
    except Exception as e:
        raise ValueError(f"No se pudo leer el CSV: {e}")

    if df.empty:
        raise ValueError("El archivo CSV está vacío.")
    
    # Validamos que existan suficientes columnas (al menos 1 feature + 1 target)
    if df.shape[1] < 2:
        raise ValueError(f"Dimensiones insuficientes: {df.shape}")

    # Separación estricta usando numpy
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    return X, y

def preprocesar(X: Features) -> Features:
    """
    Limpia y normaliza los datos manualmente.
    1. Imputa 0s con la media de la columna.
    2. Aplica normalización Z-Score: (x - mean) / std.
    """
    if X.size == 0:
        raise ValueError("Matriz de features vacía.")

    # Trabajamos sobre una copia para no afectar la referencia original
    X_proc = X.copy().astype(float)
    
    # --- A. Imputación de ceros ---
    # Recorremos columnas para reemplazar 0 con la media de los valores no-cero
    _, n_cols = X_proc.shape
    for i in range(n_cols):
        col_data = X_proc[:, i]
        if np.any(col_data == 0):
            # Calculamos media ignorando los ceros y NaNs
            mean_val = np.nanmean(np.where(col_data == 0, np.nan, col_data))
            if np.isnan(mean_val): mean_val = 0
            
            # Reemplazamos
            col_data[col_data == 0] = mean_val
            X_proc[:, i] = col_data

    # --- B. Normalización Manual (Z-Score) ---
    media = np.mean(X_proc, axis=0)
    desviacion = np.std(X_proc, axis=0)
    
    # Evitar división por cero
    desviacion[desviacion == 0] = 1.0
    
    X_norm = (X_proc - media) / desviacion

    # Validación final de sanidad
    if np.isnan(X_norm).any():
        raise ValueError("El preprocesamiento resultó en valores NaN.")

    return X_norm

def entrenar_modelo(X: Features, y: Target, **params) -> Model:
    """
    Configura y entrena el modelo RandomForest.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch de dimensiones: X={X.shape}, y={y.shape}")

    try:
        clf = RandomForestClassifier(**params)
        clf.fit(X, y)
        return clf
    except Exception as e:
        raise ValueError(f"Error en entrenamiento: {e}")

def predecir(modelo: Model, X: Features) -> Predictions:
    """
    Genera predicciones validando dimensionalidad de entrada.
    """
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz 2D para predecir.")
        
    return modelo.predict(X)

def evaluar(modelo: Model, X: Features, y: Target) -> Dict[str, float]:
    """
    Retorna métricas de rendimiento (Accuracy, Precision, Recall).
    """
    preds = predecir(modelo, X)
    
    return {
        "accuracy": float(accuracy_score(y, preds)),
        # zero_division=0 evita warnings si el modelo no predice una clase
        "precision": float(precision_score(y, preds, average='weighted', zero_division=0)),
        "recall": float(recall_score(y, preds, average='weighted', zero_division=0))
    }

# ==========================================
# 3. EJECUCIÓN (MAIN)
# ==========================================
if __name__ == "__main__":
    print("--- Iniciando Diabetes Prediction Pipeline ---")
    
    FILE_PATH = "diabetes.csv"
    
    try:
        # 1. Cargar
        if os.path.exists(FILE_PATH):
            print(f"Cargando dataset real: {FILE_PATH}")
            X_raw, y_raw = cargar_datos(FILE_PATH)
        else:
            print("Dataset no encontrado. Generando datos SINTÉTICOS para demo...")
            X_raw, y_raw = make_classification(n_samples=200, n_features=8, random_state=42)
            X_raw[0, 1] = 0 # Forzamos un cero para probar imputación logic

        # 2. Preprocesar
        print("Preprocesando y normalizando datos...")
        X_clean = preprocesar(X_raw)
        
        # 3. Entrenar
        print("Entrenando RandomForest...")
        model = entrenar_modelo(X_clean, y_raw, n_estimators=100, random_state=42, max_depth=5)
        
        # 4. Evaluar (Usamos los mismos datos por simplicidad del ejercicio)
        print("Evaluando modelo...")
        metrics = evaluar(model, X_clean, y_raw)
        
        print("\n--- Resultados del Modelo ---")
        for k, v in metrics.items():
            print(f"{k.capitalize():<12}: {v:.4f}")
            
    except Exception as e:
        print(f"\n Error Crítico: {e}")