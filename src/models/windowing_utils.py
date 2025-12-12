# Ruta: src/models/windowing_utils.py

import numpy as np
import pandas as pd


# --- 1. DEFINICIÓN DE FEATURES (Conjuntos de columnas) ---

def get_default_feature_sets():
    """
    Retorna los conjuntos de features por defecto para el modelado de energía solar.
    
    RADIACIÓN (3 variables seleccionadas):
    - glb: Radiación global horizontal (principal predictor)
    - dni: Radiación directa normal (calidad del día despejado)
    - dif: Radiación difusa (efecto de nubes)
    
    ESTADO PLANTA (1 variable binaria):
    - feriado: Indica si es día de feriado/mantenimiento (0=normal, 1=inactivo)
    
    ELIMINADAS (5 variables redundantes):
    - ghi: Redundante con glb (r > 0.99)
    - dir, dirh: Redundantes con dni
    - difh: Redundante con dif
    - sct: Baja correlación con generación
    """
    TARGET_COLUMN = 'gen_normalizada'
    FUTURE_FEATURES = [
        # RADIACIÓN: Solo 3 variables no redundantes
        'glb', 'dni', 'dif',
        # METEOROLOGÍA
        'temp', 'vel', 'shadow', 'cloud',
        # ESTADO PLANTA
        'feriado',
        # TIEMPO CÍCLICO
        'hora_sin', 'hora_cos', 'mes_sin', 'mes_cos', 'dia_año_sin', 'dia_año_cos'
    ]
    PAST_FEATURES = [TARGET_COLUMN] + FUTURE_FEATURES
    
    return TARGET_COLUMN, FUTURE_FEATURES, PAST_FEATURES


# --- 2. FUNCIÓN DE WINDOWING ---

def create_dual_stream_data(df: pd.DataFrame, n_past: int, n_future: int, feature_sets: tuple = None):
    """
    Función Genérica para crear ventanas temporales de doble entrada (histórico + pronóstico de features).

    Args:
        df (pd.DataFrame): DataFrame combinado y procesado (de data/03_processed/).
        n_past (int): Cuántos pasos temporales mirar hacia el pasado (ventana histórica).
        n_future (int): Cuántos pasos temporales predecir (ventana futura).
        feature_sets (tuple, opcional): Tupla de (TARGET_COLUMN, FUTURE_FEATURES, PAST_FEATURES). 
                                        Si es None, usa las definidas por defecto.
    
    Returns:
        tuple: (X_past, X_future, Y_target) en formato numpy (tensores 3D).
    """
    
    # 1. Cargar las definiciones de features
    if feature_sets is None:
        TARGET_COLUMN, FUTURE_FEATURES, PAST_FEATURES = get_default_feature_sets()
    else:
        TARGET_COLUMN, FUTURE_FEATURES, PAST_FEATURES = feature_sets

    # 2. Preparar los datos de entrada y salida basados en las definiciones
    data_x = df[PAST_FEATURES].values
    data_y = df[[TARGET_COLUMN]].values
    
    # Índice de las features que se asumen como 'conocidas en el futuro'
    future_feat_indices = [PAST_FEATURES.index(col) for col in FUTURE_FEATURES]

    X_past, X_future, Y_target = [], [], []
    
    # 3. Recorrer la serie temporal y crear las ventanas
    # La condición garantiza que haya suficientes datos tanto para el pasado (n_past) 
    # como para el futuro (n_future).
    for i in range(n_past, len(data_x) - n_future + 1):
        # Rama 1 (Histórico): n_past pasos antes de 'i'. Contiene generación y features históricas.
        X_past.append(data_x[i - n_past:i, :])
        
        # Rama 2 (Pronóstico de Features): n_future pasos a partir de 'i'. Solo features conocidas a futuro.
        X_future.append(data_x[i:i + n_future, future_feat_indices])
        
        # Target (Valores reales): n_future pasos a partir de 'i'. Solo el target de generación.
        Y_target.append(data_y[i:i + n_future, 0])
        
    return np.array(X_past), np.array(X_future), np.array(Y_target)


def create_tri_stream_data_alternativo(df: pd.DataFrame, n_past: int, n_future: int, 
                                        binary_features: list = None, feature_sets: tuple = None):
    """
    Función ALTERNATIVA para crear ventanas temporales de TRIPLE entrada (histórico + pronóstico + binarias).
    
    Esta función extiende create_dual_stream_data para soportar la arquitectura Tri-Stream con Gating.
    
    Args:
        df (pd.DataFrame): DataFrame combinado y procesado (de data/03_processed/).
        n_past (int): Cuántos pasos temporales mirar hacia el pasado (ventana histórica).
        n_future (int): Cuántos pasos temporales predecir (ventana futura).
        binary_features (list, opcional): Lista de nombres de columnas binarias para el gating.
                                          Si es None, usa ['feriado'] por defecto.
        feature_sets (tuple, opcional): Tupla de (TARGET_COLUMN, FUTURE_FEATURES, PAST_FEATURES).
                                        Si es None, crea features alternativas SIN 'feriado'.
    
    Returns:
        tuple: (X_past, X_future, X_binary, Y_target) en formato numpy (tensores 3D).
               - X_past: shape (n_samples, n_past, n_features_past)
               - X_future: shape (n_samples, n_future, n_features_future)
               - X_binary: shape (n_samples, n_future, n_features_binary)
               - Y_target: shape (n_samples, n_future)
    """
    
    # 1. Cargar las definiciones de features
    if feature_sets is None:
        # Usar features alternativas SIN 'feriado' para tri-stream
        TARGET_COLUMN = 'gen_normalizada'
        FUTURE_FEATURES = [
            # RADIACIÓN
            'glb', 'dni', 'dif',
            # METEOROLOGÍA
            'temp', 'vel', 'shadow', 'cloud',
            # TIEMPO CÍCLICO
            'hora_sin', 'hora_cos', 'mes_sin', 'mes_cos', 'dia_año_sin', 'dia_año_cos'
        ]
        PAST_FEATURES = [TARGET_COLUMN] + FUTURE_FEATURES
    else:
        TARGET_COLUMN, FUTURE_FEATURES, PAST_FEATURES = feature_sets
    
    # 2. Definir features binarias por defecto si no se especifican
    if binary_features is None:
        binary_features = ['feriado']  # Solo feriado para gating (shadow y cloud permanecen continuas)
    
    # Verificar que las features binarias existen en el DataFrame
    missing_features = [feat for feat in binary_features if feat not in df.columns]
    if missing_features:
        raise ValueError(f"Features binarias no encontradas en el DataFrame: {missing_features}")
    
    # 3. Preparar los datos de entrada y salida
    data_x = df[PAST_FEATURES].values
    data_y = df[[TARGET_COLUMN]].values
    data_binary = df[binary_features].values
    
    # CRÍTICO: Excluir features binarias de la Rama 2 (pronóstico continuo)
    # Las binary_features van SOLO a la Rama 3 (gating)
    future_features_continuous = [f for f in FUTURE_FEATURES if f not in binary_features]
    
    # Índice de las features continuas conocidas en el futuro (excluyendo target y binarias)
    future_feat_indices = [PAST_FEATURES.index(col) for col in future_features_continuous]
    
    X_past, X_future, X_binary, Y_target = [], [], [], []
    
    # 4. Recorrer la serie temporal y crear las ventanas
    for i in range(n_past, len(data_x) - n_future + 1):
        # Rama 1 (Histórico): n_past pasos antes de 'i'. Contiene generación y features históricas.
        X_past.append(data_x[i - n_past:i, :])
        
        # Rama 2 (Pronóstico Meteorológico): n_future pasos a partir de 'i'. Solo features continuas.
        X_future.append(data_x[i:i + n_future, future_feat_indices])
        
        # Rama 3 (Gating Binario): n_future pasos a partir de 'i'. Solo features binarias.
        X_binary.append(data_binary[i:i + n_future, :])
        
        # Target (Valores reales): n_future pasos a partir de 'i'. Solo el target de generación.
        Y_target.append(data_y[i:i + n_future, 0])
    
    return np.array(X_past), np.array(X_future), np.array(X_binary), np.array(Y_target)


# --- 3. POSTPROCESAMIENTO ---

def apply_plant_state_postprocessing(predictions: np.ndarray, X_future: np.ndarray, 
                                     future_features: list = None, verbose: bool = False) -> np.ndarray:
    """
    Aplica postprocesamiento para forzar predicciones a 0 cuando la planta está en feriado/mantenimiento.
    
    Fórmula: Predicción_Final = Predicción_Modelo × (1 - feriado)
    
    Args:
        predictions (np.ndarray): Predicciones del modelo, shape (n_samples, out_steps)
        X_future (np.ndarray): Datos futuros, shape (n_samples, out_steps, n_features)
        future_features (list, opcional): Lista de nombres de features. Si None, usa default.
        verbose (bool): Si True, muestra información de progreso
    
    Returns:
        np.ndarray: Predicciones ajustadas con mismo shape que entrada
    """
    if future_features is None:
        _, future_features, _ = get_default_feature_sets()
    
    if 'feriado' not in future_features:
        return predictions
    
    feriado_idx = future_features.index('feriado')
    
    # Extraer valores de feriado para cada timestep futuro: shape (n_samples, out_steps)
    feriado_mask = X_future[:, :, feriado_idx]
    
    predictions_adjusted = predictions * (1 - feriado_mask)
    
    if verbose:
        n_adjusted = np.sum(feriado_mask > 0)
        total_predictions = feriado_mask.size
        
        if n_adjusted > 0:
            print(f"✓ Postprocesamiento aplicado: {n_adjusted}/{total_predictions} predicciones ajustadas a 0 por feriado")
            print(f"  Porcentaje ajustado: {n_adjusted/total_predictions*100:.1f}%")
    
    return predictions_adjusted