import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path

# Imports de TensorFlow/Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, Flatten, Dense, Concatenate, Dropout, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant

# Agregar el directorio raíz al path de Python
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# --- IMPORTAMOS LAS UTILIDADES ---
# Estas funciones están definidas en src/models/windowing_utils.py
from src.models.windowing_utils import create_dual_stream_data, create_tri_stream_data_alternativo
from src.config import get_plant_config
from src.config import get_plant_config 


# --- 1. FUNCIÓN DE DEFINICIÓN DE ARQUITECTURA ---

def build_dual_stream_model(past_shape: tuple, future_shape: tuple, output_steps: int):
    """
    Define y compila el modelo BiLSTM + CNN 1D Dual-Stream para la predicción.
    """
    
    # Rama 1: Procesamiento del Histórico (Inercia del sistema)
    # Ejemplo: (24, 19)
    input_past = Input(shape=past_shape, name='Input_Historico')
    x1 = Bidirectional(LSTM(64, return_sequences=True))(input_past)
    x1 = Bidirectional(LSTM(32, return_sequences=False))(x1)
    x1 = Dropout(0.1)(x1) 

    # Rama 2: Procesamiento del Futuro (Pronóstico Meteorológico)
    # Ejemplo: (24, 18)
    input_fut = Input(shape=future_shape, name='Input_Pronostico')
    x2 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_fut)
    x2 = Flatten()(x2)
    x2 = Dense(32, activation='relu')(x2)

    # Fusión de ambas ramas
    combined = Concatenate()([x1, x2])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.1)(z)

    # Capa de Salida: output_steps neuronas para predecir todos los pasos futuros
    output = Dense(output_steps, activation='linear', name='Output_Future')(z)

    # Compilar el modelo
    model = Model(inputs=[input_past, input_fut], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    return model


def build_tri_stream_model(past_shape: tuple, future_shape: tuple, binary_shape: tuple, output_steps: int = 48):
    """
    Define y compila el modelo Tri-Stream con Gating Multiplicativo.
    
    Arquitectura:
    - Rama 1 (Inercia): BiLSTM captura dinámica histórica
    - Rama 2 (Pronóstico): Conv1D extrae patrones meteorológicos futuros
    - Rama 3 (Gating): Red densa procesa variables binarias para generar probabilidad de operación
    
    Fusión: Y_final = Y_potencial ⊗ Y_gate (multiplicación elemento a elemento)
    
    Args:
        past_shape: Shape del input histórico, ej. (24, 19)
        future_shape: Shape del input futuro continuo, ej. (48, 18)
        binary_shape: Shape del input binario futuro, ej. (48, N_binarias)
        output_steps: Número de pasos a predecir (default: 48)
    
    Returns:
        model: Modelo Keras compilado
    """
    
    # ========== RAMA 1: PROCESAMIENTO DEL HISTÓRICO (Inercia del Sistema) ==========
    # Captura la dinámica reciente y el estado operativo del sistema
    input_past = Input(shape=past_shape, name='Input_Historico')
    x1 = Bidirectional(LSTM(64, return_sequences=True))(input_past)
    x1 = Bidirectional(LSTM(32, return_sequences=False))(x1)
    x1 = Dropout(0.1)(x1)
    
    # ========== RAMA 2: PROCESAMIENTO DEL PRONÓSTICO METEOROLÓGICO ==========
    # Extrae patrones locales de radiación y condiciones futuras
    input_fut = Input(shape=future_shape, name='Input_Pronostico')
    x2 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_fut)
    x2 = Flatten()(x2)
    x2 = Dense(32, activation='relu')(x2)
    
    # ========== RAMA 3: GATING (Variables Binarias/Estado Operativo) [NUEVA] ==========
    # Procesa variables exógenas binarias (ej: is_holiday, maintenance_flag, grid_status)
    # para generar un vector de probabilidad de operación [0, 1] para cada hora
    input_binary = Input(shape=binary_shape, name='Input_Binary')
    x3 = Flatten()(input_binary)
    x3 = Dense(32, activation='relu')(x3)
    
    # CRÍTICO: Capa de salida con sigmoid para generar gate [0, 1]
    # Inicializamos el bias con valor positivo (3.0) para empezar con "compuerta abierta"
    # Esto evita gradientes nulos al inicio del entrenamiento
    output_gate = Dense(
        output_steps, 
        activation='sigmoid', 
        name='Output_Gate',
        bias_initializer=Constant(value=3.0)  # Gate empieza cerca de 1 (σ(3) ≈ 0.95)
    )(x3)
    
    # ========== FUSIÓN: GENERACIÓN POTENCIAL ==========
    # Combinamos inercia histórica + pronóstico meteorológico para estimar generación potencial
    combined = Concatenate()([x1, x2])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.1)(z)
    
    # Generación potencial (sin restricciones operativas)
    output_potential = Dense(output_steps, activation='linear', name='Output_Potential')(z)
    
    # ========== MULTIPLICACIÓN FINAL: Y_final = Y_potencial ⊗ Y_gate ==========
    # El gate modula la generación potencial según el estado operativo
    # Si gate ≈ 1: sistema operando normalmente (salida ≈ potencial)
    # Si gate ≈ 0: sistema inoperativo (salida ≈ 0)
    # Valores intermedios representan operación parcial o degradada
    output_final = Multiply(name='Output_Final')([output_potential, output_gate])
    
    # ========== COMPILACIÓN ==========
    model = Model(
        inputs=[input_past, input_fut, input_binary], 
        outputs=output_final
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='mse', 
        metrics=['mae']
    )
    
    return model


# --- 2. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO (TRI-STREAM) ---

def train_and_evaluate_model(input_path: str, output_path: str, model_name: str, 
                             in_steps: int = 24, out_steps: int = 24,
                             binary_features: list = None, verbose: bool = False):
    """
    Ejecuta el pipeline completo de preparación de datos, entrenamiento y evaluación
    usando la arquitectura Tri-Stream con Gating Multiplicativo.
    
    Args:
        binary_features: Lista de nombres de columnas binarias para gating.
                        Si es None, usa ['feriado'] por defecto.
                        shadow y cloud se mantienen como features continuas.
        verbose: Si True, muestra información detallada de progreso
    """
    
    if verbose:
        print(f"Cargando datos procesados desde: {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    if verbose:
        print(f"\nColumnas disponibles: {list(df.columns)}")
        print(f"Total de columnas: {len(df.columns)}")
        print("\n🔍 DIVISIÓN CRONOLÓGICA SIN SOLAPAMIENTO")
        print("="*60)
    
    total_timesteps = len(df)
    window_size = in_steps + out_steps  # Tamaño total de cada ventana
    
    # Calcular índices de corte en el dataset ORIGINAL
    # Dejamos gap = window_size - 1 entre cada split para evitar solapamiento
    train_end_idx = int(total_timesteps * 0.70)
    val_start_idx = train_end_idx + (window_size - 1)  # Gap para evitar solapamiento
    val_end_idx = int(total_timesteps * 0.85)
    test_start_idx = val_end_idx + (window_size - 1)   # Gap para evitar solapamiento
    
    if test_start_idx >= total_timesteps:
        raise ValueError(f"Dataset muy pequeño. Necesitas al menos {test_start_idx} registros, tienes {total_timesteps}")
    
    if verbose:
        print(f"Dataset total: {total_timesteps} registros")
        print(f"Ventana total por muestra: {window_size}h ({in_steps}h past + {out_steps}h future)")
        print(f"\nRangos temporales (con gaps para evitar solapamiento):")
        print(f"  Train: {df.index[0]} a {df.index[train_end_idx-1]} ({train_end_idx} registros)")
        print(f"  Gap:   {window_size-1} registros (evita solapamiento)")
        print(f"  Val:   {df.index[val_start_idx]} a {df.index[val_end_idx-1]} ({val_end_idx - val_start_idx} registros)")
        print(f"  Gap:   {window_size-1} registros (evita solapamiento)")
        print(f"  Test:  {df.index[test_start_idx]} a {df.index[-1]} ({total_timesteps - test_start_idx} registros)")
        print("="*60 + "\n")
    
    # Dividir el dataframe ANTES de crear ventanas
    df_train = df.iloc[:train_end_idx]
    df_val = df.iloc[val_start_idx:val_end_idx]
    df_test = df.iloc[test_start_idx:]
    
    if verbose:
        print("\nCreando ventanas TRI-STREAM para cada split...")
    
    # Definir features binarias por defecto si no se especifican
    if binary_features is None:
        binary_features = ['feriado']  # Solo feriado para gating
    
    if missing_features:
        raise ValueError(
            f"\n❌ ERROR: Features binarias no encontradas: {missing_features}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )
    
    if verbose:
        print(f"Features binarias para gating (Rama 3): {binary_features}")
        print(f"Features continuas (Ramas 1 y 2): Incluyen shadow, cloud y otras meteorológicas")
    
    X_past_train, X_fut_train, X_bin_train, Y_train = create_tri_stream_data_alternativo(
        df_train, n_past=in_steps, n_future=out_steps, binary_features=binary_features
    )
    
    X_past_val, X_fut_val, X_bin_val, Y_val = create_tri_stream_data_alternativo(
        df_val, n_past=in_steps, n_future=out_steps, binary_features=binary_features
    )
    
    X_past_test, X_fut_test, X_bin_test, Y_test = create_tri_stream_data_alternativo(
        df_test, n_past=in_steps, n_future=out_steps, binary_features=binary_features
    )

    if verbose:
        print(f"Train: {X_past_train.shape[0]} ventanas | Val: {X_past_val.shape[0]} | Test: {X_past_test.shape[0]}")
    
    # Definición del modelo
    past_shape = X_past_train.shape[1:]
    future_shape = X_fut_train.shape[1:]
    binary_shape = X_bin_train.shape[1:]
    
    model = build_tri_stream_model(past_shape, future_shape, binary_shape, out_steps)
    if verbose:
        model.summary()

    # --- 2.4 ENTRENAMIENTO ---
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print("\nIniciando Entrenamiento TRI-STREAM...")
    
    # Medir tiempo de entrenamiento
    start_time = time.time()
    
    history = model.fit(
        x=[X_past_train, X_fut_train, X_bin_train],  # TRES INPUTS
        y=Y_train,
        validation_data=([X_past_val, X_fut_val, X_bin_val], Y_val),
        epochs=100,      
        batch_size=32,    
        callbacks=[early_stop],
        verbose=1
    )
    
    training_time = time.time() - start_time
    total_epochs = len(history.history['loss'])
    avg_time_per_epoch = training_time / total_epochs
    
    print(f"\n⏱ Tiempo total de entrenamiento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)")
    print(f"⏱ Tiempo promedio por época: {avg_time_per_epoch:.2f} segundos")
    print(f"⏱ Épocas ejecutadas: {total_epochs}")

    # Evaluación
    predictions = model.predict([X_past_test, X_fut_test, X_bin_test], verbose=0)
    mse_test = np.mean((Y_test - predictions) ** 2)
    mae_test = np.mean(np.abs(Y_test - predictions))

    # Guardar modelo
    model_full_path = os.path.join(output_path, model_name)
    os.makedirs(output_path, exist_ok=True)
    model.save(model_full_path) 
    print(f"✅ Modelo guardado: {model_full_path}")
    
    # Guardar historial
    base_dir = os.path.dirname(output_path)
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    history_filename = model_name.replace('.keras', '_history.csv')
    history_path = os.path.join(figures_dir, history_filename)
    
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = range(1, len(history_df) + 1)
    history_df.to_csv(history_path, index=False)
    
    if verbose:
        print(f"✅ Historial guardado: {history_path}")
    
    # Guardar resumen
    summary_path = os.path.join(figures_dir, 'training_summary.csv')
    
    # Obtener mejores métricas
    best_val_loss = min(history.history['val_loss'])
    best_val_mae = min(history.history['val_mae'])
    
    avg_time_per_epoch = training_time / total_epochs
    
    # Crear o actualizar archivo de resumen
    summary_data = {
        'model_name': [model_name],
        'in_steps': [in_steps],
        'out_steps': [out_steps],
        'total_epochs': [total_epochs],
        'training_time_min': [training_time / 60],
        'best_val_loss': [best_val_loss],
        'best_val_mae': [best_val_mae],
        'test_mse': [mse_test],
        'test_mae': [mae_test]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    if os.path.exists(summary_path):
        existing_summary = pd.read_csv(summary_path)
        summary_df = pd.concat([existing_summary, summary_df], ignore_index=True)
    
    summary_df.to_csv(summary_path, index=False)
    if verbose:
        print(f"✅ Resumen actualizado: {summary_path}")


# --- 3. BLOQUE MAIN (Para ejecución directa/prueba) ---

if __name__ == "__main__":
    
    # --- Parámetros de Configuración ---
    ID_PLANTA = 239  # Solo necesitas cambiar esto!
    
    # Obtener configuración automática de la planta
    plant_config = get_plant_config(ID_PLANTA)
    FECHA_INICIO = plant_config['fecha_inicio']
    FECHA_FIN = plant_config['fecha_fin']
    
    in_steps = 11  # Ventana histórica (horas pasadas)
    out_steps = 48   # Horizonte de predicción (horas futuras)
    
    # Obtener el directorio base del proyecto
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Rutas absolutas
    PROCESSED_DATA_PATH = os.path.join(
        BASE_DIR, 'data', '03_processed', 
        f'DatosCombinados_{FECHA_INICIO}_a_{FECHA_FIN}_Planta{ID_PLANTA}.csv'
    )
    MODELS_OUTPUT_PATH = os.path.join(BASE_DIR, 'models')
    
    # Ejecutar la función principal de entrenamiento
    # Arquitectura TRI-STREAM con Gating Multiplicativo
    
    train_and_evaluate_model(
        input_path=PROCESSED_DATA_PATH,
        output_path=MODELS_OUTPUT_PATH,
        in_steps=in_steps,
        out_steps=out_steps,
        model_name=f'tri_stream_gating_{ID_PLANTA}_{in_steps}h_{out_steps}h.keras',
        binary_features=['feriado'],  # Solo feriado para gating (shadow y cloud quedan continuas)
        verbose=True  # Mostrar información detallada en modo directo
    )