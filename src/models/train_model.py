import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path

# Imports de TensorFlow/Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Agregar el directorio raíz al path de Python
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# --- IMPORTAMOS LAS UTILIDADES ---
# Estas funciones están definidas en src/models/windowing_utils.py
from src.models.windowing_utils import create_dual_stream_data
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


# --- 2. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO ---

def train_and_evaluate_model(input_path: str, output_path: str, model_name: str, 
                             in_steps: int = 24, out_steps: int = 24, verbose: bool = False):
    """
    Ejecuta el pipeline completo de preparación de datos, entrenamiento y evaluación.
    
    Args:
        verbose: Si True, muestra información detallada de progreso
    """
    
    if verbose:
        print(f"Cargando datos procesados desde: {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    if verbose:
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
        print("Creando ventanas para cada split...")
    
    X_past_train, X_fut_train, Y_train = create_dual_stream_data(
        df_train, n_past=in_steps, n_future=out_steps
    )
    
    X_past_val, X_fut_val, Y_val = create_dual_stream_data(
        df_val, n_past=in_steps, n_future=out_steps
    )
    
    X_past_test, X_fut_test, Y_test = create_dual_stream_data(
        df_test, n_past=in_steps, n_future=out_steps
    )

    if verbose:
        print(f"\nVentanas generadas:")
        print(f"  Train: {X_past_train.shape} - {X_past_train.shape[0]} muestras")
        print(f"  Val:   {X_past_val.shape} - {X_past_val.shape[0]} muestras")
        print(f"  Test:  {X_past_test.shape} - {X_past_test.shape[0]} muestras")
    
    past_shape = X_past_train.shape[1:]
    future_shape = X_fut_train.shape[1:]
    
    model = build_dual_stream_model(past_shape, future_shape, out_steps)
    if verbose:
        model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    if verbose:
        print("\nIniciando Entrenamiento...")
    
    # Medir tiempo de entrenamiento
    start_time = time.time()
    
    history = model.fit(
        x=[X_past_train, X_fut_train], 
        y=Y_train,
        validation_data=([X_past_val, X_fut_val], Y_val),
        epochs=100,      
        batch_size=32,    
        callbacks=[early_stop],
        verbose=1
    )
    
    training_time = time.time() - start_time
    total_epochs = len(history.history['loss'])
    avg_time_per_epoch = training_time / total_epochs
    
    if verbose:
        print(f"\n⏱ Tiempo total: {training_time:.2f}s ({training_time/60:.2f}min)")
        print(f"⏱ Tiempo promedio por época: {avg_time_per_epoch:.2f}s")
        print(f"⏱ Épocas ejecutadas: {total_epochs}")
    
    predictions = model.predict([X_past_test, X_fut_test])

    mse_test = np.mean((Y_test - predictions) ** 2)
    mae_test = np.mean(np.abs(Y_test - predictions))

    if verbose:
        print(f"\nResultados en Test (Datos Escalados):")
        print(f"MSE: {mse_test:.5f}")
        print(f"MAE: {mae_test:.5f}")

    model_full_path = os.path.join(output_path, model_name)
    os.makedirs(output_path, exist_ok=True)
    model.save(model_full_path)
    
    print(f"✓ Modelo guardado: {model_full_path}")
    
    base_dir = os.path.dirname(output_path)
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    history_filename = model_name.replace('.keras', '_history.csv')
    history_path = os.path.join(figures_dir, history_filename)
    
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = range(1, len(history_df) + 1)
    history_df.to_csv(history_path, index=False)
    
    if verbose:
        print(f"✓ Historial guardado: {history_path}")
    
    # --- 2.8 GUARDAR RESUMEN DE ENTRENAMIENTO ---
    summary_path = os.path.join(figures_dir, 'training_summary.csv')
    
    # Obtener mejores métricas
    best_val_loss = min(history.history['val_loss'])
    best_val_mae = min(history.history['val_mae'])
    best_val_loss_epoch = history.history['val_loss'].index(best_val_loss) + 1
    best_val_mae_epoch = history.history['val_mae'].index(best_val_mae) + 1
    
    # Crear o actualizar archivo de resumen
    summary_data = {
        'model_name': [model_name],
        'in_steps': [in_steps],
        'out_steps': [out_steps],
        'total_epochs': [total_epochs],
        'total_time_seconds': [training_time],
        'total_time_minutes': [training_time / 60],
        'avg_time_per_epoch': [avg_time_per_epoch],
        'best_val_loss': [best_val_loss],
        'best_val_loss_epoch': [best_val_loss_epoch],
        'best_val_mae': [best_val_mae],
        'best_val_mae_epoch': [best_val_mae_epoch],
        'test_mse': [mse_test],
        'test_mae': [mae_test]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    if os.path.exists(summary_path):
        existing_summary = pd.read_csv(summary_path)
        summary_df = pd.concat([existing_summary, summary_df], ignore_index=True)
    
    summary_df.to_csv(summary_path, index=False)
    if verbose:
        print(f"✓ Resumen actualizado: {summary_path}")


# --- 3. BLOQUE MAIN (Para ejecución directa/prueba) ---

if __name__ == "__main__":
    
    # --- Parámetros de Configuración ---
    ID_PLANTA = 305  # Solo necesitas cambiar esto!
    
    # Obtener configuración automática de la planta
    plant_config = get_plant_config(ID_PLANTA)
    FECHA_INICIO = plant_config['fecha_inicio']
    FECHA_FIN = plant_config['fecha_fin']
    
    in_steps = 24  # Ventana histórica (horas pasadas)
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
    # Ventana histórica óptima: 4 horas (mismo rendimiento que 24h, 5x más rápido)
    
    train_and_evaluate_model(
        input_path=PROCESSED_DATA_PATH,
        output_path=MODELS_OUTPUT_PATH,
        in_steps=in_steps,
        out_steps=out_steps,
        model_name=f'Finaldual_stream_lstm_cnn_{ID_PLANTA}_{in_steps}h_{out_steps}h.keras'
    )