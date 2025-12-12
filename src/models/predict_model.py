"""
Módulo para realizar predicciones con el modelo entrenado.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model

# Agregar el directorio raíz al path de Python
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.windowing_utils import create_dual_stream_data
from src.config import get_plant_config


def load_trained_model(model_path: str, verbose: bool = False):
    """Carga el modelo entrenado desde disco."""
    if verbose:
        print(f"Cargando modelo desde: {model_path}")
    model = load_model(model_path)
    return model


def make_predictions(model, X_past, X_future):
    """
    Realiza predicciones con el modelo dual-stream.
    
    Args:
        model: Modelo cargado
        X_past: Entrada histórica
        X_future: Entrada futura
    """
    predictions = model.predict([X_past, X_future], verbose=0)
    return predictions


def visualize_predictions(Y_real, Y_pred, num_samples=5, out_steps=24, save_path=None, verbose: bool = False):
    """
    Visualiza múltiples predicciones comparadas con valores reales.
    
    Args:
        Y_real: Array con valores reales
        Y_pred: Array con predicciones
        num_samples: Número de muestras aleatorias a visualizar
        out_steps: Horizonte de predicción (para el título)
        save_path: Ruta donde guardar la imagen (opcional)
        verbose: Si True, muestra información detallada
    """
    indices = np.random.choice(len(Y_real), size=min(num_samples, len(Y_real)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        axes[i].plot(Y_real[idx], label='Real', marker='o', linewidth=2)
        axes[i].plot(Y_pred[idx], label='Predicción', marker='x', linestyle='--', linewidth=2)
        axes[i].set_title(f'Predicción {out_steps}h - Muestra {idx}')
        axes[i].set_xlabel('Horas Futuras')
        axes[i].set_ylabel('Generación Normalizada')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"✓ Gráfico guardado en: {save_path}")
    
    plt.show()


def calculate_metrics(Y_real, Y_pred):
    """Calcula métricas de error."""
    mse = np.mean((Y_real - Y_pred) ** 2)
    mae = np.mean(np.abs(Y_real - Y_pred))
    rmse = np.sqrt(mse)
    
    # Error por hora
    mae_per_hour = np.mean(np.abs(Y_real - Y_pred), axis=0)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAE_por_hora': mae_per_hour
    }


def plot_error_by_hour(metrics, save_path=None, verbose: bool = False):
    """Visualiza el error MAE por cada hora de predicción."""
    mae_per_hour = metrics['MAE_por_hora']
    
    plt.figure(figsize=(12, 5))
    plt.bar(range(1, len(mae_per_hour) + 1), mae_per_hour, alpha=0.7, color='steelblue')
    plt.axhline(y=metrics['MAE'], color='r', linestyle='--', label=f'MAE Promedio: {metrics["MAE"]:.4f}')
    plt.xlabel('Hora de Predicción')
    plt.ylabel('MAE')
    plt.title('Error Absoluto Medio por Hora de Predicción')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"✓ Gráfico de error guardado en: {save_path}")
    
    plt.show()


def load_and_plot_training_history(history_path, save_path=None, verbose: bool = False):
    """
    Carga y visualiza el historial de entrenamiento guardado.
    
    Args:
        history_path: Ruta al archivo CSV del historial
        save_path: Ruta donde guardar la imagen (opcional)
        verbose: Si True, muestra información detallada
    """
    if verbose:
        print(f"Cargando historial desde: {history_path}")
    history_df = pd.read_csv(history_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de Loss
    axes[0].plot(history_df['epoch'], history_df['loss'], label='Train Loss', linewidth=2, marker='o')
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', linewidth=2, marker='s')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Evolución del Loss durante Entrenamiento')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico de MAE
    axes[1].plot(history_df['epoch'], history_df['mae'], label='Train MAE', linewidth=2, marker='o')
    axes[1].plot(history_df['epoch'], history_df['val_mae'], label='Val MAE', linewidth=2, marker='s')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Evolución del MAE durante Entrenamiento')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"✓ Gráfico de historial guardado en: {save_path}")
    
    plt.show()
    
    # Mostrar resumen solo si verbose=True
    if verbose:
        print("\n" + "="*50)
        print("RESUMEN DEL ENTRENAMIENTO")
        print("="*50)
        print(f"Épocas totales: {len(history_df)}")
        print(f"Mejor Loss Train:  {history_df['loss'].min():.5f} (época {history_df['loss'].idxmin() + 1})")
        print(f"Mejor Loss Val:    {history_df['val_loss'].min():.5f} (época {history_df['val_loss'].idxmin() + 1})")
        print(f"Mejor MAE Train:   {history_df['mae'].min():.5f} (época {history_df['mae'].idxmin() + 1})")
        print(f"Mejor MAE Val:     {history_df['val_mae'].min():.5f} (época {history_df['val_mae'].idxmin() + 1})")
        print("="*50)


if __name__ == "__main__":
    
    # --- Configuración ---
    # IMPORTANTE: Debe coincidir exactamente con train_model.py
    ID_PLANTA = 305  # Solo necesitas cambiar esto!
    
    # Obtener configuración automática de la planta
    plant_config = get_plant_config(ID_PLANTA)
    FECHA_INICIO = plant_config['fecha_inicio']
    FECHA_FIN = plant_config['fecha_fin']
    
    in_steps = 24   # Ventana histórica (horas pasadas)
    out_steps = 48  # Horizonte de predicción (horas futuras)
    
    # Obtener el directorio base del proyecto
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Rutas
    PROCESSED_DATA_PATH = os.path.join(
        BASE_DIR, 'data', '03_processed',
        f'DatosCombinados_{FECHA_INICIO}_a_{FECHA_FIN}_Planta{ID_PLANTA}.csv'
    )
    MODEL_PATH = os.path.join(BASE_DIR, 'models', f'dual_stream_lstm_cnn_{ID_PLANTA}_{in_steps}h_{out_steps}h.keras')
    HISTORY_PATH = os.path.join(BASE_DIR, 'figures', f'dual_stream_lstm_cnn_{ID_PLANTA}_{in_steps}h_{out_steps}h_history.csv')
    
    # Rutas para guardar gráficos
    FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
    
    try:
        print(f"Cargando datos desde: {PROCESSED_DATA_PATH}")
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
        
        # División cronológica sin solapamiento
        total_timesteps = len(df)
        window_size = in_steps + out_steps
        
        train_end_idx = int(total_timesteps * 0.70)
        val_start_idx = train_end_idx + (window_size - 1)
        val_end_idx = int(total_timesteps * 0.85)
        test_start_idx = val_end_idx + (window_size - 1)
        
        df_test = df.iloc[test_start_idx:]
        
        # Crear ventanas
        X_past, X_future, Y_target = create_dual_stream_data(df_test, n_past=in_steps, n_future=out_steps)
        
        X_past_test = X_past
        X_future_test = X_future
        Y_test = Y_target
        
        print(f"\n🔍 Test set: {len(X_past_test)} muestras | Periodo: {df.index[test_start_idx]} a {df.index[-1]}")
        
        # Cargar modelo
        model = load_trained_model(MODEL_PATH, verbose=True)
        
        # Cargar historial de entrenamiento
        if os.path.exists(HISTORY_PATH):
            print("\n📈 Visualizando historial de entrenamiento...")
            load_and_plot_training_history(
                HISTORY_PATH,
                save_path=os.path.join(FIGURES_DIR, 'training_history.png'),
                verbose=True
            )
        else:
            print(f"\n⚠ Historial no encontrado: {HISTORY_PATH}")
        
        # Realizar predicciones
        print("\n🚀 Realizando predicciones...")
        predictions = make_predictions(model, X_past_test, X_future_test)
        
        # Calcular métricas
        metrics = calculate_metrics(Y_test, predictions)
        print(f"✅ Test MSE: {metrics['MSE']:.5f} | MAE: {metrics['MAE']:.5f} | RMSE: {metrics['RMSE']:.5f}")
        
        # Visualizar predicciones
        print("\n📊 Generando visualizaciones...")
        visualize_predictions(
            Y_test, 
            predictions, 
            num_samples=5,
            out_steps=out_steps,
            save_path=os.path.join(FIGURES_DIR, 'predicciones_ejemplos.png'),
            verbose=True
        )
        
        # Visualizar error por hora
        plot_error_by_hour(
            metrics,
            save_path=os.path.join(FIGURES_DIR, 'error_por_hora.png'),
            verbose=True
        )
        
        print("\n✅ Predicciones completadas exitosamente")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Archivo no encontrado: {e.filename}")
        print("Ejecuta primero: make_dataset.py → build_features.py → train_model.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()