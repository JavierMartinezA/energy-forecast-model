"""
Script principal para ejecutar el pipeline completo de predicción de energía solar.

Este script ejecuta en secuencia todos los pasos necesarios desde el procesamiento
de datos hasta la evaluación del modelo.

Uso:
    python run.py
"""

import os
import sys
from pathlib import Path

# Configurar path del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_plant_config

def main():
    """Ejecuta el pipeline completo del proyecto."""
    
    print("="*60)
    print("PIPELINE DE PREDICCIÓN DE ENERGÍA SOLAR")
    print("="*60)
    
    # Parámetros comunes
    ID_PLANTA = 305  # Solo necesitas cambiar esto!
    
    # Obtener configuración automática de la planta
    plant_config = get_plant_config(ID_PLANTA)
    FECHA_INICIO = plant_config['fecha_inicio']
    FECHA_FIN = plant_config['fecha_fin']
    
    print(f"\nConfiguración:")
    print(f"  - Planta: {plant_config['nombre']} (ID {ID_PLANTA})")
    print(f"  - Período: {FECHA_INICIO} a {FECHA_FIN}")
    print()
    
    # Paso 1: Procesamiento de datos crudos
    print("\n[1/4] Ejecutando make_dataset.py...")
    print("-" * 60)
    ret = os.system(f'python "{PROJECT_ROOT}/src/data/make_dataset.py"')
    if ret != 0:
        print("❌ Error en make_dataset.py")
        return 1
    
    # Paso 2: Construcción de features
    print("\n[2/4] Ejecutando build_features.py...")
    print("-" * 60)
    ret = os.system(f'python "{PROJECT_ROOT}/src/features/build_features.py"')
    if ret != 0:
        print("❌ Error en build_features.py")
        return 1
    
    # Paso 3: Entrenamiento del modelo
    print("\n[3/4] Ejecutando train_model.py...")
    print("-" * 60)
    ret = os.system(f'python "{PROJECT_ROOT}/src/models/train_model.py"')
    if ret != 0:
        print("❌ Error en train_model.py")
        return 1
    
    # Paso 4: Predicciones y evaluación
    print("\n[4/4] Ejecutando predict_model.py...")
    print("-" * 60)
    ret = os.system(f'python "{PROJECT_ROOT}/src/models/predict_model.py"')
    if ret != 0:
        print("❌ Error en predict_model.py")
        return 1
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("\nResultados guardados en:")
    print(f"  - Modelo: models/dual_stream_lstm_cnn.keras")
    print(f"  - Gráficos: figures/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
