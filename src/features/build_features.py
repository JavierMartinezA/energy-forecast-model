"""
Módulo para la construcción y transformación de características (Feature Engineering).
"""

import pandas as pd
import numpy as np

def create_normalized_generation_feature(df_gensolar_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la característica objetivo: Generación normalizada (gen_normalizada).
    """
    df = df_gensolar_raw.copy()
    
    # Lógica de Feature Engineering: Energía Normalizada
    df['gen_normalizada'] = np.where(
        df['potencia_maxima'] > 0,
        df['gen_real_mw'] / df['potencia_maxima'],
        0
    )

    df['gen_normalizada'] = df['gen_normalizada'].replace([np.inf, -np.inf], 0)
    df['gen_normalizada'] = df['gen_normalizada'].fillna(0)
    
    return df

def create_time_features(df_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características cíclicas de tiempo (sin/cos) a partir del índice (fecha/hora).
    Estas son cruciales para el modelado de series de tiempo.
    """
    df = df_combined.copy()
    
    # Hora del día (0-23)
    df['hora_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    # Mes del año (1-12)
    df['mes_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Día del año (1-365)
    df['dia_año_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['dia_año_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    
    return df


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    
    # Agregar el directorio raíz al path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    
    from src.config import get_plant_config
    
    # --- Parámetros de Configuración ---
    ID_PLANTA = 346  # Solo necesitas cambiar esto!
    
    # Obtener configuración automática de la planta
    plant_config = get_plant_config(ID_PLANTA)
    FECHA_INICIO = plant_config['fecha_inicio']
    FECHA_FIN = plant_config['fecha_fin']
    
    # Obtener el directorio base del proyecto
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Rutas
    INTERIM_CSV_PATH = os.path.join(
        BASE_DIR, 'data', '02_interim',
        f'InterimCombinado_{FECHA_INICIO}_a_{FECHA_FIN}_Planta{ID_PLANTA}.csv'
    )
    PROCESSED_CSV_PATH = os.path.join(
        BASE_DIR, 'data', '03_processed',
        f'DatosCombinados_{FECHA_INICIO}_a_{FECHA_FIN}_Planta{ID_PLANTA}.csv'
    )
    
    try:
        df = pd.read_csv(INTERIM_CSV_PATH, index_col=0, parse_dates=True)
        
        feriado_col = df['feriado'].copy() if 'feriado' in df.columns else None
        
        df = create_normalized_generation_feature(df)
        df = create_time_features(df)
        
        if feriado_col is not None:
            df['feriado'] = feriado_col
        
        os.makedirs(os.path.dirname(PROCESSED_CSV_PATH), exist_ok=True)
        df.to_csv(PROCESSED_CSV_PATH)
        
        print(f"✓ Datos procesados guardados en: {PROCESSED_CSV_PATH}")
        print(f"  Shape: {df.shape}, Columnas: {len(df.columns)}")
        
    except FileNotFoundError:
        print(f"\nERROR: No se encontró {INTERIM_CSV_PATH}")
        print(f"Ejecuta primero: python src/data/make_dataset.py")
    except Exception as e:
        print(f"\nERROR: {e}")