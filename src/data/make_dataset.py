"""
Módulo para el preprocesamiento y limpieza de datos.
"""

import pandas as pd
import json
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import os

# Agregar el directorio raíz al path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import get_plant_config

def load_solar_data_raw(json_file_path: str) -> pd.DataFrame:
    """Carga los datos de generación solar desde un archivo JSON."""
    
    # 1. Cargar JSON directamente
    with open(json_file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    df = pd.DataFrame(raw_data['data'])
    
    # Conversión y limpieza de tipos inicial
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
    df['potencia_maxima'] = pd.to_numeric(df['potencia_maxima'], errors='coerce')
    
    # Seleccionar columnas y establecer índice
    df = df[['fecha_hora', 'potencia_maxima', 'gen_real_mw']]
    df = df.set_index('fecha_hora').sort_index()
    
    return df

def load_meteo_data_raw(csv_file_path: str) -> pd.DataFrame:
    """Carga los datos meteorológicos desde un archivo CSV."""
    
    # Leer el CSV - detectar automáticamente el separador
    df = pd.read_csv(csv_file_path, sep=None, engine='python')
    
    # Detectar la columna de fecha (puede variar entre archivos)
    posibles_columnas_fecha = ['Fecha/Hora', 'Fecha', 'fecha_hora', 'fecha', 'datetime', 'time']
    columna_fecha = None
    
    for col in posibles_columnas_fecha:
        if col in df.columns:
            columna_fecha = col
            break
    
    if columna_fecha is None:
        columna_fecha = df.columns[0]
    
    # Limpiar la columna de fecha
    if df[columna_fecha].dtype == 'object':
        # Si la columna es texto, limpiar solo la parte de fecha
        df[columna_fecha] = df[columna_fecha].astype(str).str.split(';').str[0].str.strip()
    
    try:
        df[columna_fecha] = pd.to_datetime(df[columna_fecha], dayfirst=True, errors='coerce')
    except Exception:
        df[columna_fecha] = pd.to_datetime(df[columna_fecha], format='%d-%m-%Y %H:%M', errors='coerce')
    
    # Eliminar filas con fechas inválidas
    df = df.dropna(subset=[columna_fecha])
    
    # Establecer índice y ordenar
    df = df.set_index(columna_fecha).sort_index()
    
    return df

def combine_and_normalize_meteo(df_gensolar: pd.DataFrame, df_meteo: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Combina los datos por índice y aplica normalización (MinMaxScaler) a las
    características meteorológicas.
    
    IMPORTANTE: Detecta feriados ANTES de normalizar los datos.
    """
    
    # Validación de integridad de datos
    if verbose:
        print("\n" + "="*60)
        print("VALIDACIÓN DE INTEGRIDAD DE DATOS")
        print("="*60)
        print(f"📊 Datos Meteorológicos:")
        print(f"   Rango: {df_meteo.index.min()} a {df_meteo.index.max()}")
        print(f"   Registros: {len(df_meteo)}")
        print(f"\n⚡ Datos Generación Solar:")
        print(f"   Rango: {df_gensolar.index.min()} a {df_gensolar.index.max()}")
        print(f"   Registros: {len(df_gensolar)}")
    
    # Join (Unión) de los DataFrames
    df_final = df_meteo.join(df_gensolar, how='inner')

    if df_final.empty:
        raise ValueError("El DataFrame resultante está vacío. Verifica que las fechas/índices coincidan.")
    
    if verbose:
        print(f"\n🔗 Datos Combinados (inner join):")
        print(f"   Rango: {df_final.index.min()} a {df_final.index.max()}")
        print(f"   Registros: {len(df_final)}")
        
        perdida_meteo = len(df_meteo) - len(df_final)
        perdida_solar = len(df_gensolar) - len(df_final)
        
        if perdida_meteo > 0 or perdida_solar > 0:
            print(f"\n⚠️  ADVERTENCIA: PÉRDIDA DE DATOS DETECTADA")
            print(f"   Registros meteo perdidos: {perdida_meteo} ({perdida_meteo/len(df_meteo)*100:.1f}%)")
            print(f"   Registros solar perdidos: {perdida_solar} ({perdida_solar/len(df_gensolar)*100:.1f}%)")
            
            if df_meteo.index.min() < df_gensolar.index.min():
                print(f"   ❌ FALTA datos solares desde {df_meteo.index.min()} hasta {df_gensolar.index.min()}")
            if df_meteo.index.max() > df_gensolar.index.max():
                print(f"   ❌ FALTA datos solares desde {df_gensolar.index.max()} hasta {df_meteo.index.max()}")
            if df_gensolar.index.min() < df_meteo.index.min():
                print(f"   ❌ FALTA datos meteo desde {df_gensolar.index.min()} hasta {df_meteo.index.min()}")
            if df_gensolar.index.max() > df_meteo.index.max():
                print(f"   ❌ FALTA datos meteo desde {df_meteo.index.max()} hasta {df_gensolar.index.max()}")
        else:
            print(f"   ✅ Sin pérdida de datos")
        print("="*60 + "\n")

    # ⭐ PASO CRÍTICO: Detectar feriados ANTES de normalizar
    # Crear DataFrame temporal con valores originales para detección
    df_temp = pd.DataFrame({
        'glb': df_final['glb'].copy(),
        'gen_real_mw': df_final['gen_real_mw'].copy()
    }, index=df_final.index)
    
    # Detectar feriados y obtener la columna
    df_temp = detectar_feriados(df_temp, verbose=verbose)
    df_final['feriado'] = df_temp['feriado']

    # Variables meteorológicas a normalizar (del código original)
    columnas_normalizar = ['glb', 'dir', 'dif', 'sct', 'ghi', 'dirh', 'difh', 'dni', 'temp', 'vel']

    # Filtrar solo las columnas que existen
    columnas_existentes = [col for col in columnas_normalizar if col in df_final.columns]

    # Crear scaler y normalizar
    scaler = MinMaxScaler()
    df_final[columnas_existentes] = scaler.fit_transform(df_final[columnas_existentes])
    
    return df_final


def detectar_feriados(df: pd.DataFrame, umbral_radiacion: float = 200, 
                      hora_inicio: int = 11, hora_fin: int = 15, verbose: bool = False) -> pd.DataFrame:
    """
    Detecta días feriados/mantenimiento basándose en generación cero durante mediodía solar.
    
    Lógica: Si hay radiación solar significativa (mediodía) pero generación = 0,
    es probable que sea feriado, mantenimiento o falla de la planta.
    
    Args:
        df: DataFrame con datos de generación y radiación (antes de normalizar)
        umbral_radiacion: W/m² mínimo para considerar que hay sol (default: 200)
        hora_inicio: Hora inicio del mediodía solar (default: 11)
        hora_fin: Hora fin del mediodía solar (default: 15)
    
    Returns:
        DataFrame con columna adicional 'feriado' (0=día normal, 1=feriado/mantenimiento)
    """
    df_copy = df.copy()
    
    # Extraer hora del índice
    df_copy['hora'] = df_copy.index.hour
    df_copy['fecha'] = df_copy.index.date
    
    # Filtrar solo horas de mediodía solar (11:00 - 15:00)
    mediodias = df_copy[(df_copy['hora'] >= hora_inicio) & (df_copy['hora'] <= hora_fin)]
    
    # Para cada día, verificar si hubo radiación pero NO generación
    dias_feriado = []
    for fecha in mediodias['fecha'].unique():
        datos_dia = mediodias[mediodias['fecha'] == fecha]
        
        # Condiciones para ser feriado:
        # 1. Al menos una hora con radiación > umbral
        # 2. TODAS las horas del mediodía con generación ≈ 0
        tiene_sol = (datos_dia['glb'] > umbral_radiacion).any()
        sin_generacion = (datos_dia['gen_real_mw'] < 0.01).all()
        
        if tiene_sol and sin_generacion:
            dias_feriado.append(fecha)
    
    # Crear columna feriado (0 o 1)
    df_copy['feriado'] = df_copy['fecha'].isin(dias_feriado).astype(int)
    
    # Limpiar columnas temporales
    df_copy = df_copy.drop(columns=['hora', 'fecha'])
    
    if verbose and dias_feriado:
        print(f"\n🚨 Días con posible feriado/mantenimiento detectados: {len(dias_feriado)}")
        print(f"   Fechas: {sorted(dias_feriado)[:10]}{'...' if len(dias_feriado) > 10 else ''}")
    
    return df_copy

if __name__ == "__main__":

    ID_PLANTA = 346 # Solo necesitas cambiar esto!
    
    # Obtener configuración automática de la planta
    plant_config = get_plant_config(ID_PLANTA)
    FECHA_INICIO = plant_config['fecha_inicio']
    FECHA_FIN = plant_config['fecha_fin']

    # Obtener el directorio base del proyecto (2 niveles arriba de src/data/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    SOLAR_JSON_PATH = os.path.join(BASE_DIR, 'data', '01_raw', f'generacion_solar_{FECHA_INICIO}_a_{FECHA_FIN}_central_{ID_PLANTA}.json')
    
    # Buscar el archivo meteorológico (puede tener diferentes rangos de años)
    import glob
    meteo_pattern = os.path.join(BASE_DIR, 'data', '01_raw', f'Datos*_Planta{ID_PLANTA}.csv')
    meteo_files = glob.glob(meteo_pattern)
    if not meteo_files:
        print(f"\nERROR: No se encontró archivo meteorológico para planta {ID_PLANTA}")
        print(f"Patrón buscado: {meteo_pattern}")
        sys.exit(1)
    METEO_CSV_PATH = meteo_files[0]
    
    INTERIM_CSV_PATH = os.path.join(BASE_DIR, 'data', '02_interim', f'InterimCombinado_{FECHA_INICIO}_a_{FECHA_FIN}_Planta{ID_PLANTA}.csv')

    try:
        # Cargar datos crudos
        df_gensolar_raw = load_solar_data_raw(SOLAR_JSON_PATH)
        df_meteo_raw = load_meteo_data_raw(METEO_CSV_PATH)

        # Combinar y normalizar (la detección de feriados ocurre dentro)
        df_combined_normalized = combine_and_normalize_meteo(df_gensolar_raw, df_meteo_raw, verbose=True)

        # Guardar DataFrame combinado e intermedio
        os.makedirs(os.path.dirname(INTERIM_CSV_PATH), exist_ok=True)
        df_combined_normalized.to_csv(INTERIM_CSV_PATH)

        print(f"✓ Datos combinados y normalizados guardados en: {INTERIM_CSV_PATH}")
        
    except FileNotFoundError as e:
        print(f"\nERROR: No se encontró el archivo: {e.filename}")
    except ValueError as e:
        print(f"\nERROR DE DATOS: {e}")