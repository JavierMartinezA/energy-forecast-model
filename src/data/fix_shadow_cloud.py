"""
Script para corregir valores de shadow y cloud en archivos CSV.
Convierte de escala 0-100 a escala 0-1.
"""

import pandas as pd
import os
import time
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "01_raw"

def fix_shadow_cloud_columns(csv_path: str, verbose: bool = False):
    """
    Corrige las columnas shadow y cloud convirtiéndolas a valores binarios 0 y 1.
    
    Shadow: Convierte de escala 0-100 a 0-1 (divide por 100)
    Cloud: Puede estar como string "0" o "1.00", convierte a entero 0 o 1
    
    Args:
        csv_path: Ruta al archivo CSV
        verbose: Si True, muestra información de progreso
    """
    if verbose:
        print(f"Procesando: {os.path.basename(csv_path)}")
    
    df = pd.read_csv(csv_path, sep=';')
    modificado = False
    
    if 'shadow' in df.columns:
        df['shadow'] = pd.to_numeric(df['shadow'], errors='coerce')
        max_val = df['shadow'].max()
        
        if max_val > 1:
            df['shadow'] = df['shadow'] / 100
            if verbose:
                print(f"  ✓ Shadow: {max_val} → {df['shadow'].max()} (dividido por 100)")
            modificado = True
    
    if 'cloud' in df.columns:
        df['cloud'] = pd.to_numeric(df['cloud'], errors='coerce')
        max_val = df['cloud'].max()
        
        if max_val > 1:
            df['cloud'] = df['cloud'] / 100
            if verbose:
                print(f"  ✓ Cloud: {max_val} → {df['cloud'].max()} (dividido por 100)")
        
        df['cloud'] = df['cloud'].round().astype(int)
        if verbose:
            print(f"  ✓ Cloud final: {sorted(df['cloud'].unique())} (binario)")
        modificado = True
    
    if not modificado:
        return df
    
    temp_path = csv_path.replace('.csv', '_temp.csv')
    df.to_csv(temp_path, sep=';', index=False)
    
    try:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        os.rename(temp_path, csv_path)
        if verbose:
            print(f"  ✓ Archivo actualizado exitosamente\n")
    except PermissionError:
        if verbose:
            print(f"  ⚠ No se pudo reemplazar. Archivo temporal guardado: {os.path.basename(temp_path)}\n")
    
    return df


if __name__ == "__main__":
    archivos = [
        RAW_DATA_PATH / "Datos2013-2015_Planta239.csv",
        RAW_DATA_PATH / "Datos2013-2015_Planta309.csv",
        RAW_DATA_PATH / "Datos2013-2015_Planta346.csv",
        RAW_DATA_PATH / "Datos2014-2015_Planta305.csv",
    ]
    
    archivos_procesados = 0
    
    for archivo in archivos:
        if archivo.exists():
            try:
                fix_shadow_cloud_columns(str(archivo), verbose=True)
                archivos_procesados += 1
            except Exception as e:
                print(f"❌ Error procesando {archivo.name}: {e}\n")
        else:
            print(f"⚠ Archivo no encontrado: {archivo.name}\n")
    
    if archivos_procesados > 0:
        print(f"\n✓ Procesados: {archivos_procesados}/{len(archivos)} archivos")
