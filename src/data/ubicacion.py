"""
Módulo para obtener información de ubicación de centrales solares.

Extrae coordenadas UTM de centrales desde el archivo JSON y las convierte
a coordenadas geográficas (latitud/longitud) para generar enlaces de Google Maps.

Requiere: pip install utm
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Tuple

try:
    import utm
except ImportError:
    raise ImportError(
        "La biblioteca 'utm' no está instalada. "
        "Instala con: pip install utm"
    )


# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CENTRALES_JSON_PATH = PROJECT_ROOT / "data" / "01_raw" / "centrales_solares_pre_2017.json"


def cargar_datos_centrales(json_path: Path = CENTRALES_JSON_PATH) -> Dict:
    """
    Carga los datos de centrales desde el archivo JSON.
    
    Args:
        json_path: Ruta al archivo JSON con datos de centrales
    
    Returns:
        Diccionario con los datos de centrales
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró el archivo: {json_path}\n"
            f"Ejecuta primero: python src/data/extract.py centrales"
        )


def buscar_central_por_id(id_central: int, json_path: Path = CENTRALES_JSON_PATH) -> Optional[Dict]:
    """
    Busca una central por su ID en el archivo JSON.
    
    Args:
        id_central: ID de la central a buscar
        json_path: Ruta al archivo JSON con datos de centrales
    
    Returns:
        Diccionario con los datos de la central, o None si no se encuentra
    """
    datos = cargar_datos_centrales(json_path)
    
    for central in datos.get('data', []):
        if central.get('id_central') == id_central:
            return central
    
    return None


def extraer_coordenadas_utm(central: Dict) -> Tuple[float, float, str]:
    """
    Extrae las coordenadas UTM de una central.
    
    Args:
        central: Diccionario con los datos de la central
    
    Returns:
        Tupla (coordenada_este, coordenada_norte, zona_huso)
    
    Raises:
        ValueError: Si las coordenadas no están disponibles o son inválidas
    """
    coord_este_str = central.get('coordenada_este', '')
    coord_norte_str = central.get('coordenada_norte', '')
    zona_huso = central.get('zona_huso', '')
    
    if not coord_este_str or not coord_norte_str:
        raise ValueError(
            f"Coordenadas no disponibles para la central {central.get('central', 'desconocida')}"
        )
    
    try:
        # Convertir strings a floats (reemplazar comas por puntos)
        coord_este = float(coord_este_str.replace(',', '.'))
        coord_norte = float(coord_norte_str.replace(',', '.'))
    except (ValueError, AttributeError):
        raise ValueError(
            f"Formato de coordenadas inválido: Este={coord_este_str}, Norte={coord_norte_str}"
        )
    
    return coord_este, coord_norte, zona_huso


def utm_a_latlon(este: float, norte: float, zona_huso: str) -> Tuple[float, float]:
    """
    Convierte coordenadas UTM a latitud/longitud (WGS84).
    
    Usa la biblioteca utm para conversión precisa.
    
    Args:
        este: Coordenada Este (Easting) en metros
        norte: Coordenada Norte (Northing) en metros
        zona_huso: Zona UTM (ej: "19J", "19K", "19 J")
    
    Returns:
        Tupla (latitud, longitud)
    """
    try:
        # Extraer número de zona y letra
        zona_str = zona_huso.strip().replace(' ', '')
        zona_num = int(''.join(filter(str.isdigit, zona_str)))
        zona_letra = ''.join(filter(str.isalpha, zona_str))
        
        # Determinar hemisferio: letras N-Z = Norte, C-M = Sur
        # Chile usa principalmente J, K, H (hemisferio sur)
        hemisferio = 'S' if zona_letra in 'CDEFGHJKLM' else 'N'
        
        # Convertir UTM a lat/lon
        lat, lon = utm.to_latlon(este, norte, zona_num, northern=(hemisferio == 'N'))
        
        return lat, lon
        
    except Exception as e:
        raise ValueError(f"Error en conversión UTM: {e}. Zona: {zona_huso}, Este: {este}, Norte: {norte}")


def generar_link_google_maps(lat: float, lon: float) -> str:
    """
    Genera un enlace de Google Maps para coordenadas específicas.
    
    Args:
        lat: Latitud
        lon: Longitud
    
    Returns:
        URL de Google Maps
    """
    return f"https://www.google.com/maps?q={lat},{lon}"


def obtener_ubicacion_central(id_central: int, verbose: bool = False) -> Optional[Dict]:
    """
    Función principal: obtiene coordenadas y genera link de Google Maps para una central.
    
    Args:
        id_central: ID de la central solar
        verbose: Si True, imprime información en consola
    
    Returns:
        Diccionario con información de ubicación:
        {
            'id_central': int,
            'nombre': str,
            'coordenada_este': float,
            'coordenada_norte': float,
            'zona_huso': str,
            'latitud': float,
            'longitud': float,
            'google_maps_url': str
        }
    """
    central = buscar_central_por_id(id_central)
    
    if not central:
        if verbose:
            print(f"❌ No se encontró la central con ID {id_central}")
        return None
    
    try:
        # Extraer coordenadas UTM
        coord_este, coord_norte, zona_huso = extraer_coordenadas_utm(central)
        
        # Convertir a lat/lon
        lat, lon = utm_a_latlon(coord_este, coord_norte, zona_huso)
        
        # Generar link
        google_maps_url = generar_link_google_maps(lat, lon)
        
        # Preparar resultado
        resultado = {
            'id_central': id_central,
            'nombre': central.get('central', 'N/A'),
            'propietario': central.get('propietario', 'N/A'),
            'region': central.get('region', 'N/A'),
            'comuna': central.get('comuna', 'N/A'),
            'coordenada_este': coord_este,
            'coordenada_norte': coord_norte,
            'zona_huso': zona_huso,
            'latitud': round(lat, 6),
            'longitud': round(lon, 6),
            'google_maps_url': google_maps_url
        }
        
        if verbose:
            print("="*70)
            print(f"UBICACIÓN DE CENTRAL SOLAR")
            print("="*70)
            print(f"ID:              {resultado['id_central']}")
            print(f"Nombre:          {resultado['nombre']}")
            print(f"Propietario:     {resultado['propietario']}")
            print(f"Región:          {resultado['region']}")
            print(f"Comuna:          {resultado['comuna']}")
            print()
            print(f"COORDENADAS UTM:")
            print(f"  Este:          {resultado['coordenada_este']:.2f} m")
            print(f"  Norte:         {resultado['coordenada_norte']:.2f} m")
            print(f"  Zona:          {resultado['zona_huso']}")
            print()
            print(f"COORDENADAS GEOGRÁFICAS:")
            print(f"  Latitud:       {resultado['latitud']}°")
            print(f"  Longitud:      {resultado['longitud']}°")
            print()
            print(f"GOOGLE MAPS:")
            print(f"  {resultado['google_maps_url']}")
            print("="*70)
        
        return resultado
        
    except ValueError as e:
        if verbose:
            print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        try:
            id_central = int(sys.argv[1])
        except ValueError:
            print("❌ Error: El ID de central debe ser un número entero")
            print("Uso: python src/data/ubicacion.py <ID_CENTRAL>")
            sys.exit(1)
    else:
        id_central = 305
        print(f"ℹ️  Usando ID por defecto: {id_central}")
        print("Uso: python src/data/ubicacion.py <ID_CENTRAL>\n")
    
    resultado = obtener_ubicacion_central(id_central, verbose=True)
    
    if not resultado:
        sys.exit(1)