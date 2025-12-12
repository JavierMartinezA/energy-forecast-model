import requests
import json
import time
import os
from typing import Optional, Dict, List, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

# === CONFIGURACIÓN ===
API_KEY = os.getenv("CEN_API_KEY")
if not API_KEY:
    raise ValueError("CEN_API_KEY environment variable is required")
BASE_URL = "https://sipub.api.coordinador.cl"

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "01_raw"


def get_session() -> requests.Session:
    """
    Configura una sesión HTTP con estrategia de reintentos automática.
    Esto maneja cortes intermitentes y reutiliza la conexión TCP.
    """
    session = requests.Session()
    retries = Retry(
        total=5,  # Aumentado: 5 reintentos
        backoff_factor=2,  # Aumentado: espera 2s, 4s, 8s, 16s, 32s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def consultar_generacion_solar(
    start_date: str,
    end_date: str,
    id_central: Optional[int] = None,
    api_key: str = API_KEY
) -> Optional[Dict[str, Any]]:
    """
    Consulta la generación solar real desde la API del CEN.
    
    Args:
        start_date: Fecha de inicio en formato 'YYYY-MM-DD'
        end_date: Fecha de fin en formato 'YYYY-MM-DD'
        id_central: ID de la central específica (opcional)
        api_key: Clave de API del CEN
    
    Returns:
        Diccionario con los datos de generación y metadatos, o None si falla
    """
    url = f"{BASE_URL}/generacion-real/v3/findByDate"
    headers = {"accept": "application/json"}
    
    # Parámetros base inmutables
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "tipoTecnologia": "Solar",
        "user_key": api_key,
        "page": 1
    }
    
    if id_central is not None:
        params["idCentral"] = id_central

    all_data: List[Dict] = []
    total_pages = 1
    
    # Context manager para asegurar cierre de conexión
    with get_session() as session:
        while params["page"] <= total_pages:
            try:
                response = session.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                
                payload = response.json()
                
                # Manejo de estructura de respuesta
                current_data = payload.get('data', [])
                if not current_data:
                    break
                
                all_data.extend(current_data)
                total_pages = payload.get('totalPages', 1)
                params["page"] += 1
                time.sleep(0.1)  # Reducido de 0.5s a 0.1s

            except requests.exceptions.Timeout:
                return None
                
            except requests.exceptions.ConnectionError:
                return None
                
            except requests.exceptions.HTTPError:
                if response.status_code == 429:
                    time.sleep(10)
                    continue
                if response.status_code in [401, 403, 502, 503, 504]:
                    if response.status_code in [502, 503, 504]:
                        time.sleep(5)
                        continue
                    return None
                return None
                
            except Exception:
                return None

    return {
        "data": all_data,
        "startDate": start_date,
        "endDate": end_date,
        "totalPages": total_pages,
        "totalRecords": len(all_data)
    }


def guardar_a_json(data: Dict, filename: str, output_dir: Path = RAW_DATA_PATH) -> None:
    """
    Guarda los datos en formato JSON en el directorio de datos crudos.
    
    Args:
        data: Diccionario con los datos a guardar
        filename: Nombre del archivo (sin ruta completa)
        output_dir: Directorio donde guardar el archivo
    """
    try:
        # Crear el directorio si no existe
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError:
        raise


def consultar_centrales(
    id_central: Optional[int] = None,
    api_key: str = API_KEY
) -> Optional[Dict[str, Any]]:
    """
    Consulta información de centrales eléctricas desde la API del CEN.
    
    Args:
        id_central: ID de la central específica (opcional)
        api_key: Clave de API del CEN
    
    Returns:
        Diccionario con los datos de centrales y metadatos, o None si falla
    """
    url = f"{BASE_URL}/centrales/v4/findByDate"
    headers = {"accept": "application/json"}
    
    # Parámetros base inmutables
    params = {
        "user_key": api_key,
        "page": 1
    }
    
    if id_central is not None:
        params["idCentral"] = id_central

    all_data: List[Dict] = []
    total_pages = 1
    
    # Context manager para asegurar cierre de conexión
    with get_session() as session:
        while params["page"] <= total_pages:
            try:
                response = session.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                
                payload = response.json()
                current_data = payload.get('data', [])
                if not current_data:
                    break
                
                all_data.extend(current_data)
                total_pages = payload.get('totalPages', 1)
                params["page"] += 1
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                error_str = str(e)
                if "429" in error_str:
                    time.sleep(10)
                    continue
                if any(code in error_str for code in ["502", "503", "504"]):
                    time.sleep(5)
                    continue
                return None

    return {
        "data": all_data,
        "totalPages": total_pages,
        "totalRecords": len(all_data)
    }


def filtrar_centrales_solares(
    data: Dict[str, Any],
    year_limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Filtra centrales solares de un conjunto de datos.
    
    Args:
        data: Diccionario con datos de centrales
        year_limit: Año límite para filtrar (solo centrales anteriores a este año)
    
    Returns:
        Diccionario con datos filtrados
    """
    all_data = data.get('data', [])
    filtered_data = []
    
    for item in all_data:
        tipo_central = item.get('tipo_central', '')
        fecha_oper = item.get('fecha_ent_oper', '')
        
        # Verificar tipo solar
        if tipo_central == 'Solares' or tipo_central == 'Eolica':
            # Si no hay límite de año, incluir todas las solares
            if year_limit is None:
                filtered_data.append(item)
            # Si hay límite de año, verificar fecha
            elif fecha_oper:
                try:
                    # Extraer año de la fecha formato DD-MM-YYYY
                    parts = fecha_oper.split('-')
                    if len(parts) == 3:
                        year = int(parts[2])
                        if year < year_limit:
                            filtered_data.append(item)
                except (ValueError, IndexError):
                    pass
    
    return {
        "data": filtered_data,
        "totalPages": data.get('totalPages', 1),
        "totalRecords": len(filtered_data)
    }


def extraer_centrales_solares(
    year_limit: Optional[int] = None,
    central_id: Optional[int] = None,
    save_file: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Función principal para extraer información de centrales solares.
    
    Args:
        year_limit: Año límite para filtrar (solo centrales anteriores a este año)
        central_id: ID de la central específica (opcional)
        save_file: Si True, guarda los datos en un archivo JSON
    
    Returns:
        Diccionario con los datos de centrales solares, o None si falla
    """
    result = consultar_centrales(id_central=central_id)
    
    if result:
        # Filtrar centrales solares
        result = filtrar_centrales_solares(result, year_limit=year_limit)
        
        if save_file and result:
            # Construir nombre de archivo descriptivo
            if year_limit:
                filename = f"centrales_solares_pre_{year_limit}.json"
            else:
                filename = "centrales_solares_todas.json"
            
            guardar_a_json(result, filename)
    
    return result


def extraer_datos_solar(
    start_date: str,
    end_date: str,
    central_id: Optional[int] = None,
    save_file: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Función principal para extraer datos de generación solar.
    
    Args:
        start_date: Fecha de inicio en formato 'YYYY-MM-DD'
        end_date: Fecha de fin en formato 'YYYY-MM-DD'
        central_id: ID de la central específica (opcional)
        save_file: Si True, guarda los datos en un archivo JSON
    
    Returns:
        Diccionario con los datos extraídos, o None si falla
    """
    result = consultar_generacion_solar(
        start_date=start_date,
        end_date=end_date,
        id_central=central_id
    )
    
    if result and save_file:
        # Construir nombre de archivo descriptivo
        if central_id:
            filename = f"generacion_solar_{start_date}_a_{end_date}_central_{central_id}.json"
        else:
            filename = f"generacion_solar_{start_date}_a_{end_date}_todas_centrales.json"
        
        guardar_a_json(result, filename)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extrae datos de generación solar desde API CEN')
    parser.add_argument('start_date', help='Fecha inicio (YYYY-MM-DD)')
    parser.add_argument('end_date', help='Fecha fin (YYYY-MM-DD)')
    parser.add_argument('--central-id', type=int, help='ID de la central específica')
    parser.add_argument('--centrales', action='store_true', help='Extraer info de centrales solares')
    parser.add_argument('--year-limit', type=int, help='Filtrar centrales anteriores a este año')
    
    args = parser.parse_args()
    
    if args.centrales:
        extraer_centrales_solares(year_limit=args.year_limit)
    else:
        extraer_datos_solar(
            start_date=args.start_date,
            end_date=args.end_date,
            central_id=args.central_id
        )
