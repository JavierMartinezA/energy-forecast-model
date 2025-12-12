"""
Configuración centralizada de plantas solares.

Este módulo contiene la información de cada planta (ID y fechas de datos disponibles).
Solo necesitas especificar el ID_PLANTA y el resto de la información se obtiene automáticamente.

Uso:
    from src.config import get_plant_config
    
    config = get_plant_config(239)
    print(config['fecha_inicio'])  # '2013-08-08'
    print(config['fecha_fin'])     # '2015-08-08'
"""

# Configuración de plantas validadas con datos completos
PLANTAS_CONFIG = {
    239: {
        'nombre': 'SDGX01',
        'potencia_mw': 1.28,
        'fecha_inicio': '2013-08-08',
        'fecha_fin': '2015-08-08',
        'ubicacion': 'Andacollo, Coquimbo'
    },
    309: {
        'nombre': 'Tambo Real',
        'potencia_mw': 2.94,
        'fecha_inicio': '2013-01-01',
        'fecha_fin': '2015-01-01',
        'ubicacion': 'Vicuña, Coquimbo'
    },
    346: {
        'nombre': 'Esperanza',
        'potencia_mw': 2.88,
        'fecha_inicio': '2014-01-01',
        'fecha_fin': '2015-12-20',
        'ubicacion': 'Diego de Almagro, Atacama'
    },
    305: {
        'nombre': 'Las Terrazas',
        'potencia_mw': 2.99,
        'fecha_inicio': '2014-10-30',
        'fecha_fin': '2015-10-30',
        'ubicacion': 'Tierra Amarilla, Atacama'
    }
}


def get_plant_config(plant_id: int) -> dict:
    """
    Obtiene la configuración completa de una planta por su ID.
    
    Args:
        plant_id (int): ID de la planta (239, 305, 309, 346)
    
    Returns:
        dict: Diccionario con toda la configuración de la planta
        
    Raises:
        ValueError: Si el ID de planta no está configurado
    
    Ejemplo:
        >>> config = get_plant_config(239)
        >>> print(config['fecha_inicio'])
        '2013-08-08'
    """
    if plant_id not in PLANTAS_CONFIG:
        plantas_disponibles = ', '.join(map(str, PLANTAS_CONFIG.keys()))
        raise ValueError(
            f"Planta {plant_id} no configurada. "
            f"Plantas disponibles: {plantas_disponibles}"
        )
    
    config = PLANTAS_CONFIG[plant_id].copy()
    config['id'] = plant_id
    return config


def get_available_plants() -> list:
    """
    Retorna la lista de IDs de plantas configuradas.
    
    Returns:
        list: Lista de IDs de plantas disponibles
    """
    return list(PLANTAS_CONFIG.keys())


def print_plant_info(plant_id: int = None):
    """
    Imprime información de una o todas las plantas configuradas.
    
    Args:
        plant_id (int, optional): ID de planta específica. Si es None, muestra todas.
    """
    if plant_id is None:
        print("\n" + "="*70)
        print("PLANTAS SOLARES CONFIGURADAS")
        print("="*70)
        for pid in sorted(PLANTAS_CONFIG.keys()):
            config = PLANTAS_CONFIG[pid]
            print(f"\nPlanta {pid}: {config['nombre']}")
            print(f"  Potencia: {config['potencia_mw']} MW")
            print(f"  Período: {config['fecha_inicio']} a {config['fecha_fin']}")
            print(f"  Ubicación: {config['ubicacion']}")
        print("="*70)
    else:
        config = get_plant_config(plant_id)
        print(f"\nPlanta {plant_id}: {config['nombre']}")
        print(f"  Potencia: {config['potencia_mw']} MW")
        print(f"  Período: {config['fecha_inicio']} a {config['fecha_fin']}")
        print(f"  Ubicación: {config['ubicacion']}")


if __name__ == "__main__":
    # Mostrar todas las plantas configuradas
    print_plant_info()
    
    # Ejemplo de uso
    print("\n" + "="*70)
    print("EJEMPLO DE USO")
    print("="*70)
    config = get_plant_config(239)
    print(f"\nconfig = get_plant_config(239)")
    print(f"config['fecha_inicio'] = '{config['fecha_inicio']}'")
    print(f"config['fecha_fin'] = '{config['fecha_fin']}'")
