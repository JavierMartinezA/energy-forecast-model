# Energy Forecast Model - Predicción Solar Multi-Planta

Sistema de predicción de generación solar con Deep Learning para plantas fotovoltaicas chilenas. Predice **48 horas** usando modelos BiLSTM + CNN.

## 🚀 Inicio Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar pipeline completo
python run.py
```

## 📊 Características

- **Dual-Stream BiLSTM + CNN**: Combina histórico (11-24h) con pronóstico meteorológico (48h)
- **Tri-Stream con Gating**: Modela explícitamente estado operativo (feriados, mantenimientos)
- **Multi-planta**: Validado en 4 plantas (239, 305, 309, 346) - Sistema adaptable a cualquier planta
- **41 modelos evaluados**: Benchmark exhaustivo de arquitecturas y configuraciones

## 📁 Estructura

```
├── data/                    # Datos (raw → interim → processed)
├── models/                  # Modelos entrenados (.keras)
├── figures/                 # Métricas y visualizaciones
├── notebooks/               # Análisis exploratorio
├── src/                     # Código fuente
│   ├── data/                # Procesamiento de datos
│   ├── features/            # Ingeniería de características
│   └── models/              # Entrenamiento y predicción
└── run.py                   # Pipeline completo
```

## 🎯 Resultados

| Planta | Mejor Modelo | MAE Test |
|--------|--------------|----------|
| 346 | Tri-Stream 11h | 0.0224 (2.2%) |
| 239 | Dual-Stream 24h | 0.0638 (6.4%) |
| 305 | Tri-Stream 24h | 0.0454 (4.5%) |
| 309 | Dual-Stream 11h | 0.1438 (14.4%) |

**Hallazgo clave**: Ventana de 11h logra mismo rendimiento que 24h, 2.5x más rápido.

## 💡 Tecnologías

- Python 3.13+ | TensorFlow 2.20+ | Keras
- pandas, numpy, scikit-learn, matplotlib

## 📝 Nota

Este proyecto fue desarrollado con asistencia de IA: **Claude  Sonnet 4.5** y **Gemini 3 pro**.

---

## 🏗️ Estructura del Proyecto

```
project-solar-power/
├── data/                                # Datos en diferentes etapas
│   ├── 01_raw/                          # Datos originales sin procesar
│   │   ├── generacion_solar_*.json      # Datos de generación solar (API CEN)
│   │   ├── Datos2013-2015_Planta*.csv   # Datos meteorológicos por planta
│   │   └── centrales_solares_pre_2017.json  # Metadata de plantas
│   ├── 02_interim/                      # Datos combinados y normalizados
│   │   └── InterimCombinado_*.csv       # Solar + Meteo normalizados
│   └── 03_processed/                    # Datos finales con features
│       └── DatosCombinados_*.csv        # Listo para modelado
│
├── figures/                             # Gráficos y métricas de resultados
│   ├── dual_stream_lstm_cnn_*_history.csv    # Historial entrenamiento
│   ├── training_summary.csv              # Resumen de todos los modelos
│   ├── predicciones_ejemplos.png         # Comparación predicción vs real
│   ├── error_por_hora.png                # MAE por hora de predicción
│   └── training_history.png              # Curvas de entrenamiento
│
├── models/                              # Modelos entrenados (.keras)
│   ├── dual_stream_lstm_cnn_239_24h_48h.keras  # Modelo Dual-Stream planta 239
│   ├── dual_stream_lstm_cnn_309_11h_48h.keras  # Modelo Dual-Stream planta 309
│   ├── dual_stream_lstm_cnn_346_24h_48h.keras  # Modelo Dual-Stream planta 346
│   └── tri_stream_gating_239_24h_48h.keras     # Modelo Tri-Stream planta 239
│
├── notebooks/                           # Jupyter notebooks refactorizados (reporte técnico)
│   ├── exploracion_datos.ipynb          # EDA - Key Visual Insights (serie temporal, heatmaps, correlación, ACF/PACF, outliers)
│   ├── exploracion_modelos.ipynb        # Evaluación - Métricas, curvas entrenamiento, predicciones 48h, residuales
│   └── exploracion_resultados.ipynb     # Comparación Multi-Modelo - Rankings, eficiencia, heatmaps (Dual/Tri × 4 plantas × 2 ventanas)
│
├── src/                                 # Código fuente
│   ├── data/                            # Scripts de procesamiento de datos
│   │   ├── extract.py                   # [OPCIONAL] Descarga datos de API CEN
│   │   ├── make_dataset.py              # Combina y normaliza datos
│   │   ├── fix_shadow_cloud.py          # Corrección de datos específicos
│   │   └── ubicacion.py                 # Info geográfica de plantas
│   │
│   ├── features/                        # Ingeniería de características
│   │   └── build_features.py            # Crea features temporales cíclicas
│   │
│   └── models/                          # Modelos y predicciones
│       ├── windowing_utils.py           # Creación de ventanas temporales (Dual/Tri-Stream)
│       ├── train_model.py               # Entrenamiento Dual-Stream
│       ├── train_alternativo.py         # Entrenamiento Tri-Stream con Gating
│       ├── predict_model.py             # Evaluación y visualización
│       └── Trainmodelo_multiplanta.py   # [Reservado] Entrenamiento batch
│
├── run.py                               # Orquestador del pipeline completo
├── validar_splits.py                    # Validación de splits temporales
├── verificar_datos.py                   # Verificación de integridad de datos
├── requirements.txt                     # Dependencias del proyecto
└── README.md                            # Este archivo
```

