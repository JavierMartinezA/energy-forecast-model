# AI Coding Agent Instructions - Solar Power Prediction

## Project Overview

This is a **multi-plant time-series forecasting system** for solar power generation using **Dual-Stream BiLSTM + CNN 1D** and **Tri-Stream with Gating Multiplicativo** architectures. It predicts normalized solar generation **48 hours ahead** using configurable historical data (4-24h optimizable) and 48 hours of meteorological forecasts, aligned with **Chilean Electric Coordinator (CEN) requirements**.

**Key characteristics:**
- **Multi-plant support**: Automated pipeline works with plants 239, 309, 346 by configuring `ID_PLANTA`, `FECHA_INICIO`, `FECHA_FIN`
- **Production-ready code**: Verbose mode (`verbose=False` by default) for CI/CD integration
- **Dual-Stream (baseline)**: BiLSTM + CNN 1D for general forecasting
- **Tri-Stream (advanced)**: Adds explicit operational state modeling via gating mechanism
- **Optimized features**: 13 features (8 redundant eliminated: ghi, dir, dirh, difh, sct, etc.)
- **Data leakage prevention**: Chronological splits WITH GAPS between train/val/test
- Spanish language codebase (comments, variables, documentation)
- Structured ML pipeline: raw → interim → processed → model → predictions

## Architecture & Data Flow

### Pipeline Sequence (Critical Order)

1. **`src/data/make_dataset.py`** → Loads raw solar + meteorological data, normalizes with MinMaxScaler, saves to `data/02_interim/` (verbose flag available)
2. **`src/features/build_features.py`** → Creates `gen_normalizada` target and cyclical time features (sin/cos encodings), saves to `data/03_processed/`
3. **`src/models/train_model.py`** → Creates windowed sequences WITH GAPS, trains dual-stream model, saves to `models/` (verbose flag available)
4. **`src/models/train_alternativo.py`** → Alternative Tri-Stream architecture with gating mechanism (verbose flag available)
5. **`src/models/predict_model.py`** → Generates predictions and visualizations, saves to `figures/` (verbose flag available)

**Run entire pipeline:** `python run.py` (orchestrates all steps with error handling)

### Model Architectures

#### Dual-Stream (Baseline)
```
Input 1 (Past): [in_steps × 13 features] → BiLSTM(64) → BiLSTM(32) → Dropout(0.1)
Input 2 (Future): [48 timesteps × 12 features] → Conv1D(32, k=3) → Flatten → Dense(32)
                                                      ↓
                                    Concatenate → Dense(64) → Dropout(0.1) → Dense(48)
```

#### Tri-Stream with Gating (Advanced)
```
Input 1 (Past):   [in_steps × features] → BiLSTM(64) → BiLSTM(32) → Dropout(0.1)
Input 2 (Future): [48 × features] → Conv1D(32, k=3) → Flatten → Dense(32)
Input 3 (Binary): [48 × binary_features] → Conv1D(16, k=3) → Flatten → Dense(16) → Dense(48, sigmoid)
                                    ↓                                                      ↓
                          Concatenate(1,2) → Dense(64) → Dense(48) [Generation Potential]
                                                              ↓
                                                    Multiply(Potential, Gate) → Output
```

**Why Tri-Stream?**
- Stream 1 (LSTM): Captures temporal dependencies in historical patterns
- Stream 2 (CNN): Extracts local patterns from meteorological forecasts
- Stream 3 (Gating): Models operational state (feriado, mantenimiento, etc.)
- Gate modulates generation potential: gate≈1 (normal), gate≈0 (inactive), gate∈(0,1) (degraded)

### Feature Engineering Pattern

**13 total features** (12 meteorological + 1 target):
- **Radiation (optimized):** `glb, dni, dif` (3 non-redundant features)
  - ❌ Eliminated: `ghi` (r>0.99 with glb), `dir`, `dirh` (redundant with dni), `difh` (redundant with dif), `sct` (low correlation)
- **Weather:** `temp, vel, shadow, cloud` (4 features)
- **Cyclical time encoding:** `hora_sin, hora_cos, mes_sin, mes_cos, dia_año_sin, dia_año_cos` (6 features)
- **Target:** `gen_normalizada = gen_real_mw / potencia_maxima` (normalized 0-1)
- **Binary features (Tri-Stream only):** `feriado` for gating mechanism

**Critical:** Time features use sin/cos encoding to capture cyclical patterns (24h, 12 months, 365 days). Never use raw hour/month values.

## Key Conventions & Patterns

### Naming Conventions
- Variables/functions: `snake_case` (Spanish names common: `gen_normalizada`, `fecha_hora`)
- File naming: `{Stage}{Description}_{FECHA_INICIO}_a_{FECHA_FIN}_Planta{ID}.csv`
- Model files: `.keras` format (TensorFlow 2.20+)

### Path Resolution Pattern
```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Always resolves to project root from any src/ subdirectory
```
**Why:** All scripts in `src/` subdirectories use 3-level parent resolution to find project root.

### Data Windowing Logic (`windowing_utils.py`)

```python
# Creates sliding windows for time series
for i in range(n_past, len(data_x) - n_future + 1):
    X_past.append(data_x[i - n_past:i, :])           # Historical: n_past steps BEFORE i
    X_future.append(data_x[i:i + n_future, future_feat_indices])  # Forecast: 48 steps FROM i
    Y_target.append(data_y[i:i + n_future, 0])       # Target: 48 values FROM i
```

**Critical insight:** Index `i` is the prediction point. Past window is `[i-n_past:i)`, future window is `[i:i+48)` for 48h prediction horizon. Future features exclude `gen_normalizada` (column index 0).

**Feature optimization**: Only 12 features in future stream (13 total - target = 12), down from 18 (19 - target).

### Chronological Split WITH GAPS (Prevents Data Leakage!)

**CRITICAL CHANGE**: Splits now include gaps to prevent overlap between windows:

```python
# Calculate split indices on ORIGINAL dataset (before windowing)
window_size = in_steps + out_steps  # Total window size

train_end_idx = int(total_timesteps * 0.70)        # First 70%
val_start_idx = train_end_idx + (window_size - 1)  # GAP: prevents overlap
val_end_idx = int(total_timesteps * 0.85)          # Next 15%  
test_start_idx = val_end_idx + (window_size - 1)   # GAP: prevents overlap
# Test: Last 15%

# Split BEFORE windowing
df_train = df.iloc[:train_end_idx]
df_val = df.iloc[val_start_idx:val_end_idx]
df_test = df.iloc[test_start_idx:]

# Then create windows separately for each split
```

**Why gaps are critical:** Without gaps, the last training window could include timestamps that appear in the first validation window's future forecast, causing data leakage. Gaps of `window_size - 1` guarantee complete temporal separation.

**Validation**: Use `validar_splits.py` to verify no overlap exists.

## Production-Ready Code Patterns

### Verbose Flag Pattern (Refactorización 2024-12)

**Principio:** "Silencio por defecto, verbosidad bajo demanda"

All major functions now support a `verbose` parameter for controlling output:

```python
def function_name(..., verbose: bool = False):
    """
    Args:
        verbose: Si True, muestra información detallada de progreso
    """
    # Prints condicionales (solo si verbose=True)
    if verbose:
        print("🔍 Información de debug detallada")
    
    # ... Lógica principal (siempre ejecuta) ...
    
    # Prints críticos (siempre visibles)
    print("✅ Operación completada exitosamente")
```

**Functions with verbose support:**
- `src/data/make_dataset.py`: `verbose=False` (default), shows only critical confirmations
- `src/models/train_model.py`: `train_and_evaluate_model(..., verbose=False)`
- `src/models/train_alternativo.py`: `train_and_evaluate_model(..., verbose=False)`
- `src/models/predict_model.py`: All visualization functions support `verbose`
- `src/models/windowing_utils.py`: `apply_plant_state_postprocessing(..., verbose=False)`

**Usage:**
```python
# Modo producción (silencioso)
train_and_evaluate_model(input_path, output_path, model_name)
# Output: 🚀 Entrenando... → ✅ Completado (3 líneas)

# Modo debugging (verbose)
train_and_evaluate_model(input_path, output_path, model_name, verbose=True)
# Output: Carga, splits, shapes, progreso, métricas... (30+ líneas)
```

**Benefits:**
- CI/CD friendly: Minimal logs in production
- Debugging enabled: Set `verbose=True` when needed
- Retrocompatible: Default `False` maintains API compatibility
- Consistent pattern: Applied across entire codebase

**See:** `docs/REFACTORING_SUMMARY.md` for complete refactorization report (87.5% output reduction)

## Configuration Parameters

**Global constants** (configurable per plant in each script):
```python
ID_PLANTA = 239  # Plant ID: 239, 309, 346 (validated plants with complete data)
FECHA_INICIO = '2013-08-08'  # Data start date (plant-specific)
FECHA_FIN = '2015-08-08'      # Data end date (plant-specific)
```

**Validated plants with complete datasets**:
- **239**: SDGX01 (1.28 MW) - Period: 2013-08-08 to 2015-08-08
- **309**: Tambo Real (2.94 MW) - Period: 2013-01-01 to 2015-01-01
- **346**: Esperanza (2.88 MW) - Period: 2014-01-01 to 2015-12-20

**Window sizes** (optimized configuration in `train_model.py`):
```python
in_steps = 24   # Historical window (hours) - configurable 4-24h
                # Optimization: 4h achieves same MAE as 24h, 5x faster
                # 11h: balance between performance and context
                # 24h: standard, captures full daily cycle
out_steps = 48  # Prediction horizon (hours) - FIXED (CEN requirement)
```

**Training hyperparameters:**
- Optimizer: Adam (lr=0.001)
- Loss: MSE, Metric: MAE
- Batch size: 32
- EarlyStopping: patience=15, monitor='val_loss'

## Development Workflows

### Running Tests/Training
```powershell
# Full pipeline
python run.py

# Individual steps (must run in order)
python src/data/make_dataset.py
python src/features/build_features.py
python src/models/train_model.py
python src/models/predict_model.py
```

### Debugging Data Issues
- Check `data/02_interim/` for normalization issues
- Check `data/03_processed/` for feature engineering issues
- Verify timestamp alignment: raw data uses `fecha_hora` (solar) and `Fecha/Hora` (meteo)

### Model Modification Pattern
1. Edit architecture in `build_dual_stream_model()` or `build_tri_stream_gating_model()` in respective training scripts
2. Adjust hyperparameters in `train_and_evaluate_model()` call
3. Re-run full training: `python src/models/train_model.py` (or `train_alternativo.py`)
4. Evaluate: `python src/models/predict_model.py`

## Common Pitfalls

1. **Missing data directory structure:** Scripts create directories with `os.makedirs(..., exist_ok=True)`, but ensure `data/01_raw/` exists with source files
2. **Index mismatch:** Solar and meteorological data must align on timestamps. `make_dataset.py` uses inner join—check for empty results and date range warnings
3. **Feature order matters:** `windowing_utils.py` expects `PAST_FEATURES = ['gen_normalizada'] + FUTURE_FEATURES`. Changing column order breaks indexing. Now uses only 13 features (not 19)
4. **Split configuration inconsistency:** `ID_PLANTA`, `FECHA_INICIO`, `FECHA_FIN`, `in_steps`, `out_steps` MUST match exactly between `train_model.py` and `predict_model.py`
5. **Data leakage:** Without gaps between splits, validation/test performance is artificially inflated. Always use `validar_splits.py` to verify
6. **Model loading:** Use `tensorflow.keras.models.load_model()`, not `keras.models.load_model()`. Model files now named: `dual_stream_lstm_cnn_{ID}_{in_steps}h_{out_steps}h.keras` or `tri_stream_gating_{ID}_{in_steps}h_{out_steps}h.keras`
7. **Python path issues:** Use `sys.path.insert(0, str(project_root))` pattern when importing from `src/` modules
8. **Normalization not saved:** MinMaxScaler used in `make_dataset.py` is NOT saved—must apply same transform manually for production inference
9. **Verbose mode in CI/CD:** Always use `verbose=False` (default) for production pipelines. Use `verbose=True` only for local debugging

## Integration Points

- **External data source:** CEN API (optional, see `src/data/extract.py`—not required if raw data exists). ⚠️ Requires `CEN_API_KEY` environment variable
- **TensorFlow/Keras:** Model architecture tightly coupled to Keras functional API
- **scikit-learn:** MinMaxScaler used in `make_dataset.py` (NOT saved—apply same transform manually for inference)
- **matplotlib:** Visualization in `predict_model.py` (saves to `figures/`)

## When Modifying Code

- **Adding features:** Update `get_default_feature_sets()` in `windowing_utils.py` and ensure normalization in `make_dataset.py`. Note: 8 redundant features already eliminated (ghi≈glb, dir≈dni, dirh≈dni, difh≈dif, sct)
- **Changing window sizes:** Modify both `in_steps`/`out_steps` in training and prediction scripts. Current optimal: 4h→48h (same MAE as 24h→48h, 5x faster). Standard: 24h→48h
- **Working with different plants:** Change `ID_PLANTA`, `FECHA_INICIO`, `FECHA_FIN` consistently across all scripts—pipeline auto-adapts. Only plants 239, 309, 346 validated with complete data
- **New data periods:** Update date range parameters; ensure raw data exists in `data/01_raw/` with matching filenames
- **Architecture changes:** Modify `build_dual_stream_model()` but preserve dual-input structure (past/future). Current: Dense(64) in fusion, not Dense(128)

## Multi-Plant Automation

The entire pipeline is **plant-agnostic** and automatically handles:
- Different plant capacities (normalization factor)
- Different time periods (date ranges) - must match available data
- Different geographical locations (meteorological patterns)
- Optimized features (13 total, 8 redundant eliminated)

**To train a model for a new plant:**
1. Ensure raw data exists: `generacion_solar_{FECHA_INICIO}_a_{FECHA_FIN}_central_{ID}.json` and `Datos2013-2015_Planta{ID}.csv`
2. Verify dates with `verificar_datos.py`
3. Update configuration in all scripts: `ID_PLANTA`, `FECHA_INICIO`, `FECHA_FIN`
4. Choose `in_steps` (4h optimal for speed, 24h for interpretability)
5. Run pipeline: `python run.py`
6. Model saved as: `models/dual_stream_lstm_cnn_{ID}_{in_steps}h_48h.keras`
7. Validate splits: `python validar_splits.py`

**Validated plants (with complete data):**
- Plant 239: 2013-08-08 to 2015-08-08
- Plant 309: 2013-01-01 to 2015-01-01  
- Plant 346: 2014-01-01 to 2015-12-20

**Important files:**
- `training_summary.csv`: Comparative metrics across all trained models
- `{model}_history.csv`: Training curves for each model
- Multiple `.keras` files support different configurations simultaneously

---

**Note:** This project uses Python 3.13+ and TensorFlow 2.20+. Production model predicts **48h ahead** (CEN standard) using **optimizable historical window** (4-24h). 

**Key files:** 
- `run.py` (orchestrator)
- `windowing_utils.py` (data prep with 13 features)
- `train_model.py` (modeling with gap-based splits)
- `validar_splits.py` (validation tool)
- `training_summary.csv` (comparative results)

**Recent optimizations:**
- Features reduced from 19 to 13 (8 redundant eliminated)
- Historical window optimized: 4h achieves same MAE as 24h, 5x faster
- Splits improved with gaps to prevent data leakage
- Multi-plant validation: 3 plants tested successfully

**Production-ready refactorization (Dec 2024):**
- Verbose flag pattern: `verbose=False` by default for CI/CD
- Output reduced 87.5%: ~200 prints → ~25 critical prints
- All major functions support `verbose` parameter
- See `docs/REFACTORING_SUMMARY.md` for complete report
