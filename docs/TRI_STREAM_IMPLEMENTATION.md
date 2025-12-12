# Arquitectura Tri-Stream con Gating Multiplicativo

## 📋 Resumen de la Implementación

Se ha implementado exitosamente la **arquitectura Tri-Stream con Gating Multiplicativo** en archivos separados para no interferir con el proyecto original.

## 🆕 Archivos Creados/Modificados

### 1. **`src/models/windowing_utils.py`** (modificado)
- ✅ Nueva función: `create_tri_stream_data_alternativo()`
- Genera **3 conjuntos de datos** en lugar de 2:
  - `X_past`: Datos históricos (shape: `(n_samples, in_steps, 19)`)
  - `X_future`: Pronóstico meteorológico (shape: `(n_samples, out_steps, 18)`)
  - `X_binary`: Variables binarias para gating (shape: `(n_samples, out_steps, 2)`)
  - `Y_target`: Target (shape: `(n_samples, out_steps)`)

### 2. **`src/models/train_alternativo.py`** (creado)
- ✅ Función: `build_tri_stream_model()` - Arquitectura completa tri-stream
- ✅ Función: `train_and_evaluate_model()` - Pipeline de entrenamiento modificado
- ✅ Compatible con ejecución directa: `python src/models/train_alternativo.py`

### 3. **`test_tri_stream.py`** (creado)
- ✅ Script de prueba rápida sin entrenar
- Verifica que todo funcione correctamente

## 🏗️ Arquitectura del Modelo

### Ecuación Principal
$$Y_{final} = Y_{potencial} \otimes Y_{gate}$$

### Ramas del Modelo

#### **Rama 1: Inercia Histórica**
```
Input: (batch, 24, 19)
  ↓
BiLSTM(64, return_sequences=True)
  ↓
BiLSTM(32)
  ↓
Dropout(0.1)
```

#### **Rama 2: Pronóstico Meteorológico**
```
Input: (batch, 48, 18)
  ↓
Conv1D(32, kernel=3, padding='same')
  ↓
Flatten
  ↓
Dense(32, relu)
```

#### **Rama 3: Gating (NUEVA)**
```
Input: (batch, 48, 2)  ← shadow, cloud
  ↓
Flatten
  ↓
Dense(32, relu)
  ↓
Dense(48, sigmoid, bias_init=Constant(3.0))
  ↓
Output_Gate: (batch, 48) [valores 0-1]
```

### Fusión
```
[Rama 1 + Rama 2]
  ↓
Concatenate
  ↓
Dense(64, relu)
  ↓
Dropout(0.1)
  ↓
Dense(48, linear)
  ↓
Output_Potential: (batch, 48)

Output_Final = Multiply([Output_Potential, Output_Gate])
```

## 🎯 Features Binarias Usadas

Por defecto, el modelo usa las siguientes columnas binarias del dataset:

- **`shadow`**: Indica si hay sombra (0=sin sombra, 1=con sombra)
- **`cloud`**: Indica cobertura de nubes (0=despejado, 1=nublado)

Estas variables modulan la generación potencial:
- Si `shadow=1` o `cloud=1` → Gate ≈ 0 → Generación reducida
- Si `shadow=0` y `cloud=0` → Gate ≈ 1 → Generación normal

## 🚀 Cómo Usar

### Opción 1: Script de Prueba (Rápido)
```powershell
python test_tri_stream.py
```
Esto verifica que todo funcione sin entrenar el modelo completo.

### Opción 2: Entrenamiento Completo
```powershell
python src/models/train_alternativo.py
```

O desde código:
```python
from src.models.train_alternativo import train_and_evaluate_model
from src.config import get_plant_config

ID_PLANTA = 239
plant_config = get_plant_config(ID_PLANTA)

train_and_evaluate_model(
    input_path='data/03_processed/DatosCombinados_2013-08-08_a_2015-08-08_Planta239.csv',
    output_path='models/',
    in_steps=24,
    out_steps=48,
    model_name='tri_stream_gating_239_24h_48h.keras',
    binary_features=['shadow', 'cloud']  # Personalizable
)
```

## 🔑 Características Clave

### 1. **Inicialización Inteligente del Gate**
```python
Dense(48, activation='sigmoid', bias_initializer=Constant(value=3.0))
```
- σ(3.0) ≈ 0.95 → Compuerta empieza "abierta"
- Evita gradientes nulos al inicio del entrenamiento
- El modelo aprende a "cerrar" la compuerta cuando sea necesario

### 2. **Separación de Concerns**
- **Potencial**: Qué puede generar el sistema (física + clima)
- **Gate**: Si el sistema está operativo (estado binario)
- Multiplicación elemento a elemento permite modulación hora por hora

### 3. **No Afecta el Proyecto Original**
- Todos los cambios están en archivos separados o funciones nuevas
- `train_model.py` original sin modificaciones
- `create_dual_stream_data()` original intacta
- Fácil de eliminar si no funciona

## 📊 Outputs Esperados

Al entrenar, el modelo genera:

1. **Modelo entrenado**: `models/tri_stream_gating_{ID}_{in_steps}h_{out_steps}h.keras`
2. **Historial**: `figures/tri_stream_gating_{ID}_{in_steps}h_{out_steps}h_history.csv`
3. **Resumen**: Agregado a `figures/training_summary.csv`

## ⚙️ Parámetros Configurables

En `train_alternativo.py` (línea ~350):

```python
ID_PLANTA = 239  # Cambiar a 239, 309, 346
in_steps = 24    # Ventana histórica (4-24h)
out_steps = 48   # Horizonte predicción (fijo)
binary_features = ['shadow', 'cloud']  # Personalizable
```

## 🧪 Validación

El script de prueba `test_tri_stream.py` verifica:

1. ✓ Carga de datos
2. ✓ Existencia de columnas binarias
3. ✓ Creación de ventanas tri-stream
4. ✓ Construcción del modelo (3 inputs, 1 output)
5. ✓ Predicción con batch pequeño

## 📈 Ventajas vs Dual-Stream

| Aspecto | Dual-Stream | Tri-Stream con Gating |
|---------|-------------|------------------------|
| **Inputs** | 2 (histórico, pronóstico) | 3 (histórico, pronóstico, binario) |
| **Variables binarias** | Procesadas como continuas | Stream dedicado con sigmoid |
| **Interpretabilidad** | Caja negra | Gate explícito (operativo/inoperativo) |
| **Separación física** | No | Sí (potencial × estado) |
| **Parámetros** | ~150k | ~155k (+3% overhead) |

## 🔧 Troubleshooting

### Error: "Features binarias no encontradas"
**Solución**: Verifica que `shadow` y `cloud` existen en el CSV procesado.

### Error: "Dataset muy pequeño"
**Solución**: Reduce `in_steps` o `out_steps`, o usa dataset más grande.

### Gate siempre cerca de 1
**Solución**: Esto es esperado si el sistema casi siempre está operativo. Puedes:
- Crear features binarias adicionales (ej: `is_night`, `is_holiday`)
- Verificar que las features binarias tienen variabilidad

### Peor performance que Dual-Stream
**Solución**: 
- Aumentar `patience` en EarlyStopping
- Reducir learning rate: `Adam(learning_rate=0.0005)`
- Agregar más features binarias relevantes

## 📝 Notas Importantes

1. **Las features binarias deben existir en el dataset procesado** (`data/03_processed/`)
2. **El gate modula la salida, no la reemplaza** - Si gate=0, salida=0 independientemente del potencial
3. **Bias inicial de 3.0 es crítico** - Sin esto, el gate puede quedarse en 0 durante el entrenamiento
4. **El modelo espera valores binarios [0, 1]** - No usar valores continuos en el stream binario

## 🎓 Interpretación de Resultados

Si después del entrenamiento:

- **Gate promedio ≈ 1**: Sistema operando normalmente la mayoría del tiempo
- **Gate promedio < 0.5**: Sistema frecuentemente inoperativo (revisar datos)
- **Gate varía mucho**: Modelo aprendió patrones de operación/falla correctamente

Para inspeccionar el gate entrenado:
```python
from tensorflow.keras.models import load_model, Model

model = load_model('models/tri_stream_gating_239_24h_48h.keras')
gate_model = Model(inputs=model.input, outputs=model.get_layer('Output_Gate').output)
gate_predictions = gate_model.predict([X_past, X_fut, X_bin])
```

---

**Fecha de implementación**: 2024-12-12  
**Versión TensorFlow**: 2.20+  
**Python**: 3.13+
