# Consigna de Actividad y Conceptos Clave: Predicción de Demanda M5 (Walmart)

**Procesamiento de Datos, Machine Learning y Evaluación orientada al Negocio**

**Dataset:** https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

---

## Consigna de la Actividad: Predicción de Demanda M5 (Walmart)

**Objetivo general:** Diseñar e implementar un pipeline completo de procesamiento de datos y Machine Learning para predecir las ventas diarias por unidad en un horizonte de 28 días a nivel de producto y tienda, optimizando el control de inventarios y reduciendo los costos por quiebre de stock o sobrestock.

### Fases del Proyecto

#### 1. Exploración y Procesamiento de Datos
- Integrar las tres fuentes principales del dataset (`sales_train`, `calendar` y `sell_prices`).
- Transformar la estructura de datos de formato ancho (*wide*) a formato largo (*long*) mediante técnicas de despivotado (`melt`) para adecuar los datos a modelos de series temporales.
- Optimizar el consumo de memoria RAM aplicando reducción de tipos de datos (*downcasting* de `int64` a `int16`/`int8` y `float64` a `float32`).
- Realizar un Análisis Exploratorio de Datos (EDA) identificando patrones de estacionalidad (días de la semana, fin de mes), tendencias por categoría de producto y comportamiento de precios.

#### 2. Ingeniería de Características (Feature Engineering)
- **Variables Temporales y Calendario:** Extraer día de la semana, mes, día del mes, identificadores de eventos festivos/deportivos y días de cobro de subsidios (SNAP por estado).
- **Variables de Series Temporales (Lags y Ventanas Móviles):** Construir variables de retraso ajustadas al horizonte de predicción (ej. t-28, t-29, t-35, t-42) y métricas móviles (media y desviación estándar móvil de 7, 14, 28 y 56 días).
- **Variables Monetarias:** Calcular variaciones porcentuales del precio respecto a semanas anteriores, precio relativo dentro del departamento y presencia de descuentos.

#### 3. Modelado y Validación Temporal
- Establecer un modelo base de referencia (*baseline*) utilizando un método ingenuo estacional (ej. ventas del mismo día de la semana anterior) o media móvil simple.
- Entrenar modelos avanzados basados en árboles de decisión (*Gradient Boosted Decision Trees*) como LightGBM, XGBoost o CatBoost.
- Diseñar un esquema de validación temporal estricto (*Time-Series Split* o *Expanding Window*) utilizando los últimos 28 días conocidos para validación.
- Evaluar el rendimiento empleando métricas estándar (RMSE, MAE) y métricas adaptadas a la competencia (WRMSSE).

#### 4. Interpretación y Reporte Final
- Identificar las variables con mayor peso explicativo (*feature importance*) en los pronósticos.
- Elaborar un informe ejecutivo que compare los resultados contra el baseline, detalle el impacto de los eventos de calendario en las ventas y proponga acciones para la gestión de la cadena de suministro.

### Sugerencias para el Desarrollo

- **Estrategia de Carga y Reducción del Volumen de Datos:** Dado que la transformación a formato largo genera más de 50 millones de filas, si cuentas con limitaciones de hardware, limita el historial de entrenamiento a los últimos 2 o 3 años de datos (ej. a partir del día `d_1000`) o procesa la información filtrando por estado o categoría en iteraciones iniciales.
- **Manejo del Horizonte de 28 Días:** Como el requerimiento es predecir un bloque continuo de 28 días en el futuro, no debes utilizar información de retraso inferior a 28 días (t-1, t-7) en modelos directos, ya que no estarán disponibles en el momento del pronóstico en producción a menos que utilices una estrategia autoregresiva/recursiva día a día.
- **Construcción Gradual de Modelos:** Inicia siempre evaluando el baseline simple para fijar una barra de rendimiento mínima antes de incorporar ingeniería de características compleja o búsqueda de hiperparámetros.

---

## Buenas Prácticas

### En Procesamiento de Datos

- **Prevención Estricta de Fuga de Datos (*Data Leakage*):** Las operaciones como la imputación de valores nulos, el cálculo de medias target o la normalización deben aprenderse únicamente con el conjunto de entrenamiento y aplicarse posteriormente a los conjuntos de validación y prueba.
- **Gestión Eficiente de Memoria RAM:** Convierte las columnas de texto repetitivas (como `store_id`, `state_id`, `cat_id`) al tipo `category` de Pandas. Invoca explícitamente el recolector de basura (`import gc; gc.collect()`) tras consolidar (*merges*) grandes conjuntos de datos.
- **Tratamiento de Demanda Intermitente o Venta Cero:** Diferencia los ceros correspondientes a días en que el producto no tuvo demanda de aquellos días en los que el producto aún no salía a la venta o no tenía precio asignado en la tabla `sell_prices`.

### En Predicción de Demanda con ML

- **Validación Cronológica sin Infiltración:** Evita el uso de K-Fold Cross Validation aleatorio, ya que rompe la secuencia temporal y provoca sobreajuste artificial. Utiliza siempre divisiones por bloques temporales (*Out-of-Time validation*).
- **Alineación entre Lags y el Horizonte de Predicción:** En estrategias de predicción directa (*Direct Forecasting*), el lag más reciente computable para predecir todo el período de 28 días debe ser t-28. Si utilizas lags más cortos, debes implementar una arquitectura de predicción recursiva, alimentando la predicción del día t+2 con la predicción generada para el día t+1.
- **Evaluación Orientada al Impacto del Negocio:** Considera que en el negocio minorista los costos de error no siempre son simétricos: el costo de oportunidad por quiebre de stock suele diferir del costo financiero de almacenamiento por exceso de inventario. Ajusta o evalúa los modelos considerando métricas ponderadas por volumen de venta o margen de ganancia (como la métrica oficial WRMSSE).

---

## Explicación Detallada de Conceptos Clave

### 1. Validación Cronológica sin Infiltración (*Out-of-Time Validation*)

En problemas de datos temporales, la fuga de datos (*data leakage*) ocurre cuando la información del futuro se utiliza inadvertidamente para entrenar un modelo que debe predecir el pasado o el presente.

#### El problema del K-Fold tradicional

La validación cruzada estándar (K-Fold Cross-Validation) divide el conjunto de datos de forma aleatoria. Si se aplica a series temporales, el modelo puede entrenarse con datos del día 15 para predecir el día 14. Esto invalida el proceso por dos razones:

1. **Autocorrelación temporal:** El modelo "ve" el futuro cercano y memoriza patrones que no tendría en producción.
2. **Transformaciones globales:** Calcular la media del target, aplicar escaladores (Z-score, Min-Max) o imputar nulos usando el dataset completo infiltra parámetros del conjunto de validación dentro del conjunto de entrenamiento.

#### Cómo implementarla correctamente

**Esquema de Expansión de Ventana (*Expanding Window*):**

```
Split 1: [ Train: D1-D100 ] -> [ Val: D101-D128 ]
Split 2: [ Train: D1-D128 ] -> [ Val: D129-D156 ]
Split 3: [ Train: D1-D156 ] -> [ Val: D157-D184 ]
```

- **Corte por Bloques Temporales (*Purged / Out-of-Time Split*):** El conjunto de validación debe situarse estrictamente después del conjunto de entrenamiento en la línea del tiempo.
- **Aislamiento en Preprocesamiento:** Cualquier transformación (escalado, codificación de variables categóricas, cálculo de medias móviles) debe ajustarse (*fit*) únicamente con el bloque de entrenamiento y aplicarse (*transform*) sobre el bloque de validación.

### 2. Alineación entre Lags y el Horizonte de Predicción

El horizonte de predicción (H) es la cantidad de pasos futuros que el modelo debe pronosticar a partir de un momento determinado (t₀). En la competencia M5, este horizonte es de H = 28 días.

#### El dilema del desfase (Lag)

Un lag o retraso de k días (t-k) representa la venta ocurrida k días antes del momento actual.

- Si el objetivo es predecir en el día t₀ todo el bloque continuo desde t₀+1 hasta t₀+28, la venta real del día t₀+15 aún no ha ocurrido al momento de realizar la inferencia.
- Si el modelo utiliza una característica basada en Lag_1 (t-1) para predecir el día t₀+28, el sistema fallará en producción porque no dispondrá del dato real del día t₀+27.

#### Estrategias de Alineación

| Estrategia | Descripción | Ventajas | Inconvenientes |
|---|---|---|---|
| **Directa** (*Direct Forecasting*) | Se restringe la ingeniería de características para utilizar únicamente Lags >= 28 (ej. t-28, t-29, t-35). | Es simple de implementar y no acumula errores de predicción. | Ignora la información más reciente (t-1 a t-27) al predecir los primeros días del horizonte (t+1). |
| **Recursiva** (*Autoregressive*) | Se entrena un modelo con lags cortos (t-1). Para predecir t+2, se utiliza como entrada la predicción generada para t+1. | Aprovecha las señales de alta frecuencia y de corto plazo. | Los errores de las primeras predicciones se propagan y amplifican a lo largo de los 28 días. |
| **Modelos Múltiples** (*Multi-model*) | Se entrenan 28 modelos independientes: el Modelo 1 predice t+1 usando Lag_1, el Modelo 28 predice t+28 usando Lag_28. | Optimiza la precisión específica para cada día del horizonte. | Alto costo computacional y mayor complejidad de mantenimiento. |

### 3. Evaluación Orientada al Impacto del Negocio

Las métricas estadísticas estándar como el RMSE (Error Cuadrático Medio) o el MAE (Error Absoluto Medio) tratan todos los errores por igual. Sin embargo, en la gestión de cadenas de suministro y venta minorista, los errores tienen impactos financieros asimétricos y ponderaciones distintas según el producto.

#### Asimetría del Costo de Error

El costo de equivocarse no es simétrico entre sobreestimar y subestimar la demanda:

- **Subestimar la demanda (Predicción < Real):** Genera quiebre de stock (*stockout*). Provoca pérdida directa de ventas, posible pérdida de clientes hacia la competencia y penalizaciones por incumplimiento de nivel de servicio.
- **Sobreestimar la demanda (Predicción > Real):** Genera exceso de inventario (*overstock*). Provoca costos de almacenamiento, capital de trabajo inmovilizado y riesgo de obsolescencia o caducidad.

#### Métricas Ponderadas y Escaladas (Métrica WRMSSE)

En la competencia M5 se utiliza el WRMSSE (*Weighted Root Mean Squared Scaled Error*), que resuelve los inconvenientes de las métricas tradicionales combinando dos conceptos:

1. **Escalado por volatilidad (RMSSE):** Divide el error del modelo por el error de un pronóstico ingenuo (*naive*) en la serie histórica del propio producto. Esto evita que los productos muy volátiles dominen el error global y permite evaluar productos de alta y baja rotación en una misma escala.
2. **Ponderación por valor de negocio (W):** Asigna un peso proporcional al volumen de ventas en dólares de cada serie temporal durante los últimos 28 días. Un error en un artículo de alto valor financiero impacta más en la métrica que el mismo error en un producto de bajo costo.

**Fórmula de WRMSSE:**

```
WRMSSE = ∑ [ w_i * √( (1/h * ∑ (Y_t - Y_hat_t)^2) / (1/(n-1) * ∑ (Y_t - Y_{t-1})^2) ) ]
```

Donde:

- `w_i`: peso financiero de la serie i (volumen de ventas en USD / total USD)
- `h`: horizonte de evaluación (h = 28 días)
- `n`: cantidad de datos históricos disponibles
- `Y_t`: ventas reales en el día t
- `Y_hat_t`: predicción de ventas para el día t

---

## Modelo base de referencia (*baseline model*)

Un modelo base de referencia (o *baseline model*) es la solución más simple, rápida e intuitiva que se construye al inicio de un proyecto de Machine Learning para establecer una marca de rendimiento mínima que cualquier modelo más complejo debe superar.

### ¿Para qué sirve un modelo baseline?

- **Establecer una cota inferior de rendimiento:** Define la precisión o error mínimo aceptable. Si un modelo avanzado (como LightGBM o una Red Neuronal) no supera al baseline, significa que la complejidad añadida no está aportando valor.
- **Validar el pipeline de datos:** Permite verificar rápidamente que el flujo de carga, limpieza, transformación y evaluación funciona correctamente sin esperar largos tiempos de entrenamiento.
- **Evaluar la relación costo-beneficio:** Ayuda a juzgar si una mejora marginal en métricas justifica la complejidad del código, el consumo computacional y el costo de mantenimiento en producción.

### Ejemplos según el tipo de problema

- **Series temporales y predicción de demanda:** Estrategias como el método ingenuo estacional (asumir que el próximo lunes se venderá exactamente lo mismo que el lunes anterior) o una media móvil simple de los últimos días.
- **Regresión:** Asignar a todas las predicciones futuras el valor medio o la mediana de los datos de entrenamiento.
- **Clasificación:** Predecir siempre la clase más frecuente (clase mayoritaria) para todas las observaciones.

### Regla de oro

En proyectos de ciencia de datos, el baseline se implementa en la primera etapa. Nunca se debe pasar a la ingeniería de características avanzada o a la búsqueda de hiperparámetros sin haber evaluado primero una referencia simple.

---

## Consigna estándar para el desarrollo e implementación de un modelo de predicción de la demanda

### 1. Declaración del Problema y Objetivo de Negocio

- **Objetivo principal:** Diseñar e implementar un modelo de Machine Learning capaz de pronosticar las ventas unidades por producto y ubicación en un horizonte determinado, reduciendo los costos asociados a quiebres de stock y sobrestock.
- **Impacto esperado:** Optimizar la planificación de la cadena de suministro, mejorar la gestión de inventario y maximizar el nivel de servicio al cliente.

### 2. Definición del Alcance y Datos de Entrada

- **Granularidad de predicción:** Definir la frecuencia y el nivel de agregación (por ejemplo, producto-tienda-día).
- **Horizonte de predicción (H):** Establecer la cantidad de periodos futuros a pronosticar (por ejemplo, 28 días continuos).
- **Fuentes de datos disponibles:** Incorporar el historial de ventas, tablas de calendario con eventos/festivos y variaciones históricas de precios o promociones.

### 3. Requerimientos Técnicos del Pipeline

#### A. Procesamiento de Datos y EDA

- **Estructuración:** Transformar los datos a formato largo (*long format*) para facilitar el entrenamiento de modelos de series temporales.
- **Eficiencia:** Optimizar el uso de memoria RAM reduciendo tipos de datos (*downcasting*).
- **Análisis Exploratorio:** Identificar patrones de estacionalidad, tendencia y comportamiento de la demanda intermitente o nula.

#### B. Ingeniería de Características (Feature Engineering)

- **Variables de calendario:** Extraer días de la semana, meses, festivos y eventos especiales.
- **Variables de retraso (Lags) y métricas móviles:** Construir lags alineados al horizonte de predicción (LagsH) para evitar fugas de información.
- **Variables monetarias y promocionales:** Calcular variaciones de precios y descuentos relativos.

#### C. Validación y Modelado

- **Modelo Base (Baseline):** Construir una referencia simple (por ejemplo, método ingenuo estacional o media móvil) para fijar el umbral mínimo de rendimiento.
- **Modelos avanzados:** Entrenar algoritmos basados en árboles de decisión (*Gradient Boosted Decision Trees* como LightGBM o XGBoost) o modelos específicos de series temporales.
- **Estrategia de validación:** Implementar una validación cronológica estricta (*Out-of-Time* o *Expanding Window*) aislando completamente los conjuntos de entrenamiento y validación.

#### D. Evaluación de Rendimiento

- **Métricas estadísticas:** Medir el error promedio utilizando métricas estándar como RMSE o MAE.
- **Métricas de negocio:** Evaluar los pronósticos con métricas ponderadas por volumen de venta o valor financiero (como WRMSSE) para reflejar la asimetría del costo entre sobreestimar y subestimar la demanda.

### 4. Entregables Esperados

- **Código fuente:** Scripts o notebooks reproducibles con el pipeline completo (desde el preprocesamiento hasta la inferencia).
- **Modelo entrenado:** Artefacto del modelo optimizado listo para generar pronósticos futuros.
- **Informe ejecutivo:** Documento que analice la importancia de las variables, compare el desempeño del modelo contra el baseline y proporcione recomendaciones para la toma de decisiones operativas.
