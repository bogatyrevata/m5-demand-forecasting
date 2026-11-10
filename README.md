# Predicción de demanda M5 (Walmart) – California

**Autora:** [Taisia Bogatyreva] · **Curso:** [Data Science and AI] · **Fecha:** [13.11.2026]

## 1. Objetivo del proyecto (Цель проекта)

Pipeline de Machine Learning para pronosticar las **ventas diarias en unidades a 28 días** por producto y tienda (estado de California: 4 tiendas, 12 196 series), a partir de `sales_train_validation`, `calendar` y `sell_prices` de la competencia [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy).

## 2. Contenido del repositorio (Содержимое репозитория)
```
m5-demand-forecasting/
├── notebooks/m5_forecasting.ipynb   # pipeline completo con resultados guardados
├── models/lgbm_m5_ca.txt            # modelo LightGBM entrenado
├── reports/informe_ejecutivo.md     # informe ejecutivo
├── requirements.txt                 # dependencias
├── .gitignore
└── README.md
```
**No incluido en el repositorio:** los datos (`data/`, archivos `.csv`/`.parquet`), porque las reglas de la competencia prohíben distribuirlos, y el entorno virtual `.venv/`.

## 3. Cómo obtener los datos (Как получить данные)
Los datos **no se incluyen en este repositorio** (las reglas de la competencia no permiten distribuirlos). Para reproducir el proyecto hay que descargarlos manualmente:

1. Iniciar sesión en [Kaggle](https://www.kaggle.com) y verificar la cuenta (teléfono).
2. Abrir la competencia [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) → **Join Competition** → *I Understand and Accept*.
3. En la pestaña **Data**, descargar estos tres archivos:
   - `calendar.csv`
   - `sales_train_validation.csv`
   - `sell_prices.csv`
4. Crear la carpeta `data/raw/` en la raíz del proyecto y guardar allí los tres archivos:
```bash
mkdir -p data/raw
```

Estructura esperada antes de ejecutar el notebook:
```
m5-demand-forecasting/
├── data/
│   └── raw/
│       ├── calendar.csv
│       ├── sales_train_validation.csv
│       └── sell_prices.csv
├── notebooks/
└── ...
```
Las rutas del notebook son relativas (`../data/raw`), por lo que los archivos deben estar exactamente en esa carpeta y con esos nombres. La carpeta `data/` está en `.gitignore` y no se sube a GitHub.

## 4. Instalación (Установка)
Requisitos: Python **3.12** (con 3.14 LightGBM no compiló) y, en macOS, `brew install libomp`.
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Ejecución (Запуск)
Abrir `notebooks/m5_forecasting.ipynb` (desde la carpeta `notebooks/`, las rutas son relativas) y ejecutar *Restart & Run All*. Al terminar se guarda el modelo en `models/lgbm_m5_ca.txt`. Requiere unos 4-6 GB de RAM libres.

## 6. Alcance y decisiones (Объём и решения)
Por restricción de tiempo: solo el estado CA, historial `d_1500 … d_1913` (414 días), un único corte de validación (`d_1886 … d_1913`), sin búsqueda de hiperparámetros y sin WRMSSE (se reportan RMSE y MAE).

## 7. Resultados principales (Основные результаты)
| Modelo | RMSE | MAE |
|---|---|---|
| Ingenuo estacional | 2,8315 | 1,3244 |
| Media móvil 28 días (mejor baseline) | 2,2604 | 1,1070 |
| Predicción cero | 3,8650 | 1,4764 |
| **LightGBM** | **2,1613** | 1,1097 |

LightGBM reduce el RMSE un 4,4 % frente al mejor baseline (MAE equivalente). Las medias móviles de 7 y 28 días concentran el 79,5 % de la importancia de variables. Informe completo: `reports/informe_ejecutivo.md` (sección 4.3 del notebook).

## 8. Limitaciones (Ограничения)
Un único corte de validación, alcance reducido (CA, 414 días), sin WRMSSE y efectos de eventos basados en 1-2 observaciones por evento.