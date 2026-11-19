# TFM – Predicción de la ocurrencia de infarto cerebral (ictus) mediante modelos de aprendizaje automático

Código del Trabajo Fin de Máster de **Pablo Ruiz Amezcua**, Máster realizado en el Centro Europeo de Másteres y Posgrado (CEMP). Director: Sara López.

El repositorio contiene el código necesario para reproducir todas las tablas y figuras de la memoria a partir del conjunto de datos público *Stroke Prediction Dataset*.

## Contenido

```
.
├── README.md
├── code/
│   ├── 01_prepare_eda.py      # Auditoría, limpieza y análisis descriptivo
│   ├── 02_stats_tests.py      # Contrastes de hipótesis, correlaciones y regresión logística multivariable
│   └── 03_modeling.py         # Modelos predictivos, selección de modelo, evaluación y fuga de datos
├── data/                      # Datos (no incluidos, ver instrucciones)
├── tables/                    # Salidas: tablas en CSV y resúmenes en JSON (se generan)
└── figures/                   # Salidas: figuras en PNG (se generan)
```

## Datos

Los datos no se incluyen en el repositorio.

1. Descarga el conjunto de datos *Stroke Prediction Dataset* desde Kaggle (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) o desde los materiales del Máster del CEMP.
2. Si el archivo viene comprimido, descomprímelo y copia el CSV en la carpeta `data/` con el nombre `stroke.csv`.

El fichero debe estar separado por comas y contener 5 110 filas y las columnas `id`, `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status` y `stroke`.

## Requisitos

- Python 3.13
- pandas 3.0.5
- numpy 2.5.3
- scipy 1.18.1
- scikit-learn 1.9.1
- matplotlib 3.11.2
- seaborn 0.13.2

Instalación recomendada en un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install pandas==3.0.5 numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 matplotlib==3.11.2 seaborn==0.13.2
```

## Ejecución

Ejecuta los scripts en este orden, desde la raíz del repositorio:

```bash
python code/01_prepare_eda.py
python code/02_stats_tests.py
python code/03_modeling.py
```

- **01** lee `data/stroke.csv`, genera `data/stroke_clean.csv` y produce las tablas t01–t03, t04b y las figuras f01–f05.
- **02** lee `data/stroke_clean.csv` y produce las tablas t04–t08 y `summary_stats.json`.
- **03** lee `data/stroke_clean.csv` y produce las tablas t09–t15, las figuras f06–f09 y `summary_modeling.json`. Tarda aproximadamente 1,5 minutos.

Al terminar, las tablas están en `tables/` y las figuras en `figures/`. Compáralas con las tablas y figuras de la memoria.

## Protocolo y reproducibilidad

- Semilla aleatoria común: `random_state = 42` en todos los procesos aleatorios.
- Partición estratificada 80 % entrenamiento / 20 % test. El test se evalúa una única vez.
- Selección del modelo principal por AUC-PR medio en validación cruzada estratificada de 5 pliegues, sobre el conjunto de entrenamiento.
- Umbral de decisión fijado sobre las predicciones fuera de pliegue del entrenamiento, buscando sensibilidad ≥ 0,80.
- El sobremuestreo SMOTE se implementa en `03_modeling.py` (función `smote_oversample`) y se aplica únicamente dentro de cada pliegue de entrenamiento, nunca sobre validación ni test. No se usa la biblioteca `imbalanced-learn`.
- La demostración de fuga de datos (SMOTE aplicado antes de la partición) está incluida en `03_modeling.py` con fines ilustrativos. Sus dos resultados se evalúan sobre tests distintos y no son directamente comparables.

## Licencia y uso

Código desarrollado con fines académicos para el TFM. Los datos pertenecen a sus autores originales y deben descargarse desde la fuente indicada.

