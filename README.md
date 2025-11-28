# Actividad Fundamental N.º 3 – Producto Integrador de Aprendizaje (Regresión Lineal)

## 🎯 Descripción General
Este proyecto corresponde a la **Actividad Fundamental N.º 3** de la materia **Inteligencia Artificial** de la FIME.  
Su objetivo es diseñar, implementar y documentar un sistema de **aprendizaje supervisado** capaz de predecir la 
*calidad del sueño* utilizando un modelo de **regresión lineal**, aplicado sobre el dataset público 
**Sleep Health and Lifestyle Dataset**.

El sistema implementado demuestra:

- Lectura y exploración de datos.
- Limpieza y preprocesamiento del dataset.
- Codificación de variables categóricas (LabelEncoder, OneHotEncoder).
- Normalización de variables numéricas (StandardScaler).
- Entrenamiento de un modelo de **Regresión Lineal**.
- Evaluación mediante métricas (MSE, MAE, R²) y gráfica de dispersión.
- Uso de un entorno virtual reproducible mediante `environment.yml`.
- Control de versiones con **Git y GitHub**.

---

## 1. Flujo del Código (Entorno 1, 2 y 3)

El código fue implementado en un solo archivo (`pia_regresion.py`), organizado en tres entornos siguiendo el formato solicitado durante el curso:

| Entorno | Responsabilidad Principal |
|--------|-----------------------------|
| **Entorno 1** | Carga del dataset, análisis exploratorio, revisión de tipos, nulos y estadísticas. |
| **Entorno 2** | Preprocesamiento de datos: imputación, codificación categórica y normalización numérica. |
| **Entorno 3** | Entrenamiento del modelo de regresión lineal, predicción y evaluación de métricas. |

---

## 2. Instalación y Entorno (pia_env)

El proyecto utiliza **Python 3.x** y las librerías esenciales para análisis y modelado:  
`pandas`, `numpy`, `matplotlib`, `scikit-learn`.

El entorno virtual es completamente reproducible gracias al archivo **environment.yml**.

### 🟩 Clonar el repositorio
```bash
git clone https://github.com/tu_usuario/PIA-IA-Sleep-Regression.git
cd PIA-IA-Sleep-Regression
Ejecución del Proyecto

Desde la carpeta raíz del repositorio, ejecutar:

python pia_regresion.py


El script realizará automáticamente:

Exploración del dataset (Entorno 1).

Preprocesamiento completo (Entorno 2).

Entrenamiento del modelo y evaluación de métricas (Entorno 3).

Se generará una gráfica de dispersión que compara valores reales vs predichos
de la Calidad del Sueño y se imprimirán las métricas:

MSE

MAE

R²

🧪 Resultados Principales
Métrica	Valor
MSE	0.0653
MAE	0.0774
R²	0.9583

El modelo logra explicar aproximadamente 95.83% de la variabilidad en la calidad del sueño, mostrando un excelente desempeño predictivo.

📚 Conclusiones

Este sistema demuestra la aplicación práctica del aprendizaje supervisado a través de la regresión lineal, cubriendo todas las etapas del preprocesamiento y modelado. Los resultados obtenidos confirman que las variables de estilo de vida y salud tienen una relación significativa con la calidad del sueño.

El flujo implementado puede adaptarse fácilmente a otros datasets y otros tipos de modelos supervisados (regresión polinomial, árboles, SVM, etc.).
