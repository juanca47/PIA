

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== ENTORNO 1: Carga del dataset ===")

df = pd.read_csv("data/Sleep_Health_and_Lifestyle_Dataset.csv")

print("\nDimensiones del dataset:")
print(df.shape)

print("\nPrimeras filas:")
print(df.head())

print("\nTipos de datos:")
print(df.dtypes)

print("\nValores nulos por columna:")
print(df.isna().sum())

print("\nDescripción estadística (numéricas):")
print(df.describe())

print("\nDescripción estadística (todas):")
print(df.describe(include='all'))


# =========================================================

# =========================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

print("\n=== ENTORNO 2: Preprocesamiento ===")

# ---- 0) LIMPIEZA ----
# Reemplazar NaN en Sleep Disorder
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# ---- 1) Definir columnas ----
target = "Quality of Sleep"

num_cols = [
    'Age',
    'Sleep Duration',
    'Physical Activity Level',
    'Stress Level',
    'Heart Rate',
    'Daily Steps'
]

cat_cols = [
    'Gender',
    'Occupation',
    'BMI Category',
    'Blood Pressure',
    'Sleep Disorder'
]

print("\nColumnas numéricas:", num_cols)
print("Columnas categóricas:", cat_cols)

# ---- 2) LabelEncoder (demostración) ----
df['Sleep_Disorder_LE'] = LabelEncoder().fit_transform(df['Sleep Disorder'])
print("\nLabelEncoder ejemplo (Sleep Disorder -> Sleep_Disorder_LE):")
print(df[['Sleep Disorder', 'Sleep_Disorder_LE']].head())

# ---- 3) OneHotEncoder + StandardScaler ----
try:
    ohe = OneHotEncoder(
        drop='first',
        handle_unknown='ignore',
        sparse_output=False
    )
except TypeError:
    ohe = OneHotEncoder(
        drop='first',
        handle_unknown='ignore',
        sparse=False
    )

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', ohe, cat_cols),
        ('num', StandardScaler(), num_cols)
    ]
)

X_proc = preprocessor.fit_transform(df[cat_cols + num_cols])

# ---- 4) Nombres de columnas ----
ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
scaled_cols = [c + "_scaled" for c in num_cols]
final_feature_cols = list(ohe_cols) + scaled_cols

# ---- 5) DataFrame procesado ----
df_proc = pd.DataFrame(X_proc, columns=final_feature_cols)
df_proc[target] = df[target].values

print("\nPrimeras filas de df_proc:")
print(df_proc.head())

print("\nColumnas de df_proc:")
print(df_proc.columns)


# =========================================================

# =========================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\n=== ENTORNO 3: Regresión Lineal ===")

X = df_proc.drop(columns=[target]).values
y = df_proc[target].values

# Split 70% Train / 30% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print("Tamaño Train:", X_train.shape[0])
print("Tamaño Test :", X_test.shape[0])

# ---- Entrenar modelo ----
model = LinearRegression()
model.fit(X_train, y_train)

# ---- Predicciones ----
y_pred = model.predict(X_test)

# ---- Métricas ----
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\n=== MÉTRICAS ===")
print(f"MSE : {mse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# ---- Gráfica Real vs Predicho ----
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)
plt.xlabel("Quality of Sleep (Real)")
plt.ylabel("Quality of Sleep (Predicho)")
plt.title("Regresión Lineal — Predicción de Calidad de Sueño")
plt.grid(True)
plt.show()
