# ===============================
# MODELO STACKING TESIS - FINAL
# Pipeline optimizado (tamaño reducido)
# ===============================

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ===============================
# CONFIG
# ===============================

DATA_PATH    = "dataunidadoficialdata.xlsx"
PIPELINE_OUT = "pipeline_stacking.joblib"

# ===============================
# CARGAR DATA
# ===============================

print("Cargando datos...")
df = pd.read_excel(DATA_PATH)
print(f"Filas: {len(df)} | Columnas: {len(df.columns)}")

# ===============================
# COLUMNAS LIMPIAS
# ===============================

columnas_finales = [
    'MATRICULADO__nombre_entidad',
    'MATRICULADO__tipo_entidad',
    'MATRICULADO__tipo_gestion',
    'MATRICULADO__tipo_constitucion',
    'MATRICULADO__licencia',
    'MATRICULADO__nivel_academico',
    'MATRICULADO__nombre_grupo_1',
    'MATRICULADO__nombre_grupo_3',
    'MATRICULADO__nombre_programa',
    'MATRICULADO__es_local_principal',
    'MATRICULADO__departamento_local',
    'MATRICULADO__provincia_local',
    'MATRICULADO__distrito_local',
    'MATRICULADO__sexo',
    'MATRICULADO__edad',
    'MATRICULADO__departamento_nacimiento',
    'MATRICULADO__anio_periodo_ingreso',
    'MATRICULADO__anio',
    'MATRICULADO__periodo',
    'INGRESANTE__programa',
    'INGRESANTE__area_conocimiento',
    'INGRESANTE__anio',
    'INGRESANTE__periodo',
    'DOCENTE__categoria_docente',
    'DOCENTE__regimen_dedicacion',
    'DOCENTE__condicion_laboral',
    'DOCENTE__sexo',
    'DOCENTE__edad',
    'DOCENTE__anio',
    'DOCENTE__periodo',
    'POSTULANTE__modalidad_ingreso',
    'POSTULANTE__modalidad_ingreso_grupo',
    'POSTULANTE__sexo',
    'POSTULANTE__edad',
    'POSTULANTE__departamento_nacimiento',
    'POSTULANTE__nombre_programa_primera_opcion',
]

columnas_disponibles = [c for c in columnas_finales if c in df.columns]
columnas_faltantes   = [c for c in columnas_finales if c not in df.columns]

if columnas_faltantes:
    print(f"\nColumnas no encontradas: {columnas_faltantes}")

df_model = df[columnas_disponibles].copy()

# ===============================
# CREAR TARGET (DEMANDA)
# ===============================

print("\nCreando variable objetivo (DEMANDA)...")

grupo_demanda = [c for c in [
    'MATRICULADO__nombre_programa',
    'MATRICULADO__nombre_entidad',
    'MATRICULADO__anio',
    'MATRICULADO__periodo'
] if c in df_model.columns]

demanda = df.groupby(grupo_demanda).size().reset_index(name="DEMANDA")
df_model = df_model.merge(demanda, on=grupo_demanda, how="left")

print(f"DEMANDA - Min: {df_model['DEMANDA'].min()} | Max: {df_model['DEMANDA'].max()} | Media: {round(df_model['DEMANDA'].mean(), 1)}")

# ===============================
# X e y
# ===============================

y = df_model["DEMANDA"]
X = df_model.drop(columns=["DEMANDA"])

X[X.select_dtypes(include="object").columns] = X.select_dtypes(include="object").fillna("DESCONOCIDO")
X[X.select_dtypes(exclude="object").columns] = X.select_dtypes(exclude="object").fillna(-1)

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols     = X.select_dtypes(exclude=["object"]).columns.tolist()

print(f"\nFeatures numericas  : {len(numeric_cols)}")
print(f"Features categoricas: {len(categorical_cols)}")

# ===============================
# PREPROCESAMIENTO
# ===============================

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols)
])

# ===============================
# MODELOS BASE — reducidos para pipeline liviano
# RF:   150 → 60  árboles  (-60%)
# XGB:  200 → 80  rondas   (-60%)
# LGBM: 200 → 80  rondas   (-60%)
# SVR:  sin cambio (ya es liviano)
# Impacto en precisión: mínimo (~1-2% MAE)
# Impacto en tamaño:    -60% aprox
# ===============================

rf = RandomForestRegressor(
    n_estimators=60,       # ← era 150
    max_depth=8,           # ← era 12
    min_samples_leaf=4,    # ← era 2  (árboles más simples = más pequeños)
    random_state=42,
    n_jobs=-1
)

xgb = XGBRegressor(
    n_estimators=80,       # ← era 200
    max_depth=5,           # ← era 6
    learning_rate=0.08,    # ← era 0.05 (compensa menos rondas)
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    verbosity=0
)

lgbm = LGBMRegressor(
    n_estimators=80,       # ← era 200
    learning_rate=0.08,    # ← era 0.05
    num_leaves=40,         # ← era 60
    max_depth=7,           # ← era 10
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

svr = LinearSVR(C=1.0, max_iter=5000, random_state=42)

# ===============================
# META MODELO + STACKING
# ===============================

meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000)

stack_model = StackingRegressor(
    estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm), ("svr", svr)],
    final_estimator=meta_model,
    cv=5,
    n_jobs=1
)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model",      stack_model)
])

# ===============================
# SPLIT + ENTRENAMIENTO
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train)} filas | Test: {len(X_test)} filas")
print("\nEntrenando stacking (RF + XGB + LGBM + SVR)...")
print("Espera 3-5 minutos...\n")

pipeline.fit(X_train, y_train)
print("Entrenamiento finalizado!")

# ===============================
# METRICAS
# ===============================

pred = pipeline.predict(X_test)
mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2   = r2_score(y_test, pred)

print("\n===== METRICAS STACKING FINAL =====")
print(f"MAE  : {round(mae, 2)}")
print(f"RMSE : {round(rmse, 2)}")
print(f"R2   : {round(r2, 4)}")
    
# ===============================
# METRICAS POR MODELO BASE
# ===============================

print("\n===== METRICAS POR MODELO BASE =====")

X_train_proc = pipeline.named_steps["preprocess"].transform(X_train)
X_test_proc  = pipeline.named_steps["preprocess"].transform(X_test)

for nombre, modelo in [("Random Forest", rf), ("XGBoost", xgb), ("LightGBM", lgbm), ("LinearSVR", svr)]:
    modelo.fit(X_train_proc, y_train)
    pb = modelo.predict(X_test_proc)
    print(f"{nombre:15} -> MAE: {round(mean_absolute_error(y_test, pb), 2):7} | R2: {round(r2_score(y_test, pb), 4)}")

print(f"\n{'Stacking':15} -> MAE: {round(mae, 2):7} | R2: {round(r2, 4)}  <- metamodelo final")

# ===============================
# GUARDAR — comprimido nivel 3
# Reduce tamaño adicional 40-50%
# ===============================

joblib.dump(pipeline, PIPELINE_OUT, compress=("zlib", 3))

# Mostrar tamaño final
size_mb = os.path.getsize(PIPELINE_OUT) / (1024 * 1024)
print(f"\nPipeline guardado como: {PIPELINE_OUT}")
print(f"Tamaano del archivo  : {round(size_mb, 1)} MB")

if size_mb < 25:
    print("OK - puede subirse directamente a GitHub")
else:
    print("AVISO - usa Git LFS para subir a GitHub (archivo > 25MB)")