# modelo.py
# STACKING FINAL OPTIMIZADO PARA PC DE 8 GB RAM
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ========================================================
# FUNCION PICKLEABLE
# ========================================================
def to_float32(x):
    return x.astype(np.float32)

# ========================================================
# CONFIGURACIÓN
# ========================================================
DATA_PATH   = "dataoficialunida.xlsx"
PIPELINE_OUT = "pipeline_stacking.joblib"
TARGET = "MATRICULADO__codigo_siu_programa"

# ========================================================
# CARGA
# ========================================================
print("📥 Cargando dataset...")
df = pd.read_excel(DATA_PATH)

print("Columnas encontradas:", len(df.columns))

if TARGET not in df.columns:
    raise ValueError(f"No existe la columna objetivo: {TARGET}")

# ========================================================
# Y
# ========================================================
print("🎯 Procesando variable objetivo...")

y = (
    df[TARGET]
    .astype(str)
    .value_counts()
    .reindex(df[TARGET])
    .fillna(0)
    .astype(np.float32)
)

# ========================================================
# X
# ========================================================
DROP_COLS = [
    "POSTULANTE__programa",
    "POSTULANTE__codigo_programa",
    "INGRESANTE__programa",
    "INGRESANTE__codigo_programa",
    "MATRICULADO__programa",
    "MATRICULADO__codigo_programa",
    TARGET
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

# ========================================================
# FEATURES
# ========================================================
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()

print("Variables numéricas   :", len(numeric_cols))
print("Variables categóricas:", len(categorical_cols))

# ========================================================
# PREPROCESS
# ========================================================
preprocess = ColumnTransformer(
    transformers=[
        ("num",
         Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("to32", FunctionTransformer(to_float32))
         ]),
         numeric_cols),

        ("cat",
         Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder",
             OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                min_frequency=10
             )
            )
         ]),
         categorical_cols)
    ],
    sparse_threshold=0.3
)

# ========================================================
# MODELOS BASE
# ========================================================

rf = RandomForestRegressor(
    n_estimators=60,
    max_depth=8,
    n_jobs=1,
    random_state=42
)

xgb = XGBRegressor(
    n_estimators=60,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=1
)

lgbm = LGBMRegressor(
    n_estimators=60,
    max_depth=-1,
    num_leaves=16,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1
)

svr = SVR(kernel="rbf", C=5, epsilon=0.2)

# ========================================================
# STACKING
# ========================================================
stack_model = StackingRegressor(
    estimators=[
        ("rf", rf),
        ("xgb", xgb),
        ("lgbm", lgbm),
        ("svr", svr)
    ],
    final_estimator=ElasticNet(alpha=0.5, l1_ratio=0.5),
    cv=3,
    n_jobs=1
)

# ========================================================
# PIPELINE
# ========================================================
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", stack_model)
])

# ========================================================
# SPLIT
# ========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ========================================================
# TRAIN
# ========================================================
print("\n⏳ Entrenando STACKING (modo RAM safe)...")
pipeline.fit(X_train, y_train)
print("✅ Entrenamiento completado")

# ========================================================
# SCORE
# ========================================================
r2 = pipeline.score(X_test, y_test)

print("\n============================")
print("R² DEL METAMODELO:", round(float(r2),4))
print("============================")

# ========================================================
# SAVE
# ========================================================
joblib.dump(pipeline, PIPELINE_OUT)

print("\n✅ PIPELINE GUARDADO:")
print("➡️", PIPELINE_OUT)
