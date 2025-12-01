# unirdocente.py (versión actualizada 2025)
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import re, unicodedata, os

# === RUTAS ===================================================================
DATA_DIR = Path("data/data_docente")
OUTFILE  = DATA_DIR / "docentes_limpio.csv"

# === CONFIGURACIÓN ===========================================================
AGE_MIN, AGE_MAX = 16, 50          # Ajustado según tu requerimiento
YEARS_ALLOWED = {2022, 2023, 2024, 2025}
ROWS_PER_YEAR = 5000
PAD_LEN_CODIGO = 9

# === HELPERS =================================================================
def slug_col(c):
    s = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s.lower()).strip("_")
    return re.sub(r"_+","_", s)

def read_table_robust(path):
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, dtype=str, sep="|", encoding=enc)
        except:
            pass
    return pd.read_excel(path, dtype=str)

def parse_anio_periodo(name):
    m = re.search(r"docente_(\d{4})_([iIvV]+)\.(csv|xlsx?)$", name)
    return (int(m.group(1)), m.group(2).upper()) if m else (None, None)

def normalize_codigo_entidad(x):
    if x is None or pd.isna(x):
        return pd.NA
    s = re.sub(r"[^0-9]", "", str(x).strip())
    return s.zfill(PAD_LEN_CODIGO) if s else pd.NA

def clean_age_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= AGE_MIN) & (s <= AGE_MAX))

# === COLUMNAS A ELIMINAR =====================================================
DROP_EXPLICIT = {
    "periodo_estandar", "reglamento", "departamento_nacimiento",
    "pais_entidad", "anio_nacimiento", "guid_persona",
    "grupo_investigador", "nivel_investigador",
    "fecha_calificacion_inicio", "fecha_calificacion_fin",
    "tipo_grado_titulo", "denominacion_entidad",
    "fecha_expedicion_diploma"
}

# === COLUMNAS A CONSERVAR ====================================================
KEEP_COLUMNS = [
    "codigo_entidad", "entidad", "licencia",
    "tipo_gestion", "tipo_entidad", "tipo_constitucion",
    "nivel_academico", "categoria_docente",
    "regimen_dedicacion", "condicion_laboral",
    "edad", "sexo", "nacionalidad",
    "anio", "periodo"
]

# === PROCESO =================================================================
files = sorted(list(DATA_DIR.glob("docente_*.csv")) +
               list(DATA_DIR.glob("docente_*.xlsx")))

print("Archivos encontrados:")
for f in files:
    print(" -", f.name)

frames_by_year = {y: [] for y in YEARS_ALLOWED}

for f in files:
    print(f"\nLeyendo: {f.name} ...")
    anio, periodo = parse_anio_periodo(f.name)

    # Filtrar por año permitido
    if anio not in YEARS_ALLOWED:
        print(f" -> Saltado (año fuera de rango): {anio}")
        continue

    df = read_table_robust(f)

    # Eliminar filas vacías
    df.dropna(how="all", inplace=True)

    # Normalizar columnas
    df.rename(columns={c: slug_col(c) for c in df.columns}, inplace=True)

    # Aliases para unificar nombres
    df.rename(columns={
        "codigo_inei": "codigo_entidad",
        "cod_entidad": "codigo_entidad",
    }, inplace=True)

    # Eliminar columnas explícitas
    cols_to_drop = [c for c in df.columns if c in DROP_EXPLICIT]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Normalizar código de entidad
    if "codigo_entidad" in df.columns:
        df["codigo_entidad"] = df["codigo_entidad"].apply(normalize_codigo_entidad)
        df = df[df["codigo_entidad"].notna()]

    # Limpiar edad
    if "edad" in df.columns:
        df["edad"] = clean_age_series(df["edad"])
        df = df[df["edad"].notna()]
        df["edad"] = df["edad"].astype("Int64")

    # Normalizar sexo
    if "sexo" in df.columns:
        df["sexo"] = df["sexo"].str.upper().replace({"M": "MASCULINO", "F": "FEMENINO"})

    # Agregar año/periodo del archivo
    df["anio"] = anio
    df["periodo"] = periodo

    # Tomar solo las columnas útiles que existan
    keep_now = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[keep_now].copy()

    # Limpiar strings
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    frames_by_year[anio].append(df)

# === SUBMUESTREO: 5,000 POR AÑO ==============================================
final_frames = []

for y in YEARS_ALLOWED:
    if not frames_by_year[y]:
        continue

    print(f"\n→ Uniendo año {y} ...")
    year_df = pd.concat(frames_by_year[y], ignore_index=True).drop_duplicates()

    if len(year_df) > ROWS_PER_YEAR:
        year_df = year_df.sample(ROWS_PER_YEAR, random_state=42)

    print(f"Año {y}: {len(year_df)} filas seleccionadas.")
    final_frames.append(year_df)

# === UNIR GENERAL Y GUARDAR ==================================================
full = pd.concat(final_frames, ignore_index=True).drop_duplicates()

if OUTFILE.exists():
    os.remove(OUTFILE)

full.to_csv(OUTFILE, index=False, encoding="utf-8-sig")

print("\n✔ ARCHIVO FINAL GUARDADO:", OUTFILE)
print("✔ Total filas:", len(full))
print("✔ Columnas:", list(full.columns))
