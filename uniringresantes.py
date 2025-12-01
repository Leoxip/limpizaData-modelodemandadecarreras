# uniringresantes.py (VERSIÓN FINAL)
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import re, unicodedata, os, gc

# ===================== RUTAS ============================
DATA_DIR = Path("data/data_ingresante")
OUTFILE  = DATA_DIR / "ingresantes_limpio.csv"

# ===================== CONFIG ===========================
CHUNKSIZE = 5000
AGE_MIN, AGE_MAX = 16, 50
PAD_LEN_CODIGO = 9
YEARS_ALLOWED = {2022, 2023, 2024, 2025}
ROWS_PER_YEAR = 5000

# ===================== HELPERS ==========================
def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^0-9a-zA-Z]+","_", s.lower()).strip("_")
    return re.sub(r"_+","_", s)

def parse_anio(name: str):
    m = re.search(r"ingresante_(\d{4})", name)
    return int(m.group(1)) if m else None

def normalize_codigo_entidad(x):
    if pd.isna(x): return pd.NA
    s = re.sub(r"[^0-9]","", str(x))
    return s.zfill(PAD_LEN_CODIGO) if s else pd.NA

def detect_params(path: Path):
    for enc in ["utf-8-sig","utf-8","cp1252","latin-1"]:
        for sep in ["|",";","\t",",", None]:
            try:
                pd.read_csv(path, nrows=50, dtype=str, sep=sep, encoding=enc, engine="python")
                return enc, sep
            except:
                continue
    return "utf-8-sig", "|"

# ===================== COLUMNAS =========================

# Columnas que SÍ utilizaremos
KEEP_SLUGS = {
    "codigo_inei","nombre_entidad","tipo_entidad","tipo_gestion","licenciado",
    "tipo_constitucion","nivel_academico","sexo","edad","anio_nacimiento",
    "codigo_siu_programa","nombre_programa","codigo_grupo_3","nombre_grupo_3",
    "es_sede_principal","codigo_siu_filial","departamento_filial","provincia_filial"
}

# Columnas que debemos eliminar SIN EXCEPCIÓN
DROP_COLS = {
    "cert_gravedad","des_discapacidad_de_destreza","des_discapacidad_de_disposicion",
    "des_discapacidad_de_locomocion","des_discapacidad_de_situacion",
    "des_discapacidad_de_comunicacion","des_discapacidad_de_conducta",
    "des_discapacidad_del_cuidado","guid_persona"
}

# Alias para estandarizar nombres
ALIAS = {
    "codigo_inei": "codigo_entidad",
    "nombre_entidad": "entidad",
    "codigo_siu_programa": "codigo_programa",
    "nombre_programa": "programa",
    "codigo_grupo_3": "codigo_area",
    "nombre_grupo_3": "area_conocimiento",
    "codigo_siu_filial": "codigo_filial",
}

# Orden final del dataset
OUTPUT_COLS = [
    "codigo_entidad","entidad","codigo_programa","programa","codigo_area","area_conocimiento",
    "tipo_entidad","tipo_gestion","licenciado","tipo_constitucion","nivel_academico",
    "es_sede_principal","codigo_filial","departamento_filial","provincia_filial",
    "sexo","edad","anio","periodo"
]

# ===================== LIMPIEZA POR CHUNK =========================
def clean_chunk(chunk, anio):

    # 1) Slug de columnas
    chunk.rename(columns={c: slug(c) for c in chunk.columns}, inplace=True)

    # 2) Eliminar columnas prohibidas
    cols_to_drop = [c for c in chunk.columns if c in DROP_COLS]
    if cols_to_drop:
        chunk.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # 3) Quedarnos solo con las columnas útiles
    keep = [c for c in chunk.columns if c in KEEP_SLUGS]
    chunk = chunk[keep].copy()

    # 4) Renombrar columnas
    chunk.rename(columns=ALIAS, inplace=True)

    # 5) Normalizar texto
    for c in chunk.columns:
        if c != "edad":
            chunk[c] = chunk[c].astype("string").str.strip()

    # 6) Código de entidad
    if "codigo_entidad" in chunk.columns:
        chunk["codigo_entidad"] = chunk["codigo_entidad"].apply(normalize_codigo_entidad)
        chunk = chunk[chunk["codigo_entidad"].notna()]

    # 7) Limpiar edad (16–50)
    if "edad" in chunk.columns:
        e = pd.to_numeric(chunk["edad"], errors="coerce")
        chunk["edad"] = e.where((e >= AGE_MIN) & (e <= AGE_MAX)).astype("Int64")
        chunk = chunk[chunk["edad"].notna()]

    # 8) Sexo estándar
    if "sexo" in chunk.columns:
        chunk["sexo"] = chunk["sexo"].str.upper().replace({
            "M":"MASCULINO", "F":"FEMENINO"
        })

    # 9) Año y periodo
    chunk["anio"] = anio
    chunk["periodo"] = "ANUAL"

    # 10) Completar columnas faltantes
    for col in OUTPUT_COLS:
        if col not in chunk.columns:
            chunk[col] = pd.NA

    # 11) Orden final
    return chunk[OUTPUT_COLS].drop_duplicates()

# ===================== PROCESAR TODOS LOS ARCHIVOS =========================
def main():

    files = sorted(DATA_DIR.glob("ingresante_*.csv"))

    if OUTFILE.exists():
        os.remove(OUTFILE)

    # Crear archivo vacío
    pd.DataFrame(columns=OUTPUT_COLS).to_csv(OUTFILE, index=False, encoding="utf-8-sig")

    data_by_year = {y: [] for y in YEARS_ALLOWED}

    # === Leer archivos ===
    for f in files:
        anio = parse_anio(f.name)
        if anio not in YEARS_ALLOWED:
            print(f"Saltado {f.name} (año fuera de rango)")
            continue

        enc, sep = detect_params(f)
        print(f"Procesando {f.name} (año {anio})...")

        for chunk in pd.read_csv(f, dtype=str, sep=sep, encoding=enc, chunksize=CHUNKSIZE):
            cleaned = clean_chunk(chunk, anio)
            data_by_year[anio].append(cleaned)
            gc.collect()

    # === Submuestreo 5,000 por año ===
    final_frames = []
    for y in YEARS_ALLOWED:
        if not data_by_year[y]:
            continue
        df = pd.concat(data_by_year[y], ignore_index=True).drop_duplicates()
        if len(df) > ROWS_PER_YEAR:
            df = df.sample(ROWS_PER_YEAR, random_state=42)
        final_frames.append(df)

    # === Unión final ===
    full = pd.concat(final_frames, ignore_index=True).drop_duplicates()

    # === Guardar ===
    full.to_csv(OUTFILE, index=False, encoding="utf-8-sig")

    print("\n✔ ARCHIVO FINAL GENERADO:", OUTFILE)
    print("✔ Filas finales:", len(full))
    print("✔ Columnas:", list(full.columns))

if __name__ == "__main__":
    main()
