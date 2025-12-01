# unirmatriculados.py — VERSIÓN FINAL ESTABLE 2025
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import re, unicodedata, os, gc

# ============================================================
# RUTAS
# ============================================================
DATA_DIR = Path("data/data_matriculado")
OUTFILE  = DATA_DIR / "matriculados_limpio.csv"

# ============================================================
# CONFIGURACIÓN
# ============================================================
CHUNKSIZE = 20000
YEARS_ALLOWED = {2022, 2023, 2024, 2025}
ROWS_PER_YEAR = 5000

AGE_MIN, AGE_MAX = 16, 60
PAD_LEN_CODIGO = 9

# ============================================================
# HELPERS
# ============================================================
def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^0-9a-zA-Z]+","_", s.lower()).strip("_")
    return re.sub(r"_+","_", s)

def parse_anio_periodo(name):
    m = re.search(r"matriculado_(\d{4})_?([iIvV]+)?", name)
    if not m:
        return None, None
    anio = int(m.group(1))
    periodo = m.group(2).upper() if m.group(2) else "ANUAL"
    return anio, periodo

def detect_params(path: Path):
    encodings = ["utf-8-sig","utf-8","latin-1","cp1252"]
    seps      = ["|", ";", "\t", ",", None]
    for enc in encodings:
        for sep in seps:
            try:
                pd.read_csv(path, dtype=str, nrows=50, sep=sep, encoding=enc, engine="python")
                return enc, sep
            except:
                continue
    return "latin-1", "|"

def normalize_codigo_entidad(x):
    if x is None or pd.isna(x):
        return pd.NA
    s = re.sub(r"[^0-9]", "", str(x))
    return s.zfill(PAD_LEN_CODIGO) if s else pd.NA

def clean_age(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= AGE_MIN) & (s <= AGE_MAX))

# ============================================================
# COLUMNAS A BORRAR
# ============================================================
DROP_COLUMNS = {
    "guid_persona",
    "anio_nacimiento",
    "periodo",
    "periodo_estandarizado",
    "codigo_ubigeo_inei_local",
    "cert_gravedad",
    "des_discapacidad_de_comunicacion",
    "des_discapacidad_de_conducta",
    "des_discapacidad_de_destreza",
    "des_discapacidad_de_disposicion",
    "des_discapacidad_de_locomocion",
    "des_discapacidad_de_situacion",
    "des_discapacidad_del_cuidado"
}

# ============================================================
# COLUMNAS FINALES A CONSERVAR (VALIDADAS)
# ============================================================
KEEP_COLUMNS = [
    "codigo_inei","nombre_entidad",
    "tipo_entidad","tipo_gestion","tipo_constitucion","licencia",
    "nivel_academico","periodo_lectivo",
    "codigo_siu_programa","codigo_grupo_1","nombre_grupo_1",
    "codigo_grupo_3","nombre_grupo_3","nombre_programa",
    "es_local_principal","codigo_local",
    "departamento_local","provincia_local","distrito_local",
    "sexo","edad","nacionalidad","departamento_nacimiento",
    "anio_periodo_ingreso","fecha_inicio_periodo","fecha_fin_periodo",
    "anio","periodo"
]

# ============================================================
# LIMPIEZA DE CADA CHUNK
# ============================================================
def clean_chunk(df, anio, periodo):

    df = df.copy()  # evitar SettingWithCopyWarning
    df.rename(columns={c: slug(c) for c in df.columns}, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]

    # borrar columnas indeseadas
    cols = [c for c in df.columns if c in DROP_COLUMNS]
    if cols:
        df.drop(columns=cols, inplace=True, errors="ignore")

    # normalizar código INEI
    if "codigo_inei" in df.columns:
        df["codigo_inei"] = df["codigo_inei"].apply(normalize_codigo_entidad)
        df = df[df["codigo_inei"].notna()]

    # limpiar edad
    if "edad" in df.columns:
        df["edad"] = clean_age(df["edad"])
        df = df[df["edad"].notna()].copy()
        df["edad"] = df["edad"].astype("Int64")

    # estandarizar texto
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    if "sexo" in df.columns:
        df["sexo"] = df["sexo"].str.upper().replace({"M":"MASCULINO","F":"FEMENINO"})

    # agregar año y periodo
    df["anio"] = anio
    df["periodo"] = periodo

    # completar columnas faltantes
    for col in KEEP_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[KEEP_COLUMNS].drop_duplicates()
    return df

# ============================================================
# MAIN
# ============================================================
def main():

    files = sorted(list(DATA_DIR.glob("matriculado_*.csv")) +
                   list(DATA_DIR.glob("matriculado_*.xlsx")))

    print("Archivos encontrados:")
    for f in files:
        print(" -", f.name)

    all_by_year = {y: [] for y in YEARS_ALLOWED}

    for f in files:
        anio, periodo = parse_anio_periodo(f.name)
        if anio not in YEARS_ALLOWED:
            continue

        print(f"\nProcesando: {f.name}")

        # Excel
        if f.suffix.lower() == ".xlsx":
            df = pd.read_excel(f, dtype=str)
            cleaned = clean_chunk(df, anio, periodo)
            all_by_year[anio].append(cleaned)
            continue

        # CSV
        enc, sep = detect_params(f)
        print(f" → encoding={enc}, sep='{sep}'")

        for chunk in pd.read_csv(f, dtype=str, sep=sep, encoding=enc,
                                 chunksize=CHUNKSIZE, on_bad_lines="skip"):
            chunk = chunk.dropna(how="all")
            cleaned = clean_chunk(chunk, anio, periodo)
            all_by_year[anio].append(cleaned)
            gc.collect()

    # unir todo
    final_list = []
    for y in YEARS_ALLOWED:
        if not all_by_year[y]:
            continue
        dfy = pd.concat(all_by_year[y], ignore_index=True).drop_duplicates()
        if len(dfy) > ROWS_PER_YEAR:
            dfy = dfy.sample(ROWS_PER_YEAR, random_state=42)
        print(f"Año {y}: {len(dfy)} filas.")
        final_list.append(dfy)

    full = pd.concat(final_list, ignore_index=True).drop_duplicates()

    if OUTFILE.exists():
        os.remove(OUTFILE)

    full.to_csv(OUTFILE, index=False, encoding="utf-8-sig")

    print("\n✔ Archivo final generado:", OUTFILE)
    print("✔ Total filas:", len(full))
    print("✔ Columnas:", list(full.columns))

if __name__ == "__main__":
    main()
