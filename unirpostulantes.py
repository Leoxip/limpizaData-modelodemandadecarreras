# unirpostulantes.py — VERSIÓN FINAL 2025 LIMPIEZA DEFINITIVA (20 VARIABLES)
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import re, unicodedata, os, gc

# ========= RUTAS =========
DATA_DIR = Path("data/data_postulante")
OUTFILE = DATA_DIR / "postulantes_limpio.csv"

# ========= CONFIG =========
CHUNKSIZE = 15000
YEARS_ALLOWED = {2022, 2023, 2024, 2025}
ROWS_PER_YEAR = 5000
AGE_MIN, AGE_MAX = 16, 50
PAD_LEN_CODIGO = 9

# ========= HELPERS =========

def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^0-9a-zA-Z]+","_", s.lower()).strip("_")
    return re.sub(r"_+","_", s)

def parse_year(name: str):
    m = re.search(r"postulante_(\d{4})", name)
    return int(m.group(1)) if m else None

def detect_params(path: Path):
    for enc in ["utf-8-sig","latin-1","cp1252"]:
        try:
            pd.read_csv(path, nrows=10, sep="|", encoding=enc)
            return enc
        except:
            pass
    return "latin-1"

def normalize_codigo_entidad(x):
    if pd.isna(x):
        return pd.NA
    s = re.sub(r"[^0-9]", "", str(x))
    return s.zfill(PAD_LEN_CODIGO) if s else pd.NA

def clean_age(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= AGE_MIN) & (s <= AGE_MAX))


# ========= COLUMNAS DEFINITIVAS =========

KEEP_COLS = [

    # Institucional
    "codigo_entidad","entidad","tipo_entidad","tipo_gestion",
    "licenciado","tipo_constitucion","nivel_academico",

    # Proceso admisión
    "proceso_admision",
    "modalidad_ingreso","modalidad_ingreso_grupo",
    "proceso_estandarizado","guid_persona",

    # Demografía
    "sexo","nacionalidad",
    "departamento_nacimiento","anio_nacimiento","edad",

    # Primera opción vocacional
    "codigo_siu_programa_primera_opcion",
    "codigo_grupo_1_primera_opcion",
    "nombre_grupo_1_primera_opcion",
    "nombre_programa_primera_opcion"
]


# ========= ALIAS ENTRE NOMBRES =========

ALIAS = {
    "codigo_inei": "codigo_entidad",
    "nombre_entidad": "entidad",
    "codigo_siu_programa_primer_opcion": "codigo_siu_programa_primera_opcion"
}


# ========= COLUMNAS QUE SE ELIMINAN =========

DROP_COLS = {

    # Filiales / sede
    "es_sede_principal","codigo_filial","departamento_filial","provincia_filial",
    "cert_gravedad","es_ingresante",

    # 2da opción
    "codigo_siu_programa_segunda_opcion",
    "codigo_grupo_1_segunda_opcion",
    "nombre_grupo_1_segunda_opcion",
    "codigo_grupo_3_segunda_opcion",
    "nombre_grupo_3_segunda_opcion",
    "nombre_programa_segunda_opcion",

    # 3ra opción
    "codigo_siu_programa_tercera_opcion",
    "codigo_grupo_1_tercera_opcion",
    "nombre_grupo_1_tercera_opcion",
    "codigo_grupo_3_tercera_opcion",
    "nombre_grupo_3_tercera_opcion",
    "nombre_programa_tercera_opcion",

    # Grupo 3 de primera opción
    "codigo_grupo_3_primera_opcion",
    "nombre_grupo_3_primera_opcion",

    # Discapacidades
    "des_discapacidad_de_destresa",
    "des_discapacidad_de_disposicion",
    "des_discapacidad_de_locomocion",
    "des_discapacidad_de_situacion",
    "des_discapacidad_del_cuidado",
    "des_discapacidad_de_comunicacion",
    "des_discapacidad_de_conducta"
}


# ========= LIMPIEZA POR CHUNK =========

def clean_chunk(df, anio):

    df.columns = [slug(c) for c in df.columns]

    # Aplicar alias
    for k,v in ALIAS.items():
        if k in df.columns:
            df.rename(columns={k:v}, inplace=True)

    # Drop basura
    cols_to_drop = [c for c in df.columns if c in DROP_COLS]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Filtrar columnas válidas
    df = df[[c for c in df.columns if c in KEEP_COLS]].copy()

    # Normalizaciones
    if "codigo_entidad" in df.columns:
        df["codigo_entidad"] = df["codigo_entidad"].apply(normalize_codigo_entidad)
        df = df[df["codigo_entidad"].notna()]

    if "edad" in df.columns:
        df["edad"] = clean_age(df["edad"])
        df = df[df["edad"].notna()]
        df["edad"] = df["edad"].astype("Int64")

    if "sexo" in df.columns:
        df["sexo"] = df["sexo"].str.upper().replace({
            "M":"MASCULINO",
            "F":"FEMENINO",
            "H":"MASCULINO",
            "MUJER":"FEMENINO",
            "HOMBRE":"MASCULINO"
        })

    # Año automático
    df["anio_nacimiento"] = pd.to_numeric(df.get("anio_nacimiento"), errors="coerce")

    # Completar columnas faltantes
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # Reordenar columnas
    df = df[KEEP_COLS]

    return df.drop_duplicates()


# ========= MAIN =========

def main():

    files = sorted(DATA_DIR.glob("postulante_*.csv"))

    if OUTFILE.exists():
        os.remove(OUTFILE)

    frames_by_year = {}

    for f in files:

        print("Procesando:", f.name)
        anio = parse_year(f.name)
        if anio not in YEARS_ALLOWED:
            continue

        enc = detect_params(f)

        for chunk in pd.read_csv(
                f,
                sep="|",
                encoding=enc,
                chunksize=CHUNKSIZE,
                dtype=str,
                on_bad_lines="skip"):

            chunk.dropna(how="all", inplace=True)

            cleaned = clean_chunk(chunk, anio)

            if anio not in frames_by_year:
                frames_by_year[anio] = cleaned
            else:
                frames_by_year[anio] = pd.concat(
                    [frames_by_year[anio], cleaned],
                    ignore_index=True
                ).drop_duplicates()

            gc.collect()

    final_list = []

    for y in sorted(YEARS_ALLOWED):

        if y not in frames_by_year:
            continue

        df = frames_by_year[y]

        if len(df) > ROWS_PER_YEAR:
            df = df.sample(ROWS_PER_YEAR, random_state=42)

        print(f"Año {y}: {len(df)} registros finales.")
        final_list.append(df)


    full = pd.concat(final_list, ignore_index=True)

    full.to_csv(OUTFILE, index=False, encoding="utf-8-sig")

    print("\n✔ CSV FINAL CREADO:", OUTFILE)
    print("✔ Total filas:", len(full))
    print("✔ Columnas:", list(full.columns))


if __name__ == "__main__":
    main()
