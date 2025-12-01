# unir_data_limpia.py (VERSIÓN FINAL 2025)
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

# ===================== RUTAS =====================
BASE = Path("data")
OUT_XLSX = "dataoficialunida.xlsx"
OUT_CSV  = "dataoficialunida.csv"

# Archivos limpios finales
FILES = {
    "INGRESANTE":  BASE / "data_ingresante" / "ingresantes_limpio.csv",
    "DOCENTE":     BASE / "data_docente" / "docentes_limpio.csv",
    "MATRICULADO": BASE / "data_matriculado" / "matriculados_limpio.csv",
    "POSTULANTE":  BASE / "data_postulante" / "postulantes_limpio.csv",
}

# Máximo de filas por dataset (AJUSTADO A 20K)
MAX_ROWS = 20000

# ===================== PROCESAR =====================
dfs = {}
for grp, path in FILES.items():
    print(f"📥 Leyendo {grp}: {path}")

    # Cargar
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")

    # Tomar máximo 20K filas
    df = df.head(MAX_ROWS).copy()

    # Prefijar columnas por grupo
    df = df.add_prefix(f"{grp}__")

    dfs[grp] = df
    print(f" - {grp}: {df.shape[0]} filas, {df.shape[1]} columnas")

# ===================== ALINEAR A MISMA LONGITUD =====================
max_len = max(len(df) for df in dfs.values())
print(f"\n📌 Longitud final unificada: {max_len} filas\n")

for grp in dfs:
    dfs[grp] = dfs[grp].reindex(range(max_len))

# ===================== UNIÓN HORIZONTAL =====================
df_final = pd.concat(dfs.values(), axis=1)

# ===================== EXPORTAR =====================
print(f"📤 Exportando a Excel: {OUT_XLSX}")
df_final.to_excel(OUT_XLSX, index=False, engine="openpyxl")

print(f"📤 Exportando a CSV: {OUT_CSV}")
df_final.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("\n✅ UNIÓN COMPLETADA")
print(f"📌 Filas: {len(df_final)} | Columnas: {len(df_final.columns)}")
print(f"📁 Archivos generados:")
print(f"  - {OUT_XLSX}")
print(f"  - {OUT_CSV}")
