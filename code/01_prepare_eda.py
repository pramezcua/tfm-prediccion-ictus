"""
01_prepare_eda.py
Auditoría de calidad, limpieza y análisis exploratorio del TFM de predicción de ictus.

Entrada : data/stroke.csv        (Kaggle, "Stroke Prediction Dataset")
Salida  : data/stroke_clean.csv, tables/t01-t03*.csv, tables/summary_eda.json,
          figures/f01-f05*.png

Ejecución (desde la raíz del repositorio):
    python code/01_prepare_eda.py
"""
import os
import json

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN = os.path.join(BASE, "data", "stroke.csv")
DATA_OUT = os.path.join(BASE, "data", "stroke_clean.csv")
TAB = os.path.join(BASE, "tables")
FIG = os.path.join(BASE, "figures")
for d in (TAB, FIG):
    os.makedirs(d, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
PALETTE = ["#3B6FA0", "#D98E33"]
TARGET = "stroke"
NUM_VARS = ["age", "avg_glucose_level", "bmi"]
CAT_VARS = ["gender", "hypertension", "heart_disease", "ever_married",
            "work_type", "Residence_type", "smoking_status"]

summary = {}

# ---------------------------------------------------------------- 1. Carga y auditoría
# El CSV de Kaggle está separado por comas.
df = pd.read_csv(DATA_IN, sep=",")
summary["n_raw"] = int(len(df))

audit = pd.DataFrame({
    "n_missing": df.isna().sum(),
    "pct_missing": (df.isna().mean() * 100).round(2),
    "dtype": df.dtypes.astype(str),
    "n_unique": df.nunique(),
})
audit.to_csv(os.path.join(TAB, "t01_auditoria_calidad.csv"))

summary["n_duplicated_id"] = int(df["id"].duplicated().sum())
summary["n_missing_bmi"] = int(df["bmi"].isna().sum())
summary["pct_missing_bmi"] = round(float(df["bmi"].isna().mean() * 100), 2)
summary["n_gender_other"] = int((df["gender"] == "Other").sum())

# ---------------------------------------------------------------- 2. Limpieza
df = df.drop(columns=["id"]).copy()

df["age_group"] = pd.cut(
    df["age"], bins=[0, 18, 40, 60, 80, 100],
    labels=["0-18", "19-40", "41-60", "61-80", "81+"],
    right=True, include_lowest=True,
)
# Imputación DESCRIPTIVA del IMC: mediana estratificada por grupo de edad.
# Se usa solo para EDA y contrastes. El modelado imputa dentro del pipeline,
# ajustado solo con datos de entrenamiento (ver 03_modeling.py).
med_by_age = df.groupby("age_group", observed=True)["bmi"].median()
df["bmi_imputed_eda"] = df["bmi"]
for grp, med in med_by_age.items():
    mask = (df["age_group"] == grp) & df["bmi"].isna()
    df.loc[mask, "bmi_imputed_eda"] = med

summary["n_after_cleaning"] = int(len(df))
df.drop(columns=["age_group"]).to_csv(DATA_OUT, index=False)

# ---------------------------------------------------------------- 3. Tabla 2: numéricas
g0 = df[df[TARGET] == 0]
g1 = df[df[TARGET] == 1]
rows = []
for v in ["age", "avg_glucose_level", "bmi_imputed_eda"]:
    s = df[v]
    rows.append({
        "Variable": v,
        "Global_media_DE": f"{s.mean():.2f} ± {s.std():.2f}",
        "Global_mediana_RIQ": f"{s.median():.2f} [{s.quantile(.25):.2f}-{s.quantile(.75):.2f}]",
        "Sin_ictus_media_DE": f"{g0[v].mean():.2f} ± {g0[v].std():.2f}",
        "Ictus_media_DE": f"{g1[v].mean():.2f} ± {g1[v].std():.2f}",
    })
pd.DataFrame(rows).to_csv(os.path.join(TAB, "t02_descriptivos_numericos.csv"), index=False)

# ---------------------------------------------------------------- 4. Tabla 3: categóricas
cat_rows = []
for v in CAT_VARS:
    for cat, sub in df.groupby(v, observed=True):
        n = len(sub)
        n1 = int((sub[TARGET] == 1).sum())
        cat_rows.append({
            "Variable": v,
            "Categoria": str(cat),
            "n": n,
            "pct_global": round(n / len(df) * 100, 2),
            "n_ictus": n1,
            "pct_ictus_en_categoria": round(n1 / n * 100, 2),
        })
pd.DataFrame(cat_rows).to_csv(os.path.join(TAB, "t03_descriptivos_categoricos.csv"), index=False)

# ---------------------------------------------------------------- 5. Balance de clases
bal = df[TARGET].value_counts()
summary["n_stroke"] = int(bal.get(1, 0))
summary["n_no_stroke"] = int(bal.get(0, 0))
summary["pct_stroke"] = round(float(bal.get(1, 0) / len(df) * 100), 3)

# ---------------------------------------------------------------- 6. Figuras
# f01: distribución de la edad por grupo
fig, ax = plt.subplots(figsize=(6.5, 4))
sns.histplot(data=df, x="age", hue=TARGET, palette=PALETTE, bins=30,
             stat="density", common_norm=False, alpha=0.6, ax=ax)
ax.set_xlabel("Edad (años)")
ax.set_ylabel("Densidad")
ax.legend_.set_title("Ictus")
for t, lab in zip(ax.legend_.texts, ["No", "Sí"]):  # hue ordenado 0 (no), 1 (sí)
    t.set_text(lab)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f01_edad_densidad.png"), dpi=200)
plt.close(fig)

# f02: glucosa e IMC por grupo
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
sns.boxplot(data=df, x=TARGET, y="avg_glucose_level", hue=TARGET, palette=PALETTE,
            legend=False, ax=axes[0])
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["No", "Sí"])
axes[0].set_xlabel("Ictus")
axes[0].set_ylabel("Glucosa media (mg/dL)")
sns.boxplot(data=df, x=TARGET, y="bmi_imputed_eda", hue=TARGET, palette=PALETTE,
            legend=False, ax=axes[1])
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["No", "Sí"])
axes[1].set_xlabel("Ictus")
axes[1].set_ylabel("IMC (kg/m²)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f02_glucosa_imc_boxplot.png"), dpi=200)
plt.close(fig)

# f03: prevalencia por categoría (etiquetas en español)
pairs = [("hypertension", "Hipertensión"), ("heart_disease", "Cardiopatía"),
         ("smoking_status", "Tabaquismo"), ("work_type", "Tipo de trabajo")]
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, (v, label) in zip(axes.flat, pairs):
    rates = (df.groupby(v, observed=True)[TARGET].mean() * 100).sort_values(ascending=False)
    ax.bar(rates.index.astype(str), rates.values, color=PALETTE[0])
    ax.set_title(label)
    ax.set_ylabel("% con ictus")
    ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f03_prevalencia_categoricas.png"), dpi=200)
plt.close(fig)

# f04: correlación de Spearman entre numéricas y binarias
corr_vars = ["age", "avg_glucose_level", "bmi_imputed_eda", "hypertension", "heart_disease", TARGET]
corr = df[corr_vars].astype(float).corr(method="spearman")
corr.round(3).to_csv(os.path.join(TAB, "t04b_matriz_spearman_figura4.csv"))
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={"label": "ρ de Spearman"}, ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f04_correlacion_spearman.png"), dpi=200)
plt.close(fig)

# f05: balance de clases
fig, ax = plt.subplots(figsize=(4, 4))
bal_plot = bal.rename({0: "No ictus", 1: "Ictus"})
bal_plot.plot(kind="bar", color=PALETTE, ax=ax, rot=0)
ax.set_ylabel("Nº de personas")
ax.set_xlabel("")
for i, val in enumerate(bal_plot.values):
    ax.text(i, val + 30, str(val), ha="center")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f05_balance_clases.png"), dpi=200)
plt.close(fig)

with open(os.path.join(TAB, "summary_eda.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("EDA OK")
