"""
02_stats_tests.py
Contrastes de hipótesis, medidas de correlación/asociación y regresión logística
multivariable (OR ajustadas) con penalización ridge débil.

La regresión logística es descriptiva (asociaciones ajustadas). El modelo
predictivo se entrena y evalúa por separado en 03_modeling.py.

Entrada : data/stroke_clean.csv
Salida  : tables/t04-t08*.csv, tables/summary_stats.json

Ejecución (desde la raíz del repositorio):
    python code/02_stats_tests.py
"""
import os
import json
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(BASE, "tables")
os.makedirs(TAB, exist_ok=True)

TARGET = "stroke"
NUM_VARS = ["age", "avg_glucose_level", "bmi_imputed_eda"]
CAT_VARS = ["gender", "hypertension", "heart_disease", "ever_married",
            "work_type", "Residence_type", "smoking_status"]

df = pd.read_csv(os.path.join(BASE, "data", "stroke_clean.csv"))
results = {}


def cramers_v(chi2, n, r, c):
    return float(np.sqrt(chi2 / (n * min(r - 1, c - 1))))


# ---------------------------------------------------------------- 1. Numéricas frente a ictus
rng = np.random.default_rng(42)
g0_all = df.loc[df[TARGET] == 0]
g1_all = df.loc[df[TARGET] == 1]
rows = []
for v in NUM_VARS:
    g0 = g0_all[v].dropna().to_numpy()
    g1 = g1_all[v].dropna().to_numpy()
    # Shapiro-Wilk sobre submuestra de hasta 4 999 observaciones (límite del test)
    sub0 = rng.choice(g0, size=min(len(g0), 4999), replace=False)
    shapiro_p0 = stats.shapiro(sub0).pvalue
    U, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    n1, n0 = len(g1), len(g0)
    r_rb = 2 * U / (n1 * n0) - 1  # positivo si el grupo con ictus tiene valores mayores
    pooled_sd = np.sqrt(((n1 - 1) * g1.var(ddof=1) + (n0 - 1) * g0.var(ddof=1)) / (n1 + n0 - 2))
    d = (g1.mean() - g0.mean()) / pooled_sd
    rows.append({
        "Variable": v,
        "Mediana_sin_ictus": round(float(np.median(g0)), 2),
        "Mediana_ictus": round(float(np.median(g1)), 2),
        "U": float(U),
        "p_valor": p,
        "r_rango_biserial": round(r_rb, 3),
        "d_Cohen": round(d, 3),
        "Shapiro_p_sin_ictus": shapiro_p0,
    })
pd.DataFrame(rows).to_csv(os.path.join(TAB, "t04_test_numericas_vs_ictus.csv"), index=False)

# ---------------------------------------------------------------- 2. Categóricas frente a ictus
rows = []
for v in CAT_VARS:
    tab = pd.crosstab(df[v], df[TARGET])
    chi2, p, dof, expected = stats.chi2_contingency(tab)
    test = "Chi-cuadrado"
    if tab.shape == (2, 2) and (expected < 5).any():
        _, p = stats.fisher_exact(tab.values)
        test = "Fisher exacto"
    OR = np.nan
    if tab.shape == (2, 2):
        # OR de la segunda categoría frente a la primera (filas), para ictus (columnas)
        a, b = tab.iloc[1, 1], tab.iloc[1, 0]
        c, d = tab.iloc[0, 1], tab.iloc[0, 0]
        OR = (a * d) / (b * c) if b * c > 0 else np.nan
    rows.append({
        "Variable": v,
        "Test": test,
        "chi2": round(float(chi2), 2),
        "gl": int(dof),
        "p_valor": p,
        "V_Cramer": round(cramers_v(chi2, tab.values.sum(), *tab.shape), 3),
        "OR_cruda_segunda_vs_primera_categoria": OR,
    })
pd.DataFrame(rows).to_csv(os.path.join(TAB, "t05_test_categoricas_vs_ictus.csv"), index=False)

# ---------------------------------------------------------------- 3. Spearman (numéricas y binarias)
spearman = df[NUM_VARS + ["hypertension", "heart_disease", TARGET]].astype(float).corr(method="spearman")
spearman.round(3).to_csv(os.path.join(TAB, "t06_matriz_spearman.csv"))

# ---------------------------------------------------------------- 4. V de Cramér entre categóricas
rows = []
for v1, v2 in itertools.combinations(CAT_VARS, 2):
    tab = pd.crosstab(df[v1], df[v2])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    rows.append({
        "Variable1": v1,
        "Variable2": v2,
        "V_Cramer": round(cramers_v(chi2, tab.values.sum(), *tab.shape), 3),
        "p_valor": p,
    })
pd.DataFrame(rows).sort_values("V_Cramer", ascending=False).to_csv(
    os.path.join(TAB, "t07_cramerv_categoricas.csv"), index=False)

# ---------------------------------------------------------------- 5. Regresión logística (OR ajustadas)
# Diseño: numéricas + dummies (referencia = primera categoría alfabética).
# Separación cuasi-completa (gender=Other, work_type=Never_worked: 0 eventos)
# -> penalización L2 débil (lambda = 1) sobre todos los coeficientes salvo el intercepto.
num = df[NUM_VARS].astype(float)
dummies = pd.get_dummies(df[CAT_VARS].astype(str), drop_first=True).astype(float)
Xdf = pd.concat([num, dummies], axis=1)
Xdf.insert(0, "Intercepto", 1.0)
X = Xdf.to_numpy()
names = list(Xdf.columns)
y = df[TARGET].to_numpy(dtype=float)

LAMBDA = 1.0
mask = np.ones(X.shape[1])
mask[0] = 0.0  # no penalizar el intercepto


def neg_penalized_loglik(b):
    z = X @ b
    ll = np.sum(y * z - np.logaddexp(0.0, z))
    return -ll + 0.5 * LAMBDA * np.sum((mask * b) ** 2)


def gradient(b):
    z = X @ b
    p = 1.0 / (1.0 + np.exp(-z))
    return -X.T @ (y - p) + LAMBDA * mask * b


opt = minimize(neg_penalized_loglik, np.zeros(X.shape[1]), jac=gradient,
               method="BFGS", options={"maxiter": 1000, "gtol": 1e-8})
beta = opt.x

z = X @ beta
p_hat = 1.0 / (1.0 + np.exp(-z))
W = p_hat * (1.0 - p_hat)
fisher = (X * W[:, None]).T @ X + LAMBDA * np.diag(mask)
cov = np.linalg.inv(fisher)
se = np.sqrt(np.diag(cov))
z_scores = beta / se
p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

glm_table = pd.DataFrame({
    "coef": beta,
    "SE": se,
    "OR": np.exp(beta),
    "IC95_inf": np.exp(beta - 1.96 * se),
    "IC95_sup": np.exp(beta + 1.96 * se),
    "p_valor": p_values,
}, index=names)
glm_table.to_csv(os.path.join(TAB, "t08_regresion_logistica_multivariable.csv"))

ll_full = np.sum(y * z - np.logaddexp(0.0, z))  # sin penalización
p0 = y.mean()
ll_null = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
results["glm_pseudo_r2_mcfadden"] = float(1 - ll_full / ll_null)
results["glm_aic"] = float(2 * len(beta) - 2 * ll_full)
results["glm_n"] = int(len(y))
results["glm_converged"] = bool(np.max(np.abs(gradient(beta))) < 1e-3)
results["glm_max_abs_gradient"] = float(np.max(np.abs(gradient(beta))))
results["glm_ridge_lambda"] = LAMBDA

with open(os.path.join(TAB, "summary_stats.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(pd.read_csv(os.path.join(TAB, "t04_test_numericas_vs_ictus.csv")).to_string())
print(pd.read_csv(os.path.join(TAB, "t05_test_categoricas_vs_ictus.csv")).to_string())
print(glm_table.round(4).to_string())
print(json.dumps(results, indent=2, ensure_ascii=False))
print("STATS OK")
