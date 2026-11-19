"""
03_modeling.py
Modelos predictivos de ictus sin fuga de información.

PROTOCOLO (fijado antes de consultar el conjunto de test):
  1. Partición estratificada 80/20. El test NO se usa en ninguna decisión.
  2. Ajuste de hiperparámetros por validación cruzada de 5 pliegues en el
     entrenamiento, con AUC-PR como criterio.
  3. Modelo principal = mayor AUC-PR medio en validación cruzada.
  4. Umbral de decisión = mayor umbral con sensibilidad >= 0,80 en las
     predicciones fuera de pliegue (out-of-fold) del entrenamiento.
  5. Evaluación única en el test: umbral 0,5 para todos los modelos, y el
     umbral del protocolo para el modelo principal.
  6. El SMOTE se aplica solo dentro de cada pliegue de entrenamiento
     (SmoteLogReg lo hace dentro de fit()).

Demostración de fuga (SMOTE antes del reparto): solo con fines comparativos.

Entrada : data/stroke_clean.csv
Salida  : tables/t09-t13*.csv, tables/summary_modeling.json, figures/f06-f09*.png

Ejecución (desde la raíz del repositorio):
    python code/03_modeling.py
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from sklearn.base import BaseEstimator, ClassifierMixin, clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

RS = 42
np.random.seed(RS)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(BASE, "tables")
FIG = os.path.join(BASE, "figures")
for d in (TAB, FIG):
    os.makedirs(d, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
PALETTE = ["#3B6FA0", "#D98E33", "#5F9E6E", "#B4485A", "#8464A8"]

NUM = ["age", "avg_glucose_level", "bmi"]
CAT = ["gender", "hypertension", "heart_disease", "ever_married",
       "work_type", "Residence_type", "smoking_status"]
TARGET = "stroke"
THRESHOLD_TARGET_SENS = 0.80


# ---------------------------------------------------------------- Datos
def load_xy():
    df = pd.read_csv(os.path.join(BASE, "data", "stroke_clean.csv"))
    X = df[NUM + CAT].copy()
    for c in ["hypertension", "heart_disease"]:
        X[c] = X[c].astype(str)
    y = df[TARGET].to_numpy(dtype=int)
    return X, y


def build_preprocessor():
    """Imputación y escalado de numéricas; imputación y one-hot de categóricas.
    Siempre se ajusta dentro de fit() del modelo, nunca con datos de test."""
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUM),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])


# ---------------------------------------------------------------- SMOTE propio
def smote_oversample(X, y, k=5, seed=RS):
    """SMOTE (Chawla et al., 2002) sobre matrices densas.
    Genera observaciones sintéticas de la clase minoritaria por interpolación
    con uno de sus k vecinos más próximos, hasta igualar las dos clases."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    minority = classes[np.argmin(counts)]
    n_new = int(counts.max() - counts.min())
    if n_new <= 0:
        return X, y
    X_min = X[y == minority]
    k_use = min(k, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k_use + 1).fit(X_min)
    _, idx = nn.kneighbors(X_min)
    base = rng.integers(0, len(X_min), n_new)
    neighbor = idx[base, rng.integers(1, k_use + 1, n_new)]  # columna 0 es el propio punto
    gap = rng.random((n_new, 1))
    synthetic = X_min[base] + gap * (X_min[neighbor] - X_min[base])
    return np.vstack([X, synthetic]), np.concatenate([y, np.full(n_new, minority)])


class SmoteLogReg(BaseEstimator, ClassifierMixin):
    """Regresión logística con SMOTE dentro de fit(). Compatible con GridSearchCV
    y clone(): el remuestreo solo ve los datos que recibe fit()."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.pre_ = build_preprocessor()
        Xe = self.pre_.fit_transform(X)
        Xb, yb = smote_oversample(Xe, np.asarray(y))
        self.clf_ = LogisticRegression(C=self.C, max_iter=2000, random_state=RS).fit(Xb, yb)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.pre_.transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------- Utilidades de evaluación
def boot_ci(y, scores, metric, n=1000, seed=RS):
    rng = np.random.default_rng(seed)
    n_obs = len(y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        yt = y[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(metric(yt, scores[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def specificity(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def evaluate(name, y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc_lo, auc_hi = boot_ci(y, proba, roc_auc_score)
    rec_lo, rec_hi = boot_ci(y, pred, lambda a, b: recall_score(a, b, zero_division=0))
    return {
        "Modelo": name,
        "Umbral": round(float(thr), 4),
        "Exactitud": accuracy_score(y, pred),
        "Exactitud_balanceada": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Sensibilidad": recall_score(y, pred, zero_division=0),
        "Sensibilidad_IC_inf": rec_lo,
        "Sensibilidad_IC_sup": rec_hi,
        "Especificidad": specificity(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "AUC_ROC": roc_auc_score(y, proba),
        "AUC_ROC_IC_inf": auc_lo,
        "AUC_ROC_IC_sup": auc_hi,
        "AUC_PR": average_precision_score(y, proba),
        "Brier": brier_score_loss(y, proba),
        "VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp),
    }


def threshold_for_sensitivity(y, proba, target=THRESHOLD_TARGET_SENS):
    """Mayor umbral con sensibilidad >= target (se recorre de mayor a menor)."""
    for t in np.linspace(0.99, 0.01, 400):
        if recall_score(y, (proba >= t).astype(int), zero_division=0) >= target:
            return float(t)
    return 0.5


def ap_scorer(est, X, y):
    """AUC-PR con la columna de la clase positiva. Explícito para evitar
    diferencias entre versiones de scikit-learn con estimadores propios."""
    return average_precision_score(y, est.predict_proba(X)[:, 1])


def oof_proba(estimator, X, y, cv):
    """Predicciones fuera de pliegue dentro del entrenamiento."""
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        m = clone(estimator).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


# ---------------------------------------------------------------- 1. Partición
X, y = load_xy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RS)
summary = {
    "n_train": int(len(y_train)), "n_test": int(len(y_test)),
    "prevalencia_train": float(y_train.mean()), "prevalencia_test": float(y_test.mean()),
    "casos_test": int(y_test.sum()),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

# ---------------------------------------------------------------- 2. Búsqueda de hiperparámetros (solo entrenamiento)
candidates = {
    "LR sin balancear": (
        Pipeline([("prep", build_preprocessor()),
                  ("clf", LogisticRegression(max_iter=2000, random_state=RS))]),
        {"clf__C": [0.01, 0.1, 1, 10]}),
    "LR balanceada": (
        Pipeline([("prep", build_preprocessor()),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS))]),
        {"clf__C": [0.01, 0.1, 1, 10]}),
    "LR + SMOTE": (
        SmoteLogReg(),
        {"C": [0.01, 0.1, 1, 10]}),
    "Random Forest (balanceado)": (
        Pipeline([("prep", build_preprocessor()),
                  ("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                                 random_state=RS, n_jobs=-1))]),
        {"clf__n_estimators": [300, 600], "clf__max_depth": [4, 8, None],
         "clf__min_samples_leaf": [1, 5, 10]}),
    "Gradient Boosting (balanceado)": (
        Pipeline([("prep", build_preprocessor()),
                  ("clf", HistGradientBoostingClassifier(class_weight="balanced", random_state=RS))]),
        {"clf__max_iter": [150, 300], "clf__learning_rate": [0.03, 0.1],
         "clf__max_depth": [3, 6, None]}),
}

fitted = {}
cv_rows = []
for name, (est, grid) in candidates.items():
    gs = GridSearchCV(est, grid, scoring=ap_scorer, cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    fitted[name] = gs.best_estimator_
    cv_rows.append({"Modelo": name, "Hiperparametros": str(gs.best_params_),
                    "AUC_PR_CV_medio": gs.best_score_})

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
fitted["Baseline (mayoritaria)"] = baseline
cv_rows.append({"Modelo": "Baseline (mayoritaria)", "Hiperparametros": "-", "AUC_PR_CV_medio": np.nan})

cv_df = pd.DataFrame(cv_rows).sort_values("AUC_PR_CV_medio", ascending=False)
cv_df.to_csv(os.path.join(TAB, "t09_hiperparametros_cv.csv"), index=False)

# ---------------------------------------------------------------- 3. Modelo principal y umbral (sin test)
primary_name = cv_df.iloc[0]["Modelo"]
primary = fitted[primary_name]
oof = oof_proba(primary, X_train, y_train, cv)
thr_primary = threshold_for_sensitivity(y_train, oof)
summary.update({
    "modelo_principal": primary_name,
    "umbral_protocolo": thr_primary,
    "sensibilidad_oof_con_umbral": float(recall_score(y_train, (oof >= thr_primary).astype(int))),
})

# ---------------------------------------------------------------- 4. Evaluación única en test
test_rows = []
test_proba = {}
for name, est in fitted.items():
    proba = est.predict_proba(X_test)[:, 1]
    test_proba[name] = proba
    test_rows.append(evaluate(name, y_test, proba, 0.5))
test_rows.append(evaluate(f"{primary_name} (umbral del protocolo)",
                          y_test, test_proba[primary_name], thr_primary))
test_df = pd.DataFrame(test_rows).sort_values("AUC_ROC", ascending=False)
test_df.to_csv(os.path.join(TAB, "t10_comparativa_modelos_test.csv"), index=False)

cm_df = test_df[["Modelo", "VN", "FP", "FN", "VP"]]
cm_df.to_csv(os.path.join(TAB, "t11_matrices_confusion_test.csv"), index=False)

# ---------------------------------------------------------------- 5. Figuras de rendimiento
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for i, name in enumerate(test_proba):
    fpr, tpr, _ = roc_curve(y_test, test_proba[name])
    axes[0].plot(fpr, tpr, label=name, color=PALETTE[i % len(PALETTE)], lw=1.6)
    prec, rec, _ = precision_recall_curve(y_test, test_proba[name])
    axes[1].plot(rec, prec, label=name, color=PALETTE[i % len(PALETTE)], lw=1.6)
axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
axes[0].set_xlabel("1 - Especificidad (FPR)")
axes[0].set_ylabel("Sensibilidad (TPR)")
axes[0].set_title("Curvas ROC (test)")
axes[0].legend(fontsize=6.5, loc="lower right")
axes[1].axhline(y_test.mean(), ls="--", color="grey", lw=1, label="Prevalencia base")
axes[1].set_xlabel("Recall (sensibilidad)")
axes[1].set_ylabel("Precisión")
axes[1].set_title("Curvas precisión-recall (test)")
axes[1].legend(fontsize=6.5, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f06_roc_pr_curvas.png"), dpi=200)
plt.close(fig)

# Matrices de confusión a umbral 0,5 de los dos modelos con mayor AUC-ROC en test
top2 = [n for n in test_df["Modelo"] if n in test_proba][:2]
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, name in zip(axes, top2):
    cm = confusion_matrix(y_test, (test_proba[name] >= 0.5).astype(int), labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No ictus", "Ictus"], yticklabels=["No ictus", "Ictus"])
    ax.set_title(name, fontsize=8)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f08_matrices_confusion.png"), dpi=200)
plt.close(fig)

# Calibración del modelo principal (test)
frac_pos, mean_pred = calibration_curve(y_test, test_proba[primary_name], n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(mean_pred, frac_pos, "o-", color=PALETTE[0], label=primary_name)
ax.plot([0, 1], [0, 1], "--", color="grey")
ax.set_xlabel("Probabilidad predicha media")
ax.set_ylabel("Frecuencia observada de ictus")
ax.set_title("Curva de calibración (test)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f07_calibracion.png"), dpi=200)
pd.DataFrame({"prob_predicha_media": mean_pred, "frecuencia_observada": frac_pos}).to_csv(
    os.path.join(TAB, "t15_calibracion_modelo_principal.csv"), index=False)
plt.close(fig)

# ---------------------------------------------------------------- 6. Importancia de variables
# 6a. Gini de Random Forest (figura, descriptiva)
rf_pipe = fitted["Random Forest (balanceado)"]
rf_names = rf_pipe.named_steps["prep"].get_feature_names_out()
rf_imp = pd.DataFrame({"Variable": rf_names,
                       "Importancia_Gini": rf_pipe.named_steps["clf"].feature_importances_})
rf_imp = rf_imp.sort_values("Importancia_Gini", ascending=False)
rf_imp.to_csv(os.path.join(TAB, "t12_importancia_rf_gini.csv"), index=False)

fig, ax = plt.subplots(figsize=(6.5, 5))
top = rf_imp.head(12).iloc[::-1]
ax.barh(top["Variable"], top["Importancia_Gini"], color=PALETTE[0])
ax.set_xlabel("Importancia (Random Forest, Gini)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f09_importancia_variables.png"), dpi=200)
plt.close(fig)

# 6b. Permutación sobre test del modelo principal, agregada por variable original
pi = permutation_importance(primary, X_test, y_test, n_repeats=30,
                            random_state=RS, scoring=ap_scorer)
perm = pd.DataFrame({"Variable": X_test.columns,
                     "Importancia_media": pi.importances_mean,
                     "Importancia_DE": pi.importances_std}).sort_values("Importancia_media", ascending=False)
perm.to_csv(os.path.join(TAB, "t13_importancia_permutacion_modelo_principal.csv"), index=False)

# ---------------------------------------------------------------- 7. Demostración de fuga de datos
# A) Diseño CON fuga: SMOTE sobre TODO el conjunto, después reparto 80/20.
#    El test resultante está balanceado (~50 % de positivos).
pre_all = build_preprocessor()
X_all_enc = pre_all.fit_transform(X)
Xb_all, yb_all = smote_oversample(X_all_enc, y)
Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(
    Xb_all, yb_all, test_size=0.20, stratify=yb_all, random_state=RS)
rf_leak = RandomForestClassifier(n_estimators=300, random_state=RS).fit(Xl_tr, yl_tr)
p_leak = rf_leak.predict_proba(Xl_te)[:, 1]

# B) Diseño CORRECTO con la misma configuración base: SMOTE solo en entrenamiento,
#    evaluado sobre el test ORIGINAL (prevalencia ~4,9 %).
pre_ok = build_preprocessor()
Xe_tr = pre_ok.fit_transform(X_train)
Xe_te = pre_ok.transform(X_test)
Xb_ok, yb_ok = smote_oversample(Xe_tr, y_train)
rf_ok = RandomForestClassifier(n_estimators=300, random_state=RS).fit(Xb_ok, yb_ok)
p_ok = rf_ok.predict_proba(Xe_te)[:, 1]

leak_rows = [
    {"Diseno": "Con fuga (SMOTE antes del reparto)", "Conjunto_test": "balanceado (SMOTE)",
     "n_test": int(len(yl_te)), "prevalencia_test": float(yl_te.mean()),
     "AUC_ROC": roc_auc_score(yl_te, p_leak),
     "Sensibilidad": recall_score(yl_te, (p_leak >= 0.5).astype(int), zero_division=0),
     "Exactitud": accuracy_score(yl_te, (p_leak >= 0.5).astype(int))},
    {"Diseno": "Sin fuga (SMOTE solo en entrenamiento)", "Conjunto_test": "original",
     "n_test": int(len(y_test)), "prevalencia_test": float(y_test.mean()),
     "AUC_ROC": roc_auc_score(y_test, p_ok),
     "Sensibilidad": recall_score(y_test, (p_ok >= 0.5).astype(int), zero_division=0),
     "Exactitud": accuracy_score(y_test, (p_ok >= 0.5).astype(int))},
]
# NOTA: ambas filas usan test distintos; no son comparables directamente.
# La comparación válida requiere evaluar ambos diseños sobre el mismo test original.
pd.DataFrame(leak_rows).to_csv(os.path.join(TAB, "t14_demostracion_fuga.csv"), index=False)

# ---------------------------------------------------------------- 8. Resumen
summary.update({
    "auc_roc_modelo_principal_test": float(test_df.loc[test_df["Modelo"] == primary_name, "AUC_ROC"].iloc[0]),
    "leak_demo": leak_rows,
})
with open(os.path.join(TAB, "summary_modeling.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

print(cv_df.to_string(index=False))
print(test_df.round(3).to_string(index=False))
print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
print("MODELING OK")
