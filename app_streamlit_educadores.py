
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score


st.set_page_config(page_title="Regresión Polinomial – Educadores", page_icon="📘", layout="wide")

st.title("📘 Regresión Polinomial para Educadores")
st.write(
    "Ejemplo completo con el dataset de 1000 registros: **horas_preparacion, tamano_clase, anos_experiencia → eficacia_docente**. "
    "Puedes usar el dataset por defecto o subir tu propia versión con las mismas columnas."
)

# ---- Data loading ----
default_path = "dataset_regresion_polinomial_educadores.csv"
uploaded = st.file_uploader("Sube un CSV (opcional)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    # Si el archivo no se encuentra en el directorio actual, intentar ruta de trabajo alternativa (ej. sandbox)
    if not os.path.exists(default_path):
        alt_path = "dataset_regresion_polinomial_educadores.csv"
        if os.path.exists(alt_path):
            df = pd.read_csv(alt_path)
        else:
            st.error("No se encontró el dataset por defecto. Sube un CSV para continuar.")
            st.stop()
    else:
        df = pd.read_csv(default_path)

required_cols = ["horas_preparacion", "tamano_clase", "anos_experiencia", "eficacia_docente"]
if any(c not in df.columns for c in required_cols):
    st.error(f"El CSV debe contener las columnas: {required_cols}")
    st.stop()

st.subheader("1) Vista rápida de datos")
st.write("Primeras filas del dataset:")
st.dataframe(df.head(10))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas", f"{len(df):,}")
c2.metric("Horas prep. (media)", f"{df['horas_preparacion'].mean():.2f}")
c3.metric("Tamaño clase (media)", f"{df['tamano_clase'].mean():.2f}")
c4.metric("Experiencia (media)", f"{df['anos_experiencia'].mean():.2f}")

st.markdown("---")
st.subheader("2) Parámetros del modelo")

with st.form(key="params"):
    degree = st.slider("Grado polinomial", min_value=1, max_value=5, value=3, step=1)
    model_type = st.selectbox("Tipo de modelo", ["RidgeCV", "LassoCV", "LinearRegression"])
    #model_type = st.selectbox("Tipo de modelo", ["JULIO", "SILVIA", "RUTH","GLADYS"])
    test_size = st.slider("Tamaño de test (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
    random_state = st.number_input("Random State", min_value=0, value=42, step=1)
    kfold = st.slider("K-Fold CV (para *CV*)", min_value=3, max_value=10, value=5)
    submit = st.form_submit_button("Entrenar modelo")

# ---- Prepare data ----
X = df[["horas_preparacion", "tamano_clase", "anos_experiencia"]].values
y = df["eficacia_docente"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# ---- Build pipeline ----
if model_type == "RidgeCV":
    # alpha grid razonable
    alphas = np.logspace(-3, 3, 13)
    estimator = RidgeCV(alphas=alphas, cv=kfold)
elif model_type == "LassoCV":
    alphas = np.logspace(-3, 1, 13)
    estimator = LassoCV(alphas=alphas, cv=kfold, random_state=random_state, max_iter=10000)
else:
    estimator = LinearRegression()

pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
    ("scaler", StandardScaler()),
    ("est", estimator)
])

# ---- Train ----
if submit:
    pipe.fit(X_train, y_train)

    # Cross-Validation RMSE (solo para modelos con CV explícito usamos el score de la tubería)
    kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
    # Nota: cross_val_score usa score por defecto = r2; calculamos RMSE manualmente
    cv_rmse = []
    for tr_idx, va_idx in kf.split(X_train):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        pipe.fit(X_tr, y_tr)
        y_va_hat = pipe.predict(X_va)
        cv_rmse.append(np.sqrt(mean_squared_error(y_va, y_va_hat)))
    cv_rmse_mean = np.mean(cv_rmse)

    y_tr_hat = pipe.predict(X_train)
    y_te_hat = pipe.predict(X_test)

    rmse_tr = np.sqrt(mean_squared_error(y_train, y_tr_hat))
    rmse_te = np.sqrt(mean_squared_error(y_test, y_te_hat))
    r2_tr = r2_score(y_train, y_tr_hat)
    r2_te = r2_score(y_test, y_te_hat)

    st.markdown("### 3) Resultados del modelo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE (CV)", f"{cv_rmse_mean:.3f}")
    m2.metric("RMSE (Train)", f"{rmse_tr:.3f}")
    m3.metric("RMSE (Test)", f"{rmse_te:.3f}")
    m4.metric("R² (Test)", f"{r2_te:.3f}")

    if model_type in ["RidgeCV", "LassoCV"]:
        st.write("**Modelo seleccionado:**", model_type)
        if model_type == "RidgeCV":
            st.write("Alpha óptimo (RidgeCV):", float(pipe.named_steps["est"].alpha_))
        else:
            st.write("Alpha óptimo (LassoCV):", float(pipe.named_steps["est"].alpha_))

    st.markdown("---")
    st.subheader("4) Gráficos de diagnóstico")

    # 4.1 Predicción vs Real
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_te_hat)
    ax1.set_xlabel("Real (Test)")
    ax1.set_ylabel("Predicción (Test)")
    ax1.set_title("Predicción vs Real (Test)")
    st.pyplot(fig1)

    # 4.2 Residuos
    residuals = y_test - y_te_hat
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_te_hat, residuals)
    ax2.axhline(0)
    ax2.set_xlabel("Predicción (Test)")
    ax2.set_ylabel("Residuo")
    ax2.set_title("Residuos vs Predicción (Test)")
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("5) Curvas por variable (parciales)")

    # Helper: rango y función para PDP aproximado (manteniendo otras vars en su media)
    means = X_train.mean(axis=0)
    feature_names = ["horas_preparacion", "tamano_clase", "anos_experiencia"]
    ranges = [
        np.linspace(df["horas_preparacion"].min(), df["horas_preparacion"].max(), 100),
        np.linspace(df["tamano_clase"].min(), df["tamano_clase"].max(), 100),
        np.linspace(df["anos_experiencia"].min(), df["anos_experiencia"].max(), 100),
    ]

    for i, fname in enumerate(feature_names):
        grid = ranges[i]
        X_ref = np.tile(means, (len(grid), 1))
        X_ref[:, i] = grid
        y_grid = pipe.predict(X_ref)

        fig, ax = plt.subplots()
        ax.plot(grid, y_grid)
        ax.set_xlabel(fname)
        ax.set_ylabel("Eficacia docente (predicha)")
        ax.set_title(f"Curva parcial: {fname}")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("6) Exportar modelo entrenado (opcional)")
    st.write("Puedes reentrenar y luego exportar con `joblib` si deseas. Aquí solo mostramos el código de ejemplo:")
    st.code("""
    import joblib
    joblib.dump(pipe, "modelo_educadores.pkl")
    # Cargar después:
    # pipe = joblib.load("modelo_educadores.pkl")
    """)

else:
    st.info("Configura los parámetros y presiona **Entrenar modelo** para ver resultados.")
