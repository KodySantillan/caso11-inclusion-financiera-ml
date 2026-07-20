import streamlit as st
import joblib
import numpy as np
from supabase import create_client, Client

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Inclusión Financiera SBS - Caso 11",
    page_icon="🏦",
    layout="centered"
)

# ── Conexión segura a Supabase ───────────────────────────────────────────────
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# ── Carga del modelo ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/modelo_entrenado.pkl")

model = load_model()

# ── Interfaz principal ───────────────────────────────────────────────────────
st.title("🏦 Clasificador de Inclusión Financiera")
st.subheader("Caso 11 — SBS Perú | Algoritmo: KNN (K=9)")
st.markdown("""
Ingresa los indicadores financieros de una provincia para clasificar
su nivel de inclusión financiera: **Bajo**, **Medio** o **Alto**.
""")

st.divider()

# ── Formulario de entrada ────────────────────────────────────────────────────
st.markdown("### 📊 Indicadores de la Provincia")

col1, col2 = st.columns(2)

with col1:
    puntos_atencion = st.number_input(
        "Puntos de atención por 10,000 adultos",
        min_value=0.0, max_value=100.0, value=5.0, step=0.1,
        help="Oficinas + ATMs + corresponsales por cada 10,000 adultos"
    )
    pct_deudores = st.number_input(
        "% Adultos con crédito activo",
        min_value=0.0, max_value=100.0, value=15.0, step=0.1,
        help="Porcentaje de adultos con algún crédito formal vigente"
    )
    pct_cuentas_ahorro = st.number_input(
        "% Adultos con cuenta de ahorro",
        min_value=0.0, max_value=100.0, value=25.0, step=0.1,
        help="Porcentaje de adultos con cuenta de ahorro activa"
    )
    atm_x10k = st.number_input(
        "Cajeros ATM por 10,000 adultos",
        min_value=0.0, max_value=50.0, value=2.0, step=0.1,
        help="Número de cajeros automáticos por cada 10,000 adultos"
    )

with col2:
    corresponsales_x10k = st.number_input(
        "Corresponsales por 10,000 adultos",
        min_value=0.0, max_value=50.0, value=3.0, step=0.1,
        help="Agentes corresponsales bancarios por cada 10,000 adultos"
    )
    credito_promedio = st.number_input(
        "Crédito promedio (soles)",
        min_value=0.0, max_value=100000.0, value=5000.0, step=100.0,
        help="Monto promedio de crédito en soles de la provincia"
    )
    indice_pobreza = st.number_input(
        "Índice de pobreza (%)",
        min_value=0.0, max_value=100.0, value=40.0, step=0.1,
        help="Porcentaje de pobreza monetaria de la provincia"
    )
    acceso_internet = st.number_input(
        "Acceso a internet (%)",
        min_value=0.0, max_value=100.0, value=20.0, step=0.1,
        help="Porcentaje de población con acceso a internet"
    )

st.divider()

# ── Botón de predicción ──────────────────────────────────────────────────────
if st.button("🔍 Clasificar Nivel de Inclusión Financiera", type="primary"):

    datos = np.array([[
        puntos_atencion, pct_deudores, pct_cuentas_ahorro,
        atm_x10k, corresponsales_x10k, credito_promedio,
        indice_pobreza, acceso_internet
    ]])

    prediccion = model.predict(datos)[0]
    proba = model.predict_proba(datos)[0]

    # Mapeo de etiquetas
    etiquetas = {0: "🔴 Bajo", 1: "🟡 Medio", 2: "🟢 Alto"}
    colores   = {0: "error",  1: "warning", 2: "success"}
    etiqueta  = etiquetas[prediccion]

    # Mostrar resultado
    st.markdown("### 📈 Resultado de la Clasificación")
    if prediccion == 0:
        st.error(f"**Nivel de Inclusión Financiera: {etiqueta}**")
        st.markdown("""
        ⚠️ Esta provincia presenta alta vulnerabilidad financiera.
        Se recomienda priorizar la apertura de corresponsales bancarios
        y programas de educación financiera del Estado.
        """)
    elif prediccion == 1:
        st.warning(f"**Nivel de Inclusión Financiera: {etiqueta}**")
        st.markdown("""
        📊 Esta provincia tiene acceso financiero parcial.
        Se recomienda fortalecer los canales digitales y
        expandir la cobertura de cajeros corresponsales.
        """)
    else:
        st.success(f"**Nivel de Inclusión Financiera: {etiqueta}**")
        st.markdown("""
        ✅ Esta provincia tiene buen nivel de acceso financiero.
        Se recomienda mantener la infraestructura actual y
        promover productos financieros más sofisticados.
        """)

    # Probabilidades por clase
    st.markdown("#### Probabilidades por clase (KNN K=9):")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🔴 Bajo",  f"{proba[0]*100:.1f}%")
    col_b.metric("🟡 Medio", f"{proba[1]*100:.1f}%")
    col_c.metric("🟢 Alto",  f"{proba[2]*100:.1f}%")

    # Guardar en Supabase
    payload = {
        "inputs_usuario": {
            "puntos_atencion_x10k": puntos_atencion,
            "pct_deudores": pct_deudores,
            "pct_cuentas_ahorro": pct_cuentas_ahorro,
            "atm_x10k": atm_x10k,
            "corresponsales_x10k": corresponsales_x10k,
            "credito_promedio_soles": credito_promedio,
            "indice_pobreza_pct": indice_pobreza,
            "acceso_internet_pct": acceso_internet
        },
        "resultado_prediccion": str(etiqueta)
    }

    try:
        supabase.table("predicciones_log").insert(payload).execute()
        st.caption("✓ Consulta registrada en Supabase.")
    except Exception as e:
        st.caption(f"Aviso: No se pudo registrar en base de datos: {e}")

# ── Información del modelo ───────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ Información del Modelo"):
    st.markdown("""
    **Algoritmo:** K-Vecinos Cercanos (KNN) con K=9
    **Accuracy:** 95.83% sobre conjunto de prueba (24 provincias)
    **Dataset:** 120 provincias peruanas con indicadores SBS 2024
    **Variables:** 8 indicadores financieros y socioeconómicos
    **Preprocesamiento:** KNN Imputer + StandardScaler
    **Clases:** Bajo (0) | Medio (1) | Alto (2)
    """)

st.markdown("---")
st.caption("Caso 11 — Análisis de Inclusión Financiera SBS Perú | Machine Learning - UPLA 2026")
