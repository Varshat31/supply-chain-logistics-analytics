import os
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# SUPPLY CHAIN INTELLIGENCE - STREAMLIT DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

# --------------------------- Styling ------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1220; color: #e8eef8; }
section[data-testid="stSidebar"] { background: #101b2d; border-right: 1px solid #22334b; }
.block-container { padding-top: 1.5rem; max-width: 1500px; }
h1,h2,h3 { font-family: 'Manrope', sans-serif !important; color: #f4f7fb !important; }
.hero { padding: 24px 28px; border: 1px solid #263b56; border-radius: 18px;
        background: linear-gradient(115deg,#14263d 0%,#102b38 55%,#163b43 100%); margin-bottom: 20px; }
.hero-kicker { color:#58d6c4; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; }
.hero-title { font-family:'Manrope',sans-serif; color:#fff; font-size:30px; font-weight:800; margin:5px 0; }
.hero-sub { color:#b6c6d9; font-size:14px; }
.metric { background:#121f32; border:1px solid #263a53; border-radius:15px; padding:17px 18px; min-height:112px; }
.metric-label { color:#9eb0c6; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1px; }
.metric-value { color:#f6f9ff; font-family:'Manrope',sans-serif; font-size:27px; font-weight:800; margin-top:8px; }
.metric-note { color:#56d6c3; font-size:11px; margin-top:3px; }
.section-head { color:#f4f7fb; font-family:'Manrope',sans-serif; font-size:18px; font-weight:700; margin:14px 0 8px; }
div[data-testid="stDataFrame"] { border:1px solid #263a53; border-radius:12px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background:#142238; border-radius:9px 9px 0 0; color:#b8c7d9; padding:10px 18px; }
.stTabs [aria-selected="true"] { background:#1b3c4a !important; color:#62e0cc !important; }
div[data-testid="stAlert"] { border-radius:12px; }
</style>
""", unsafe_allow_html=True)


# --------------------------- Helpers ------------------------

def fmt_int(value):
    """Format numeric KPI values with thousands separators."""
    try:
        if pd.isna(value):
            return "—"
        return f"{int(value):,}"
    except (TypeError, ValueError, OverflowError):
        return "—"


def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=330):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c7d4e5", family="DM Sans"),
        margin=dict(l=12, r=12, t=42, b=12),
        legend=dict(orientation="h", y=1.12, x=0),
        title_font=dict(color="#f3f7fc", size=15),
    )
    fig.update_xaxes(gridcolor="#26364b", zerolinecolor="#26364b")
    fig.update_yaxes(gridcolor="#26364b", zerolinecolor="#26364b")
    return fig


def read_csv_candidates(paths):
    for path in paths:
        try:
            if path.exists():
                return pd.read_csv(path, low_memory=False)
        except Exception:
            pass
    return None


def load_orders():
    uploaded = st.session_state.get("uploaded_orders")
    if uploaded is not None:
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, low_memory=False), "Uploaded CSV"
        except Exception as exc:
            st.sidebar.error(f"Could not read uploaded CSV: {exc}")

    df = read_csv_candidates([
        OUTPUT_DIR / "powerbi_orders.csv",
        OUTPUT_DIR / "predictions.csv",
        OUTPUT_DIR / "model_predictions.csv",
        ROOT / "powerbi_orders.csv",
    ])
    return df, "Project outputs" if df is not None else "No CSV loaded"


def find_model():
    preferred = [
        MODEL_DIR / "delivery_risk_model.joblib",
        MODEL_DIR / "best_model.joblib",
        MODEL_DIR / "model.joblib",
        MODEL_DIR / "pipeline.joblib",
        ROOT / "delivery_risk_model.joblib",
        ROOT / "model.joblib",
    ]
    for p in preferred:
        if p.exists():
            try:
                return joblib.load(p), p
            except Exception:
                pass

    if MODEL_DIR.exists():
        for p in sorted(list(MODEL_DIR.glob("*.joblib")) + list(MODEL_DIR.glob("*.pkl"))):
            try:
                return joblib.load(p), p
            except Exception:
                continue
    return None, None


def find_training_data():
    candidates = [
        ROOT / "DataCoSupplyChainDataset.csv",
        ROOT / "data" / "DataCoSupplyChainDataset.csv",
        ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv",
        ROOT / "dataset" / "DataCoSupplyChainDataset.csv",
        ROOT / "datasets" / "DataCoSupplyChainDataset.csv",
    ]
    return read_csv_candidates(candidates)


def find_col(df, options):
    if df is None:
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for option in options:
        if option.lower() in lookup:
            return lookup[option.lower()]
    return None


def safe_numeric(series, default=0):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.median()) if len(vals) else default


def safe_mode(series, default="Unknown"):
    vals = series.dropna()
    return vals.mode().iloc[0] if len(vals) else default


def get_feature_names(model):
    # A fitted sklearn Pipeline generally exposes feature_names_in_.
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)

    # A pipeline's first preprocessing step may retain the original names.
    for _, step in getattr(model, "steps", []):
        names = getattr(step, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return None


def get_classes(model):
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = getattr(getattr(model, "named_steps", {}).get("model", None), "classes_", None)
    return list(classes) if classes is not None else [0, 1]


def make_input_form(model, reference_df):
    """
    Show only the shipment details an operator can reasonably provide.
    All remaining model features are automatically filled using typical
    values from the reference dataset so the model still receives its
    complete training-time input schema.
    """
    feature_names = get_feature_names(model)
    if not feature_names:
        st.error(
            "The saved model does not expose its original input feature names. "
            "Please check the feature list used when training the model."
        )
        return None

    target_like = {
        "late_delivery_risk", "actual_late_risk", "delivery_status",
        "predicted_late_risk", "predicted_late_risk_probability", "risk_band"
    }
    feature_names = [
        f for f in feature_names
        if str(f).strip().lower() not in target_like
    ]

    def normalise(name):
        return re.sub(r"[^a-z0-9]", "", str(name).lower())

    # Match the user-friendly controls to the actual column names used by
    # the trained model. The other model columns are auto-filled below.
    user_fields = [
        {
            "label": "Shipping mode",
            "aliases": ["Shipping Mode"],
            "kind": "category",
        },
        {
            "label": "Order region",
            "aliases": ["Order Region"],
            "kind": "category",
        },
        {
            "label": "Scheduled shipping days",
            "aliases": [
                "Days for shipment (scheduled)",
                "Days For Shipment Scheduled",
            ],
            "kind": "number",
            "default": 4.0,
        },
        {
            "label": "Product price",
            "aliases": ["Order Item Product Price", "Product Price"],
            "kind": "number",
            "default": 60.0,
        },
        {
            "label": "Quantity",
            "aliases": ["Order Item Quantity", "Quantity"],
            "kind": "number",
            "default": 1.0,
        },
        {
            "label": "Discount rate",
            "aliases": ["Order Item Discount Rate", "Discount Rate"],
            "kind": "number",
            "default": 0.10,
        },
    ]

    # First create a complete model row using median/mode values from the
    # available reference dataset.
    values = {}
    actual_feature_lookup = {normalise(f): f for f in feature_names}

    for feature in feature_names:
        ref_col = find_col(reference_df, [feature]) if reference_df is not None else None
        series = reference_df[ref_col] if ref_col is not None else None

        if series is not None:
            numeric = pd.to_numeric(series, errors="coerce")
            is_numeric = numeric.notna().mean() > 0.85
            if is_numeric:
                values[feature] = safe_numeric(series, 0.0)
            else:
                values[feature] = safe_mode(series, "Unknown")
        else:
            values[feature] = 0.0

    st.markdown("### Enter shipment details")
    st.caption(
        "Only six details are required. The remaining technical model inputs "
        "are automatically populated with typical values from the available dataset."
    )

    visible_fields = []
    for spec in user_fields:
        matched = None
        for alias in spec["aliases"]:
            matched = actual_feature_lookup.get(normalise(alias))
            if matched is not None:
                break
        if matched is not None:
            visible_fields.append((spec, matched))

    if not visible_fields:
        st.error(
            "None of the simplified shipment fields match this model's input columns. "
            "The model feature names may differ from the expected DataCo dataset."
        )
        return None

    columns = st.columns(2)
    for i, (spec, feature) in enumerate(visible_fields):
        ref_col = find_col(reference_df, [feature]) if reference_df is not None else None
        ref_series = reference_df[ref_col] if ref_col is not None else None

        with columns[i % 2]:
            if spec["kind"] == "category":
                if ref_series is not None:
                    choices = sorted(
                        ref_series.dropna().astype(str).unique().tolist()
                    )
                    choices = choices[:250]
                else:
                    choices = []

                if choices:
                    default = str(values.get(feature, choices[0]))
                    if default not in choices:
                        default = choices[0]
                    values[feature] = st.selectbox(
                        spec["label"],
                        choices,
                        index=choices.index(default),
                        key=f"simple_live_{normalise(feature)}",
                    )
                else:
                    values[feature] = st.text_input(
                        spec["label"],
                        value=str(values.get(feature, "Unknown")),
                        key=f"simple_live_{normalise(feature)}",
                    )
            else:
                default = float(values.get(feature, spec.get("default", 0.0)))
                # Use a sensible example value if the reference data is absent
                # or its median is zero.
                if default == 0 and "default" in spec:
                    default = float(spec["default"])
                values[feature] = st.number_input(
                    spec["label"],
                    min_value=0.0,
                    value=default,
                    step=1.0 if spec["label"] != "Discount rate" else 0.01,
                    key=f"simple_live_{normalise(feature)}",
                )

    st.info(
        "Note: values such as customer IDs, coordinates, dates, department details "
        "and other training columns are filled automatically. They are not required "
        "from the person using the dashboard."
    )

    return pd.DataFrame([values], columns=feature_names)


def prediction_probability(model, row, prediction):
    try:
        probabilities = model.predict_proba(row)[0]
        classes = get_classes(model)
        positive_index = 1
        if 1 in classes:
            positive_index = classes.index(1)
        elif "1" in classes:
            positive_index = classes.index("1")
        return float(probabilities[positive_index])
    except Exception:
        return float(prediction)


def get_col(df, *names):
    return find_col(df, list(names))


def normalized_risk(value):
    if pd.isna(value):
        return "Unknown"
    try:
        v = int(float(value))
        return "High" if v == 1 else "Low"
    except Exception:
        text = str(value).strip().lower()
        if "high" in text or "late" in text or text in ("yes", "true"):
            return "High"
        if "low" in text or "on time" in text or text in ("no", "false"):
            return "Low"
        return str(value)


# --------------------------- Sidebar ------------------------

st.sidebar.markdown("## 📦 Supply Chain")
st.sidebar.caption("INTELLIGENCE & DELIVERY RISK")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Overview",
        "Logistics Analytics",
        "Delivery Risk",
        "Live Prediction",
        "Model Performance",
    ],
    label_visibility="visible",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data source")
st.sidebar.caption("Predictions generated by the Python machine-learning pipeline.")
uploaded_file = st.sidebar.file_uploader(
    "Or upload powerbi_orders.csv", type=["csv"], key="orders_uploader"
)
if uploaded_file is not None:
    st.session_state["uploaded_orders"] = uploaded_file

orders, data_source = load_orders()
model, model_path = find_model()
training_data = find_training_data()

if training_data is not None:
    reference_data = training_data
elif orders is not None:
    reference_data = orders
else:
    reference_data = None

st.sidebar.caption(f"Active data: {data_source}")
st.sidebar.caption(f"ML model: {model_path.name if model_path else 'Not found'}")


# --------------------------- Header -------------------------

st.markdown("""
<div class="hero">
  <div class="hero-kicker">Operations intelligence • Machine learning enabled</div>
  <div class="hero-title">Supply Chain Command Centre</div>
  <div class="hero-sub">Monitor order risk, understand logistics patterns and identify shipments that may need attention.</div>
</div>
""", unsafe_allow_html=True)


# ---------------------- Data preparation --------------------

if orders is not None:
    orders.columns = [str(c).strip() for c in orders.columns]

    risk_col = get_col(orders, "Risk_Band", "Predicted_Risk_Band", "Predicted_Late_Risk")
    prob_col = get_col(orders, "Predicted_Late_Risk_Probability", "Late_Risk_Probability", "Risk_Probability")
    actual_col = get_col(orders, "Actual_Late_Risk", "Late_delivery_risk", "Late_Delivery_Risk")
    predicted_col = get_col(orders, "Predicted_Late_Risk", "Prediction", "Predicted")
    shipping_col = get_col(orders, "Shipping_Mode", "Shipping Mode")
    region_col = get_col(orders, "Order_Region", "Order Region")
    record_col = get_col(orders, "Record_ID", "Order_Id", "Order ID")

    if risk_col is not None:
        orders["_risk_label"] = orders[risk_col].apply(normalized_risk)
    elif predicted_col is not None:
        orders["_risk_label"] = orders[predicted_col].apply(normalized_risk)
    else:
        orders["_risk_label"] = "Unknown"

    if prob_col is not None:
        orders["_risk_probability"] = pd.to_numeric(orders[prob_col], errors="coerce")
    else:
        orders["_risk_probability"] = np.nan

    if shipping_col is None:
        shipping_col = None
    if region_col is None:
        region_col = None
else:
    risk_col = prob_col = actual_col = predicted_col = shipping_col = region_col = record_col = None


# ---------------------- Page: Overview ----------------------

if page == "Executive Overview":
    st.header("Executive Overview")

    if orders is None:
        st.warning("No prediction output CSV was found. Run `python src/train_model.py` first, or upload `outputs/powerbi_orders.csv` in the sidebar.")
        st.info("Expected file: outputs/powerbi_orders.csv")
    else:
        total = len(orders)
        high = int((orders["_risk_label"] == "High").sum())
        late_actual = int(pd.to_numeric(orders[actual_col], errors="coerce").fillna(0).eq(1).sum()) if actual_col else 0
        avg_prob = orders["_risk_probability"].mean()
        high_pct = high / total * 100 if total else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Total records", fmt_int(total), "Orders in current dataset")
        with c2: metric_card("Actual late records", fmt_int(late_actual) if actual_col else "N/A", "Based on actual outcome column")
        with c3: metric_card("High-risk predictions", fmt_int(high), f"{high_pct:.1f}% of records")
        with c4: metric_card("Average risk probability", f"{avg_prob:.1%}" if pd.notna(avg_prob) else "N/A", "Model-estimated probability")

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-head">Shipment risk distribution</div>', unsafe_allow_html=True)
            risk_counts = orders["_risk_label"].value_counts().rename_axis("Risk").reset_index(name="Orders")
            if not risk_counts.empty:
                fig = px.pie(risk_counts, names="Risk", values="Orders", hole=0.58,
                             color="Risk", color_discrete_map={"Low": "#55d2bf", "High": "#f16f7a", "Medium": "#f5bd62", "Unknown": "#8293aa"})
                st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
        with right:
            st.markdown('<div class="section-head">Orders by shipping mode</div>', unsafe_allow_html=True)
            if shipping_col:
                ship_counts = orders[shipping_col].astype(str).value_counts().rename_axis("Shipping Mode").reset_index(name="Orders")
                fig = px.bar(ship_counts, x="Orders", y="Shipping Mode", orientation="h",
                             color="Orders", color_continuous_scale=["#24566a", "#58d6c4"])
                st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
            else:
                st.info("Shipping mode column is not present in the output CSV.")

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-head">Risk probability distribution</div>', unsafe_allow_html=True)
            probs = orders["_risk_probability"].dropna()
            if not probs.empty:
                fig = px.histogram(probs, nbins=25, labels={"value": "Risk probability", "count": "Orders"})
                fig.update_traces(marker_color="#58d6c4")
                st.plotly_chart(chart_layout(fig, 300), use_container_width=True)
            else:
                st.info("No probability column was found.")
        with right:
            st.markdown('<div class="section-head">Orders by region</div>', unsafe_allow_html=True)
            if region_col:
                region_counts = orders[region_col].astype(str).value_counts().head(12).rename_axis("Region").reset_index(name="Orders")
                fig = px.bar(region_counts, x="Orders", y="Region", orientation="h", color="Orders",
                             color_continuous_scale=["#24566a", "#58d6c4"])
                st.plotly_chart(chart_layout(fig, 300), use_container_width=True)
            else:
                st.info("Order region column is not present in the output CSV.")

        st.markdown('<div class="section-head">Recent shipment records</div>', unsafe_allow_html=True)
        display_cols = [c for c in [
            record_col, shipping_col, region_col, actual_col, predicted_col, prob_col, risk_col
        ] if c is not None and c in orders.columns]
        if not display_cols:
            display_cols = list(orders.columns[:10])
        st.dataframe(orders[display_cols].head(100), use_container_width=True, height=360)


# ---------------------- Page: Logistics ---------------------

elif page == "Logistics Analytics":
    st.header("Logistics Analytics")
    if orders is None:
        st.warning("Load `outputs/powerbi_orders.csv` or upload a CSV from the sidebar.")
    else:
        f1, f2 = st.columns(2)
        with f1:
            if shipping_col:
                modes = ["All"] + sorted(orders[shipping_col].dropna().astype(str).unique().tolist())
                selected_mode = st.selectbox("Shipping mode", modes)
            else:
                selected_mode = "All"
        with f2:
            if region_col:
                regions = ["All"] + sorted(orders[region_col].dropna().astype(str).unique().tolist())
                selected_region = st.selectbox("Order region", regions)
            else:
                selected_region = "All"

        filtered = orders.copy()
        if shipping_col and selected_mode != "All":
            filtered = filtered[filtered[shipping_col].astype(str) == selected_mode]
        if region_col and selected_region != "All":
            filtered = filtered[filtered[region_col].astype(str) == selected_region]

        a, b, c = st.columns(3)
        with a: metric_card("Filtered orders", fmt_int(len(filtered)))
        with b: metric_card("High-risk orders", fmt_int((filtered["_risk_label"] == "High").sum()))
        with c:
            p = filtered["_risk_probability"].mean()
            metric_card("Mean risk probability", f"{p:.1%}" if pd.notna(p) else "N/A")

        left, right = st.columns(2)
        with left:
            if shipping_col:
                temp = filtered[shipping_col].astype(str).value_counts().rename_axis("Mode").reset_index(name="Orders")
                fig = px.bar(temp, x="Mode", y="Orders", color="Orders", color_continuous_scale=["#24566a", "#58d6c4"])
                st.plotly_chart(chart_layout(fig), use_container_width=True)
        with right:
            if region_col:
                temp = filtered[region_col].astype(str).value_counts().head(15).rename_axis("Region").reset_index(name="Orders")
                fig = px.bar(temp, x="Orders", y="Region", orientation="h", color="Orders", color_continuous_scale=["#24566a", "#58d6c4"])
                st.plotly_chart(chart_layout(fig), use_container_width=True)

        st.subheader("Filtered shipment data")
        st.dataframe(filtered.head(500), use_container_width=True, height=420)
        st.download_button(
            "Download filtered records",
            filtered.to_csv(index=False).encode("utf-8"),
            "filtered_supply_chain_orders.csv",
            "text/csv",
        )


# ---------------------- Page: Delivery Risk -----------------

elif page == "Delivery Risk":
    st.header("Delivery Risk Analysis")
    if orders is None:
        st.warning("Load `outputs/powerbi_orders.csv` or upload a CSV from the sidebar.")
    else:
        high_only = st.checkbox("Show only high-risk shipments", value=False)
        filtered = orders.copy()
        if high_only:
            filtered = filtered[filtered["_risk_label"] == "High"]

        if region_col:
            risk_region = filtered.groupby([region_col, "_risk_label"]).size().reset_index(name="Orders")
            fig = px.bar(risk_region, x=region_col, y="Orders", color="_risk_label", barmode="stack",
                         color_discrete_map={"Low": "#55d2bf", "High": "#f16f7a", "Medium": "#f5bd62", "Unknown": "#8293aa"})
            st.plotly_chart(chart_layout(fig, 390), use_container_width=True)

        if prob_col:
            st.subheader("Highest predicted risk")
            sort_col = "_risk_probability"
            top = filtered.sort_values(sort_col, ascending=False).head(100)
        else:
            top = filtered[filtered["_risk_label"] == "High"].head(100)

        st.dataframe(top, use_container_width=True, height=420)
        st.download_button(
            "Download risk records",
            top.to_csv(index=False).encode("utf-8"),
            "delivery_risk_records.csv",
            "text/csv",
        )


# ---------------------- Page: Live Prediction ----------------

elif page == "Live Prediction":
    st.header("Live Shipment Prediction")
    st.write("Enter shipment information and let the trained machine-learning pipeline estimate late-delivery risk.")

    if model is None:
        st.error("No trained model file was found.")
        st.markdown(
            "Place your saved model (`.joblib` or `.pkl`) inside the project's `models` folder. "
            "The app checks common model filenames automatically."
        )
        st.code("E:\\Supply_Chain_Logistics_Analytics\\models\\")
    else:
        st.success(f"Loaded trained model: `{model_path.name}`")
        if training_data is None:
            st.warning("Original training CSV was not found. Dropdown choices will be limited. If your model uses categorical inputs, place DataCoSupplyChainDataset.csv in the project root or data/raw folder.")
        input_row = make_input_form(model, reference_data)

        if input_row is not None:
            if st.button("🔮 Predict delivery risk", type="primary", use_container_width=True):
                try:
                    prediction = model.predict(input_row)[0]
                    probability = prediction_probability(model, input_row, prediction)
                    is_late = str(prediction).strip().lower() in ("1", "1.0", "true", "yes", "late")
                    risk_label = "HIGH" if is_late else "LOW"

                    st.markdown("---")
                    st.subheader("Prediction result")
                    r1, r2, r3 = st.columns(3)
                    with r1: metric_card("Late delivery risk", "YES" if is_late else "NO", "Model classification")
                    with r2: metric_card("Risk probability", f"{probability:.1%}", "Estimated probability of class 1")
                    with r3: metric_card("Risk band", risk_label, "Based on predicted class")

                    if is_late:
                        st.error("⚠️ High risk: this shipment may be delivered late. Consider monitoring or prioritising it.")
                    else:
                        st.success("✅ Low risk: the model does not predict a late delivery for this shipment.")

                    result = input_row.copy()
                    result["Predicted_Late_Risk"] = int(is_late)
                    result["Predicted_Late_Risk_Probability"] = probability
                    result["Risk_Band"] = "High" if is_late else "Low"
                    st.dataframe(result, use_container_width=True)
                    st.download_button(
                        "Download this prediction",
                        result.to_csv(index=False).encode("utf-8"),
                        "single_shipment_prediction.csv",
                        "text/csv",
                    )

                    # Keep a session history of interactive predictions.
                    history = st.session_state.get("live_prediction_history", [])
                    history.append(result.copy())
                    st.session_state["live_prediction_history"] = history[-100:]
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.info("The input columns or data types may not match the columns used during model training. Share the traceback and train_model.py if this occurs.")

        history = st.session_state.get("live_prediction_history", [])
        if history:
            st.subheader("Recent live predictions")
            st.dataframe(pd.concat(history, ignore_index=True), use_container_width=True)


# ---------------------- Page: Model Performance -------------

elif page == "Model Performance":
    st.header("Model Performance")
    st.write("Evaluation metrics from the training pipeline.")

    metrics_path = OUTPUT_DIR / "model_metrics.json"
    report_path = OUTPUT_DIR / "classification_report.csv"
    metrics_csv_path = OUTPUT_DIR / "model_metrics.csv"

    found_metrics = False
    if metrics_path.exists():
        try:
            import json
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            cols = st.columns(min(4, max(1, len(metrics))))
            for i, (key, value) in enumerate(metrics.items()):
                if isinstance(value, (int, float)):
                    with cols[i % len(cols)]:
                        metric_card(str(key).replace("_", " ").title(), f"{value:.3f}")
                    found_metrics = True
        except Exception as exc:
            st.warning(f"Could not read model_metrics.json: {exc}")

    if report_path.exists():
        st.subheader("Classification report")
        try:
            report = pd.read_csv(report_path)
            st.dataframe(report, use_container_width=True)
            found_metrics = True
        except Exception as exc:
            st.warning(f"Could not read classification_report.csv: {exc}")

    if metrics_csv_path.exists():
        st.subheader("Saved metrics")
        try:
            st.dataframe(pd.read_csv(metrics_csv_path), use_container_width=True)
            found_metrics = True
        except Exception:
            pass

    if not found_metrics:
        st.info("No separate metrics file was found in outputs/. Your training terminal showed ROC-AUC = 0.826109 and accuracy = 0.71. These are the reported training results, not recalculated here.")
        st.markdown("**Model file:** " + (str(model_path) if model_path else "Not found"))
        if model is not None:
            st.success("A trained model file is available and can be used for live predictions.")


# --------------------------- Footer -------------------------

st.markdown("---")
st.caption("Supply Chain Intelligence • Machine-learning delivery-risk analytics • Local project dashboard")
