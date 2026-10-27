import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------- Page setup ----------
st.set_page_config(
    page_title="Laptop Price Intelligence",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom styling ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0b1220 0%, #101b2d 100%);
    color: #e5edf8;
}
[data-testid="stHeader"] {
    background: rgba(11,18,32,0.9);
}
[data-testid="stSidebar"] {
    background: #111c2e;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
.hero {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(120deg, #17365f, #164e63);
    border: 1px solid #315579;
    margin-bottom: 24px;
}
.hero h1 {
    color: #ffffff;
    font-size: 2.35rem;
    margin-bottom: 8px;
}
.hero p {
    color: #d7e7f7;
    font-size: 1rem;
}
.panel {
    padding: 18px;
    background: #162338;
    border: 1px solid #2b3c54;
    border-radius: 16px;
    margin-bottom: 12px;
}
.small-muted {
    color: #9fb1c9;
    font-size: 0.9rem;
}
div[data-testid="stMetric"] {
    background: #162338;
    border: 1px solid #2b3c54;
    padding: 16px;
    border-radius: 14px;
}
div[data-testid="stMetricLabel"] {
    color: #a9bdd5;
}
div[data-testid="stMetricValue"] {
    color: #ffffff;
}
h2, h3 {
    color: #f1f6ff;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load files ----------
@st.cache_resource
def load_model():
    model = joblib.load("/content/laptop_price_model.joblib")
    with open("/content/model_info.json", "r") as f:
        info = json.load(f)
    return model, info

@st.cache_data
def load_data():
    data_path = Path("/content/laptop_data.csv")
    eval_path = Path("/content/model_evaluation.csv")

    data = pd.read_csv(data_path) if data_path.exists() else pd.DataFrame()
    evaluation = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    return data, evaluation

model, info = load_model()
data, evaluation = load_data()

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>💻 Laptop Price Intelligence</h1>
    <p>Machine learning-powered price estimation, market insights, and model evaluation.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Academic project prototype • Estimates listed laptop prices from hardware specifications"
)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Price Estimator", "Data Insights", "Model Performance", "About Project"]
)
st.sidebar.divider()
st.sidebar.markdown("### Model information")
st.sidebar.write("**Algorithm:**", info.get("model_name", "Regression model"))
if info.get("mae") is not None:
    st.sidebar.write("**Test MAE:** ₹{:,.0f}".format(float(info["mae"])))

st.sidebar.caption("Predictions are estimates, not guaranteed selling prices.")

# ---------- Price estimator ----------
if page == "Price Estimator":
    st.subheader("Estimate a laptop price")
    st.write("Enter the hardware details and the seller's asking price.")

    with st.form("prediction_form"):
        values = {}
        numeric_features = info.get("numeric_features", [])
        categorical_features = info.get("categorical_features", [])
        defaults = info.get("defaults", {})

        # Organized form layout
        left, right = st.columns(2, gap="large")

        for i, col in enumerate(info["features"]):
            default = defaults.get(col, "")
            label = col.replace("_", " ").title()
            container = left if i % 2 == 0 else right

            with container:
                if col in numeric_features:
                    try:
                        default_value = max(0.0, float(default))
                    except (ValueError, TypeError):
                        default_value = 0.0

                    values[col] = st.number_input(
                        label,
                        min_value=0.0,
                        value=default_value,
                        step=1.0,
                        key=f"feature_{col}"
                    )
                else:
                    values[col] = st.text_input(
                        label,
                        value=str(default),
                        key=f"feature_{col}"
                    )

        asking_price = st.number_input(
            "Seller's asking price (₹)",
            min_value=0.0,
            value=30000.0,
            step=500.0
        )

        submitted = st.form_submit_button(
            "✨ Estimate price",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        input_df = pd.DataFrame([values], columns=info["features"])
        fair_price = max(0.0, float(model.predict(input_df)[0]))
        mae = max(0.0, float(info.get("mae", 0)))

        if fair_price > 0 and asking_price > 0:
            difference = asking_price - fair_price
            percentage = difference / fair_price * 100

            if percentage > 15:
                verdict = "OVERPRICED"
                color = "#f87171"
                explanation = "The asking price is more than 15% above the model estimate."
            elif percentage < -15:
                verdict = "GOOD DEAL"
                color = "#4ade80"
                explanation = "The asking price is more than 15% below the model estimate. Check condition carefully."
            else:
                verdict = "FAIR PRICE"
                color = "#60a5fa"
                explanation = "The asking price is within 15% of the model estimate."
        else:
            difference = asking_price - fair_price
            percentage = None
            verdict = "CHECK INPUT"
            color = "#fbbf24"
            explanation = "Enter a positive asking price and verify the input values."

        st.divider()
        st.subheader("📊 Your estimation")

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated price", f"₹{fair_price:,.0f}")
        c2.metric("Asking price", f"₹{asking_price:,.0f}")
        c3.metric(
            "Difference",
            f"{percentage:+.1f}%" if percentage is not None else "N/A"
        )

        st.markdown(
            f"""<div class="panel">
                <div class="small-muted">DEAL VERDICT</div>
                <h2 style="color:{color}; margin-top:8px;">{verdict}</h2>
                <p>{explanation}</p>
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown("**Illustrative error band**")
        st.write(
            f"₹{max(0, fair_price - mae):,.0f} – ₹{fair_price + mae:,.0f}"
        )
        st.caption(
            "This range uses the test-set MAE as a simple reference, "
            "not a statistically calibrated prediction interval."
        )

# ---------- Dataset charts ----------
elif page == "Data Insights":
    st.subheader("📈 Dataset insights")
    st.write("Explore the laptop listings used in the model prototype.")

    if data.empty or "price" not in data.columns:
        st.warning("Dataset file is missing. Run the artifact-saving cell in Colab first.")
    else:
        prices = pd.to_numeric(data["price"], errors="coerce").dropna()
        prices = prices[prices > 0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Dataset records", f"{len(data):,}")
        m2.metric("Median listed price", f"₹{prices.median():,.0f}")
        m3.metric("Brands represented", str(data["brand"].nunique()) if "brand" in data.columns else "N/A")

        chart1, chart2 = st.columns(2, gap="large")

        with chart1:
            st.markdown("#### Price distribution")
            fig = px.histogram(
                prices.to_frame(name="Price"),
                x="Price",
                nbins=35,
                title="How listed prices are distributed",
                labels={"Price": "Listed price (₹)"},
                template="plotly_dark"
            )
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with chart2:
            st.markdown("#### Average price by brand")
            if "brand" in data.columns:
                brand_data = data[["brand", "price"]].copy()
                brand_data["price"] = pd.to_numeric(brand_data["price"], errors="coerce")
                brand_data = brand_data.dropna()
                brand_data = brand_data[brand_data["price"] > 0]
                brand_avg = (
                    brand_data.groupby("brand", as_index=False)["price"]
                    .mean()
                    .sort_values("price", ascending=False)
                    .head(10)
                )
                fig = px.bar(
                    brand_avg,
                    x="brand",
                    y="price",
                    title="Top 10 brands by average listed price",
                    labels={"brand": "Brand", "price": "Average price (₹)"},
                    template="plotly_dark"
                )
                fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("The dataset does not contain a brand column.")

        st.markdown("#### Explore records")
        st.dataframe(data.head(100), use_container_width=True, hide_index=True)

# ---------- Model performance ----------
elif page == "Model Performance":
    st.subheader("🎯 Model evaluation")
    st.write("Evaluation uses the held-out test set from model training.")

    if evaluation.empty:
        st.warning("Evaluation file not found. Run the artifact-saving cell first.")
    else:
        actual = evaluation["Actual Price"]
        predicted = evaluation["Predicted Price"]
        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        denom = ((actual - actual.mean()) ** 2).sum()
        r2 = 1 - ((actual - predicted) ** 2).sum() / denom if denom else float("nan")

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"₹{mae:,.0f}")
        m2.metric("RMSE", f"₹{rmse:,.0f}")
        m3.metric("R²", f"{r2:.3f}" if np.isfinite(r2) else "N/A")

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### Actual vs predicted")
            fig = px.scatter(
                evaluation,
                x="Actual Price",
                y="Predicted Price",
                title="Predictions compared with actual prices",
                template="plotly_dark",
                opacity=0.65
            )
            low = float(min(actual.min(), predicted.min()))
            high = float(max(actual.max(), predicted.max()))
            fig.add_trace(go.Scatter(
                x=[low, high], y=[low, high],
                mode="lines",
                name="Ideal prediction",
                line=dict(dash="dash")
            ))
            fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### Prediction error distribution")
            error_df = pd.DataFrame({
                "Prediction Error": actual - predicted
            })
            fig = px.histogram(
                error_df,
                x="Prediction Error",
                nbins=30,
                title="Actual price minus predicted price",
                template="plotly_dark"
            )
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Lower MAE and RMSE are better. R² should be interpreted alongside "
            "the error metrics and the dataset's limitations."
        )

# ---------- About ----------
else:
    st.subheader("📘 About the project")
    st.markdown("""
    <div class="panel">
        <h3>Project objective</h3>
        <p>Estimate laptop listed prices from selected hardware specifications
        using supervised machine learning.</p>
        <h3>Workflow</h3>
        <p>Data collection → cleaning → exploratory analysis → regression
        training → evaluation → interactive prediction dashboard.</p>
        <h3>Limitations</h3>
        <p>The current dataset contains listed laptop prices and is not verified
        second-hand transaction data. Therefore, this prototype does not establish
        reliable second-hand resale prices. Laptop age, battery health, warranty,
        cosmetic condition, and local resale demand may materially affect actual
        used-market prices.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Features used by the model")
    st.write(", ".join(info.get("features", [])))

st.divider()
st.caption(
    "Laptop Price Intelligence • Machine Learning Mini Project • "
    "Predictions are estimates and should not be treated as guaranteed market values."
)
