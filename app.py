import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EstateAI | House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,0.12), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(139,92,246,0.10), transparent 30%),
        #07111f;
    color: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* HERO */

.hero {
    padding: 35px;
    border-radius: 28px;
    background: linear-gradient(
        135deg,
        rgba(15,23,42,0.98),
        rgba(23,37,84,0.95)
    );
    border: 1px solid rgba(96,165,250,0.25);
    margin-bottom: 25px;
    animation: fadeIn 0.8s ease;
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 17px;
}

/* CARDS */

.card {
    background: rgba(15,23,42,0.78);
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 22px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* PRICE CARD */

.price-card {
    background: linear-gradient(
        135deg,
        #172554,
        #1e3a8a,
        #312e81
    );
    border: 1px solid rgba(96,165,250,0.55);
    border-radius: 28px;
    padding: 45px 30px;
    text-align: center;
    animation: priceAppear 0.7s ease;
}

.price {
    font-size: 52px;
    font-weight: 850;
    margin: 0;
}

/* METRICS */

.metric-card {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(96,165,250,0.5);
}

.metric-title {
    color: #94a3b8;
    font-size: 13px;
}

.metric-value {
    font-size: 23px;
    font-weight: 750;
    margin-top: 7px;
}

/* BADGE */

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(59,130,246,0.14);
    border: 1px solid rgba(96,165,250,0.3);
    color: #bfdbfe;
    font-size: 13px;
}

/* ANIMATION */

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes priceAppear {
    from {
        opacity: 0;
        transform: scale(0.94) translateY(12px);
    }

    to {
        opacity: 1;
        transform: scale(1) translateY(0);
    }
}

/* BUTTON */

div.stButton > button {
    width: 100%;
    min-height: 50px;
    border-radius: 13px;
    font-size: 16px;
    font-weight: 700;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #0b1628;
    border-right: 1px solid rgba(148,163,184,0.12);
}

/* TABLE */

[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path("models/best_model.joblib")

if not MODEL_PATH.exists():

    st.error(
        "Model not found. Please check that "
        "models/best_model.joblib exists."
    )

    st.stop()

model = joblib.load(MODEL_PATH)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="badge">
AI-POWERED REAL ESTATE ANALYTICS
</div>

<div class="hero-title">
🏠 EstateAI
</div>

<div class="hero-subtitle">
Intelligent house price estimation using machine learning.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏠 Property Configuration")

    st.caption(
        "Enter the property details to generate a price estimate."
    )

    area_sqft = st.number_input(
        "Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1200,
        step=50
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    location_score = st.slider(
        "Location Score",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    property_age = st.number_input(
        "Property Age (years)",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

    distance_city_km = st.number_input(
        "Distance from City (km)",
        min_value=0.0,
        max_value=100.0,
        value=8.0,
        step=0.5
    )

    near_school = st.selectbox(
        "Near School",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    near_metro = st.selectbox(
        "Near Metro",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    crime_rate_index = st.number_input(
        "Crime Rate Index",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

    st.write("")

    predict_button = st.button(
        "🔮 Generate Price Prediction",
        type="primary"
    )

    clear_history = st.button(
        "🗑️ Clear Prediction History"
    )

    if clear_history:

        st.session_state.prediction_history = []

        st.success(
            "Prediction history cleared."
        )


# ============================================================
# PROPERTY OVERVIEW
# ============================================================

left, right = st.columns([1.45, 1])


with left:

    st.markdown(
        '<div class="section-title">📋 Property Overview</div>',
        unsafe_allow_html=True
    )

    overview = pd.DataFrame({
        "Feature": [
            "Area",
            "Bedrooms",
            "Bathrooms",
            "Location Score",
            "Property Age",
            "Distance from City",
            "Near School",
            "Near Metro",
            "Crime Rate Index"
        ],

        "Value": [
            f"{area_sqft:,} sq ft",
            bedrooms,
            bathrooms,
            f"{location_score:.1f}/10",
            f"{property_age} years",
            f"{distance_city_km:.1f} km",
            "Yes" if near_school else "No",
            "Yes" if near_metro else "No",
            f"{crime_rate_index:.1f}"
        ]
    })

    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True
    )


with right:

    st.markdown(
        '<div class="section-title">🤖 Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <span class="badge">
    SELECTED MODEL
    </span>

    <h3>RBF Support Vector Regression</h3>

    <p style="color:#94a3b8;">
    Non-linear regression model used for house price prediction.
    </p>

    <hr>

    <b>Pipeline</b>

    <p style="color:#94a3b8;">
    Feature Scaling → RBF SVR → Prediction
    </p>

    <b>Prediction Type</b>

    <p style="color:#94a3b8;">
    Regression
    </p>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = pd.DataFrame({

        "area_sqft": [area_sqft],

        "bedrooms": [bedrooms],

        "bathrooms": [bathrooms],

        "location_score": [location_score],

        "property_age": [property_age],

        "distance_city_km": [distance_city_km],

        "near_school": [near_school],

        "near_metro": [near_metro],

        "crime_rate_index": [crime_rate_index]
    })


    # Prediction

    prediction = float(
        model.predict(input_data)[0]
    )


    # History

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    history_item = {

        "Time": timestamp,

        "Area": area_sqft,

        "Bedrooms": bedrooms,

        "Bathrooms": bathrooms,

        "Location Score": location_score,

        "Property Age": property_age,

        "Distance (km)": distance_city_km,

        "Near School": "Yes" if near_school else "No",

        "Near Metro": "Yes" if near_metro else "No",

        "Crime Index": crime_rate_index,

        "Predicted Price": prediction
    }


    st.session_state.prediction_history.insert(
        0,
        history_item
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">💰 AI Valuation Result</div>',
        unsafe_allow_html=True
    )


    # ONLY HOUSE VALUE INSIDE THE BIG BOX

    st.markdown(
        f"""
        <div class="price-card">
            <div class="price">
                ₹{prediction:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    # ========================================================
    # PRICE METRICS
    # ========================================================

    price_lakh = prediction / 100000

    price_crore = prediction / 10000000

    price_per_sqft = prediction / area_sqft


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Estimated Price
            </div>

            <div class="metric-value">
            ₹{prediction:,.0f}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Price in Lakhs
            </div>

            <div class="metric-value">
            ₹{price_lakh:.2f} L
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m3:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Price in Crores
            </div>

            <div class="metric-value">
            ₹{price_crore:.2f} Cr
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m4:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Estimated Price / Sq Ft
            </div>

            <div class="metric-value">
            ₹{price_per_sqft:,.0f}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROPERTY INSIGHTS
    # ========================================================

    st.markdown(
        '<div class="section-title">🔎 Property Insights</div>',
        unsafe_allow_html=True
    )


    insight1, insight2, insight3 = st.columns(3)


    with insight1:

        if location_score >= 7:
            location_text = "Strong location score"
        else:
            location_text = "Moderate location score"

        st.info(
            f"📍 **Location**\n\n"
            f"{location_text}: {location_score:.1f}/10"
        )


    with insight2:

        accessibility = []

        if near_school:
            accessibility.append("school")

        if near_metro:
            accessibility.append("metro")


        if accessibility:

            text = (
                "Near " +
                " and ".join(accessibility)
            )

        else:

            text = "No nearby selected amenities"


        st.info(
            f"🚇 **Accessibility**\n\n{text}"
        )


    with insight3:

        if property_age <= 10:

            age_text = "Relatively new property"

        else:

            age_text = "Older property"


        st.info(
            f"🏗️ **Property Age**\n\n"
            f"{age_text}: {property_age} years"
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">🕘 Prediction History</div>',
    unsafe_allow_html=True
)


if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )


    display_df = history_df.copy()


    display_df["Predicted Price"] = (
        display_df["Predicted Price"]
        .apply(lambda x: f"₹{x:,.0f}")
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    csv = history_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Prediction History",

        data=csv,

        file_name="prediction_history.csv",

        mime="text/csv"
    )


else:

    st.markdown(
        """
        <div class="card">

        <h4>No predictions yet</h4>

        <p style="color:#94a3b8;">
        Enter property details and generate your first prediction.
        Your prediction history will appear here.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-title">⚡ How EstateAI Works</div>',
    unsafe_allow_html=True
)


step1, step2, step3 = st.columns(3)


with step1:

    st.markdown(
        """
        <div class="card">

        <h3>01 · Property Data</h3>

        <p style="color:#94a3b8;">
        Enter property characteristics such as area,
        bedrooms, bathrooms and location information.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with step2:

    st.markdown(
        """
        <div class="card">

        <h3>02 · ML Pipeline</h3>

        <p style="color:#94a3b8;">
        The saved machine learning pipeline processes
        the property features.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with step3:

    st.markdown(
        """
        <div class="card">

        <h3>03 · Price Estimate</h3>

        <p style="color:#94a3b8;">
        EstateAI generates an estimated property value
        using the trained regression model.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "EstateAI • Machine Learning House Price Prediction • "
    "RBF Support Vector Regression"
)