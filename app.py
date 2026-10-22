import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Churn Analytics Hub",
    page_icon="🔮",
    layout="wide"
)

# 2. Add Custom Modern CSS Styling
st.markdown("""
    <style>
    /* Styling the Main Welcome Header */
    .welcome-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .welcome-subtitle {
        font-size: 1.4rem;
        color: #A3A8B4;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 40px;
    }
    
    /* Sleek Container Cards */
    .feature-card {
        background-color: #1E2330;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 20px;
        height: 100%;
    }
    .feature-card h3 {
        color: #FFFFFF !important;
        margin-top: 0px;
    }
    .feature-card p {
        color: #CDCED2;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Hero Header Section
st.markdown("<h1 class='welcome-header'>🔮 Customer Churn Analytics Hub</h1>", unsafe_allow_html=True)
st.markdown("<p class='welcome-subtitle'>Predictive Intelligence to Protect and Scale Your Revenue Base</p>", unsafe_allow_html=True)

st.write("---")

# 4. Interactive Key Feature Cards (3 Columns)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='feature-card'>
        <h3>📊 Advanced Analytics</h3>
        <p>Deep-dive into core user behavior metrics. Track engagement trends, drop-off rates, and usage patterns before users walk away.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-card'>
        <h3>🤖 AI Churn Predictor</h3>
        <p>Leverage trained machine learning pipelines to flag high-risk accounts instantly based on modern tenure and active flags.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-card'>
        <h3>💡 Smart Retention</h3>
        <p>Don't just look at errors—uncover prescriptive recommendations to re-engage accounts based on calculated risk variables.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# 5. Prominent, Wide Interactive Launch Button
col_left, col_mid, col_right = st.columns([1, 2, 1])
with col_mid:
    st.info("⚡ System initialized. Ready to upload telemetry or evaluate active clients.")
    if st.button("🚀 LAUNCH SYSTEM DASHBOARD", type="primary", use_container_width=True):
        st.switch_page("pages/main.py")
