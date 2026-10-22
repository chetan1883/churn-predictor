import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Churn Radar | Customer Retention Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Look and feel
# ------------------------------------------------------------------
SIGNAL, RISK, WARN, OK = "#4CC9F0", "#FF6B5E", "#FFC857", "#3DDC97"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=Figtree:wght@400;500;600&display=swap');
:root{--ink:#0B1B2B;--panel:#12283B;--line:#23425C;--text:#E8EEF4;--muted:#8FA6BA;--signal:#4CC9F0;--risk:#FF6B5E;}
html, body, .stApp, [class*="css"]{font-family:'Figtree',system-ui,sans-serif;}
.stApp{background:var(--ink);color:var(--text);}
h1,h2,h3,h4{font-family:'Bricolage Grotesque',system-ui,sans-serif !important;letter-spacing:-0.02em;color:var(--text);}
#MainMenu, footer, header[data-testid="stHeader"]{visibility:hidden;height:0;}
.block-container{max-width:1200px;padding-top:1.4rem;padding-bottom:3rem;}
.hero{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:2rem;padding:2.2rem 2.4rem;margin-bottom:1.6rem;border:1px solid var(--line);border-radius:18px;background:var(--panel);}
.hero h1{font-size:clamp(1.7rem,4.2vw,2.9rem);line-height:1.08;margin:0 0 .7rem 0;max-width:22ch;}
.hero p{color:var(--muted);max-width:58ch;margin:0;font-size:1.05rem;line-height:1.55;}
.bars{display:flex;align-items:flex-end;gap:9px;height:96px;}
.bars span{width:17px;border-radius:4px;background:var(--signal);}
.bars span.off{background:var(--risk);opacity:.35;}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem;}
[data-testid="stMetricLabel"]{color:var(--muted);}
.stTabs [data-baseweb="tab-list"]{gap:.25rem;border-bottom:1px solid var(--line);overflow-x:auto;}
.stTabs [data-baseweb="tab"]{padding:.65rem 1rem;color:var(--muted);white-space:nowrap;}
.stTabs [aria-selected="true"]{color:var(--text);}
.note{border-left:3px solid var(--signal);padding:.3rem 0 .3rem 1rem;margin:.6rem 0 1.2rem 0;color:var(--muted);}
.action{border:1px solid var(--line);border-radius:10px;padding:.7rem 1rem;margin:.4rem 0;background:var(--panel);}
.stButton>button, .stDownloadButton>button{border-radius:10px;font-weight:600;}
@media (max-width:640px){.hero{padding:1.4rem 1.2rem;}.bars{height:64px;}.bars span{width:12px;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def style(fig, height=380):
    """Apply the site theme to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, sans-serif", color="#E8EEF4"),
        title_font=dict(family="Bricolage Grotesque, sans-serif", size=18),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    fig.update_xaxes(gridcolor="#1B3349", zerolinecolor="#1B3349")
    fig.update_yaxes(gridcolor="#1B3349", zerolinecolor="#1B3349")
    return fig


# ------------------------------------------------------------------
# Data and model
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")


@st.cache_data
def load_data():
    data = pd.read_csv("churn.csv")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    return data.dropna()


model = load_model()
df = load_data()
FEATURES = [c for c in df.columns if c not in ("customerID", "Churn")]
YES_NO = ["No", "Yes"]
PAYMENTS = ["Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"]


@st.cache_data
def holdout():
    """The 20% of customers the model never saw during training."""
    X = df[FEATURES]
    y = (df["Churn"] == "Yes").astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test, model.predict_proba(X_test)[:, 1], model.predict(X_test)


def churn_by(col):
    g = df.groupby(col).agg(
        customers=("Churn", "size"),
        churned=("Churn", lambda s: (s == "Yes").sum()),
    ).reset_index()
    g[col] = g[col].astype(str)
    g["Churn rate (%)"] = g["churned"] / g["customers"] * 100
    return g.sort_values("Churn rate (%)", ascending=False)


def rate(mask):
    return (df.loc[mask, "Churn"] == "Yes").mean() * 100


def risk_band(p):
    if p >= 0.60:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


BAND_COLOR = {"High": RISK, "Medium": WARN, "Low": OK}


def suggestions(row):
    tips = []
    if row["Contract"] == "Month-to-month":
        tips.append("Offer a discount or perk for moving to a 1- or 2-year contract.")
    if row["PaymentMethod"] == "Electronic check":
        tips.append("Encourage automatic payment with a small incentive.")
    if row["TechSupport"] == "No":
        tips.append("Offer a free trial of tech support.")
    if row["InternetService"] == "Fiber optic":
        tips.append("Review service quality and pricing for fiber customers.")
    if row["tenure"] <= 12:
        tips.append("New customer: start an onboarding and early-loyalty programme.")
    return tips or ["No major risk factors. Keep the relationship warm with regular engagement."]


# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------
churn_rate = (df["Churn"] == "Yes").mean() * 100
one_in = round(100 / churn_rate)
lit = round(5 * (1 - churn_rate / 100))
bars = "".join(
    f'<span class="{"" if i < lit else "off"}" style="height:{h}px"></span>'
    for i, h in enumerate([24, 42, 60, 78, 96])
)
st.markdown(
    '<div class="hero"><div>'
    f"<h1>1 in {one_in} customers leaves. This tool shows who is next.</h1>"
    "<p>Churn Radar scores every telecom customer for the risk of leaving, explains why, "
    "and estimates what a retention campaign is worth.</p>"
    f'</div><div class="bars" aria-hidden="true">{bars}</div></div>',
    unsafe_allow_html=True,
)

tab_over, tab_ins, tab_pred, tab_bulk, tab_impact, tab_model = st.tabs(
    ["Overview", "Insights", "Predict", "Bulk scoring", "Business impact", "Model"]
)

# ---------------- OVERVIEW ----------------
with tab_over:
    churned = df[df["Churn"] == "Yes"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers analysed", f"{len(df):,}")
    c2.metric("Customers who left", f"{len(churned):,}")
    c3.metric("Churn rate", f"{churn_rate:.1f}%")
    c4.metric("Monthly revenue lost", f"${churned['MonthlyCharges'].sum():,.0f}")

    st.subheader("Three findings that drive the recommendations")
    f1, f2, f3 = st.columns(3)
    f1.metric("Month-to-month contracts", f"{rate(df['Contract'] == 'Month-to-month'):.1f}%")
    f1.caption(f"Two-year contracts churn at {rate(df['Contract'] == 'Two year'):.1f}%.")
    f2.metric("Customers paying by electronic check",
              f"{rate(df['PaymentMethod'] == 'Electronic check'):.1f}%")
    f2.caption(f"Automatic payers churn at {rate(df['PaymentMethod'].str.contains('automatic')):.1f}%.")
    f3.metric("Customers in their first year", f"{rate(df['tenure'] <= 12):.1f}%")
    f3.caption(f"Customers beyond a year churn at {rate(df['tenure'] > 12):.1f}%.")

    st.subheader("What you can do here")
    st.markdown(
        "- **Insights:** explore why customers leave, with interactive charts.\n"
        "- **Predict:** change one customer's details and watch the risk update live.\n"
        "- **Bulk scoring:** upload a customer list and get it ranked by risk.\n"
        "- **Business impact:** see what a retention campaign would earn.\n"
        "- **Model:** check how reliable the predictions are."
    )
    with st.expander("Project method and limitations"):
        st.markdown(
            f"""
**Data:** {len(df):,} telecom customers, {len(FEATURES)} attributes (demographics, services, billing, contract).

**Method:** descriptive analysis of churn patterns, then two classifiers (logistic regression and random forest) trained on 80% of customers and tested on the remaining 20%. Classes were balanced so the model pays attention to the minority group of churners. The best model by ROC-AUC powers this site.

**Limitations:** one sample dataset from a fictional telecom, a single point in time, and no information on why a customer left. Predictions show risk, not cause, and the campaign figures on the Business impact tab are simulations based on the assumptions you set.
"""
        )

# ---------------- INSIGHTS ----------------
with tab_ins:
    st.subheader("Why customers leave")
    factor = st.selectbox(
        "Compare churn across",
        ["Contract", "InternetService", "PaymentMethod", "TechSupport",
         "OnlineSecurity", "PaperlessBilling", "Dependents", "SeniorCitizen"],
    )
    g = churn_by(factor)
    top, bottom = g.iloc[0], g.iloc[-1]
    fig = px.bar(
        g, x=factor, y="Churn rate (%)", text_auto=".1f",
        color="Churn rate (%)", hover_data={"customers": True, "churned": True},
        color_continuous_scale=[[0, OK], [0.5, WARN], [1, RISK]],
        title=f"Churn rate by {factor}",
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(style(fig), use_container_width=True)
    st.markdown(
        f'<div class="note"><b>{top[factor]}</b> customers churn at {top["Churn rate (%)"]:.1f}%, '
        f'compared with {bottom["Churn rate (%)"]:.1f}% for <b>{bottom[factor]}</b>.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    band = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                  labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    tb = (df.assign(Band=band).groupby("Band", observed=True)["Churn"]
          .apply(lambda s: (s == "Yes").mean() * 100)
          .reset_index(name="Churn rate (%)"))
    tb["Band"] = tb["Band"].astype(str)
    fig = px.line(tb, x="Band", y="Churn rate (%)", markers=True,
                  title="Churn falls the longer a customer stays")
    fig.update_traces(line_color=SIGNAL, marker_size=10)
    fig.update_xaxes(title="Tenure (months)")
    left.plotly_chart(style(fig, 340), use_container_width=True)

    fig = px.histogram(df, x="MonthlyCharges", color="Churn", barmode="overlay",
                       nbins=40, opacity=0.7,
                       color_discrete_map={"Yes": RISK, "No": SIGNAL},
                       title="Monthly charges: churned vs retained")
    right.plotly_chart(style(fig, 340), use_container_width=True)

    pivot = df.pivot_table(index="Contract", columns="InternetService",
                           values="Churn", aggfunc=lambda s: (s == "Yes").mean() * 100)
    fig = px.imshow(pivot, text_auto=".1f", aspect="auto",
                    color_continuous_scale=[[0, "#12283B"], [1, RISK]],
                    title="Where risks combine: churn rate (%) by contract and internet service")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(style(fig, 320), use_container_width=True)

# ---------------- PREDICT ----------------
with tab_pred:
    st.subheader("Score one customer")
    st.caption("Change any detail and the risk updates instantly.")
    col_in, col_out = st.columns([1.15, 1])

    with col_in:
        st.markdown("**Profile**")
        a, b = st.columns(2)
        gender = a.selectbox("Gender", ["Female", "Male"])
        senior = b.selectbox("Senior citizen", YES_NO)
        partner = a.selectbox("Partner", YES_NO)
        dependents = b.selectbox("Dependents", YES_NO)
        tenure = st.slider("Tenure (months)", 0, 72, 6)

        st.markdown("**Services**")
        a, b = st.columns(2)
        phone = a.selectbox("Phone service", YES_NO, index=1)
        lines = b.selectbox("Multiple lines",
                            ["No", "Yes"] if phone == "Yes" else ["No phone service"])
        internet = a.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        addon = ["No", "Yes"] if internet != "No" else ["No internet service"]
        security = b.selectbox("Online security", addon)
        backup = a.selectbox("Online backup", addon)
        protection = b.selectbox("Device protection", addon)
        support = a.selectbox("Tech support", addon)
        tv = b.selectbox("Streaming TV", addon)
        movies = a.selectbox("Streaming movies", addon)

        st.markdown("**Account**")
        a, b = st.columns(2)
        contract = a.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = b.selectbox("Paperless billing", YES_NO, index=1)
        payment = a.selectbox("Payment method", PAYMENTS)
        monthly = b.number_input("Monthly charges ($)", 18.0, 120.0, 80.0, step=1.0)

    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": round(monthly * tenure, 2),
    }])[FEATURES]
    prob = model.predict_proba(row)[0, 1]
    band_name = risk_band(prob)

    with col_out:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 46}},
            title={"text": f"{band_name} churn risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": BAND_COLOR[band_name]},
                "bgcolor": "#12283B",
                "borderwidth": 0,
                "steps": [{"range": [0, 40], "color": "#17384A"},
                          {"range": [40, 60], "color": "#1D4258"},
                          {"range": [60, 100], "color": "#244C66"}],
            },
        ))
        st.plotly_chart(style(gauge, 300), use_container_width=True)
        st.metric("Expected revenue at risk (next 12 months)",
                  f"${monthly * 12 * prob:,.0f}")

        st.markdown("**Suggested retention actions**")
        for tip in suggestions(row.iloc[0]):
            st.markdown(f'<div class="action">{tip}</div>', unsafe_allow_html=True)

        st.markdown("**What if we act?**")
        options = []
        if contract == "Month-to-month":
            options.append(("Move to a 2-year contract", {"Contract": "Two year"}))
        if payment == "Electronic check":
            options.append(("Switch to automatic payment",
                            {"PaymentMethod": "Bank transfer (automatic)"}))
        if internet != "No" and support == "No":
            options.append(("Add tech support", {"TechSupport": "Yes"}))
        if options:
            cols = st.columns(len(options))
            for c, (label, change) in zip(cols, options):
                alt = row.copy()
                for k, v in change.items():
                    alt.loc[0, k] = v
                new_p = model.predict_proba(alt)[0, 1]
                c.metric(label, f"{new_p * 100:.0f}%",
                         f"{(new_p - prob) * 100:+.0f} pts",
                         delta_color="inverse")
        else:
            st.caption("This customer already has the strongest retention settings.")

# ---------------- BULK SCORING ----------------
with tab_bulk:
    st.subheader("Rank a customer list by churn risk")
    st.write("Upload a CSV with the same columns as the training data. "
             "`customerID` is optional and `Churn` is ignored.")
    st.download_button("Download a sample file to try",
                       df.drop(columns="Churn").head(200).to_csv(index=False),
                       "sample_customers.csv", "text/csv")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        data = pd.read_csv(file)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error("These columns are missing from your file: " + ", ".join(missing))
        else:
            ids = data["customerID"] if "customerID" in data.columns else pd.Series(range(1, len(data) + 1))
            feats = data[FEATURES].copy()
            feats["TotalCharges"] = pd.to_numeric(feats["TotalCharges"], errors="coerce").fillna(0)
            probs = model.predict_proba(feats)[:, 1]
            out = pd.DataFrame({
                "Customer": ids,
                "Churn probability (%)": (probs * 100).round(1),
                "Risk level": [risk_band(p) for p in probs],
                "Monthly charges ($)": feats["MonthlyCharges"].values,
            }).sort_values("Churn probability (%)", ascending=False)
            counts = out["Risk level"].value_counts()
            a, b, c = st.columns(3)
            a.metric("High risk", int(counts.get("High", 0)))
            b.metric("Medium risk", int(counts.get("Medium", 0)))
            c.metric("Low risk", int(counts.get("Low", 0)))
            fig = px.histogram(out, x="Churn probability (%)", nbins=20,
                               title="Distribution of churn risk")
            fig.update_traces(marker_color=SIGNAL)
            st.plotly_chart(style(fig, 300), use_container_width=True)
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("Download scored list", out.to_csv(index=False),
                               "churn_scores.csv", "text/csv")

# ---------------- BUSINESS IMPACT ----------------
with tab_impact:
    st.subheader("What is a retention campaign worth?")
    X_test, y_test, proba, _ = holdout()
    st.caption(f"Simulated on {len(y_test):,} customers the model never saw during training. "
               "Adjust the assumptions to match a real campaign.")
    a, b, c = st.columns(3)
    share = a.slider("Share of customers contacted (highest risk first)", 5, 50, 20, format="%d%%")
    success = b.slider("Offer success rate", 5, 80, 30, format="%d%%") / 100
    cost_each = c.slider("Cost of offer per customer ($)", 0, 150, 25)

    order = np.argsort(-proba)
    k = int(len(order) * share / 100)
    actual = y_test.to_numpy()[order]
    monthly_sorted = X_test["MonthlyCharges"].to_numpy()[order]

    caught = actual[:k].sum()
    saved = (actual[:k] * monthly_sorted[:k]).sum() * 12 * success
    cost = k * cost_each
    net = saved - cost

    base_rate = y_test.mean()
    avg_monthly_churner = X_test.loc[y_test == 1, "MonthlyCharges"].mean()
    caught_rand = base_rate * k
    saved_rand = caught_rand * avg_monthly_churner * 12 * success
    net_rand = saved_rand - cost

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers contacted", f"{k:,}")
    m2.metric("Churners reached", f"{int(caught):,}",
              f"{caught - caught_rand:+.0f} vs random", delta_color="normal")
    m3.metric("Annual revenue saved", f"${saved:,.0f}")
    m4.metric("Net benefit", f"${net:,.0f}",
              f"ROI {net / cost * 100:,.0f}%" if cost > 0 else None)

    left, right = st.columns(2)
    fig = go.Figure(go.Bar(
        x=["Contact customers at random", "Contact highest-risk customers"],
        y=[net_rand, net],
        marker_color=[WARN, OK],
        text=[f"${net_rand:,.0f}", f"${net:,.0f}"], textposition="outside",
    ))
    fig.update_layout(title="Net benefit of the campaign")
    left.plotly_chart(style(fig, 340), use_container_width=True)

    share_contacted = np.arange(0, len(actual) + 1) / len(actual) * 100
    share_caught = np.concatenate([[0], np.cumsum(actual)]) / actual.sum() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=share_contacted, y=share_caught, name="Model",
                             line=dict(color=SIGNAL, width=3)))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Random",
                             line=dict(color="#8FA6BA", dash="dash")))
    fig.update_layout(title="Share of churners reached",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title="Customers contacted (%)")
    fig.update_yaxes(title="Churners reached (%)")
    right.plotly_chart(style(fig, 340), use_container_width=True)

    reached = share_caught[k]
    st.markdown(
        f'<div class="note">Contacting the top {share}% by risk reaches about '
        f'<b>{reached:.0f}%</b> of all churners, versus {share}% if customers were picked at random.</div>',
        unsafe_allow_html=True,
    )

# ---------------- MODEL ----------------
with tab_model:
    st.subheader("How reliable are the predictions?")
    X_test, y_test, proba, pred = holdout()
    est = model.named_steps["model"]
    nice = {"LogisticRegression": "Logistic regression",
            "RandomForestClassifier": "Random forest"}
    st.caption(f"Selected model: {nice.get(type(est).__name__, type(est).__name__)}, "
               f"evaluated on {len(y_test):,} unseen customers.")

    m = st.columns(5)
    m[0].metric("Accuracy", f"{accuracy_score(y_test, pred):.2f}")
    m[1].metric("Precision", f"{precision_score(y_test, pred):.2f}")
    m[2].metric("Recall", f"{recall_score(y_test, pred):.2f}")
    m[3].metric("F1 score", f"{f1_score(y_test, pred):.2f}")
    m[4].metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.2f}")
    st.markdown(
        '<div class="note">Recall matters most for retention: it is the share of customers '
        "who really left that the model flagged in time.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name="Model", line=dict(color=SIGNAL, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance",
                             line=dict(color="#8FA6BA", dash="dash")))
    fig.update_layout(title="ROC curve", legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title="False positive rate")
    fig.update_yaxes(title="True positive rate")
    left.plotly_chart(style(fig, 360), use_container_width=True)

    cm = confusion_matrix(y_test, pred)
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                    x=["Predicted: stay", "Predicted: leave"],
                    y=["Actual: stay", "Actual: leave"],
                    color_continuous_scale=[[0, "#12283B"], [1, SIGNAL]],
                    title="Confusion matrix")
    fig.update_coloraxes(showscale=False)
    right.plotly_chart(style(fig, 360), use_container_width=True)

    names = model.named_steps["prep"].get_feature_names_out()
    values = est.feature_importances_ if hasattr(est, "feature_importances_") else np.abs(est.coef_[0])
    imp = pd.DataFrame({
        "Feature": [n.split("__", 1)[1].replace("_", " = ", 1) for n in names],
        "Importance": values,
    }).nlargest(12, "Importance").sort_values("Importance")
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                 title="What the model relies on most")
    fig.update_traces(marker_color=SIGNAL)
    st.plotly_chart(style(fig, 440), use_container_width=True)

st.caption("Built with Python, scikit-learn, Plotly and Streamlit. Data: IBM Telco Customer Churn sample dataset.")