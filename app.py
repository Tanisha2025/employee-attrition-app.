"""HR Attrition Risk Dashboard  (run:  streamlit run app.py)"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import INPUT_COLUMNS, CATEGORICAL, add_features
from recommendations import get_recommendations, RATING

st.set_page_config(page_title="Attrition risk check", page_icon="🧭", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.verdict {padding: 1rem 1.25rem; border-radius: 8px; border-left: 6px solid var(--c);
          background: color-mix(in srgb, var(--c) 10%, transparent);}
.verdict h2 {margin: 0; font-size: 1.5rem;}
.verdict p {margin: .25rem 0 0 0;}
.rec {padding: .65rem .9rem; border-radius: 6px; margin-bottom: .5rem;
      border-left: 4px solid var(--c); background: color-mix(in srgb, var(--c) 8%, transparent);}
.rec b {display: block;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    path = Path(__file__).parent / "attrition_model.joblib"
    if not path.exists():
        st.error("Model file not found. Run the training cell in the notebook first.")
        st.stop()
    return joblib.load(path)


B = load_bundle()
THR = B["threshold"]
OPT = B["options"]
DEF = B["defaults"]

PRIORITY_COLOR = {1: "#c0392b", 2: "#e08e0b", 3: "#2e86ab"}
PRIORITY_LABEL = {1: "Act now", 2: "Act soon", 3: "Good practice"}


# ------------------------------------------------------------------ helpers
def risk_band(p):
    if p >= THR:
        return "High", "#c0392b"
    if p >= 0.6 * THR:
        return "Medium", "#e08e0b"
    return "Low", "#2e8b57"


def gauge(p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
               "steps": [{"range": [0, 60 * THR], "color": "#bfe3cd"},
                         {"range": [60 * THR, 100 * THR], "color": "#f6dba5"},
                         {"range": [100 * THR, 100], "color": "#efb4ad"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=0))
    return fig


def pretty(name, feat_row):
    name = name.split("__", 1)[1]
    for c in CATEGORICAL:
        if name.startswith(c + "_"):
            return f"{c} = {name[len(c) + 1:]}"
    val = feat_row.get(name)
    if val is None:
        return name
    return f"{name} ({val:.2f})" if isinstance(val, float) and val % 1 else f"{name} ({val:g})"


def explain(df_row):
    """Per-employee contribution of every feature (exact for a linear model)."""
    z = B["preprocessor"].transform(df_row)[0]
    contrib = B["coef"] * (z - B["mean_z"])
    feat_row = add_features(df_row).iloc[0].to_dict()
    return pd.DataFrame({"feature": [pretty(n, feat_row) for n in B["feature_names"]],
                         "impact": contrib}).sort_values("impact")


def predict_proba(df):
    return B["pipeline"].predict_proba(df[INPUT_COLUMNS])[:, 1]


def contrib_chart(exp):
    top = pd.concat([exp.head(5), exp.tail(6)]).drop_duplicates("feature")
    colors = ["#2e8b57" if v < 0 else "#c0392b" for v in top["impact"]]
    fig = go.Figure(go.Bar(x=top["impact"], y=top["feature"], orientation="h",
                           marker_color=colors))
    fig.update_layout(height=380, margin=dict(l=0, r=10, t=10, b=0),
                      xaxis_title="Pushes towards staying  ←    →  pushes towards leaving")
    return fig


def rating(label, key, default=3):
    return st.select_slider(label, options=[1, 2, 3, 4], value=default,
                            format_func=lambda x: RATING[x], key=key)


# ------------------------------------------------------------------ layout
st.title("Employee attrition risk check")
st.caption("Enter an employee's details to see how likely they are to leave, "
           "why, and what HR can do about it.")

tab_single, tab_batch, tab_model = st.tabs(
    ["Check one employee", "Check many (CSV)", "About the model"])

# ================================================================= SINGLE
with tab_single:
    with st.form("employee"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Role**")
            department = st.selectbox("Department", OPT["Department"])
            job_role = st.selectbox("Job role", OPT["JobRole"])
            job_level = st.selectbox("Job level", [1, 2, 3, 4, 5], index=1)
            income = st.number_input("Monthly income", 1000, 30000, int(DEF["MonthlyIncome"]), 100)
            overtime = st.radio("Works overtime", ["No", "Yes"], horizontal=True)
            travel = st.selectbox("Business travel", OPT["BusinessTravel"])
            distance = st.slider("Distance from home (km)", 1, 30, int(DEF["DistanceFromHome"]))
        with c2:
            st.markdown("**How they feel about work**")
            job_sat = rating("Job satisfaction", "js")
            env_sat = rating("Environment satisfaction", "es")
            wlb = rating("Work-life balance", "wlb")
            involvement = rating("Job involvement", "ji")
            stock = st.selectbox("Stock option level", [0, 1, 2, 3])
            training = st.slider("Trainings last year", 0, 6, 2)
        with c3:
            st.markdown("**Experience**")
            total_years = st.number_input("Total working years", 0, 45, int(DEF["TotalWorkingYears"]))
            years_co = st.number_input("Years at this company", 0, 45, int(DEF["YearsAtCompany"]))
            promo = st.number_input("Years since last promotion", 0, 20, int(DEF["YearsSinceLastPromotion"]))
            mgr = st.number_input("Years with current manager", 0, 20, int(DEF["YearsWithCurrManager"]))
            n_comp = st.number_input("Companies worked at before", 0, 15, int(DEF["NumCompaniesWorked"]))
        go_btn = st.form_submit_button("Check attrition risk", type="primary")

    if go_btn:
        if years_co > total_years:
            st.error("Years at this company cannot be more than total working years.")
            st.stop()
        row = {"Department": department, "JobRole": job_role, "BusinessTravel": travel,
               "OverTime": overtime, "JobLevel": job_level, "MonthlyIncome": income,
               "DistanceFromHome": distance, "JobSatisfaction": job_sat,
               "EnvironmentSatisfaction": env_sat, "WorkLifeBalance": wlb,
               "JobInvolvement": involvement, "StockOptionLevel": stock,
               "TotalWorkingYears": total_years, "NumCompaniesWorked": n_comp,
               "YearsAtCompany": years_co, "YearsSinceLastPromotion": promo,
               "YearsWithCurrManager": mgr, "TrainingTimesLastYear": training}
        df_row = pd.DataFrame([row])[INPUT_COLUMNS]
        p = float(predict_proba(df_row)[0])
        band, color = risk_band(p)
        leaves = p >= THR

        st.divider()
        left, right = st.columns([1, 1.3])
        with left:
            st.plotly_chart(gauge(p), width="stretch")
            headline = "Likely to leave" if leaves else "Likely to stay"
            st.markdown(
                f"<div class='verdict' style='--c:{color}'><h2>{headline}</h2>"
                f"<p>{band} risk · model score {p:.0%} "
                f"(flag raised from {THR:.0%})</p></div>", unsafe_allow_html=True)
        with right:
            st.subheader("What is driving this result")
            exp = explain(df_row)
            st.plotly_chart(contrib_chart(exp), width="stretch")

        st.subheader("Recommended HR actions")
        if not leaves:
            st.caption("Risk is below the alert level. These are preventive actions.")
        for r in get_recommendations(row, B["peer_income"]):
            c = PRIORITY_COLOR[r["priority"]]
            st.markdown(
                f"<div class='rec' style='--c:{c}'><b>{r['area']}: {r['issue']} "
                f"({PRIORITY_LABEL[r['priority']]})</b>{r['action']}</div>",
                unsafe_allow_html=True)
        st.caption("This is decision support, not a verdict on the person. "
                   "Use it to start a conversation, not to make hiring, firing or pay decisions.")

# ================================================================== BATCH
with tab_batch:
    st.write("Upload a CSV with one row per employee. Any extra columns "
             "(name, ID…) are kept in the results.")
    st.download_button("Download CSV template",
                       pd.DataFrame(columns=INPUT_COLUMNS).to_csv(index=False),
                       "attrition_template.csv", "text/csv")
    up = st.file_uploader("Employee CSV", type="csv")
    if up:
        data = pd.read_csv(up)
        missing = [c for c in INPUT_COLUMNS if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            probs = predict_proba(data)
            out = data.copy()
            out["LeaveProbability"] = probs.round(3)
            out["Prediction"] = np.where(probs >= THR, "Likely to leave", "Likely to stay")
            out["RiskBand"] = [risk_band(p)[0] for p in probs]
            out = out.sort_values("LeaveProbability", ascending=False)
            m1, m2, m3 = st.columns(3)
            m1.metric("Employees", len(out))
            m2.metric("High risk", int((out["RiskBand"] == "High").sum()))
            m3.metric("Share flagged", f"{(probs >= THR).mean():.0%}")
            st.dataframe(out, width="stretch", hide_index=True)
            st.download_button("Download results", out.to_csv(index=False),
                               "attrition_predictions.csv", "text/csv")

# ================================================================== MODEL
with tab_model:
    m = B["metrics"]
    st.subheader("How well does it work? (held-out test set)")
    a, b, c, d = st.columns(4)
    a.metric("Recall (leavers caught)", f"{m['recall']:.0%}")
    b.metric("Precision (flags that are right)", f"{m['precision']:.0%}")
    c.metric("ROC AUC", f"{m['roc_auc']:.2f}")
    d.metric("Accuracy", f"{m['accuracy']:.0%}")
    tn, fp, fn, tp = np.array(m["confusion_matrix"]).ravel()
    st.write(f"Of the employees who really left, the model flagged **{tp}** and missed **{fn}**. "
             f"It also flagged **{fp}** people who stayed.")
    st.subheader("Model comparison (cross-validation)")
    st.dataframe(B["cv_table"].round(3), hide_index=True, width="stretch")
    st.subheader("Most influential factors overall")
    imp = pd.DataFrame({"feature": B["feature_names"], "coef": B["coef"]}).sort_values("coef")
    imp = pd.concat([imp.head(8), imp.tail(8)])
    st.plotly_chart(go.Figure(go.Bar(
        x=imp["coef"], y=imp["feature"].str.split("__").str[1], orientation="h",
        marker_color=["#2e8b57" if v < 0 else "#c0392b" for v in imp["coef"]]))
        .update_layout(height=450, margin=dict(l=0, r=10, t=10, b=0)),
        width="stretch")
    st.info("Gender, marital status and age are intentionally excluded so the model "
            "cannot treat people differently because of them. The sample IBM dataset is "
            "fictional: retrain on your own HR data before real use.")
