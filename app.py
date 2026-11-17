"""HR Attrition Risk Dashboard  (run:  streamlit run app.py)"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import INPUT_COLUMNS, CATEGORICAL, ENGINEERED, add_features
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
        st.error("Model file not found. Run `python train_model.py` first.")
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


FRIENDLY = {"AvgSatisfaction": "Overall satisfaction",
            "TenureRatio": "Share of career spent here",
            "PromotionGap": "Time since promotion vs tenure",
            "IncomePerLevel": "Pay for job level",
            "JobHopping": "How often they changed jobs"}


def pretty(name, feat_row):
    name = name.split("__", 1)[1]
    for c in CATEGORICAL:
        if name.startswith(c + "_"):
            return f"{re.sub(r'(?<!^)(?=[A-Z])', ' ', c)}: {name[len(c) + 1:]}"
    if name in FRIENDLY:
        return FRIENDLY[name]
    label = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    val = feat_row.get(name)
    return label if val is None else f"{label} ({val:g})"


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

SHOW_BATCH = True   # "Check many (CSV)" tab  -> set False to hide
SHOW_MODEL = True   # "About the model" tab  -> set False to hide

tab_names = ["Check one employee"]
if SHOW_BATCH:
    tab_names.append("Check many (CSV)")
if SHOW_MODEL:
    tab_names.append("About the model")
_tabs = st.tabs(tab_names)
tab_single = _tabs[0]
tab_batch = _tabs[tab_names.index("Check many (CSV)")] if SHOW_BATCH else None
tab_model = _tabs[tab_names.index("About the model")] if SHOW_MODEL else None

# ================================================================= SINGLE
def pick(options, wanted):
    return wanted if wanted in options else options[0]


BASE = {
    "f_department": OPT["Department"][0], "f_job_role": OPT["JobRole"][0],
    "f_job_level": 2, "f_income": int(DEF["MonthlyIncome"]), "f_overtime": "No",
    "f_job_sat": 3, "f_env_sat": 3, "f_wlb": 3,
    "f_total_years": int(DEF["TotalWorkingYears"]), "f_years_co": int(DEF["YearsAtCompany"]),
    "f_promo": int(DEF["YearsSinceLastPromotion"]),
    "f_travel": pick(OPT["BusinessTravel"], "Travel_Rarely"),
    "f_distance": int(DEF["DistanceFromHome"]), "f_involvement": 3,
    "f_stock": int(DEF["StockOptionLevel"]), "f_n_comp": int(DEF["NumCompaniesWorked"]),
    "f_mgr": int(DEF["YearsWithCurrManager"]), "f_training": int(DEF["TrainingTimesLastYear"]),
}
EX_LEAVE = {**BASE, "f_department": pick(OPT["Department"], "Sales"),
            "f_job_role": pick(OPT["JobRole"], "Sales Representative"),
            "f_job_level": 1, "f_income": 2300, "f_overtime": "Yes",
            "f_job_sat": 1, "f_env_sat": 1, "f_wlb": 1, "f_total_years": 4,
            "f_years_co": 1, "f_promo": 0, "f_travel": pick(OPT["BusinessTravel"], "Travel_Frequently"),
            "f_distance": 24, "f_involvement": 2, "f_stock": 0, "f_n_comp": 5,
            "f_mgr": 0, "f_training": 0}
EX_STAY = {**BASE, "f_department": pick(OPT["Department"], "Research & Development"),
           "f_job_role": pick(OPT["JobRole"], "Manager"),
           "f_job_level": 3, "f_income": 9500, "f_overtime": "No",
           "f_job_sat": 4, "f_env_sat": 4, "f_wlb": 3, "f_total_years": 16,
           "f_years_co": 10, "f_promo": 1, "f_travel": pick(OPT["BusinessTravel"], "Travel_Rarely"),
           "f_distance": 4, "f_involvement": 3, "f_stock": 1, "f_n_comp": 2,
           "f_mgr": 6, "f_training": 3}

for k, v in BASE.items():
    st.session_state.setdefault(k, v)


def load_example(values):
    for k, v in values.items():
        st.session_state[k] = v


def radio_rating(label, key, help_text=None):
    return st.radio(label, [1, 2, 3, 4], format_func=lambda x: RATING[x],
                    horizontal=True, key=key, help=help_text)


with tab_single:
    b1, b2, b3, _sp = st.columns([1.2, 1.2, 1, 3])
    b1.button("Fill example: likely to leave", on_click=load_example, args=(EX_LEAVE,))
    b2.button("Fill example: likely to stay", on_click=load_example, args=(EX_STAY,))
    b3.button("Reset form", on_click=load_example, args=(BASE,))

    with st.form("employee", border=False):
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown("**Role and pay**")
                st.selectbox("Department", OPT["Department"], key="f_department")
                st.selectbox("Job role", OPT["JobRole"], key="f_job_role")
                st.selectbox("Job level", [1, 2, 3, 4, 5], key="f_job_level",
                             help="1 = entry level, 5 = most senior")
                st.number_input("Monthly income", 0, 500000, step=100, key="f_income",
                                help="The model learned from incomes between about 1,000 and 20,000. "
                                     "Use the same scale.")
                st.radio("Works overtime regularly?", ["No", "Yes"], horizontal=True,
                         key="f_overtime")
        with right:
            with st.container(border=True):
                st.markdown("**How they feel about work**")
                st.caption("From your latest survey or a recent one-to-one.")
                radio_rating("Job satisfaction", "f_job_sat")
                radio_rating("Satisfaction with work environment", "f_env_sat")
                radio_rating("Work-life balance", "f_wlb")

        with st.container(border=True):
            st.markdown("**Time and career**")
            t1, t2, t3 = st.columns(3)
            t1.number_input("Total years of work experience", 0, 45, key="f_total_years")
            t2.number_input("Years at this company", 0, 45, key="f_years_co")
            t3.number_input("Years since last promotion", 0, 20, key="f_promo",
                            help="Use years at this company if never promoted.")

        with st.expander("More details (optional, typical values are already filled in)"):
            m1, m2, m3 = st.columns(3)
            m1.selectbox("Business travel", OPT["BusinessTravel"], key="f_travel")
            m2.slider("Distance from home (km)", 1, 30, key="f_distance")
            m3.number_input("Companies worked at before", 0, 15, key="f_n_comp")
            m4, m5, m6 = st.columns(3)
            m4.number_input("Years with current manager", 0, 20, key="f_mgr")
            m5.slider("Trainings attended last year", 0, 6, key="f_training")
            m6.selectbox("Stock option level", [0, 1, 2, 3], key="f_stock",
                         help="0 = no stock options")
            radio_rating("Job involvement", "f_involvement",
                         "How engaged the employee is in their day-to-day work.")

        go_btn = st.form_submit_button("Check attrition risk", type="primary")

    if go_btn:
        S = st.session_state
        if S.f_years_co > S.f_total_years:
            st.error("Years at this company cannot be more than total years of experience.")
            st.stop()
        row = {"Department": S.f_department, "JobRole": S.f_job_role,
               "BusinessTravel": S.f_travel, "OverTime": S.f_overtime,
               "JobLevel": S.f_job_level, "MonthlyIncome": S.f_income,
               "DistanceFromHome": S.f_distance, "JobSatisfaction": S.f_job_sat,
               "EnvironmentSatisfaction": S.f_env_sat, "WorkLifeBalance": S.f_wlb,
               "JobInvolvement": S.f_involvement, "StockOptionLevel": S.f_stock,
               "TotalWorkingYears": S.f_total_years, "NumCompaniesWorked": S.f_n_comp,
               "YearsAtCompany": S.f_years_co, "YearsSinceLastPromotion": S.f_promo,
               "YearsWithCurrManager": S.f_mgr, "TrainingTimesLastYear": S.f_training}
        if S.f_income > 25000:
            st.warning("Monthly income is above the range the model was trained on "
                       "(about 1,000 to 20,000), so this result may be less reliable.")
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
        recs = get_recommendations(row, B["peer_income"])

        def show(r):
            c = PRIORITY_COLOR[r["priority"]]
            st.markdown(
                f"<div class='rec' style='--c:{c}'><b>{r['area']}: {r['issue']} "
                f"({PRIORITY_LABEL[r['priority']]})</b>{r['action']}</div>",
                unsafe_allow_html=True)

        for r in recs[:4]:
            show(r)
        if len(recs) > 4:
            with st.expander(f"{len(recs) - 4} more suggestions"):
                for r in recs[4:]:
                    show(r)
        st.caption("This is decision support, not a verdict on the person. "
                   "Use it to start a conversation, not to make hiring, firing or pay decisions.")

# ================================================================== BATCH
if SHOW_BATCH:
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
if SHOW_MODEL:
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
