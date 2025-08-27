"""
Streamlit App: Causal Inference for Policy Evaluation (Job Training → Earnings)
Author: You

Quick start
-----------
1) Create a new virtual env and install deps:
   pip install -r requirements.txt

2) Run the app:
   streamlit run app.py

Notes
-----
- This app uses a small, classic dataset (Lalonde/NSW). We try to load it from the
  `causaldata` package. If unavailable, we generate a light synthetic fallback so the
  app still runs. Replace `load_data()` with your preferred source if needed.
- Methods included: IPW, PSM, optional DML (if `econml` is installed).
- Added: bootstrap 95% CI, propensity overlap plot, subgroup effects, download buttons,
  and assumptions/limitations expander.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import streamlit as st

# ----------------------------
# Data Loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    try:
        from causaldata import lalonde
        df = lalonde.load_pandas().data.copy()
        return df
    except Exception:
        rng = np.random.default_rng(7)
        n = 600
        age = rng.integers(18, 55, size=n)
        educ = rng.integers(6, 18, size=n)
        black = rng.integers(0, 2, size=n)
        hisp = rng.integers(0, 2, size=n)
        married = rng.integers(0, 2, size=n)
        re74 = rng.normal(5000 + 200*educ - 50*age, 2000, size=n).clip(0)
        re75 = re74 + rng.normal(0, 1000, size=n)
        logits = -3 + 0.06*educ - 0.03*age + 0.5*black - 0.3*married + 0.0002*re75
        p = 1/(1+np.exp(-logits))
        treat = rng.binomial(1, p)
        tau = 1500 + 40*educ - 10*age + 300*married - 200*black
        noise = rng.normal(0, 1500, size=n)
        re78 = (3000 + 250*educ - 60*age + 0.2*re75
                + treat * tau + noise).clip(0)
        df = pd.DataFrame({
            'treat': treat,
            'age': age,
            'educ': educ,
            'black': black,
            'hispan': hisp,
            'married': married,
            're74': re74,
            're75': re75,
            're78': re78
        })
        return df

# ----------------------------
# Helpers & Estimators
# ----------------------------

def build_ps_model(confounders):
    numeric_cols = [c for c in confounders if confounders[c] == 'num']
    cat_cols = [c for c in confounders if confounders[c] == 'cat']
    pre = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', 'passthrough', cat_cols)
    ], remainder='drop')
    model = Pipeline([
        ('pre', pre),
        ('clf', LogisticRegression(max_iter=200))
    ])
    return model


def bootstrap_ci(df, x_cols, est_fn, B=300, alpha=0.05):
    vals = []
    n = len(df)
    for _ in range(B):
        samp = df.sample(n, replace=True)
        r = est_fn(samp, x_cols)
        if r["ATE"] is not None:
            vals.append(r["ATE"])
    if len(vals) == 0:
        return None, None, None
    lo, hi = np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)])
    return float(np.mean(vals)), float(lo), float(hi)


def estimate_ipw(df, x_cols):
    X = df[x_cols]
    T = df['treat'].values
    Y = df['re78'].values
    confounders = {c: ('num' if np.issubdtype(df[c].dtype, np.number) else 'cat') for c in x_cols}
    ps_model = build_ps_model(confounders)
    ps_model.fit(X, T)
    e = ps_model.predict_proba(X)[:, 1].clip(1e-3, 1-1e-3)
    w_t = T / e
    w_c = (1 - T) / (1 - e)
    mu1 = np.sum(w_t * Y) / np.sum(w_t)
    mu0 = np.sum(w_c * Y) / np.sum(w_c)
    ate = float(mu1 - mu0)
    return {'method': 'IPW','ATE': ate,'extra': {'ps': e.tolist()}}


def estimate_psm(df, x_cols):
    X = df[x_cols]
    T = df['treat'].values
    Y = df['re78'].values
    confounders = {c: ('num' if np.issubdtype(df[c].dtype, np.number) else 'cat') for c in x_cols}
    ps_model = build_ps_model(confounders)
    ps_model.fit(X, T)
    pscore = ps_model.predict_proba(X)[:, 1].reshape(-1, 1)
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return {'method': 'PSM (1-NN on propensity)', 'ATE': None, 'extra': {}}
    nn_control = NearestNeighbors(n_neighbors=1).fit(pscore[control_idx])
    _, idx = nn_control.kneighbors(pscore[treated_idx])
    matched_ctrl = control_idx[idx[:, 0]]
    att = np.mean(Y[treated_idx] - Y[matched_ctrl])
    nn_treated = NearestNeighbors(n_neighbors=1).fit(pscore[treated_idx])
    _, idx_c = nn_treated.kneighbors(pscore[control_idx])
    matched_trt = treated_idx[idx_c[:, 0]]
    atc = np.mean(Y[matched_trt] - Y[control_idx])
    ate = 0.5 * (att + atc)
    return {'method': 'PSM (1-NN on propensity)','ATE': float(ate),'extra': {}}


def estimate_dml(df, x_cols):
    try:
        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestRegressor
        X = df[x_cols].values
        T = df['treat'].values
        Y = df['re78'].values
        model_y = RandomForestRegressor(n_estimators=200, random_state=7)
        model_t = RandomForestRegressor(n_estimators=200, random_state=7)
        dml = LinearDML(model_y=model_y, model_t=model_t, featurizer=None, cv=3, random_state=7)
        dml.fit(Y, T, X=X)
        ate = float(np.mean(dml.effect(X)))
        return {'method': 'Double ML (econml)', 'ATE': ate, 'extra': {}}
    except Exception as e:
        return {'method': 'Double ML (econml)', 'ATE': None, 'extra': {'error': str(e)}}


def subgroup_effects(df, x_cols, estimator_fn, groupby_col, bins=None, min_n=30):
    results = []
    dfg = df.copy()
    if bins is not None and pd.api.types.is_numeric_dtype(dfg[groupby_col]):
        dfg[groupby_col] = pd.cut(dfg[groupby_col], bins=bins)
    for g, part in dfg.groupby(groupby_col):
        if part['treat'].nunique() < 2 or len(part) < min_n:
            continue
        try:
            est = estimator_fn(part, x_cols)
            if est['ATE'] is not None:
                results.append({'group': str(g), 'ATE': est['ATE']})
        except Exception:
            continue
    return pd.DataFrame(results)

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Causal Inference – Policy Evaluation (Job Training)", layout="wide")
st.title("Causal Inference for Policy Evaluation: Job Training → Earnings")
st.caption("Estimate treatment effects with IPW, PSM, and optional Double ML.")

df = load_data()
outcome_col = 're78'
treat_col = 'treat'
candidate_confounders = ['age','educ','black','hispan','married','re74','re75']
existing = [c for c in candidate_confounders if c in df.columns]

with st.sidebar:
    st.header("Controls")
    method = st.selectbox("Estimation method", ["IPW","PSM (1-NN on propensity)","Double ML (econml)"])
    cols = st.multiselect("Confounders (X)", options=existing, default=existing)
    show_subgroups = st.checkbox("Show subgroup effects (CATE-ish)", value=True)
    subgroup = st.selectbox("Subgroup variable", options=[c for c in existing if df[c].dtype != 'O'])
    nbins = st.slider("Bins (if numeric)", 3, 10, 4)

x_cols = cols if len(cols) > 0 else existing

if method == "IPW":
    main_res = estimate_ipw(df, x_cols)
elif method.startswith("PSM"):
    main_res = estimate_psm(df, x_cols)
else:
    main_res = estimate_dml(df, x_cols)

kpi = main_res['ATE']
boot_mean, ci_lo, ci_hi = bootstrap_ci(df, x_cols, estimate_ipw if method=="IPW" else (estimate_psm if method.startswith("PSM") else estimate_dml), B=300, alpha=0.05)
ci_text = f"${ci_lo:,.0f} … ${ci_hi:,.0f}" if ci_lo is not None else "N/A"

col1,col2,col3 = st.columns(3)
col1.metric("Estimated ATE (USD)", f"${kpi:,.0f}" if kpi is not None else "N/A")
col2.metric("95% CI", ci_text)
col3.metric("Treated (n)", int((df[treat_col]==1).sum()))

all_results = [estimate_ipw(df, x_cols), estimate_psm(df, x_cols), estimate_dml(df, x_cols)]
all_df = pd.DataFrame([{ "Method": r['method'], "ATE": r['ATE']} for r in all_results])
bar = px.bar(all_df, x="Method", y="ATE", title="Estimated Treatment Effect by Method (ATE)")
st.plotly_chart(bar, use_container_width=True)

if method == "IPW":
    confounders = {c: ('num' if np.issubdtype(df[c].dtype, np.number) else 'cat') for c in x_cols}
    ps_model = build_ps_model(confounders)
    ps_model.fit(df[x_cols], df['treat'])
    e = ps_model.predict_proba(df[x_cols])[:,1].clip(1e-3,1-1e-3)
    ps_df = pd.DataFrame({"Propensity": e, "treat": df[treat_col]})
    fig_ps = px.histogram(ps_df, x="Propensity", color="treat", nbins=30, barmode="overlay", title="Propensity Score Overlap (Common Support)")
    st.plotly_chart(fig_ps, use_container_width=True)

left,right = st.columns([0.6,0.4])
with left:
    fig = px.histogram(df, x=outcome_col, color=treat_col, nbins=40, barmode='overlay', title="Outcome Distribution (Earnings re78) – Treated vs Control")
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Conceptual Causal Graph (DAG)")
    st.graphviz_chart("""
        digraph G {
          rankdir=LR;
          X[label="Confounders (X)"];
          T[label="Treatment (Training)"];
          Y[label="Outcome (Earnings)"];
          X -> T; X -> Y; T -> Y;
        }
        """)

if show_subgroups and subgroup:
    if pd.api.types.is_numeric_dtype(df[subgroup]):
        bins = pd.interval_range(start=float(df[subgroup].min()), end=float(df[subgroup].max()), periods=nbins)
    else:
        bins = None
    est_fn = estimate_ipw if method=="IPW" else (estimate_psm if method.startswith("PSM") else estimate_dml)
    sub_df = subgroup_effects(df, x_cols, est_fn, subgroup, bins=bins, min_n=30)
    if not sub_df.empty:
        sub_chart = px.bar(sub_df, x='group', y='ATE', title=f"Subgroup Effects by {subgroup}")
        st.plotly_chart(sub_chart, use_container_width=True)
        st.dataframe(sub_df)
    else:
        st.info("Not enough overlap/sample size across subgroups. Try fewer bins or a different variable.")

st.markdown("---")
st.subheader("Interpretation / Notes")
st.markdown("""
- **ATE (Average Treatment Effect)** reflects the estimated change in earnings if everyone received the training versus nobody.
- Results differ by method due to assumptions (IPW: correct propensity; PSM: good matches; DML: flexible ML nuisance).
- Subgroup charts explore heterogeneous effects (CATE-style summaries).
- Dashboard is for **policy evaluation storytelling**: pair with a short write-up.
""")

with st.expander("Assumptions & Limitations", expanded=False):
    st.markdown("""
- **Unconfoundedness**: Treatment assignment random given X.
- **Overlap (Common Support)**: Each unit has non-zero prob. of treatment/control.
- **SUTVA**: No interference.
**Limitations**: Possible bias, small treated group, sensitivity to method.
""")

st.download_button("Download ATE results (CSV)", all_df.to_csv(index=False).encode(), file_name="treatment_effects.csv", mime="text/csv")

st.caption("Built with Streamlit + Plotly • Methods: IPW, PSM, optional DML • Dataset: Lalonde/NSW (or synthetic fallback)")
