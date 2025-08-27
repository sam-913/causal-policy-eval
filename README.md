# 📊 Causal Inference for Policy Evaluation: Job Training → Earnings

An interactive **Streamlit dashboard** that applies causal inference methods to evaluate the impact of a U.S. job training program on participants’ earnings (Lalonde/NSW dataset).

This project demonstrates how causal inference can move beyond correlation to provide **policy-relevant, interpretable insights**.

---

## 🚀 Features
- **Treatment Effect Estimation**
  - Inverse Propensity Weighting (IPW)
  - Propensity Score Matching (PSM, 1-NN)
  - Double Machine Learning (DML, optional)
- **Interpretability**
  - Average Treatment Effect (ATE) with bootstrap 95% CI
  - Outcome distribution plots (treated vs. control)
  - Propensity score overlap diagnostics
  - Subgroup (CATE-ish) analysis
- **Transparency**
  - Assumptions & Limitations panel
  - Downloadable CSV outputs for reproducibility

---

## 🛠️ Tech Stack
- **Python** (pandas, numpy, scikit-learn, statsmodels)
- **Causal Inference Libraries** (`causaldata`, optional `econml`)
- **Visualization** (Plotly, Graphviz)
- **Dashboard** (Streamlit)

---

## 📂 Dataset
- **Source:** Lalonde/NSW job training dataset (`causaldata` package).
- **Treatment (`treat`)**: Participation in training (1 = yes, 0 = no).
- **Outcome (`re78`)**: Earnings in 1978.
- **Confounders**: Age, education, prior earnings (`re74`, `re75`), race, marital status.

If the dataset is unavailable, the app falls back to a synthetic dataset so it always runs.

---

## ⚡ Quickstart

Clone the repo and run locally:
```bash
git clone https://github.com/sam-913/causal-policy-eval.git
cd causal-policy-eval
python -m venv .venv && source .venv/bin/activate   # or use conda


pip install -r requirements.txt
streamlit run app.py
