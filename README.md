# Causal Inference for Policy Evaluation: Job Training → Earnings

Interactive Streamlit dashboard estimating the impact of a job training program on earnings using the Lalonde/NSW dataset.

## Methods
- IPW (Inverse Propensity Weighting)
- PSM (Propensity Score Matching, 1-NN)
- Double ML (optional, econml)

## Run locally
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

## Deploy (Streamlit Cloud)
Push repo → New app → select your repo/branch → Deploy.  
If build fails on econml, remove/comment it from requirements.
