# Credit Risk Cascade — v2 (trained on the new dataset)

Same cascade design as before, retrained on your new `historical_dataset.csv`
(no code changes needed — same 49-column schema):

```
applicant data --> Model 1 (default probability)
                      |
                      v
                 Model 2 (loan grade A-E)   [uses Model 1's output as a feature]
                      |
                      v
                 Model 3 (interest rate %)  [uses Model 1 + Model 2's outputs]
```

## What's inside

```
project/
├── train.py              # trains all 3 models, saves them to artifacts/
├── artifacts/             # trained models + metrics.json (already built)
├── app/main.py            # FastAPI service (/predict, /health)
├── requirements.txt
├── Dockerfile
└── data/                  # your two CSVs (not shipped in the Docker image)
```

## Results — big improvement over the first dataset

| Model | Metric | v1 (old data) | **v2 (new data)** |
|---|---|---|---|
| Model 1 — Default Probability | ROC-AUC | 0.60 | **0.71** |
| Model 1 — Default Probability | PR-AUC | 0.23 | **0.36** |
| Model 2 — Loan Grade (A-E) | Accuracy | 0.31 | **0.80** |
| Model 2 — Loan Grade (A-E) | Macro-F1 | 0.31 | **0.81** |
| Model 3 — Interest Rate | MAE | 3.5 pp | **0.95 pp** |
| Model 3 — Interest Rate | R² | 0.28 | **0.90** |

Out-of-time on `current_dataset.csv` (real drift baked in: wage inflation,
+0.75pp rate pass-through, higher gig-worker share): Model 1 AUC 0.68,
Model 2 accuracy 0.78, Model 3 MAE 1.08pp — a modest, expected drop vs. the
historical validation split, not a red flag.

**Why v2 is so much stronger:** I checked the new data dictionary — the
targets are now generated from an "OBSERVABLE composite" (i.e., built from
columns that are actually in your dataset) with less random noise mixed in,
instead of a hidden latent-risk variable with heavy unexplainable noise.
Concretely, feature correlations with `loan_grade` roughly doubled (e.g.
`loan_stacking_count_90d` went from 0.31 to 0.52, `credit_utilization` from
0.29 to 0.49). That's what's driving the jump in accuracy — this dataset is
just much more learnable, not a modeling trick.

A quick sanity check on the deployed pipeline: a strong synthetic applicant
(salaried, ₹45k/month, CIBIL 720, no delinquency) now gets a clean, decisive
prediction — Grade A with 97% confidence, 9.96% interest rate — instead of
the muddled, near-random probabilities the old dataset produced for the same
profile.

## Run it locally

```bash
pip install -r requirements.txt

# (re)train — only needed if you change the data or the feature list
python train.py

# start the API
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive test UI.

## Deploying it — easiest first

**1. Railway (recommended)** — push to GitHub (artifacts/*.joblib included,
they're a few MB), then railway.app → New Project → Deploy from GitHub repo.
It auto-detects the `Dockerfile` and builds. Public URL in ~2 minutes.

**2. ngrok (fastest, no real deployment)** — run the API locally, then
`ngrok http 8000` for an instant public tunnel. Good for quick demos; dies
when your laptop sleeps.

**3. Render** — same idea as Railway, choose "Docker" environment. Free tier
has a ~30s cold start after inactivity.

## Known limitations / next steps

- **Model 1's probabilities still aren't calibrated** (`class_weight="balanced"`
  trades calibration for minority-class recall). Treat as a relative risk
  score; wrap in `sklearn.calibration.CalibratedClassifierCV` to fix.
- **Still using `HistGradientBoostingClassifier/Regressor`** (built into
  scikit-learn) instead of XGBoost/LightGBM/CatBoost, because this sandbox
  has no internet access to install them. Same algorithm family; swapping
  in the real libraries later is a small change in `train.py` once you're
  on a machine with internet.
- No hyperparameter tuning done yet — sane defaults only. With this much
  cleaner signal, a tuning pass (grid/random search over `max_iter`,
  `max_depth`, `learning_rate`) should push Model 1's AUC and Model 2's
  accuracy up further — that's now a good next step since the ceiling
  clearly isn't the data anymore.
- The API derives `loan_to_income_ratio` and `disposable_income` internally
  if you don't supply them, so your frontend form doesn't need to compute
  those.
