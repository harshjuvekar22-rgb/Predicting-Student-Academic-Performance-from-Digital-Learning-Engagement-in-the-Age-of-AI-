# 🎓 Predicting Student Academic Performance from Digital Learning Engagement in the Age of AI

Regression and deep-learning models that predict a student's **final score (0–100)** from LMS (VLE)
engagement, assessment behaviour, background and AI-use variables — with an interactive Streamlit app
(prediction, "at-risk" flag, what-if analysis, and model diagnostics).

> ⚠️ **The bundled dataset is synthetic.** It is OULAD-*style* (same kinds of features that can be derived from the
> Open University Learning Analytics Dataset) plus three AI-use columns that OULAD lacks. Findings describe the
> simulation, not real students. Use `prepare_oulad.py` to build a **real** table from OULAD.

## Results on the synthetic data (20 % hold-out, n = 1,600)

| Model | MAE | RMSE | R² | 5-fold CV R² | At-risk precision / recall (<55) |
|---|---|---|---|---|---|
| Linear Regression | 5.579 | 7.077 | 0.789 | 0.799 ± 0.008 | 0.80 / 0.70 |
| Ridge Regression | 5.579 | 7.077 | 0.789 | 0.799 ± 0.008 | 0.80 / 0.70 |
| Random Forest | 6.196 | 7.795 | 0.744 | 0.752 ± 0.012 | 0.79 / 0.66 |
| Gradient Boosting | 5.728 | 7.230 | 0.780 | 0.790 ± 0.008 | 0.82 / 0.70 |
| Deep Neural Network (128-64-32) | 5.761 | 7.260 | 0.778 | 0.789 ± 0.008 | 0.81 / 0.70 |

The simulated relationships are mostly additive, so linear regression is already near-optimal and the deep
network matches it rather than beating it. On real, messier data results will differ — run `train.py --data`.

## Structure
```
├── app.py                    # Streamlit app (adapts to whatever columns the dataset has)
├── edu_model.py              # preprocessing, models, what-if helpers
├── train.py                  # trains/evaluates, writes models/ and reports/
├── prepare_oulad.py          # builds a real-data table from the OULAD CSVs (optional --cutoff for early warning)
├── train_keras.py            # OPTIONAL TensorFlow version (not tested in authoring sandbox)
├── data/generate_dataset.py  # synthetic generator
├── data/student_engagement_ai.csv
├── models/  reports/  requirements.txt  .streamlit/config.toml
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                     # models + data are already included
# optional: python data/generate_dataset.py && python train.py
```

## Use the REAL OULAD data
1. Download `anonymisedData.zip` from https://analyse.kmi.open.ac.uk/open-dataset (CC BY 4.0 — cite Kuzilek et al., 2017).
2. Unzip into `data/oulad/` (not committed to git — see `.gitignore`).
3. `python prepare_oulad.py` → `data/oulad_features.csv`   (add `--cutoff 28` for an early-warning model)
4. `python train.py --data data/oulad_features.csv`
5. Commit the new `models/`, `reports/` and `data/oulad_features.csv`, push, and Streamlit redeploys.

## Deploy: GitHub → Streamlit Community Cloud
```bash
git init && git add . && git commit -m "Student engagement & AI performance predictor"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
Then go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, file `app.py` → **Deploy**.
If the host's scikit-learn version can't load `models/models.joblib`, the app retrains automatically on first load.

## Ethics
Predictions are associations, not causes. Do not use them for grading, admissions or other high-stakes decisions.
Audit for subgroup fairness before any real deployment, and keep a human in the loop.
