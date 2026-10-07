"""
Build a student-level feature table from the REAL Open University Learning Analytics Dataset (OULAD).

1. Download "anonymisedData.zip" from https://analyse.kmi.open.ac.uk/open-dataset  (CC BY 4.0 licence;
   please cite Kuzilek, Hlosta & Zdrahal, 2017, Scientific Data 4:170171).
2. Unzip the CSVs into  data/oulad/   (studentInfo, studentVle, vle, studentAssessment, assessments, courses).
3. Run:   python prepare_oulad.py                      -> data/oulad_features.csv
          python prepare_oulad.py --cutoff 28          -> early-warning table using only the first 28 days
4. Train: python train.py --data data/oulad_features.csv

Target `final_score` = weight-averaged score of the student's submitted TMA/CMA assessments (0-100).
OULAD has NO AI-use variables, so those columns are simply absent; the app adapts automatically.
Demographics such as gender, region, disability and IMD band are deliberately left out (fairness).
NOTE: this script follows the documented OULAD schema; check the output on your copy of the data.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["code_module", "code_presentation", "id_student"]
TYPES = {"forumng": "clicks_forum", "resource": "clicks_resource", "quiz": "clicks_quiz",
         "oucontent": "clicks_content", "homepage": "clicks_homepage"}


def main(src, out, cutoff):
    src = Path(src)
    info = pd.read_csv(src / "studentInfo.csv")
    courses = pd.read_csv(src / "courses.csv")
    vle = pd.read_csv(src / "vle.csv")[["id_site", "code_module", "code_presentation", "activity_type"]]
    svle = pd.read_csv(src / "studentVle.csv")
    assess = pd.read_csv(src / "assessments.csv")
    sa = pd.read_csv(src / "studentAssessment.csv")

    # ---------------- target + assessment behaviour (TMA/CMA only; exams are missing for some courses)
    assess = assess[assess["assessment_type"].isin(["TMA", "CMA"])]
    sa = sa.merge(assess, on="id_assessment", how="inner")
    sa = sa[(sa["is_banked"] == 0)] if "is_banked" in sa else sa
    sa = sa.dropna(subset=["score"])
    sa["weight"] = sa["weight"].replace(0, np.nan).fillna(1.0)
    sa["w_score"] = sa["score"] * sa["weight"]
    sa["days_early"] = sa["date"] - sa["date_submitted"]

    # id_student only identifies a student within a module-presentation, so group on all KEYS
    g = sa.groupby(KEYS)
    tgt = pd.DataFrame({
        "final_score": g["w_score"].sum() / g["weight"].sum(),
        "submission_timeliness_days": g["days_early"].mean(),
        "n_submitted": g.size(),
    }).reset_index()
    n_assess = assess.groupby(["code_module", "code_presentation"]).size().rename("n_assess").reset_index()
    tgt = tgt.merge(n_assess, on=["code_module", "code_presentation"], how="left")
    tgt["assessments_submitted_pct"] = (100 * tgt["n_submitted"] / tgt["n_assess"]).clip(upper=100)
    tgt["submission_timeliness_days"] = tgt["submission_timeliness_days"].clip(-30, 30)
    tgt = tgt.drop(columns=["n_submitted", "n_assess"])

    # ---------------- VLE engagement
    if cutoff is not None:
        svle = svle[svle["date"] <= cutoff]
    svle = svle.merge(vle, on=["id_site", "code_module", "code_presentation"], how="left")
    svle["col"] = svle["activity_type"].map(TYPES).fillna("clicks_other")
    by_type = (svle.pivot_table(index=KEYS, columns="col", values="sum_click", aggfunc="sum", fill_value=0)
               .reindex(columns=list(TYPES.values()) + ["clicks_other"], fill_value=0).reset_index())
    daily = svle.groupby(KEYS).agg(active_days=("date", "nunique")).reset_index()
    svle["week"] = np.floor(svle["date"] / 7)
    weeks = svle.groupby(KEYS).agg(active_weeks=("week", "nunique")).reset_index()
    early = (svle[svle["date"] < 28].groupby(KEYS)["sum_click"].sum().rename("clicks_first_4_weeks").reset_index())

    df = (info[KEYS + ["highest_education", "age_band", "num_of_prev_attempts", "studied_credits"]]
          .merge(by_type, on=KEYS, how="left").merge(daily, on=KEYS, how="left")
          .merge(weeks, on=KEYS, how="left").merge(early, on=KEYS, how="left")
          .merge(courses, on=["code_module", "code_presentation"], how="left")
          .merge(tgt, on=KEYS, how="inner"))                     # keep only students with a score

    num_fill = list(TYPES.values()) + ["clicks_other", "active_days", "active_weeks", "clicks_first_4_weeks"]
    df[num_fill] = df[num_fill].fillna(0)
    total_weeks = (df["module_presentation_length"] / 7).clip(lower=1)
    df["active_weeks_pct"] = (100 * df["active_weeks"] / total_weeks).clip(upper=100).round(1)
    df = df.rename(columns={"code_module": "course_module"})

    if cutoff is not None:      # early-warning mode: drop everything that would be unknown at the cutoff
        df = df.drop(columns=["assessments_submitted_pct", "submission_timeliness_days"])
        df["active_weeks_pct"] = (100 * df["active_weeks"] / max(cutoff / 7, 1)).clip(upper=100).round(1)

    keep = ["course_module", "highest_education", "age_band", "num_of_prev_attempts", "studied_credits",
            "active_days", "active_weeks_pct", "clicks_content", "clicks_resource", "clicks_forum",
            "clicks_quiz", "clicks_homepage", "clicks_other", "clicks_first_4_weeks",
            "assessments_submitted_pct", "submission_timeliness_days", "final_score"]
    df = df[[c for c in keep if c in df.columns]]
    for c in df.columns:
        if c.startswith("clicks_") or c in ("active_days",):
            df[c] = df[c].astype(int)
    df = df.dropna()
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/oulad")
    ap.add_argument("--out", default="data/oulad_features.csv")
    ap.add_argument("--cutoff", type=int, default=None, help="use only VLE activity up to this day (early warning)")
    a = ap.parse_args()
    main(a.src, a.out, a.cutoff)
