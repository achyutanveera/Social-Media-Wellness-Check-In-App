"""Shared preprocessing for training and serving.

Mirrors the Psyber Mentalytics notebook (Social_Media_Mental_Health_Code.ipynb):
  - impact_sum = sum of the 12 Likert items (Q9-Q20)
  - risk label = thresholds on impact_sum (>=44 higher, >=32.31 medium, else lower)
  - model features: age, depressed, restless, compare_to_others, impact_sum,
    platform_sum, plus one-hot of gender, rs, avg_time_per_day, age_group
Training and the API both go through these functions so the columns always match.
"""
import re
import pandas as pd

ITEMS = ["wo_purpose", "distracted", "restless", "distracted_scale", "worry_scale",
         "concentration", "compare_to_others", "compare_feelings", "validation",
         "depressed", "daily_activity_flux", "sleeping_issues"]

LIKERT = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Very Often": 5}

# Platforms counted in platform_sum (the notebook's nine; "Other" is not counted)
PLATFORMS = ["Facebook", "Instagram", "TikTok", "Snapchat", "YouTube",
             "Twitter", "Pinterest", "Discord", "Reddit"]
PLATFORM_ALIASES = {"Twitter/X": "Twitter"}

# Notebook's labels after the rename in cell 30
TIME_LABELS = ["<1 hrs", "1-2 hrs", "2-3 hrs", "3-4 hrs", "4-5 hrs", ">5 hrs"]
KAGGLE_TIME = {"Less than an Hour": "<1 hrs", "Between 1 and 2 hours": "1-2 hrs",
               "Between 2 and 3 hours": "2-3 hrs", "Between 3 and 4 hours": "3-4 hrs",
               "Between 4 and 5 hours": "4-5 hrs", "More than 5 hours": ">5 hrs"}
APP_TIME = {"Less than 1 hour": "<1 hrs", "1-2 hours": "1-2 hrs", "2-3 hours": "2-3 hrs",
            "3-4 hours": "3-4 hrs", "4-5 hours": "4-5 hrs", "More than 5 hours": ">5 hrs"}

GENDERS = ["Female", "Male", "Other"]
RS = ["Divorced", "In a relationship", "Married", "Single"]
AGE_GROUPS = ["<=20", "21-30", "31-40", "41-50", ">50"]

HIGH_CUT = 44
MED_CUT = 32.31

NUMBER_TO_KEY = {1: "age", 2: "gender", 3: "rs", 4: "occupation", 7: "sm_type",
                 8: "avg_time_per_day", 9: "wo_purpose", 10: "distracted", 11: "restless",
                 12: "distracted_scale", 13: "worry_scale", 14: "concentration",
                 15: "compare_to_others", 16: "compare_feelings", 17: "validation",
                 18: "depressed", 19: "daily_activity_flux", 20: "sleeping_issues"}


def risk_from_impact(s):
    if s >= HIGH_CUT:
        return "higher"
    if s >= MED_CUT:
        return "medium"
    return "lower"


def age_group(age):
    if age <= 20:
        return "<=20"
    if age <= 30:
        return "21-30"
    if age <= 40:
        return "31-40"
    if age <= 50:
        return "41-50"
    return ">50"


def platform_count(names):
    found = set()
    for n in names:
        n = PLATFORM_ALIASES.get(n.strip(), n.strip())
        if n in PLATFORMS:
            found.add(n)
    return len(found)


def from_kaggle_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw smmh.csv -> tidy frame with the 18 app keys plus platform_sum."""
    rename = {}
    for col in raw.columns:
        m = re.match(r"\s*(\d+)\.", str(col))
        if m and int(m.group(1)) in NUMBER_TO_KEY:
            rename[col] = NUMBER_TO_KEY[int(m.group(1))]
    df = raw.rename(columns=rename)[list(NUMBER_TO_KEY.values())].copy()
    df["gender"] = df["gender"].apply(lambda x: x if x in ("Male", "Female") else "Other")
    df["avg_time_per_day"] = df["avg_time_per_day"].map(KAGGLE_TIME)
    df["platform_sum"] = df["sm_type"].fillna("").apply(lambda s: platform_count(s.split(",")))
    df["age"] = df["age"].astype(float).astype(int)
    df = df.dropna(subset=["avg_time_per_day", "rs"] + ITEMS).reset_index(drop=True)
    for c in ITEMS:
        df[c] = df[c].astype(int)
    return df


def from_app(r: dict) -> pd.DataFrame:
    """One check-in as saved by the app (string answers) -> one-row tidy frame."""
    row = {
        "age": int(r["age"]),
        "gender": r["gender"],
        "rs": r["rs"],
        "avg_time_per_day": APP_TIME[r["avg_time_per_day"]],
        "platform_sum": platform_count(r.get("sm_type", [])),
    }
    for k in ITEMS:
        row[k] = LIKERT[r[k]]
    return pd.DataFrame([row])


def build_features(df: pd.DataFrame, with_impact: bool = True) -> pd.DataFrame:
    """Tidy frame -> model matrix. Fixed category lists give fixed columns."""
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"]
    out["depressed"] = df["depressed"]
    out["restless"] = df["restless"]
    out["compare_to_others"] = df["compare_to_others"]
    if with_impact:
        out["impact_sum"] = df[ITEMS].sum(axis=1)
    out["platform_sum"] = df["platform_sum"]
    ag = df["age"].apply(age_group)
    for prefix, series, cats in (("gender", df["gender"], GENDERS), ("rs", df["rs"], RS),
                                 ("avg_time_per_day", df["avg_time_per_day"], TIME_LABELS),
                                 ("age_group", ag, AGE_GROUPS)):
        for c in cats:
            out[f"{prefix}_{c}"] = (series == c).astype(int)
    return out
