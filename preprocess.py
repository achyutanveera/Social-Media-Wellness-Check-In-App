#Mirrors the Psyber Mentalytics notebook (Social_Media_Mental_Health_Code.ipynb):
#Impact_sum: sum of the responses to the questions
#Risk label: (>=44 higher, >32 medium, else lower)
#Model features: age, depressed, restless, compare_to_others, impact_sum,
#Platform_sum, plus one-hot encoding of gender, rs, avg_time_per_day, age_group
#Training and API scripts both go through these functions so the columns always match
import re
import pandas as pd
items = ["wo_purpose", "distracted", "restless", "distracted_scale", "worry_scale",
         "concentration", "compare_to_others", "compare_feelings", "validation",
         "depressed", "daily_activity_flux", "sleeping_issues"]
likert = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Very Often": 5}
#Platforms counted in platform_sum (the notebook's nine, "Other" is not counted)
platforms = ["Facebook", "Instagram", "TikTok", "Snapchat", "YouTube",
             "Twitter", "Pinterest", "Discord", "Reddit"]
platform_aliases = {"Twitter/X": "Twitter"}

#Notebook's labels after the rename in cell 30 of original code
time_labels = ["<1 hrs", "1-2 hrs", "2-3 hrs", "3-4 hrs", "4-5 hrs", ">5 hrs"]
kaggle_time = {"Less than an Hour": "<1 hrs", "Between 1 and 2 hours": "1-2 hrs",
               "Between 2 and 3 hours": "2-3 hrs", "Between 3 and 4 hours": "3-4 hrs",
               "Between 4 and 5 hours": "4-5 hrs", "More than 5 hours": ">5 hrs"}
app_time = {"Less than 1 hour": "<1 hrs", "1-2 hours": "1-2 hrs", "2-3 hours": "2-3 hrs",
            "3-4 hours": "3-4 hrs", "4-5 hours": "4-5 hrs", "More than 5 hours": ">5 hrs"}
genders = ["Female", "Male", "Other"]
rs = ["Divorced", "In a relationship", "Married", "Single"]
age_groups = ["<=20", "21-30", "31-40", "41-50", ">50"]
high_cut = 44
med_cut = 32
number_to_key = {1: "age", 2: "gender", 3: "rs", 4: "occupation", 7: "sm_type",
                 8: "avg_time_per_day", 9: "wo_purpose", 10: "distracted", 11: "restless",
                 12: "distracted_scale", 13: "worry_scale", 14: "concentration",
                 15: "compare_to_others", 16: "compare_feelings", 17: "validation",
                 18: "depressed", 19: "daily_activity_flux", 20: "sleeping_issues"}
def risk_from_impact(s):
    if s >= high_cut:
        return "higher"
    if s > med_cut:
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
        n = platform_aliases.get(n.strip(), n.strip())
        if n in platforms:
            found.add(n)
    return len(found)
def from_kaggle_raw(raw: pd.DataFrame) -> pd.DataFrame:
    #Raw smmh.csv -> tidy frame with the 18 app keys plus platform_sum
    rename = {}
    for col in raw.columns:
        m = re.match(r"\s*(\d+)\.", str(col))
        if m and int(m.group(1)) in number_to_key:
            rename[col] = number_to_key[int(m.group(1))]
    df = raw.rename(columns=rename)[list(number_to_key.values())].copy()
    df["gender"] = df["gender"].apply(lambda x: x if x in ("Male", "Female") else "Other")
    df["avg_time_per_day"] = df["avg_time_per_day"].map(kaggle_time)
    df["platform_sum"] = df["sm_type"].fillna("").apply(lambda s: platform_count(s.split(",")))
    df["age"] = df["age"].astype(float).astype(int)
    df = df.dropna(subset=["avg_time_per_day", "rs"] + items).reset_index(drop=True)
    for c in items:
        df[c] = df[c].astype(int)
    return df
def from_app(r: dict) -> pd.DataFrame:
    #One check-in as saved by the app (string answers) -> one-row tidy frame
    row = {
        "age": int(r["age"]),
        "gender": r["gender"],
        "rs": r["rs"],
        "avg_time_per_day": app_time[r["avg_time_per_day"]],
        "platform_sum": platform_count(r.get("sm_type", [])),
    }
    for k in items:
        row[k] = likert[r[k]]
    return pd.DataFrame([row])
def build_features(df: pd.DataFrame, with_impact: bool = True) -> pd.DataFrame:
    #Tidy frame -> model matrix. Fixed category lists give fixed columns
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"]
    out["depressed"] = df["depressed"]
    out["restless"] = df["restless"]
    out["compare_to_others"] = df["compare_to_others"]
    if with_impact:
        out["impact_sum"] = df[items].sum(axis=1)
    out["platform_sum"] = df["platform_sum"]
    ag = df["age"].apply(age_group)
    for prefix, series, cats in (("gender", df["gender"], genders), ("rs", df["rs"], rs),
                                 ("avg_time_per_day", df["avg_time_per_day"], time_labels),
                                 ("age_group", ag, age_groups)):
        for c in cats:
            out[f"{prefix}_{c}"] = (series == c).astype(int)
    return out
