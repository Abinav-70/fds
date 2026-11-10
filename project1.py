# %% [markdown]
# # Foundations of Data Science (23CSE351) - Course Project
# **Self-reported diabetes and health indicators: a statistical and logistic-regression
# analysis of the CDC BRFSS 2015 survey**
#
# Sections: 1 Preprocessing | 2 Visualization | 3 Hypothesis testing |
#           4 Logistic regression (train vs test) | 5 Predict function | 6 Results summary
# Course-project rubric: Preprocessing 3 | Visualization 3 | Results & presentation 4 | Viva 10
#
# Needs: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn.
# Put this file and diabetes_binary_health_indicators_BRFSS2015.csv in the same folder.
# Figures are written to a `figures/` sub-folder.

# %% 0. Setup
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CSV_PATH = BASE / "diabetes_binary_health_indicators_BRFSS2015.csv"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

TARGET = "Diabetes_binary"
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def save_fig(fig, name):
    """Save a figure as PNG into figures/ and show it."""
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"saved figures/{name}")


def fmt_p(p):
    """p-values this small underflow to 0.0, so print them as a bound."""
    return "< 1e-300" if p < 1e-300 else f"{p:.2e}"


# %% [markdown]
# # 1. Data preprocessing

# %% 1.1 Load, types, value ranges
# Syllabus: pandas DataFrames, data cleaning, categorical data in numeric form.
# All categorical columns already come as numeric codes (0/1 or ordinal 1..k), so no
# separate encoding step is needed; ordinal codes are used as numbers in the model.
df_raw = pd.read_csv(CSV_PATH)
print("Shape:", df_raw.shape)
print("\nDtypes as loaded:\n", df_raw.dtypes.value_counts())
types_before = ", ".join(f"{n} x {t}" for t, n in df_raw.dtypes.value_counts().items())

# Column groups (BRFSS codebook)
YES_NO = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
          "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
          "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"]
BINARY = YES_NO + ["Sex", TARGET]                      # 0/1 columns
ORDINAL = {"GenHlth": (1, 5), "Age": (1, 13), "Education": (1, 6), "Income": (1, 8)}
DAYS = ["MentHlth", "PhysHlth"]                        # days in last 30 -> bounded 0-30
FEATURES = [c for c in df_raw.columns if c != TARGET]

# Every column arrives as float64 although almost all are really integer codes.
# Check that the values are legal, then convert to int (BMI stays numeric/float).
assert (df_raw != df_raw.round()).sum().sum() == 0, "non-integer values found"
for c in BINARY:
    assert set(df_raw[c].unique()) <= {0.0, 1.0}, f"{c} is not 0/1"
for c, (lo, hi) in ORDINAL.items():
    assert df_raw[c].between(lo, hi).all(), f"{c} outside {lo}-{hi}"
for c in DAYS:
    assert df_raw[c].between(0, 30).all(), f"{c} outside 0-30"
int_cols = [c for c in df_raw.columns if c != "BMI"]
df_raw[int_cols] = df_raw[int_cols].astype(int)
print("\nAll value-range checks passed. Dtypes now:\n", df_raw.dtypes.value_counts())

print("\nDescriptive statistics (five-point summary + mean/std):")
print(df_raw.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2))

# %% 1.2 Missing values
# Syllabus: missing data treatment (isna / dropna / fillna).
n_missing = int(df_raw.isna().sum().sum())
print("Missing cells:", n_missing)
if n_missing:
    df_raw = df_raw.dropna()          # not triggered for this file
# Result: 0 missing. This is the pre-cleaned Kaggle/UCI version of BRFSS, in which
# 'refused / don't know' codes were already removed, so no fillna/dropna is needed.

# %% 1.3 Duplicates
# Syllabus: data cleaning.
n_dup = int(df_raw.duplicated().sum())
dup_rows = df_raw[df_raw.duplicated(keep=False)]
print(f"Duplicate rows (2nd+ copies): {n_dup:,} ({n_dup / len(df_raw):.1%})")
print(f"Diabetes rate in all rows that belong to a duplicate group: "
      f"{dup_rows[TARGET].mean():.1%}  (whole dataset: {df_raw[TARGET].mean():.1%})")
print(f"Diabetes rate if duplicates are dropped: "
      f"{df_raw.drop_duplicates()[TARGET].mean():.1%}")

# DECISION: KEEP the duplicates.
#  * There is no respondent ID. 21 of the 22 columns are 0/1 or small ordinal codes,
#    so two different people giving identical answers is expected, not an entry error.
#  * The duplicates are overwhelmingly non-diabetic ("typical healthy" answer patterns).
#    Dropping them removes real healthy respondents and inflates prevalence 13.9% -> 15.3%.
#  * Section 4.6 re-runs the model with duplicates removed to show how much that choice matters
#    (AUC changes by about 0.01; recall is unchanged).
df = df_raw.copy()

# %% 1.4 Outliers: IQR capping of BMI (and NOT of the 0-30 day columns)
# Syllabus: data cleaning, measures of spread (quartiles, IQR), skewness and kurtosis.
q1, q3 = df_raw["BMI"].quantile([0.25, 0.75])
iqr = q3 - q1
bmi_lo, bmi_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_lo, n_hi = int((df_raw.BMI < bmi_lo).sum()), int((df_raw.BMI > bmi_hi).sum())
print(f"BMI Q1={q1}, Q3={q3}, IQR={iqr}; fences = [{bmi_lo}, {bmi_hi}]")
print(f"BMI below lower fence: {n_lo:,} | above upper fence: {n_hi:,} "
      f"({(n_lo + n_hi) / len(df_raw):.1%} of rows)")
print(f"Raw BMI range: {df_raw.BMI.min():.0f}-{df_raw.BMI.max():.0f}; "
      f"skew={df_raw.BMI.skew():.2f}, excess kurtosis={df_raw.BMI.kurt():.2f}")

df["BMI"] = df_raw["BMI"].clip(lower=bmi_lo, upper=bmi_hi)       # capping (winsorising)
print(f"Capped BMI range: {df.BMI.min()}-{df.BMI.max()}; "
      f"skew={df.BMI.skew():.2f}, excess kurtosis={df.BMI.kurt():.2f}")

# Trade-off to be aware of: capping flattens genuinely severe obesity (BMI > 41.5)
hi_mask = df_raw.BMI > bmi_hi
print(f"Diabetes rate among rows with BMI > {bmi_hi}: {df_raw.loc[hi_mask, TARGET].mean():.1%} "
      f"vs {df_raw[TARGET].mean():.1%} overall -> these are real, high-risk people, "
      "so capping (not deleting) is the right choice.")

# MentHlth / PhysHlth: bounded 0-30 days, so they cannot contain impossible values.
# The IQR rule would still flag them, because most people report 0 days while a
# real group reports all 30 days (a U-shaped, not a bell-shaped, distribution).
print("\nWhy the day-count columns are NOT capped:")
for c in DAYS:
    cq1, cq3 = df_raw[c].quantile([0.25, 0.75])
    fence = cq3 + 1.5 * (cq3 - cq1)
    flagged = int((df_raw[c] > fence).sum())
    print(f"  {c}: IQR rule would flag {flagged:,} rows (> {fence:.1f} days); "
          f"{(df_raw[c] == 30).mean():.1%} of respondents report exactly 30 days; "
          f"{(df_raw[c] == 0).mean():.1%} report 0 days")

# %% 1.5 Class imbalance
counts = df[TARGET].value_counts().sort_index()
share = df[TARGET].value_counts(normalize=True).sort_index()
print(pd.DataFrame({"count": counts.values, "share": share.round(4).values},
                   index=["No diabetes (0)", "Diabetes (1)"]))
print(f"Imbalance ratio (majority:minority) = {counts[0] / counts[1]:.1f} : 1")
print(f"Always-predict-'no' accuracy = {share[0]:.4f}  <- accuracy alone is misleading here")

# %% 1.6 Data discretization + groupby / crosstab
# Syllabus: data discretization (pd.cut), grouping (groupby), crosstab.
# BMI is turned into the four standard WHO groups. Used for the plot 02b and as an example
# of discretization; the model itself keeps BMI as a number (no information thrown away).
bmi_group = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, np.inf], right=False,
                   labels=["Underweight", "Normal", "Overweight", "Obese"])
bmi_tbl = df.groupby(bmi_group, observed=True)[TARGET].agg(people="count", diabetes_rate="mean")
bmi_tbl["diabetes_rate"] = (bmi_tbl["diabetes_rate"] * 100).round(1)
print(bmi_tbl)
print("\nRow percentages (crosstab):")
print(pd.crosstab(bmi_group, df[TARGET], normalize="index").round(3))

# %% 1.7 Data transformation: z-score and min-max normalization (illustration)
# Syllabus: data transformation / normalization (min-max, z-score).
bmi_z = (df["BMI"] - df["BMI"].mean()) / df["BMI"].std()
bmi_mm = (df["BMI"] - df["BMI"].min()) / (df["BMI"].max() - df["BMI"].min())
print(pd.DataFrame({"raw": df_raw["BMI"], "capped": df["BMI"], "z-score": bmi_z, "min-max": bmi_mm})
      .describe().loc[["mean", "std", "min", "max"]].round(3))
# The MODEL uses z-scores (section 4.1), fitted on the training set only. z-score is chosen
# because it makes odds ratios comparable "per standard deviation"; min-max would depend on
# the two most extreme values.

# %% 1.8 Preprocessing summary (before vs after)
print(pd.DataFrame({
    "raw file": [f"{len(df_raw):,}", int(df_raw.isna().sum().sum()), f"{n_dup:,}",
                 types_before, f"{df_raw.BMI.min():.0f} to {df_raw.BMI.max():.0f}",
                 round(df_raw.BMI.skew(), 2), round(df_raw.BMI.kurt(), 2)],
    "after preprocessing": [f"{len(df):,}", int(df.isna().sum().sum()), f"{n_dup:,} (kept)",
                            "21 int, 1 float (BMI)", f"{df.BMI.min()} to {df.BMI.max()}",
                            round(df.BMI.skew(), 2), round(df.BMI.kurt(), 2)]},
    index=["rows", "missing cells", "duplicate rows", "column types", "BMI range",
           "BMI skewness", "BMI excess kurtosis"]).to_string())

# %% [markdown]
# # 2. Visualization (all figures use the BMI-capped data `df`)

# %% 2.1 Class balance (bar chart + pie chart)
overall = df[TARGET].mean() * 100          # overall diabetes rate in %, reused in later plots
# Exploration uses the full cleaned data (descriptive only, nothing is fitted here).
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
bars = axes[0].bar(["No diabetes", "Diabetes"], counts.values, color=["#4C72B0", "#C44E52"])
for b, n, sh in zip(bars, counts.values, share.values):
    axes[0].text(b.get_x() + b.get_width() / 2, n, f"{n:,}\n({sh:.1%})", ha="center", va="bottom")
axes[0].set_ylim(0, counts.max() * 1.15)
axes[0].set_ylabel("Respondents")
axes[0].set_title("Class balance (bar chart)")
axes[1].pie(counts.values, labels=["No diabetes", "Diabetes"], colors=["#4C72B0", "#C44E52"],
            autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white"})
axes[1].set_title("Class balance (pie chart)")
save_fig(fig, "01_class_balance.png")

# %% 2.2 BMI by outcome (box plot + overlaid histograms)
label = df[TARGET].map({0: "No diabetes", 1: "Diabetes"}).rename("Outcome")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(x=label, y=df["BMI"], hue=label, order=["No diabetes", "Diabetes"],
            palette=["#4C72B0", "#C44E52"], legend=False, ax=axes[0])
axes[0].set_title("BMI by outcome (box plot)")
sns.histplot(x=df["BMI"], hue=label, hue_order=["No diabetes", "Diabetes"],
             palette=["#4C72B0", "#C44E52"], stat="density", common_norm=False,
             bins=np.arange(12.5, 43.5, 1), alpha=0.45, ax=axes[1])   # one bar per whole BMI value
axes[1].set_ylim(0, 0.14)
axes[1].annotate("all BMI > 41.5,\ncapped at 41.5", xy=(42, 0.098), xytext=(37.5, 0.122),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2), fontsize=9, ha="center")
axes[1].set_title("BMI distribution by outcome (overlaid, each scaled to 1)")
axes[1].set_xlabel(f"BMI (capped to [{bmi_lo}, {bmi_hi}] by the IQR rule)")
save_fig(fig, "02_bmi_by_outcome.png")

# %% 2.2b Diabetes rate by BMI group (discretized BMI)
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(bmi_tbl.index.astype(str), bmi_tbl["diabetes_rate"], color="#CCB974")
for b, v, n in zip(bars, bmi_tbl["diabetes_rate"], bmi_tbl["people"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%\n(n={n:,})", ha="center", fontsize=8)
ax.axhline(overall, color="k", ls="--", lw=1, label=f"Overall {overall:.1f}%")
ax.set_ylim(0, bmi_tbl["diabetes_rate"].max() * 1.25)
ax.set_ylabel("% with diabetes")
ax.set_xlabel("BMI group (WHO cut-offs: 18.5, 25, 30)")
ax.set_title("Diabetes rate by BMI group")
ax.legend()
save_fig(fig, "02b_diabetes_rate_by_bmi_group.png")

# %% 2.3 Diabetes rate by age group
AGE_LABELS = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
              "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
rate_age = df.groupby("Age")[TARGET].mean() * 100
fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(AGE_LABELS, rate_age.values, color="#55A868")
for b, v in zip(bars, rate_age.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
ax.axhline(overall, color="k", ls="--", lw=1, label=f"Overall {overall:.1f}%")
ax.set_xlabel("Age group (years)")
ax.set_ylabel("% with diabetes")
ax.set_title("Diabetes rate by age group")
ax.legend()
save_fig(fig, "03_diabetes_rate_by_age.png")

# %% 2.4 Diabetes rate by income
INCOME_LABELS = ["<10k", "10-15k", "15-20k", "20-25k", "25-35k", "35-50k", "50-75k", "75k+"]
rate_inc = df.groupby("Income")[TARGET].mean() * 100
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(INCOME_LABELS, rate_inc.values, color="#8172B2")
for b, v in zip(bars, rate_inc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
ax.axhline(overall, color="k", ls="--", lw=1, label=f"Overall {overall:.1f}%")
ax.set_xlabel("Annual household income (US$, lowest to highest)")
ax.set_ylabel("% with diabetes")
ax.set_title("Diabetes rate by income group")
ax.legend()
save_fig(fig, "04_diabetes_rate_by_income.png")

# %% 2.5 Key yes/no risk factors
PRETTY = {"HighBP": "High blood pressure", "HighChol": "High cholesterol",
          "CholCheck": "Cholesterol checked (5 yrs)", "Smoker": "Smoked 100+ cigarettes",
          "Stroke": "Ever had stroke", "HeartDiseaseorAttack": "Heart disease / attack",
          "PhysActivity": "Physical activity (30 days)", "Fruits": "Eats fruit daily",
          "Veggies": "Eats vegetables daily", "HvyAlcoholConsump": "Heavy drinker",
          "AnyHealthcare": "Has health coverage", "NoDocbcCost": "Skipped doctor (cost)",
          "DiffWalk": "Difficulty walking"}
rf = pd.DataFrame({
    "yes": [df.loc[df[c] == 1, TARGET].mean() * 100 for c in YES_NO],
    "no": [df.loc[df[c] == 0, TARGET].mean() * 100 for c in YES_NO]}, index=YES_NO)
rf = rf.assign(gap=rf.yes - rf.no).sort_values("gap")         # biggest gap ends up on top
fig, ax = plt.subplots(figsize=(9, 7))
y = np.arange(len(rf))
ax.barh(y + 0.2, rf.yes, height=0.4, color="#C44E52", label="Yes")
ax.barh(y - 0.2, rf.no, height=0.4, color="#4C72B0", label="No")
ax.set_yticks(y)
ax.set_yticklabels([PRETTY[c] for c in rf.index])
ax.axvline(overall, color="k", ls="--", lw=1, label=f"Overall {overall:.1f}%")
ax.set_xlabel("% with diabetes")
ax.set_title("Diabetes rate by yes/no risk factor")
ax.legend(loc="lower right")
save_fig(fig, "05_risk_factors.png")

# %% 2.6 Correlation plot
corr = df.corr()
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool), k=1), annot=True,
            fmt=".2f", annot_kws={"size": 6.5}, cmap="coolwarm", center=0,
            vmin=-1, vmax=1, linewidths=0.3, ax=ax)
ax.set_title("Pearson correlation matrix (binary/ordinal columns: read as rough strength only)")
save_fig(fig, "06_correlation_heatmap.png")
fc = corr.drop(index=TARGET, columns=TARGET).abs()
fc = fc.where(~np.eye(len(fc), dtype=bool), 0)
a_, b_ = fc.stack().idxmax()
print(f"Data reduction check: the strongest feature-feature correlation is {fc.loc[a_, b_]:.2f} "
      f"({a_} vs {b_}); no pair is near 1, so no feature is redundant and all 21 are kept.")
print("Correlation of each column with the outcome:")
print(corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3).to_string())

# %% [markdown]
# # 3. Hypothesis testing
# alpha = 0.05 everywhere. With n = 253,680, almost any difference gives p ~ 0, so
# the *effect size* is what tells us whether a difference matters.

# %% 3.1 t-test on BMI: diabetic vs non-diabetic (Welch, two-tailed) + Cohen's d
# H0: mean BMI is the same in both groups.  H1: the means differ.
bmi_d = df.loc[df[TARGET] == 1, "BMI"]
bmi_n = df.loc[df[TARGET] == 0, "BMI"]
tt = stats.ttest_ind(bmi_d, bmi_n, equal_var=False)    # Welch: unequal n and variance
ci_lo, ci_hi = tt.confidence_interval(0.95)             # CI of (mean_diabetic - mean_not)
n1, n2 = len(bmi_d), len(bmi_n)
pooled_sd = np.sqrt(((n1 - 1) * bmi_d.var() + (n2 - 1) * bmi_n.var()) / (n1 + n2 - 2))
cohens_d = (bmi_d.mean() - bmi_n.mean()) / pooled_sd
d_label = ("negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5
           else "medium" if abs(cohens_d) < 0.8 else "large")
print(f"Mean BMI  diabetic = {bmi_d.mean():.2f} (n={n1:,}) | not diabetic = {bmi_n.mean():.2f} (n={n2:,})")
print(f"Difference = {bmi_d.mean() - bmi_n.mean():.2f} BMI points, 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
print(f"Welch t = {tt.statistic:.1f}, p = {fmt_p(tt.pvalue)}")
print(f"Cohen's d = {cohens_d:.2f} ({d_label}; rule of thumb 0.2 small, 0.5 medium, 0.8 large)")
raw_d = df_raw.loc[df_raw[TARGET] == 1, "BMI"]
raw_n = df_raw.loc[df_raw[TARGET] == 0, "BMI"]
print(f"(Check on uncapped BMI: difference = {raw_d.mean() - raw_n.mean():.2f}, "
      f"Welch p = {fmt_p(stats.ttest_ind(raw_d, raw_n, equal_var=False).pvalue)})")

# %% 3.2 Chi-square tests for the yes/no risk factors + Cramer's V
# H0: the factor and diabetes are independent.  H1: they are associated.
# correction=False so that V = sqrt(chi2 / n) is exactly the phi coefficient for a 2x2 table.
rows = []
for c in YES_NO:
    ct = pd.crosstab(df[c], df[TARGET])
    chi2, p, dof, expected = stats.chi2_contingency(ct, correction=False)
    assert expected.min() >= 5, "chi-square assumption (expected counts >= 5) violated"
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    rows.append({"factor": PRETTY[c],
                 "rate_if_yes_%": df.loc[df[c] == 1, TARGET].mean() * 100,
                 "rate_if_no_%": df.loc[df[c] == 0, TARGET].mean() * 100,
                 "chi2": chi2, "p": fmt_p(p), "p_raw": p, "cramers_V": v})
chi_tbl = pd.DataFrame(rows).sort_values("cramers_V", ascending=False).reset_index(drop=True)
chi_tbl["strength"] = pd.cut(chi_tbl.cramers_V, [0, 0.1, 0.3, 0.5, 1],
                             labels=["negligible", "small", "medium", "large"], right=False)
print(chi_tbl.drop(columns="p_raw").round(3).to_string())
alpha_bonf = 0.05 / len(YES_NO)
print(f"\n{(chi_tbl.p_raw < alpha_bonf).sum()} of {len(YES_NO)} tests are significant even after "
      f"Bonferroni correction (alpha = {alpha_bonf:.4f}).")
print("Cramer's V (df=1) rule of thumb: 0.1 small, 0.3 medium, 0.5 large. "
      "Significant does NOT mean strong: compare the p column with the V column.")

# %% [markdown]
# # 4. Logistic regression

# %% 4.1 Split, train-only BMI fences, train-only scaling (no leakage)
def prepare_split(data, seed=SEED):
    """Stratified 80/20 split, then everything is *fitted on the training part only*:
    BMI IQR fences and the StandardScaler. The test set is only transformed."""
    X, y = data[FEATURES].copy(), data[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    q1_, q3_ = X_tr["BMI"].quantile([0.25, 0.75])
    lo_, hi_ = q1_ - 1.5 * (q3_ - q1_), q3_ + 1.5 * (q3_ - q1_)
    X_tr = X_tr.assign(BMI=X_tr["BMI"].clip(lo_, hi_))
    X_te = X_te.assign(BMI=X_te["BMI"].clip(lo_, hi_))     # train fences applied to test
    scaler = StandardScaler().fit(X_tr)                    # fit on train ONLY
    to_df = lambda X_: pd.DataFrame(scaler.transform(X_), columns=FEATURES, index=X_.index)
    return {"X_tr": X_tr, "X_te": X_te, "Xs_tr": to_df(X_tr), "Xs_te": to_df(X_te),
            "y_tr": y_tr, "y_te": y_te, "scaler": scaler, "bmi_fences": (lo_, hi_)}


def fit_models(s):
    plain = LogisticRegression(max_iter=1000).fit(s["Xs_tr"], s["y_tr"])
    balanced = LogisticRegression(max_iter=1000, class_weight="balanced").fit(s["Xs_tr"], s["y_tr"])
    return plain, balanced


def score(y_true, pred, proba):
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred, zero_division=0),
            "AUC": roc_auc_score(y_true, proba)}


split = prepare_split(df_raw)
X_tr, X_te, Xs_tr, Xs_te = split["X_tr"], split["X_te"], split["Xs_tr"], split["Xs_te"]
y_tr, y_te, scaler = split["y_tr"], split["y_te"], split["scaler"]
print(f"Train: {len(y_tr):,} rows, {y_tr.mean():.4f} diabetic | "
      f"Test: {len(y_te):,} rows, {y_te.mean():.4f} diabetic  (stratified -> same share)")
print(f"BMI fences from TRAIN only: {split['bmi_fences']} (whole-data fences were ({bmi_lo}, {bmi_hi}))")
print("Scaler fitted on train only: train means ~0 ->", np.allclose(Xs_tr.mean(), 0, atol=1e-8),
      "| test means not exactly 0 ->", not np.allclose(Xs_te.mean(), 0, atol=1e-8))

# %% 4.1b Split check: is the test set representative of the training set?
split_tbl = pd.DataFrame({
    "rows": [len(y_tr), len(y_te)],
    "diabetic": [int(y_tr.sum()), int(y_te.sum())],
    "diabetic_share": [y_tr.mean(), y_te.mean()],
    "mean_BMI": [X_tr.BMI.mean(), X_te.BMI.mean()],
    "mean_Age_code": [X_tr.Age.mean(), X_te.Age.mean()]}, index=["Train (80%)", "Test (20%)"])
print(split_tbl.round(4).to_string())
smd = ((X_te.mean() - X_tr.mean()) / X_tr.std()).abs()
print(f"Largest standardised mean difference (test vs train) over all 21 features: "
      f"{smd.max():.3f} ({smd.idxmax()}); values below 0.1 mean the two sets look alike.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
bars = axes[0].bar(["Train (80%)", "Test (20%)"], [len(y_tr), len(y_te)], color=["#8FA8C8", "#C44E52"])
for b, n, sh in zip(bars, [len(y_tr), len(y_te)], [y_tr.mean(), y_te.mean()]):
    axes[0].text(b.get_x() + b.get_width() / 2, n, f"{n:,} rows\n{sh:.2%} diabetic", ha="center", va="bottom")
axes[0].set_ylim(0, len(y_tr) * 1.18)
axes[0].set_ylabel("Respondents")
axes[0].set_title("Stratified 80/20 split: same diabetes share in both")
bins_ = np.arange(12.5, 43.5, 1)
axes[1].hist(X_tr.BMI, bins=bins_, density=True, alpha=0.55, color="#8FA8C8", label="Train")
axes[1].hist(X_te.BMI, bins=bins_, density=True, alpha=0.55, color="#C44E52", label="Test")
axes[1].set_xlabel("BMI (capped with the training-set fences)")
axes[1].set_ylabel("Density")
axes[1].set_title("BMI distribution: train vs test")
axes[1].legend()
save_fig(fig, "07_train_test_split_check.png")

# %% 4.2 Fit plain and class_weight="balanced" models; predict on TRAIN and TEST
# Models are fitted on the training set only. Predicting on the training set is a diagnostic
# (over-fitting check); the TEST numbers are the ones to report.
plain, balanced = fit_models(split)
MODELS = {"Plain": plain, "Balanced": balanced}
SETS = {"Train": (Xs_tr, y_tr), "Test": (Xs_te, y_te)}
proba = {(m, s): MODELS[m].predict_proba(X_)[:, 1] for m in MODELS for s, (X_, _) in SETS.items()}
pred = {(m, s): MODELS[m].predict(X_) for m in MODELS for s, (X_, _) in SETS.items()}   # threshold 0.5
proba_plain, proba_bal = proba[("Plain", "Test")], proba[("Balanced", "Test")]          # test shortcuts
pred_plain, pred_bal = pred[("Plain", "Test")], pred[("Balanced", "Test")]

# %% 4.3 Metrics: train vs test, against the always-"no" baseline
rows = {}
for s, (X_, y_) in SETS.items():
    rows[("Always 'no' (baseline)", s)] = score(y_, np.zeros(len(y_), dtype=int), np.zeros(len(y_)))
for m in MODELS:
    for s, (X_, y_) in SETS.items():
        rows[(m, s)] = score(y_, pred[(m, s)], proba[(m, s)])
results = pd.DataFrame(rows).T
results.index.names = ["model", "set"]
print(results.round(4).to_string())
print("\nHeadline numbers are the TEST rows. Baseline precision/F1 are 0 because it never says 'yes'.")
gap = pd.concat({m: results.loc[m].loc["Train"] - results.loc[m].loc["Test"] for m in MODELS}, axis=1).T
print("\nTrain minus test (near 0 = no sign of over-fitting):")
print(gap.round(4).to_string())

cms = {(m, s): confusion_matrix(y_, pred[(m, s)]) for m in MODELS for s, (_, y_) in SETS.items()}
for m in MODELS:
    tn, fp, fn, tp = cms[(m, "Test")].ravel()
    print(f"\nConfusion matrix, {m} model, TEST set [rows = actual, cols = predicted]")
    print(pd.DataFrame(cms[(m, "Test")], index=["actual no", "actual yes"], columns=["pred no", "pred yes"]))
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

# Why only the PLAIN model may be used for probabilities:
print(f"\nMean predicted probability on the test set: plain = {proba_plain.mean():.3f}, "
      f"balanced = {proba_bal.mean():.3f}, true share = {y_te.mean():.3f}")
print("-> balanced weights act as if diabetes were 50% of the data, so its 'probabilities' are inflated.")

# %% 4.4 Train-vs-test plots: confusion matrices, ROC curves, metric comparison
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for i, m in enumerate(MODELS):
    for j, s in enumerate(SETS):
        cm = cms[(m, s)]
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([[f"{cm[a, b]:,}\n({pct[a, b]:.1f}% of row)" for b in range(2)] for a in range(2)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=axes[i, j],
                    xticklabels=["Pred no", "Pred yes"], yticklabels=["Actual no", "Actual yes"])
        axes[i, j].set_title(f"{m} model, {s} set (threshold 0.5)")
fig.tight_layout()
save_fig(fig, "08_confusion_matrices.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, m in zip(axes, MODELS):
    for s, ls, col in [("Train", "--", "#8FA8C8"), ("Test", "-", "#C44E52")]:
        y_ = SETS[s][1]
        fpr, tpr, _ = roc_curve(y_, proba[(m, s)])
        ax.plot(fpr, tpr, ls=ls, lw=2, color=col,
                label=f"{s} (AUC = {roc_auc_score(y_, proba[(m, s)]):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(f"ROC curve, {m} model")
    ax.legend(loc="lower right")
fig.tight_layout()
save_fig(fig, "09_roc_curves.png")

METRICS = ["accuracy", "precision", "recall", "F1", "AUC"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
x = np.arange(len(METRICS))
for ax, m in zip(axes, MODELS):
    for k, (s, col) in enumerate([("Train", "#8FA8C8"), ("Test", "#C44E52")]):
        vals = results.loc[(m, s), METRICS].values.astype(float)
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, width=0.38, color=col, label=s)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    base = results.loc[("Always 'no' (baseline)", "Test"), "accuracy"]
    ax.axhline(base, color="k", ls="--", lw=1, label=f"Always-'no' accuracy {base:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{m} model: train vs test")
    ax.legend(loc="upper right", fontsize=8)
axes[0].set_ylabel("Score")
fig.tight_layout()
save_fig(fig, "10_train_vs_test_metrics.png")

# %% 4.5 Odds ratios (plain model)
# Features were z-scored, so exp(coef) = multiplicative change in the ODDS of diabetes
# for a +1 standard-deviation increase in that feature, other features held fixed.
# For a 0/1 variable, "1 SD" is only ~0.4-0.5 of the yes/no gap, so the second column
# (per 1 raw unit = yes vs no for binary; one step for ordinal; one BMI point) is easier to read.
coefs = pd.Series(plain.coef_[0], index=FEATURES)
or_tbl = pd.DataFrame({"OR_per_SD": np.exp(coefs),
                       "OR_per_1_raw_unit": np.exp(coefs / scaler.scale_),
                       "SD_of_feature": scaler.scale_}, index=FEATURES)
or_tbl = or_tbl.reindex(coefs.abs().sort_values(ascending=False).index)
print(or_tbl.round(3).to_string())
print(f"\nIntercept (log-odds at the mean of every feature): {plain.intercept_[0]:.3f}")

fig, ax = plt.subplots(figsize=(8.5, 7))
o = or_tbl.iloc[::-1]
ax.barh(o.index, o.OR_per_SD - 1, left=1,
        color=np.where(o.OR_per_SD > 1, "#C44E52", "#4C72B0"))
ax.set_xscale("log")
ax.set_xlim(0.75, 2.1)
ax.set_xticks([0.8, 1, 1.25, 1.5, 2])
ax.set_xticklabels(["0.8", "1", "1.25", "1.5", "2"])
ax.minorticks_off()
ax.axvline(1, color="k", lw=1)
for i, v in enumerate(o.OR_per_SD):
    ax.text(v * (1.02 if v > 1 else 0.98), i, f"{v:.2f}", va="center",
            ha="left" if v > 1 else "right", fontsize=8)
ax.set_xlabel("Odds ratio per +1 SD (log scale; 1 = no effect)")
ax.set_title("Plain logistic regression: odds ratios per standard deviation")
save_fig(fig, "11_odds_ratios.png")

# %% 4.6 Sensitivity check: what if the duplicate rows had been dropped?
twins = X_te.merge(X_tr.drop_duplicates(), how="left", on=FEATURES, indicator=True)
print(f"Share of test rows whose exact answer pattern also appears in the training set: "
      f"{(twins['_merge'] == 'both').mean():.1%}")

s_dd = prepare_split(df_raw.drop_duplicates())     # de-duplicate BEFORE splitting
plain_dd, bal_dd = fit_models(s_dd)
sens = pd.DataFrame({
    "Plain, all rows (kept)": score(y_te, pred_plain, proba_plain),
    "Plain, duplicates dropped": score(s_dd["y_te"], plain_dd.predict(s_dd["Xs_te"]),
                                       plain_dd.predict_proba(s_dd["Xs_te"])[:, 1]),
    "Balanced, all rows (kept)": score(y_te, pred_bal, proba_bal),
    "Balanced, duplicates dropped": score(s_dd["y_te"], bal_dd.predict(s_dd["Xs_te"]),
                                          bal_dd.predict_proba(s_dd["Xs_te"])[:, 1])}).T
print(sens.round(4).to_string())
print("Note: the two test sets differ (different rows, different prevalence), so compare AUC and "
      "recall only roughly.")

# %% [markdown]
# # 5. Predict function for one new person

# %% 5.1 Predict function
# Codes used by the survey (all inputs are the dataset's own codes):
#   Age 1..13 = 18-24, 25-29, ..., 75-79, 80+       GenHlth 1 excellent .. 5 poor
#   Education 1..6 (1 = never/kindergarten ... 6 = college graduate)
#   Income 1..8 (1 = < $10k ... 8 = >= $75k)         Sex 0 female, 1 male
#   yes/no columns: 1 = yes, 0 = no                  MentHlth/PhysHlth: bad days in last 30 (0-30)
VALID_RANGE = {c: (df_raw[c].min(), df_raw[c].max()) for c in FEATURES}   # observed ranges


def age_group(years):
    """Convert age in years to the survey's 13-level Age code."""
    if years < 18:
        raise ValueError("The survey covers adults (18+) only.")
    return 13 if years >= 80 else (years - 25) // 5 + 2 if years >= 25 else 1


def predict_diabetes_risk(person):
    """person: dict with all 21 feature names (see codes above).
    Returns the PLAIN model's probability (the only trustworthy probability) and the
    BALANCED model's yes/no screening flag (high recall, many false alarms)."""
    missing = [c for c in FEATURES if c not in person]
    if missing:
        raise ValueError(f"missing inputs: {missing}")
    for c in FEATURES:
        lo_, hi_ = VALID_RANGE[c]
        if not lo_ <= person[c] <= hi_:
            raise ValueError(f"{c}={person[c]} outside the range seen in the data ({lo_}-{hi_})")
        if c != "BMI" and person[c] != round(person[c]):
            raise ValueError(f"{c} must be a whole-number code")
    x = pd.DataFrame([person])[FEATURES].astype(float)   # fixes column order
    x["BMI"] = x["BMI"].clip(*split["bmi_fences"])        # same train-only fences
    x_s = scaler.transform(x)                              # same train-only scaler
    prob = float(plain.predict_proba(x_s)[0, 1])
    flag = bool(balanced.predict(x_s)[0])
    base = float(y_tr.mean())
    return {"probability_plain_model": round(prob, 3),
            "x_times_average_risk": round(prob / base, 2),
            "screening_flag_balanced_model": "refer for a blood-sugar test" if flag else "no flag",
            "note": f"average risk in this sample is {base:.1%}; educational model, not a diagnosis"}


# %% 5.2 Examples
person_a = dict(HighBP=0, HighChol=0, CholCheck=1, BMI=23, Smoker=0, Stroke=0,
                HeartDiseaseorAttack=0, PhysActivity=1, Fruits=1, Veggies=1,
                HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0, GenHlth=1,
                MentHlth=0, PhysHlth=0, DiffWalk=0, Sex=0, Age=age_group(28),
                Education=6, Income=8)
person_b = dict(HighBP=1, HighChol=1, CholCheck=1, BMI=34, Smoker=1, Stroke=0,
                HeartDiseaseorAttack=1, PhysActivity=0, Fruits=0, Veggies=1,
                HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0, GenHlth=4,
                MentHlth=5, PhysHlth=15, DiffWalk=1, Sex=1, Age=age_group(63),
                Education=4, Income=3)
for name, p_ in [("Person A (28 y, healthy)", person_a), ("Person B (63 y, several risk factors)", person_b)]:
    print(name, "->", predict_diabetes_risk(p_))

# %% 6. Results summary and conclusion
test_tbl = results.xs("Test", level="set")
base_acc = test_tbl.loc["Always 'no' (baseline)", "accuracy"]
top_v = chi_tbl.iloc[0]
print("RESULTS SUMMARY")
print(f"Data: {len(df_raw):,} adults, {df_raw[TARGET].mean():.1%} diabetic; stratified split "
      f"{len(y_tr):,} train / {len(y_te):,} test.")
print(f"BMI: {bmi_d.mean():.2f} (diabetic) vs {bmi_n.mean():.2f} (not); Cohen's d = {cohens_d:.2f}.")
print(f"Yes/no factors: strongest is {top_v.factor} (Cramer's V = {top_v.cramers_V:.2f}); "
      f"{int((chi_tbl.cramers_V >= 0.1).sum())} of {len(chi_tbl)} have V >= 0.1, although all p-values are tiny.")
print("\nTEST-set performance:")
print(test_tbl.round(3).to_string())
print(f"\nTrain-vs-test AUC gap: plain {gap.loc['Plain', 'AUC']:+.4f}, balanced {gap.loc['Balanced', 'AUC']:+.4f}")
print("\nCONCLUSION")
print(f"- Diabetes is most strongly linked to {', '.join(or_tbl.index[:3])} "
      "(largest odds ratios per standard deviation); these are associations, not causes.")
print(f"- The model ranks people well (test AUC {test_tbl.loc['Plain', 'AUC']:.2f}) and does not "
      "over-fit: train and test scores are almost identical.")
print(f"- Plain accuracy ({test_tbl.loc['Plain', 'accuracy']:.3f}) barely beats always saying 'no' "
      f"({base_acc:.3f}) and finds only "
      f"{test_tbl.loc['Plain', 'recall']:.0%} of diabetic people. The balanced model finds "
      f"{test_tbl.loc['Balanced', 'recall']:.0%} but only {test_tbl.loc['Balanced', 'precision']:.0%} of its "
      "'yes' calls are correct. Choose by the cost of a missed case vs a false alarm.")
print("- Limits: self-reported 2015 US survey data, cross-sectional, no survey weights.")

# %% 7. Files written
print("\nFigures written to:", FIG_DIR)
for f in sorted(FIG_DIR.glob("*.png")):
    print("  ", f.name)