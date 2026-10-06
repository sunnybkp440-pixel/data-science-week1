import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer
sns.set_theme(style="whitegrid")

# 1. Data acquisition
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
df = df.drop(columns="target")
print("Shape:", df.shape)
print(df.dtypes.value_counts())
print("Missing total:", int(df.isnull().sum().sum()))
print("Duplicates:", int(df.duplicated().sum()))
print(df["diagnosis"].value_counts())

# 2. Cleaning
raw_rows = len(df)
df = df.drop_duplicates().reset_index(drop=True)
df.columns = [c.replace(" ", "_") for c in df.columns]
num = df.select_dtypes("number").columns
# simulate-check: any impossible values (<=0 for size features)
print("Non-positive values:", int((df[num] <= 0).sum().sum()))
# outliers by IQR
q1, q3 = df[num].quantile(.25), df[num].quantile(.75)
iqr = q3 - q1
mask = (df[num] < q1 - 1.5*iqr) | (df[num] > q3 + 1.5*iqr)
out_counts = mask.sum().sort_values(ascending=False)
print("Rows with >=1 outlier:", int(mask.any(axis=1).sum()))
print(out_counts.head(5))
# winsorise (cap) instead of dropping
clean = df.copy()
clean[num] = clean[num].clip(q1 - 1.5*iqr, q3 + 1.5*iqr, axis=1)
print("Rows before/after:", raw_rows, len(clean))

# 3. EDA
print(clean[["mean_radius","mean_area","mean_texture","mean_concavity"]].describe().round(2))
print(clean.groupby("diagnosis")[["mean_radius","mean_area","mean_concavity"]].mean().round(2))
corr = clean[num].corr()
tgt = clean.assign(y=(clean.diagnosis=="Malignant").astype(int))[list(num)+["y"]].corr()["y"].drop("y").sort_values()
print(tgt.head(5).round(2)); print(tgt.tail(5).round(2))
hi = corr.where(np.triu(np.ones(corr.shape),1).astype(bool)).stack().sort_values(ascending=False)
print(hi.head(3).round(3))

pal = {"Malignant":"#d62728","Benign":"#1f77b4"}
plt.figure(figsize=(6,4)); sns.countplot(x="diagnosis", data=clean, palette=pal, hue="diagnosis", legend=False)
plt.title("Class distribution"); plt.tight_layout(); plt.savefig("figs/1_class.png", dpi=150); plt.close()

plt.figure(figsize=(6,4)); sns.histplot(data=clean, x="mean_radius", hue="diagnosis", kde=True, palette=pal)
plt.title("Mean radius by diagnosis"); plt.tight_layout(); plt.savefig("figs/2_hist.png", dpi=150); plt.close()

# before/after outliers boxplot
fig, ax = plt.subplots(1,2, figsize=(9,4), sharey=True)
sns.boxplot(y=df["area_error"], ax=ax[0], color="#ff9896"); ax[0].set_title("area_error (before)")
sns.boxplot(y=clean["area_error"], ax=ax[1], color="#98df8a"); ax[1].set_title("area_error (after capping)")
plt.tight_layout(); plt.savefig("figs/3_box.png", dpi=150); plt.close()

plt.figure(figsize=(6,4.5)); sns.scatterplot(data=clean, x="mean_radius", y="mean_concavity", hue="diagnosis", palette=pal, alpha=.7)
plt.title("Mean radius vs mean concavity"); plt.tight_layout(); plt.savefig("figs/4_scatter.png", dpi=150); plt.close()

cols = ["mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness","mean_compactness","mean_concavity","mean_concave_points"]
plt.figure(figsize=(8,6.5)); sns.heatmap(clean[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size":7})
plt.title("Correlation (mean features)"); plt.tight_layout(); plt.savefig("figs/5_heat.png", dpi=150); plt.close()
