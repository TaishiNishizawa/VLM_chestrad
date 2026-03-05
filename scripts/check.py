import pandas as pd

df = pd.read_csv("/gs/gsfs0/shared-lab/duong-lab/MIMIC/mimic-cxr-2.0.0-split.csv")
print(df["split"].value_counts(dropna=False))

# if you have subject_id / study_id columns:
print("unique subjects:", df.groupby("split")["subject_id"].nunique())
print("unique studies:", df.groupby("split")["study_id"].nunique())