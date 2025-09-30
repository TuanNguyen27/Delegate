# math500_split.py

from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split
import os

SEED = 42
TRAIN_PCT, VAL_PCT, TEST_PCT = 0.70, 0.15, 0.15

# 1) Load as pandas for stratification
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
df = ds.to_pandas()

# 2) Train vs (val+test), stratify by subject
df_train, df_valtest = train_test_split(
    df, test_size=(1 - TRAIN_PCT), random_state=SEED, stratify=df["subject"]
)

# 3) Val vs Test, stratify again by subject
df_val, df_test = train_test_split(
    df_valtest,
    test_size=TEST_PCT / (VAL_PCT + TEST_PCT),
    random_state=SEED,
    stratify=df_valtest["subject"]
)

# Print counts
print(f"Total samples: {len(df)}")
print(f"Train: {len(df_train)}")
print(f"Validation: {len(df_val)}")
print(f"Test: {len(df_test)}")

# 4) Back to 🤗 datasets (optional) + save
train_ds = Dataset.from_pandas(df_train, preserve_index=False)
val_ds   = Dataset.from_pandas(df_val,   preserve_index=False)
test_ds  = Dataset.from_pandas(df_test,  preserve_index=False)

splits = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
print(splits)

# Save to CSV
os.makedirs("math500", exist_ok=True)
df_train.to_csv("math500/train.csv", index=False)
df_val.to_csv("math500/validation.csv", index=False)
df_test.to_csv("math500/test.csv", index=False)
print("Saved to math500/{train,validation,test}.csv")
