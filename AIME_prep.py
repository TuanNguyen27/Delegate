import pandas as pd
import kagglehub
import os

# Download Dataset
path = kagglehub.dataset_download("hemishveeraboina/aime-problem-set-1983-2024")


# Pick the correct CSV file (adjust after checking file list)
csv_path = os.path.join(path, "AIME_Dataset_1983_2024.csv")   # replace with real filename
df = pd.read_csv(csv_path)

# Filter rows 2013 - 2018
train_df = df[df["Year"].between(2013, 2018)]

# Val split: 2019–2021 (subsample 30 if you want exactly 12.5%)
val_df = df[df["Year"].between(2019, 2021)].sample(n=30, random_state=42)

# Test split: 2022–2024 (subsample 30 for balance)
test_df = df[df["Year"].between(2022, 2024)].sample(n=30, random_state=42)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

# Save if needed
train_df.to_csv("AIME/AIME_train.csv", index=False)
val_df.to_csv("AIME/AIME_val.csv", index=False)
test_df.to_csv("AIME/AIME_test.csv", index=False)
