import pandas as pd
import kagglehub
import os

# Download Dataset
path = kagglehub.dataset_download("hemishveeraboina/aime-problem-set-1983-2024")

# Pick the correct CSV file (adjust after checking file list)
csv_path = os.path.join(path, "AIME_Dataset_1983_2024.csv")   # replace with real filename
df = pd.read_csv(csv_path)

# Train split: 2013–2018
train_df = df[df["Year"].between(2020,2022)]

# Val split: 2019–2021
val_df = df[df["Year"]== 2023]

# Test split: 2022–2024
test_df = df[df["Year"] == 2024]

# Print number of problems
total_samples = len(train_df)+len(val_df)+len(test_df)
print("Total samples", total_samples)
print("Train sample portion:", len(train_df)*100/total_samples)
print("Validation portion:", len(val_df)*100/total_samples)
print("Test portion:", (len(test_df)*100/total_samples))

# (Optional) Save them
train_df.to_csv("AIME/AIME_train.csv", index=False)
val_df.to_csv("AIME/AIME_val.csv", index=False)
test_df.to_csv("AIME/AIME_test.csv", index=False)