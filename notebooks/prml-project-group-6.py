# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv (3.14.3.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pattern Recognition and Machine Learning
#
# ## Project
#
# ### Comparative Study of kNN and Weighted kNN Classification on the Palmer Penguins Dataset

# %% [markdown]
# #### Completed by: **Group 6**
#
# | Index | Full Name |
# | ----- | ----- |
# | 66m/25 | Ivona Petrović |
# | 144m/25 | Vojin Đorđević |

# %% [markdown]
# **Imports and data loading.** We use `pandas` for loading and manipulating the tabular dataset, `os` for file system operations, and `numpy` for saving the final feature/label arrays. `load_penguins_data` fetches the Palmer Penguins dataset from a public CSV source and we save an untouched raw copy to `../data/raw/penguins_.csv` before any cleaning.

# %%
import pandas as pd
import os
import numpy as np


# %%
def load_penguins_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    df = pd.read_csv(url)
    
    print("Raw shape:", df.shape)
    
    return df

df = load_penguins_data()
df.to_csv("../data/raw/penguins_.csv", index=False)

# %% [markdown]
# **Inspecting the raw data.** We preview the first rows to check column names and types, count missing values per column, and look at the rows/indices where values are missing, to decide how to handle them.

# %%
df.head()

# %%
missing_counts = df.isnull().sum().to_frame(name="missing_count")
missing_counts

# %%
rows_with_missing = df[df.isnull().any(axis=1)]
rows_with_missing

# %%
missing_indices = df[df.isnull().any(axis=1)].index
missing_indices 

# %% [markdown]
# **Cleaning the data.** Since missing entries are a small fraction of the dataset, we drop them with `dropna()` rather than imputing, producing `df_clean`. Shapes before/after are printed to quantify how much data was lost.

# %%
df_clean = df.dropna()

print("Before:", df.shape)
print("After:", df_clean.shape)

# %% [markdown]
# **Feature/target preparation and saving.** We select the four numeric measurements (bill length, bill depth, flipper length, body mass) as features `X`, and species as the target. Species names are mapped to integers $\{0,1,2\}$ to produce `y_encoded`, while `df_eda` is kept as an independent copy for later exploratory analysis. Finally, the cleaned dataset and feature/label arrays are saved to disk for use in subsequent notebook stages.

# %%
X = df_clean[[ "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
y = df_clean["species"]

# %%
label_map = {
    "Adelie": 0,
    "Chinstrap": 1,
    "Gentoo": 2
}

y_encoded = y.map(label_map)

df_eda = df_clean.copy()  # separate dataset for EDA (kept independent from modeling data)

# %%
os.makedirs("../data", exist_ok=True)

df_clean.to_csv("../data/penguins_clean.csv", index=False)
np.save("../data/X.npy", X.values)
np.save("../data/y.npy", y_encoded.values)
