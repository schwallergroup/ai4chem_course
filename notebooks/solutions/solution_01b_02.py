import pandas as pd

# Reading the ESOL dataset into a DataFrame
df = pd.read_csv("data/delaney-processed.csv")

# Inspecting the first 5 rows of the DataFrame
print(df.head())
