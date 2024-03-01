# Accessing a specific column
solubility = df["measured log solubility in mols per litre"]
print(solubility)

# Calculating statistics on a column
mean_solubility = solubility.mean()
print(mean_solubility)
