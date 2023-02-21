toluene = Chem.MolFromSmiles('c1ccccc1C') #insert toluene SMILES

#Now, create the fingerprints of theobromine and toluene
toluene_fp = AllChem.GetMorganFingerprintAsBitVect(toluene) #insert corresponding values
theobromine_fp =  AllChem.GetMorganFingerprintAsBitVect(theobromine) #same for theobromine

#Now we calculate Tanimoto Similarity
sim1 = FingerprintSimilarity(caffeine, toluene) #insert fingerprints to compare
sim2 = FingerprintSimilarity(caffeine, theobromine) #same than before
