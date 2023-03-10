toluene = Chem.MolFromSmiles('c1ccccc1C') #insert toluene SMILES

#Now, create the fingerprints of theobromine and toluene
toluene_fp = AllChem.GetMorganFingerprintAsBitVect(toluene, 2, nBits=1024) #insert corresponding values
theobromine_fp =  AllChem.GetMorganFingerprintAsBitVect(theobromine, 2, nBits=1024) #same for theobromine

#Now we calculate Tanimoto Similarity
sim1 = FingerprintSimilarity(caffeine_fp, toluene_fp) #insert fingerprints to compare
sim2 = FingerprintSimilarity(caffeine_fp, theobromine_fp) #same than before