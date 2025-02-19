toluene = Chem.MolFromSmiles('c1ccccc1C') #insert toluene SMILES

#Now, create the fingerprints of theobromine and toluene
toluene_fp = mfpgen.GetFingerprint(toluene) #insert corresponding values
theobromine_fp =  mfpgen.GetFingerprint(theobromine) #same for theobromine

#Now we calculate Tanimoto Similarity
sim1 = FingerprintSimilarity(caffeine_fp, toluene_fp) #insert fingerprints to compare
sim2 = FingerprintSimilarity(caffeine_fp, theobromine_fp) #same than before