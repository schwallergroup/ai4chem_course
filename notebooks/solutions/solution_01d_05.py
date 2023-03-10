###First part

#create pattern
phenyl = Chem.MolFromSmiles('c1ccccc1')

#apply HasSubstructureMatch to each molecule (we use a lambda function here)
df['phenyl'] = df['Molecule'].apply(lambda x: x.HasSubstructMatch(phenyl)) #use phenyl object for the query


###Second part
ring = Chem.MolFromSmarts('[r]') #look for the SMARTS specification corresponding to any ring

df['ring'] = df['Molecule'].apply(lambda x: x.HasSubstructMatch(ring)) #proceed as in the previous case
