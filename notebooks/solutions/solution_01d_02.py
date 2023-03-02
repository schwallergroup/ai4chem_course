mol = Chem.MolFromSmiles(smiles)#create a mol object from input smiles 

canonical_smiles = Chem.MolToSmiles(mol)#convert the previous mol object to SMILES using Chem.MolToSmiles()
