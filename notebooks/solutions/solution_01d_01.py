theobromine = Chem.MolFromSmiles('CN1C=NC2=C1C(=O)NC(=O)N2C') # insert SMILES
xanthine = Chem.MolFromSmiles('C1=NC2=C(N1)C(=O)NC(=O)N2') #insert SMILES


mols = [caffeine, theobromine, xanthine] #create a list containing the 3 mol objects we have created
names = ['caffeine', 'theobromine', 'xanthine'] #create a list containing the names of the 3 molecules

#Now we create the GridImage
grid = Draw.MolsToGridImage(mols, legends=names) #pass the 'mols' list here and create the image
