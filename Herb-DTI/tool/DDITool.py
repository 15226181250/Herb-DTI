





def getDrugToSMILESDict(inputFileName):

    # inputFileName = '../data/structureData/drug/drug_smiles'
    print('Fetching drug to SMILES mapping ...')
    drug_to_smiles_dict = {}
    f = open(file=inputFileName, mode='r').readlines()
    for line in f:
        # CIDm00007501	C=CC1=CC=CC=C1
        drug_id, drug_smiles_enc = line.split('\t')
        drug_smiles_enc = drug_smiles_enc.strip()
        drug_to_smiles_dict[drug_id] = drug_smiles_enc.strip()
    return drug_to_smiles_dict






if __name__ == '__main__':
    print(getDrugToSMILESDict())























