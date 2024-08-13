import gensim





def getPhenotypeDrugList():
    filePath = '../data/phenotypeData/drug/drug_list'
    drugList = open(filePath).readlines()
    drugList = [drug.strip() for drug in drugList]

    # drugModelFilePath = '../model/phenotypeFeature/sideEffectEmbeddingModel'
    # drugModel = gensim.models.Word2Vec.load(drugModelFilePath)
    # drugModel = drugModel.wv
    #
    # resultList = []
    # for i in drugList:
    #     for j in drugModel.key_to_index.keys():
    #         if 'CIDm' in j and i == j:
    #             resultList.append(i)

    return drugList

def getPhenotypeProteinList():
    filePath = '../data/phenotypeData/target/GO_list'
    proteinList = open(filePath).readlines()
    proteinList = [protein.strip() for protein in proteinList]

    # proteinModelFilePath = '../model/phenotypeFeature/GOEmbeddingModel'
    # proteinModel = gensim.models.Word2Vec.load(proteinModelFilePath)
    # proteinModel = proteinModel.wv
    #
    # resultList = []
    # for i in proteinList:
    #     for j in proteinModel.key_to_index.keys():
    #         if '9606.ENSP' in j and i == j:
    #             resultList.append(i)

    return proteinList



if __name__ == '__main__':
    # a = getPhenotypeDrugList()
    a = getPhenotypeProteinList()
    for i in a:
        print(i)



















