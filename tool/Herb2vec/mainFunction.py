from generateVector import *
from generateGraph import *


if __name__ == '__main__':

    # 可选择drug   GO    MP   uberon
    # association_file= '../../data/phenotypeData/target/GO_association_file'
    association_file= '../../data/phenotypeData/drug/drug_association_file'

    axiom_file = '../../data/mateData/Herb2vec/axiom/axiomsorig.lst'

    embedding_size= 200 # TODO 单词生成的向量大小为200

    # 可选择drug   GO    MP   uberon
    # outfile = '../../model/phenotypeFeature/GOEmbeddingModel'
    outfile = '../../model/phenotypeFeature/sideEffectEmbeddingModel'
    # 可选择drug   GO    MP   uberon
    # entity_list = '../../data/entityData/target/GO_list'
    entity_list = '../../data/phenotypeData/drug/drug_list'

    num_workers = 12

    file_pre = '../../process/Herb2vecCache/'



    print('99999999999')
    G = generate_graph(association_file,axiom_file)
    print('33333333333333333------------------------------------------333333333333333')
    # TODO 单词生成的向量大小为200
    getNodeVector(G,inputFline=entity_list,outputFile=outfile, embedding_size=embedding_size, file_pre=file_pre, num_workers=num_workers)
    print('44444444444444444-------------------------------------------44444444444444444')
