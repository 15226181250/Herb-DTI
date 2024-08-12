## 模型数据加载类
import numpy as np
from math import sqrt, log2
from scipy import stats
import gensim
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import sys
import pickle
import DTITool
import phenotypeDataTool
import PPITool
# 提供DTI网络数据(模型训练集 或 测试集)
class DTINetworkData:
    def __init__(self, config):
        self.config = config
        print("Loading data ...")
        # 获取药物列表
        if config.herbTest == 'true':
            # TODO 待完善
            pass
            print('!!!!!!!!!!!!!!!!!!!!')
            # self.drug_list = np.array(PhenomeNET_DL2vec_utils.get_PhenomeNET_drug_list())
        else:
            self.drugList = np.array(DTITool.getDrugList(config.mode))
        print(len(self.drugList), "drugs present")
        # get protein lists for each ontology
        # 获取基因本体论的蛋白质列表
        self.GOProteinList = phenotypeDataTool.getPhenotypeProteinList()
        # 获取DTI网络
        self.DTIGraph = DTITool.getDTIGraph(mode=config.mode)
        # 获取PPI网络
        self.PPIGraph = PPITool.getPPIGraph(minScore=config.PPIMinScore)
        # 获取蛋白质列表 (PPI网络节点 交 DTI网络节点 交 基因本体论蛋白质列表)
        self.proteinList = np.array(list(set(self.PPIGraph.nodes()) & set(self.DTIGraph.nodes()) & set(self.GOProteinList)))
        if config.includeMolecularFeatures == 'true':
            print("Building molecular features")
            # 构建药物分子特征
            if config.drugMode == 'trfm' or config.drugMode == 'rnn':
                drugFilePath = '../model/structureFeature/' + config.drugMode + 'DrugMolecularMapping.pkl'
                with open(file=drugFilePath, mode='rb') as f:
                    # 药物分子嵌入
                    self.drugMoleculeEncodings = pickle.load(f)
            else:
                print("No valid mode selected for drug to SMILES encoding.")
                raise ValueError
            # 构建蛋白质分子特征
            proteinFilePath = '../model/structureFeature/proteinSequenceMapping.pkl'
            with open(file=proteinFilePath, mode='rb') as f:
                self.proteinMoleculeEncodings = pickle.load(f)
            # 更新蛋白质列表
            self.proteinList = np.array(list(set(self.proteinList) & set(self.proteinMoleculeEncodings.keys())))
            # 蛋白质分子嵌入
            self.proteinMoleculeEncodings = torch.Tensor([self.proteinMoleculeEncodings[protein] for protein in self.proteinList])
            print(len(self.proteinList), "proteins present with mol_pred_intersection.\n")
        # PPI data
        print("Loading PPI graph ...")
        # subgraph()返回在“节点”上诱导的子图的子图视图
        # 图的诱导子图包含“节点”中的节点以及这些节点之间的边。
        self.PPIGraph = self.PPIGraph.subgraph(self.proteinList) # 根据蛋白质列表构建子图视图
        # calculate dimensions of network 计算网络尺寸
        # DTI网络蛋白质数量
        self.numProteins = len(self.proteinList)
        # DTI网络药物数量
        self.numDrugs = len(self.drugList)
        config.numDrugs = self.numDrugs
        config.numProteins = self.numProteins
    def buildData(self, config):
        # 打印PPI网络的 节点数 和 边数
        print('PPI_graph nodes/edges:', len(self.PPIGraph.nodes()), len(self.PPIGraph.edges()))
        print("Building index dict ...")
        # 构建 蛋白质和索引 对应的字典
        self.proteinToIndexDictionary = {protein: index for index, protein in enumerate(self.proteinList)}
        print("Building edge list ...")
        # 构建 PPI网络 边的列表
        forward_edges_list = [(self.proteinToIndexDictionary[node1], self.proteinToIndexDictionary[node2]) for node1, node2 in list(self.PPIGraph.edges())]
        backward_edges_list = [(self.proteinToIndexDictionary[node1], self.proteinToIndexDictionary[node2]) for node2, node1 in list(self.PPIGraph.edges())]
        # np.transpose()转置数组
        self.edgeList = torch.tensor(np.transpose(np.array(forward_edges_list + backward_edges_list)), dtype=torch.long)
        # TODO 调试
        print('*********************************************************************************************')
        # print(self.edgeList)
        # print(len(self.edgeList[0]))
        # print(len(self.edgeList[1]))
        # print(len(self.proteinList))
        # print(self.edgeList[0])
        # print(forward_edges_list)
        print(len(forward_edges_list))
        print(len(np.array(forward_edges_list + backward_edges_list)))
        print(np.array(forward_edges_list + backward_edges_list))
        print(np.transpose(np.array(forward_edges_list + backward_edges_list)))
        # 定义PPI网络特征为1
        self.numPPIFeatures = 1
        print('Building edge feature attributes ...')
        # 构建 PPI网络 边的特征列表
        forward_edge_feature_list = [self.PPIGraph[node1][node2]['score']/1000 for node1, node2 in list(self.PPIGraph.edges())]
        backward_edge_feature_list = [self.PPIGraph[node1][node2]['score']/1000 for node2, node1 in list(self.PPIGraph.edges())]
        self.edgeAttr = torch.tensor(forward_edge_feature_list + backward_edge_feature_list, dtype=torch.float)# .view(-1,1)
        # TODO 调试
        print('*********************************************************************************************')
        # print(self.edgeList)
        # print(len(self.edgeList[0]))
        # print(len(self.edgeList[1]))
        # print(len(self.proteinList))
        # print(self.edgeList[0])
        # print(forward_edges_list)
        print(len(self.edgeAttr))
        print('*********************************************************************************************')
        # DTI data
        # TODO 构建 药物-蛋白质 矩阵(DTI网络)
        if not config.herbTest == 'true':
            print("Loading DTI links ...")
            # y_dti_data：药物-蛋白质 矩阵(DTI网络)
            #       prot1 prot2 prot3 prot4 prot5 prot6 prot7
            # drug1   0     1     1     0     0     1     0
            # drug2   1     0     1     1     0     0     0
            # drug3   1     1     0     0     1     1     0
            arrDTIData = DTITool.getDTIs(drugList=self.drugList, proteinList=self.proteinList, mode=config.mode)
            # .reshape()可以用于改变一个数组的形状
            self.arrDTIData = arrDTIData.reshape((len(self.drugList), len(self.proteinList)))
        print(self.arrDTIData.shape)
        print("Building feature matrix ...")
        # TODO 构建蛋白质训练集特征矩阵
        self.trainProteins = config.trainProteins
        self.trainMask = np.zeros(self.numProteins)
        # TODO 给当前训练集中蛋白质对应的索引位置赋值为1
        self.trainMask[self.trainProteins] = 1 # TODO self.train_prots是个列表
        # 构建 药物-蛋白质 特征矩阵
        self.featureMatrix = np.zeros((self.numDrugs, self.numProteins))
        # 更新PPI网络特征为200
        self.numPPIFeatures = 200
        # 构建药物分子特征嵌入 张量
        if config.includeMolecularFeatures:
            self.drugMoleculeEncodings = torch.Tensor([self.drugMoleculeEncodings[drug] for drug in self.drugList])
        print("Finished.\n")
        Herb2vecPathPrefix = '../model/phenotypeFeature/'
        # 表型特征嵌入模型
        drugModelFilePath = Herb2vecPathPrefix + 'sideEffectEmbeddingModel'
        GOModelFilePath = Herb2vecPathPrefix + 'GOEmbeddingModel'
        # load models
        # 加载模型
        # Gensim是在做自然语言处理时较为经常用到的一个工具库
        drugModel = gensim.models.Word2Vec.load(drugModelFilePath)
        GOModel = gensim.models.Word2Vec.load(GOModelFilePath)
        # Build wordvector dicts
        # TODO 构建词向量字典
        # wv: 是类gensim.models.keyedvectors.Word2VecKeyedVectors生产的对象，在word2vec是一个属性
        # 为了在不同的训练算法（Word2Vec，Fastext，WordRank，VarEmbed）之间共享单词向量查询代码，
        # gensim将单词向量的存储和查询分离为一个单独的类 KeyedVectors，包含单词和对应向量的映射。可以通过它进行词向量的查询
        # Example:
        # model_w2v.wv.most_similar("深度学习")  # 找最相似的词
        # model_w2v.wv.get_vector("深度学习")  # 查看向量
        # model_w2v.wv.syn0  #  model_w2v.wv.vectors 一样都是查看向量
        # model_w2v.wv.vocab  # 查看词和对应向量
        # model_w2v.wv.index2word  # 每个index对应的词
        drugModel = drugModel.wv
        GOModel = GOModel.wv
        # 表型特征向量大小为200
        # 存储 药物和蛋白质 的向量列表
        drugEmbeddings = []
        GOEmbeddings = []
        for protein in self.proteinList:
            proteinID = protein
            # index2word 和 key_to_index 差不多，都是将函数变量转换成字典。形式不同主要是gensim的版本不同导致的。
            # .keys()用于获取对象自身所有的可枚举的属性值
            if proteinID in GOModel.key_to_index.keys():
                GOEmbeddings.append(GOModel[proteinID])
            else:
                GOEmbeddings.append(torch.zeros((200)))
        for drugID in self.drugList:
            if drugID in drugModel.key_to_index.keys():
                # drugModel[drugID]获取到的是药物对应的向量（向量的长度为200）
                drugEmbeddings.append((drugModel[drugID]))
            else:
                drugEmbeddings.append(torch.zeros((200)))
        # torch.Tensor()是一个类，是默认张量类型torch.FloatTensor()的别名，用于生成一个单精度浮点类型的张量。
        # 将向量列表转换成张量
        self.drugEmbeddings = torch.Tensor(drugEmbeddings)
        self.GOEmbeddings = torch.Tensor(GOEmbeddings)
        # torch.cat()函数将两个张量（tensor）按指定维度拼接在一起
        self.proteinEmbeddings = self.GOEmbeddings
        print('#####################################')
    def get(self):
        dataList = []
        indices = list(range(self.numDrugs))
        for drugIndex in indices:
            # build protein mask
            # y: 取 药物-蛋白质 矩阵中的一行（即对应一个药物和所有蛋白质的关系） EG:
            #       prot1 prot2 prot3 prot4 prot5 prot6 prot7
            # drug1   1     1     0     0     1     1     0
            # y的长度为DTI网络中蛋白质个数
            y = torch.tensor(self.arrDTIData[drugIndex, :]).view(-1) # X.view(-1) 将X里面的所有维度数据转化成一维，并且按先后顺序排列。
            # 药物分子特征向量长度 1024
            molecularDrugFeature = self.drugMoleculeEncodings[drugIndex,:]
            # torch_geometric.data.Data  图数据的构造器：用于记录和表示一张图信息
            # x: 用于存储每个节点的特征，形状是[num_nodes, num_node_features]
            # edge_index: 用于存储节点之间的边，形状是 [2, num_edges]
            # edge_attr: 存储边的特征。形状是[num_edges, num_edge_features]
            # pos: 存储节点的坐标，形状是[num_nodes, num_dimensions]
            # y: 存储样本标签。如果是每个节点都有标签，那么形状是[num_nodes, *]；如果是整张图只有一个标签，那么形状是[1, *]
            # Data对象不仅仅限制于这些属性，我们可以通过(full_PPI_graph.扩展属性名)来扩展Data，以张量保存三维网格中三角形的连接性
            fullPPIGraph = Data(x=self.proteinEmbeddings, # [蛋白质1向量, 蛋白质2向量, 蛋白质3向量...]
                                  edge_index=self.edgeList, # [(蛋白质1_index,蛋白质2_index), (蛋白质1_index,蛋白质3_index)...]
                                  edge_attr=self.edgeAttr, # [蛋白质1和蛋白质2关系得分, 蛋白质1和蛋白质3关系得分...]
                                  y=y) # [1, 1, 0, 0, 1...] 一个药物和所有蛋白质的关系
            fullPPIGraph.drug_feature = self.drugEmbeddings[drugIndex, :]
            # 是否考虑 药物和蛋白质 分子特征
            if self.config.includeMolecularFeatures:
                fullPPIGraph.drug_mol_feature = molecularDrugFeature
                fullPPIGraph.protein_mol_feature = self.proteinMoleculeEncodings
            # TODO 为每一个药物都生成一个Data(图数据的构造器)，并添加到列表里
            dataList.append(fullPPIGraph)
        # TODO 返回所有药物的Data组成的列表
        return dataList



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 给parser实例添加属性  parser.add_argument()
    parser.add_argument("--herbTest", type=str, default='false')
    parser.add_argument("--mode", type=str, default='')
    parser.add_argument("--PPIMinScore", type=int, default=700)
    parser.add_argument("--includeMolecularFeatures", type=str, default='true')
    parser.add_argument("--drugMode", type=str, default='trfm')
    config = parser.parse_args()
    a = DTINetworkData(config=config)
    a.buildData(config=config)















