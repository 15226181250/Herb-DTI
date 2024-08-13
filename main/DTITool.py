## DTI工具类 ##
import numpy as np
import networkx as nx
import gensim
import torch
import sklearn.metrics as metrics
import pickle
import os
from tqdm import tqdm
import phenotypeDataTool
# 创建DTI网络图
def writeDTIGraph(min_score=700, mode=''):
    dti_graph = nx.Graph()
    print("Loading chemical stereo to mono mapping...")
    # 读取化合物立体结构和平面结构对应文件
    # {'CIDs91758688': 'CIDm91758688'}
    # stereo_mono_mapping = DDI_utils.get_chemical_stereo_to_normal_mapping()
    print("Done.\n")
    print("Parsing human drug-protein-links data ...")
    filePath = "../data/databaseData/STITCH/9606.protein_chemical.links.transfer.v5.0.tsv"
    with open(file=filePath, mode='r') as f:
        f.readline()
        for line in tqdm(f, total=15473940):
            split_line = line.split('\t')
            # 药物
            drug = split_line[0].strip()
            if 's' in drug:
                continue
            # 蛋白质
            target = split_line[1]
            # 计算药物和蛋白质之间的得分
            score = None
            if mode=='experimental': # 实验数据
                score = int((1- (1-int(split_line[2])/1000) * (1-int(split_line[3])/1000))*1000)
            elif mode=='database': # 数据库数据
                # 2: experimental_direct  3: experimental_transferred  6: database_direct  7: database_transferred
                score = int((1- (1-int(split_line[2])/1000) * (1-int(split_line[3])/1000) * (1-int(split_line[6])/1000) * (1-int(split_line[7])/1000))*1000)
            else:
                # 10: combined_score
                score = int(split_line[-1])
            # 使用药物和蛋白质之间的得分大于700的边和点构建人类DTI网络图
            if score >= min_score:
                dti_graph.add_node(drug)
                dti_graph.add_node(target)
                dti_graph.add_edge(drug, target, score=score)
    print("Finished.\n")
    print('num_nodes', len(dti_graph.nodes()))
    print('num_edges', len(dti_graph.edges()))
    print("Writing human only DTI-graph to disk ...")
    filename = "../data/graphData/DTIGraph/only_"+(mode+'_' if mode else '')+"DTI_graph"
    with open(file=filename+'.pkl', mode='wb') as f:
        pickle.dump(dti_graph, f, pickle.HIGHEST_PROTOCOL)
# 拿到DTI网络图
def getDTIGraph(mode=''):
    filePath = "../data/graphData/DTIGraph/only_"+(mode+'_' if mode else '')+"DTI_graph"
    with open(file=filePath + '.pkl', mode='rb') as f:
        return pickle.load(f)
# 拿到训练要用的药物列表
def getDrugList(mode=''):
    dtiGraph = getDTIGraph(mode=mode)
    # filePath = '../data/entityData/drug/drug_entity_list'
    filePath = '../data/phenotypeData/drug/drug_list'
    drugList = open(filePath).readlines()
    drugList = [drug.strip() for drug in drugList]
    return np.array([drug for drug in drugList if drug in dtiGraph.nodes()])
def getDTIs(drugList,
            proteinList,
            mode=''):
    # 拿到人类DTI网络图
    DTIGraph = getDTIGraph(mode=mode)
    # 创建 药物-蛋白质 矩阵
    arrData = np.zeros((len(drugList), len(proteinList)))
    for i in range(len(proteinList)):
        protein = proteinList[i]
        if protein not in DTIGraph.nodes():
            continue
        for drug in DTIGraph.neighbors(protein):
            if drug not in drugList:
                continue
            j = list(drugList).index(drug)
            # 药物和蛋白质之间有联系则将矩阵对应位置的值由0改为1
            arrData[j, i] = 1
    # 返回 药物-蛋白质 矩阵
    return np.array(arrData, dtype=np.int8)
# TODO 评价指标
def dti_auroc(y_true, y_pred):
    if y_true.sum() == 0 or (1-y_true).sum() == 0:
        return metrics.accuracy_score(y_true=y_true, y_pred=y_pred.round())
    return metrics.roc_auc_score(y_true, y_pred)
def dti_auprc(y_true, y_pred):
    p,r, t = metrics.precision_recall_curve(y_true, y_pred)
    # return metrics.auc(r, p)
    return metrics.average_precision_score(y_true, y_pred, average='weighted')
def dti_mcc(y_true, y_pred):
    return metrics.matthews_corrcoef(y_true, y_pred)
def dti_f1_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)
def micro_AUC_per_prot(y_true, y_pred, num_drugs):
    y_true = y_true.reshape(num_drugs, -1)
    y_pred = y_pred.reshape(num_drugs, -1)
    num_prots = y_true.shape[1]
    prot_wise_auc = []
    for prot_index in range(num_prots):
        if y_true[:, prot_index].sum() == 0:
            prot_wise_auc.append(metrics.accuracy_score(y_true=y_true, y_pred=y_pred.round()))
        prot_wise_auc.append(dti_auroc(y_true[:, prot_index], y_pred[:, prot_index]))
    prot_wise_auc = np.array(prot_wise_auc)
    return prot_wise_auc.mean()
def micro_AUC_per_prot_DT_pairs(y_true, y_pred, num_drugs, indices):
    # microAUC only for DT pair splitting scheme
    y_true = y_true.reshape(num_drugs, -1)
    y_pred = y_pred.reshape(num_drugs, -1)
    num_prots = y_true.shape[1]
    prot_wise_auc = []
    prot_wise_index_list = [[]] * num_prots
    # collect all interactors for each protein w.r.t. indices
    for index in indices:
        print(index, len(prot_wise_index_list))
        prot_wise_index_list[index%num_drugs].append(index)
    for prot_index in range(num_prots):
        if prot_wise_index_list[prot_index]:
            pair_indices = np.array(prot_wise_index_list[prot_index])
            prot_y_true = y_true.flatten()[pair_indices]
            prot_y_pred = y_pred.flatten()[pair_indices]
            prot_wise_auc.append(dti_auroc(prot_y_true, prot_y_pred))
    prot_wise_auc = np.array(prot_wise_auc)
    return prot_wise_auc.mean()
def micro_AUC_per_drug(y_true, y_pred, num_drugs):
    y_true = y_true.reshape(num_drugs, -1)
    y_pred = y_pred.reshape(num_drugs, -1)
    num_prots = y_true.shape[1]
    drug_wise_auc = []
    for drug_index in range(num_drugs):
        if y_true[drug_index, :].sum() == 0:
            pass
        drug_wise_auc.append(dti_auroc(y_true[drug_index, :], y_pred[drug_index, :]))
    drug_wise_auc = np.array(drug_wise_auc)
    return drug_wise_auc.mean()



if __name__ == '__main__':
    # writeDTIGraph(min_score=700)
    a = list(getDrugList())
    print(len(a))











