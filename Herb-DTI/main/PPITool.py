## PPI网络构建类 ##
import numpy as np
import math
import networkx as nx
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import queue
import threading
import sys
import os
from Bio import SeqIO
# 只保留蛋白质-蛋白质 > 700 的数据
# 输出：9606.ENSP00000000233 9606.ENSP00000432568 903
def pruneSTRINGDB(minScore=700,
                             mode=''):
    filePath = "../data/databaseData/STRING/9606.protein.links.full.v11.0.txt"
    outFilePath = "../data/graphData/DDIGraph/9606.protein.links." + str(minScore) + "_min_score.v11.0.txt"
    print("Processing raw human protein links file ...")
    # 处理原始人类蛋白质链接文件
    p = 0.041 # see STRING documentation
    with open(file=filePath, mode='r') as f, open(file=outFilePath, mode='w') as outFile:
        head = f.readline()
        outFile.write(head)
        counter = 0
        for line in f:
            counter += 1
            if counter % 1000000 == 0:
                print("Processed lines:", counter)
            split_line = line.strip().split(' ')
            if mode=='experimental':
                experimental_score = (1-int(split_line[-6])/1000) * (1-int(split_line[-7])/1000)
                database_score = (1-int(split_line[-5])/1000) * (1-int(split_line[-4])/1000)
                experimental_score = int(1000 * (1-experimental_score * database_score))
                if experimental_score < minScore:
                    continue
                outFile.write(split_line[0]+" "+ split_line[1]+" "+str(experimental_score)+'\n')
            else:
                total_score = int(split_line[15])/1000
                total_score_nop = (total_score-p)/(1-p)
                txt_score = int(split_line[14])/1000
                txt_score_nop = (txt_score - p)/(1-p)
                total_score_updated_nop = 1 - (1-total_score_nop)/(1-txt_score_nop)
                total_score_updated = total_score_updated_nop + p * (1-total_score_updated_nop)
                if total_score_updated * 1000 < minScore:
                    continue
                outFile.write(split_line[0]+" "+ split_line[1]+" "+str(int(total_score_updated*1000))+'\n')
    print("Finished.")
# 构建PPI网络
def writePPIGraph(minScore=700):
    prunedPPIFile = "../data/graphData/DDIGraph/9606.protein.links." + str(minScore) + "_min_score.v11.0.txt"
    print("Building PPI graph ...")
    # 使用networkx库构建图
    PPI_graph = nx.Graph()
    num_lines = sum(1 for line in open(prunedPPIFile, 'r'))
    with open(file=prunedPPIFile, mode='r') as f:
        f.readline() # skip header
        for line in tqdm(f, total=num_lines):
            split_line = line.split(' ')
            node_1 = split_line[0]
            node_2 = split_line[1]
            score = int(split_line[-1])
            # 图的点就是蛋白质
            PPI_graph.add_node(node_1)
            PPI_graph.add_node(node_2)
            # 图的边就是两个蛋白质之间的得分
            PPI_graph.add_edge(node_1, node_2, score=score)
    print("Finished.")
    print('nodes', len(PPI_graph.nodes()))
    print('edges', len(PPI_graph.edges()))
    print("Writing PPI graph to disk ...")
    graph_filename = "../data/graphData/DDIGraph/PPI_graph_"+str(minScore)+"_min_score"
    with open(file=graph_filename+'.pkl', mode='wb') as f:
        pickle.dump(PPI_graph, f, pickle.HIGHEST_PROTOCOL)
    print("Finished writing {}.\n".format(graph_filename))
# 获取PPI网络
def getPPIGraph(minScore=700):
    filename = "../data/graphData/DDIGraph/PPI_graph_" + str(minScore) + "_min_score"
    with open(file= filename+'.pkl', mode='rb') as f:
        return pickle.load(f)
# 为蛋白质序列生成fasta文件
def writeProteinFasta(protein_list):
    filePath = "../data/databaseData/STRING/9606.protein.sequences.v11.0.fa"
    return_sequences = []  # Setup an empty list
    for record in SeqIO.parse(filePath, "fasta"):
        if record.id in protein_list:
            return_sequences.append(record)
    print("Found {} PPI protein sequences of {}".format(len(return_sequences), len(protein_list)))
    SeqIO.write(return_sequences, "../data/mateData/DeepGOPlus/PPI_graph_protein_seqs.fasta", "fasta")



if __name__ == '__main__':
    # writeProteinFasta()
    # pruneSTRINGDB()
    writePPIGraph()


















