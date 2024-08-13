## 生成向量类 ##
import random
import gensim
import multiprocessing as mp
from threading import Lock
WALK_LENGTH = 30
WALK_NUMBER = 100
WORK_NUMBER = 48
def beginRandomWalk(G, nodes, num_walks=WALK_NUMBER, num_length=WALK_LENGTH, file_pre=''):
    print("开始随机游走")
    all_walks_list = []
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
    # 同时列出数据和数据下标，一般用在 for 循环当中
    for index, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        epsilon = 0.000001
        for i in range(num_walks):
            curr_node = node
            walk_accumulate=[]
            for j in range(num_length):
                # 当前节点的所有邻居节点
                neighbours = list(G.neighbors(curr_node))
                neighbour_types = [G.edges[curr_node, neighbour]['type'] for neighbour in neighbours]
                # 统计关系为   HasAssociation(实体与其表型特征的关系表示)   的边的总数
                # TODO 添加
                if 'HasAssociation' not in neighbour_types:
                    # continue
                    weight_vec = [1 for type in neighbour_types]
                else:
                    num_HasAssociations = neighbour_types.index('HasAssociation')
                    # make HasAssociation and non-HasAssociation equally likely
                    # 使 HasAssociation 和非 HasAssociation 的可能性相等    为了满足随机游走定义
                    HasAssociation_weight = 0.01 / (num_HasAssociations + epsilon)
                    non_HasAssociation_weight = 0.99 / (len(neighbours)-num_HasAssociations + epsilon)
                    # build weight vector
                    # 构建权重矢量
                    weight_vec = [(HasAssociation_weight if type=='HasAssociation' else non_HasAssociation_weight) for type in neighbour_types]
                # k为选取次数
                # weights：设置相对权重，它的值是一个列表，设置之后，每一个成员被抽取到的概率就被确定了。
                # 例如：weights=[1,1,1,1,1]，那么第一个元素的权重就是1/1+1+1+1+1 = 1/5；
                # weights=[1,2,3,4,5]，那么第二个元素的权重就是2/1+2+3+4+5 = 2/15
                next_node = random.choices(population=neighbours, weights=weight_vec, k=1)[0]
                type_nodes = G.edges[curr_node, next_node]["type"]
                if curr_node == node:
                    walk_accumulate.append(curr_node)
                walk_accumulate.append(type_nodes)
                walk_accumulate.append(next_node)
                curr_node = next_node
            all_walks_list.append(walk_accumulate)
        if index % 100 == 0:
            print("Done walks for", index, "nodes")
    saveAsFile(all_walks_list, file_pre)
def runRandomWalk(nodes, G, num_workers=WORK_NUMBER, num_length=WALK_LENGTH, file_pre=''):
    global data_pairs
    length = len(nodes) // num_workers
    # print("length: "+str(length))
    # target：进程的目标函数，即要执行的任务       args：给目标函数传递参数
    processes = [mp.Process(target=beginRandomWalk, args=(G, nodes[(index) * length:(index + 1) * length], num_workers, num_length, file_pre)) for index
                 in range(num_workers-1)]
    processes.append(mp.Process(target=beginRandomWalk, args=(G, nodes[(num_workers-1) * length:len(nodes) - 1], num_workers, num_length, file_pre)))
    for p in processes:
        p.start()
        print('------进程 '+str(p.pid)+' 进程已开启------')
    for p in processes:
        p.join()
lock = Lock()
def saveAsFile(all_walks_list, file_pre=''):
    with lock:
        with open(file_pre + "walks.txt", "a") as fp:
            for walks in all_walks_list:
                for step in walks:
                    fp.write(str(step)+" ")
                fp.write("\n")
# 基因节点向量
def getNodeVector(graph, inputFline, outputFile, embedding_size, file_pre='', num_workers=WORK_NUMBER):
    node_list = []
    with open(inputFline, "r") as f:
        for line in f.readlines():
            data = line.strip().split()
            for da in data:
                # print('------------')
                # print(da)
                node_list.append(da)
    nodes_G = [n for n in graph.nodes()]
    node_set = set(node_list)
    nodes= [n for n in node_set]
    # .subgraph(): 根据节点，生成图的子图，边保留
    G = graph.subgraph(nodes_G)
    runRandomWalk(nodes,G, num_workers=num_workers, num_length=WALK_LENGTH, file_pre=file_pre)
    print("start to train the word2vec models")
    # 开始训练 word2vec 模型
    sentences = gensim.models.word2vec.LineSentence(file_pre+"walks.txt")
    # cbow: 用窗口内的上下文词预测中心词
    # skip-gram: 用窗口内的中心词预测上下文词
    # negative：每次采多少样本（负采样，同时考虑窗口外的词，优化算法）
    # vector_size: 词向量维度大小
    # min_count：把词频率小于多少的去掉
    # epochs：模型训练的迭代次数
    model=gensim.models.Word2Vec(sentences,sg=1, min_count=1, vector_size=embedding_size, window=10,epochs=30,workers=num_workers)
    model.save(outputFile)







































