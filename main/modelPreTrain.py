## 模型预训练类
import subprocess
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn import metrics
import gensim
from tqdm import tqdm
import argparse
import sys
import pickle
import torch
import torch.nn as NN
import torch.utils.data as data
import DTITool
import PPITool
import phenotypeDataTool
from modelBuildFunction import *
# 预训练数据加载器
class PreTrainDataBuilder:
    def __init__(self, config):
        self.config = config
        print("Loading data ...")
        if config.herbTest == 'true':
            # TODO 待开发
            pass
        else:
            self.drugList = np.array(DTITool.getDrugList(config.mode))
        print(len(self.drugList), "drugs present")
        # get protein lists for each ontology
        # Gene Ontology 基因本体论
        GOProteinList = phenotypeDataTool.getPhenotypeProteinList()
        # 拿到DTI网络图
        self.DTIGraph = DTITool.getDTIGraph(mode=config.mode)
        # 获取人类蛋白质网络
        self.PPIGraph = PPITool.getPPIGraph(minScore=700)
        self.proteinList = np.array(list(set(self.PPIGraph.nodes()) & set(self.DTIGraph.nodes()) & (set(GOProteinList))))
        if config.herbTest == 'true':
            # TODO 待开发
            pass
        print(len(self.proteinList), "proteins present\n")
        # calculate dimensions of network
        self.numProteins = len(self.proteinList)
        self.numDrugs = len(self.drugList)
        config.numProteins = self.numProteins
        config.numDrugs = self.numDrugs
    def build_data(self, config):
        # DTI data
        if not config.herbTest == 'true':
            print("Loading DTI links...")
            arrDTIData = DTITool.getDTIs(drugList=self.drugList, proteinList=self.proteinList, mode=config.mode)
            self.arrDTIData = arrDTIData.reshape((len(self.drugList), len(self.proteinList)))
        self.featureMatrix = np.zeros((self.numDrugs, self.numProteins))
        Herb2vecPathPrefix = '../model/phenotypeFeature/'
        drugModelFilePath = Herb2vecPathPrefix + 'sideEffectEmbeddingModel'
        GOModelFilePath = Herb2vecPathPrefix + 'GOEmbeddingModel'
        # load models
        drugModel = gensim.models.Word2Vec.load(drugModelFilePath)
        GOModel = gensim.models.Word2Vec.load(GOModelFilePath)
        # Build wordvector dicts
        drugModel = drugModel.wv
        GOModel = GOModel.wv
        drugEmbeddings = []
        GOEmbeddings = []
        for protein in self.proteinList:
            proteinID = protein
            if proteinID in GOModel.key_to_index.keys():
                GOEmbeddings.append(GOModel[proteinID])
            else:
                GOEmbeddings.append(torch.zeros((200)))
        for drugID in self.drugList:
            # drugModel[drugID]获取到的是药物对应的向量（向量的长度为200）
            drugEmbeddings.append((drugModel[drugID]))
        self.drugEmbeddings = torch.Tensor(drugEmbeddings)
        self.GOEmbeddings = torch.Tensor(GOEmbeddings)
        print(len(self.drugEmbeddings))
        print(len(self.GOEmbeddings))
        print('1111111111111')
        self.proteinEmbeddings = self.GOEmbeddings
        print('#####################################')
        print("Finished.\n")
    def get(self, indices):
        dataList = []
        # indices = list(range(self.num_drugs))
        for index in tqdm(indices):
            drug_index = index // self.numProteins
            protein_index = index % self.numProteins
            # build protein mask
            y = int(self.arrDTIData[drug_index, protein_index])
            drugFeature = self.drugEmbeddings[drug_index, :]
            proteinFeature = self.proteinEmbeddings[protein_index, :]
            # additional
            dataList.append((torch.cat((drugFeature, proteinFeature), 0).float(), y))
        return dataList
    def __len__(self):
        return self.numDrugs * self.numProteins
class DTIGraphDataset(data.Dataset):
    def __init__(self, data_list):
        super(DTIGraphDataset, self).__init__()
        self.data_list = data_list
    def __getitem__(self, idx):
        return self.data_list[idx]
    def __len__(self):
        return len(self.data_list)
    def _download(self):
        pass
    def _process(self):
        pass
class PreTrainNetwork(NN.Module):
    def __init__(self):
        super(PreTrainNetwork, self).__init__()
        self.dropout = NN.Dropout(0.5)
        # siamese network approach 连体的神经网络
        # Siamese Network 是一种神经网络的框架，而不是具体的某种网络，就像seq2seq一样，具体实现上可以使用RNN也可以使用CNN
        # Siamese Network有两个结构相同，且共享权值的子网络
        # 分别接收两个输入X1与X2，将其转换为向量Gw(X1)与Gw(X2)，再通过某种距离度量的方式计算两个输出向量的距离Ew
        # TODO 药物
        # nn.Sequential 一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中
        self.model = NN.Sequential(
            # nn.Linear 用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是二维张量
            NN.Linear(200, 256),
            # nn.Dropout(0.2),
            # BatchNorm是深度网络中经常用到的加速神经网络训练，加速收敛速度及稳定性的算法，是深度网络训练必不可少的一部分,几乎成为标配
            # BatchNorm 即批规范化，是为了将每个batch的数据规范化为统一的分布，帮助网络训练， 对输入数据做规范化
            NN.BatchNorm1d(256),
            # 在深度学习中，nn.leakyrelu是一种激活函数，它可以用于增强神经网络的非线性特征提取能力。
            # 在人工神经网络的每一层中，需要选择一个激活函数。激活函数将输入值进行一个非线性转换，使神经网络能够更好地适应非线性数据
            NN.LeakyReLU(0.2, inplace=True),
            NN.Linear(256, 200),
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(50),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 1),
            # nn.Sigmoid()
        )
        # TODO 蛋白质
        self.model2 = NN.Sequential(
            # NN.Linear(600, 256),
            NN.Linear(200, 256),
            # nn.Dropout(0.5),
            NN.BatchNorm1d(256, affine=True),
            NN.LeakyReLU(0.2, inplace=True),
            NN.Linear(256, 200),
            # nn.BatchNorm1d(200),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(200, 200),
            # nn.Sigmoid()
        )
        # CosineSimilarity函数计算两个高维特征图(B,C,H,W)中各个像素位置的特征相似度
        self.sim = NN.CosineSimilarity(dim=1)
    def forward(self, x):
        # view()函数就是用来改变tensor的形状的，
        # 例如将2行3列的tensor变为1行6列，其中-1表示会自适应的调整剩余的维度
        # 药物
        p1 = self.model(x[:,:200]).view(-1, 200)
        # 蛋白质
        d1 = self.model2(x[:,200:]).view(-1, 200)
        s1 = self.sim(p1, d1)
        out = s1.reshape(-1, 1)
        # out = self.output_sig(s1)
        # sigmoid是激活函数的一种，它会将样本值映射到0到1之间。
        return out.sigmoid()
def train(config, model, device, train_loader, optimizer, epoch, neg_to_pos_ratio):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    sys.stdout.flush()
    model.train()
    return_loss = 0
    for batch_idx, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        features = features.to(device)
        labels = labels.float().to(device)
        # TODO 调试
        # print('000000000000000000000')
        # print(batch_idx)
        # print('000000000000000000000')
        # print(features)
        # print('000000000000000000000')
        # print(labels)
        output = model(features)
        import modelBuildFunction
        loss = modelBuildFunction.BCELossClassWeights(input=output, target=labels.view(-1,1), pos_weight=neg_to_pos_ratio)
        loss = loss /(config.numProteins * config.numDrugs)
        return_loss += loss
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * output.size(0),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            sys.stdout.flush()
    return return_loss
def predicting(model, device, loader, round=False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(loader):
            features = features.to(device)
            output = model(features)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, labels.view(-1, 1).float().cpu()), 0)
    if round:
        return total_labels.round().numpy().flatten(),total_preds.round().numpy().flatten()
    else:
        return total_labels.round().numpy().flatten(),total_preds.numpy().flatten()
def siamese_drug_protein_network(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.numProteins <= 0:
        config.numProteins = None
    dtiData = PreTrainDataBuilder(config)
    # generate indices for proteins
    kf = KFold(n_splits=config.numFolds, random_state=42, shuffle=True)
    X = np.zeros((dtiData.numProteins, 1))
    # build for help matrix for indices
    helpMatrix = np.arange(dtiData.numDrugs * dtiData.numProteins)
    helpMatrix = helpMatrix.reshape((dtiData.numDrugs, dtiData.numProteins))
    results = []
    fold = 0
    for trainProteinIndices, testProteinIndices in kf.split(X):
        fold += 1
        if config.fold != -1 and config.fold != fold:
            continue
        print("Fold:", fold)
        dtiData.build_data(config)
        # build train data over whole dataset with help matrix
        trainIndices = helpMatrix[:, trainProteinIndices].flatten()
        testIndices = helpMatrix[:, testProteinIndices].flatten()
        print(trainIndices.shape, testIndices.shape)
        trainDataset = dtiData.get(trainIndices)
        testDataset = dtiData.get(testIndices)
        trainDataset = DTIGraphDataset(trainDataset)
        testDataset = DTIGraphDataset(testDataset)
        print('len(trainDataset)', len(trainDataset))
        # Calculate weights
        positives = dtiData.arrDTIData.flatten()[trainIndices].sum()
        negToPosRatio = (len(trainIndices) - positives) / positives
        trainLoader = data.DataLoader(trainDataset, batch_size=config.batchSize, shuffle=True)
        testLoader = data.DataLoader(testDataset, batch_size=config.batchSize)
        model = PreTrainNetwork()
        model = NN.DataParallel(model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        sys.stdout.flush()
        ret = None
        best_AUROC = 0
        for epoch in range(1, config.numEpochs + 1):
            loss = train(config=config, model=model, device=device, train_loader=trainLoader, optimizer=optimizer, epoch=epoch, neg_to_pos_ratio=negToPosRatio)
            print('Train Loss:', loss)
            if epoch%2 == 0:
                print('Predicting for validation data...')
                file='../model/preTrainModel/process/pred_results_' + str(config.numEpochs)+'_epochs'
                with open(file=file, mode='a') as f:
                    trainLabels, trainPredictions = predicting(model, device, trainLoader)
                    print('Train: Acc, ROC_AUC, MicroAUC, f1, matthews_corrcoef',
                          metrics.accuracy_score(trainLabels, trainPredictions.round()),
                          DTITool.dti_auroc(trainLabels, trainPredictions),
                          DTITool.micro_AUC_per_prot(trainLabels, trainPredictions, config.numDrugs),
                          metrics.average_precision_score(trainLabels, trainPredictions),
                          DTITool.dti_f1_score(trainLabels, trainPredictions.round()),
                          DTITool.dti_mcc(trainLabels, trainPredictions.round()))#@TODO, file=f)
                    testLabels, testPredictions = predicting(model, device, testLoader)
                    print('Test: Acc, ROC_AUC, MicroAUC, f1, matthews_corrcoef',
                          metrics.accuracy_score(testLabels, testPredictions.round()),
                          DTITool.dti_auroc(testLabels, testPredictions),
                          DTITool.micro_AUC_per_prot(testLabels, testPredictions, config.numDrugs),
                          metrics.average_precision_score(testLabels, testPredictions),
                          DTITool.dti_f1_score(testLabels, testPredictions.round()),
                          DTITool.dti_mcc(testLabels, testPredictions.round()))#@TODO, file=f)
                    test_AUROC = DTITool.micro_AUC_per_prot(testLabels, testPredictions, config.numDrugs)
                    if test_AUROC > best_AUROC:
                        print('New best test MicroAUC:', test_AUROC)
                        # New best test MicroAUC: 0.7434683597877955
                        state_dict_path = '../model/preTrainModel/pred_fold_'+str(fold)+'_model'
                        torch.save(model.state_dict(), state_dict_path)
                        # TODO 修改
                        best_AUROC = test_AUROC
            sys.stdout.flush()


if __name__ == '__main__':
    # Add parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--numProteins", type=int, default=-1)
    # parser.add_argument("--arch", type=str, default='GCNConv')
    # parser.add_argument("--node_features", type=str, default='simple')

    parser.add_argument("--numEpochs", type=int, default=20)
    parser.add_argument("--batchSize", type=int, default=1024)
    parser.add_argument("--numFolds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--mode", type=str, default='')
    parser.add_argument("--fold", type=int, default=3)

    parser.add_argument("--herbTest", type=str, default='false')

    config = parser.parse_args()

    siamese_drug_protein_network(config)































