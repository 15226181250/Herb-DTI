# 模型构建类
import sys
from tqdm import tqdm
import time
import numpy as np
import networkx as nx
import math
from sklearn.model_selection import KFold
from sklearn import metrics
import torch
import torch_geometric.nn as NN
import torch_geometric.loader as DATA
import argparse
from modelDataLoader import *
from modelBuildFunction import *
import DTITool
from matplotlib import pyplot as plt

def quickenedMissingTargetPredictor(config):
    device = torch.device(type='cpu', index=0)
    print('________________________________')
    print(device)
    print('________________________________')
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    config.num_gpus = num_gpus
    np.random.seed(42)
    # build data
    config.numProteins = None if config.numProteins == -1 else config.numProteins
    # TODO 构建DTI数据生成器实例
    networkData = DTINetworkData(config=config)
    # 获取DTI网络药物和蛋白质数量
    numDrugs = networkData.numDrugs
    numProteins = networkData.numProteins
    # TODO 给出蛋白质和药物数量
    config.numProteins = numProteins
    config.numDrugs = numDrugs
    # dataset is present in dimension (numDrugs * numProteins)
    # KFold()用于划分数据集，训练集和验证集
    # n_splits：默认为3，表示将数据划分为多少份，即k折交叉验证中的k；（训练集：n_splits-1，测试集：1）
    # shuffle：默认为False，表示是否需要打乱顺序，如果设置为True，则会先打乱顺序再做划分，如果为False，会直接按照顺序做划分；
    # random_state：默认为None，表示随机数的种子，只有当shuffle设置为True的时候才会生效。
    kf = KFold(n_splits=config.numFolds, random_state=42, shuffle=True)  # config.num_folds=5
    # 生成蛋白质索引数组
    X = np.zeros((numProteins, 1))
    # build for help matrix for indices
    # 为蛋白质索引数组构建帮助矩阵
    helpMatrix = np.arange(numDrugs * numProteins)
    helpMatrix = helpMatrix.reshape((numDrugs, numProteins))
    print('Model:', config.arch)  # config.arch='GCNConv'(图卷积)
    results = []
    fold = 0
    for trainProteinIndices, testProteinIndices in kf.split(X):
        fold += 1
        if config.fold != -1 and fold != config.fold:  # config.fold=3
            continue
        print("Fold:", fold)
        config.trainProteins = trainProteinIndices  # 蛋白质训练集
        networkData.buildData(config)  # TODO 调用DTI数据生成器的构建数据函数
        # build train data over whole dataset with help matrix
        print('Fetching data...')
        trainDataset = networkData.get()  # TODO 获取DTI数据
        print('\nDone.\n')
        trainDataset = DTIGraphDataset(trainDataset)  # 将获取到的DTI数据封装为DTIGraphDataset对象
        # Calculate weights
        # TODO 计算权重
        # y_dti_data: 药物-蛋白质 矩阵；    train_protein_indices: 蛋白质训练集
        positives = networkData.arrDTIData[:, trainProteinIndices].sum()  # .sum()求矩阵中所有数的和
        # len_to_sum_ratio：蛋白质训练集和所有药组成的矩阵A - 矩阵A中蛋白质和药物有关系的点的和 / 矩阵A中蛋白质和药物有关系的点的和
        # 即：矩阵A中蛋白质和药物无关系的点的和 / 矩阵A中蛋白质和药物有关系的点的和（矩阵中0的和 / 矩阵中1的和）
        lenToSumRatio = (networkData.numDrugs * len(
            trainProteinIndices) - positives) / positives  # Get negatives/positives ratio
        print('Neg/pos ratio:', lenToSumRatio)
        # trainMask为蛋白质训练集特征数组，[0，1，1，1，1，1，0，1，1]，当前训练集中蛋白质对应的索引位置赋值为1
        trainMask = networkData.trainMask
        testMask = 1 - networkData.trainMask
        # build DataLoaders
        print('Building data loader ...')
        # TODO 构建数据加载器    DataListLoader继承torch.utils.data.DataLoader()用来把训练数据分成多个小组,
        # 此函数每次抛出一组数据。直至把所有的数据都抛出。就是做一个数据的初始化)
        # train_dataset为训练用的数据，包括PPI网络，药物和靶点的表型特征向量和结构特征向量等
        # config.batch_size=50(默认) 每次抛出多少数据(一组数据的大小)
        # 返回的结果为已经分好的小组组成的DataListLoader对象，可使用enumerate()来获取每组数据
        # import torch.utils.data as data1
        train_loader = DATA.DataLoader(trainDataset, batch_size=config.batchSize, shuffle=True)
        # train_loader = DATA.DataLoader(trainDataset, batch_size=config.batchSize, shuffle=True, drop_last=True)
        # TODO 调试BUG
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        # for batch_idx, data in enumerate(train_loader):
        #     print(batch_idx)
        #     print(data)
        #     print('8**********')
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('Initializing model ...')
        model = ModelFeatureNetwork(config,
                                    num_drugs=0,  # 药物数
                                    num_prots=networkData.numProteins,  # 蛋白质数
                                    num_features=networkData.numPPIFeatures,  # 药物特征数
                                    conv_method=config.arch)  # 卷积网络类型(GCN、GEN、GAT)
        # model = NN.DataParallel(model).to(device)  # 多GPU计算
        model.to(device=device) # CPU计算
        print("model total parameters", sum(p.numel() for p in model.parameters()))
        # Adam参数优化器     lr学习率
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # , momentum=0.9)
        # 刷新缓冲区
        sys.stdout.flush()
        ret = None
        best_AUROC = 0
        print('_____________________________________________')
        print(config.num_epochs)
        print('_____________________________________________')
        # 开始训练，训练 num_epochs 次
        for epoch in range(1, config.num_epochs + 1):
            loss = modelTrain(config=config,  # 参数
                              model=model,  # 待训练模型
                              device=device,  # 运行设备 CPU
                              trainLoader=train_loader,  # 数据加载器
                              optimizer=optimizer,  # 参数优化器
                              epoch=epoch,  # 训练批次
                              neg_to_pos_ratio=lenToSumRatio,  # 权重比
                              train_mask=trainMask)  # 特征矩阵
            print('Train loss:', loss)
            # 刷新缓冲区
            sys.stdout.flush()

            '''
            # TODO 绘制ROC曲线
            labels, predictions = modelPrediction(model, device, train_loader, round=False)
            # get train and test predictions
            # train_labels = labels.reshape((numDrugs, numProteins))[:, trainMask == 1].flatten()
            # train_predictions = predictions.reshape((numDrugs, numProteins))[:, trainMask == 1].flatten()
            test_labels = labels.reshape((numDrugs, numProteins))[:, trainMask == 0].flatten()
            test_predictions = predictions.reshape((numDrugs, numProteins))[:, trainMask == 0].flatten()

            # 假设y_true和y_score已经准备好了
            y_true = test_labels
            y_score = test_predictions

            # 计算ROC曲线的值
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)

            # 绘制ROC曲线
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic example')
            plt.legend(loc="lower right")
            plt.show()
            '''

            if epoch % 3 == 0:
            # if epoch % 1 == 0:
                print('Predicting for validation data...')
                # 预测验证数据
                file = '../model/resultDTIModel/pred_' + config.arch + '_fold_' + str(config.fold) + '_results'
                with open(file=file, mode='a') as f:
                    labels, predictions = modelPrediction(model, device, train_loader, round=False)
                    # get train and test predictions
                    train_labels = labels.reshape((numDrugs, numProteins))[:, trainMask == 1].flatten()
                    train_predictions = predictions.reshape((numDrugs, numProteins))[:, trainMask == 1].flatten()
                    test_labels = labels.reshape((numDrugs, numProteins))[:, trainMask == 0].flatten()
                    test_predictions = predictions.reshape((numDrugs, numProteins))[:, trainMask == 0].flatten()
                    print('pred_eval', train_labels.max(), train_predictions.max(), train_labels.min(),
                          train_predictions.min(), train_predictions.shape)
                    print('pred_eval', test_labels.max(), test_predictions.max(), test_labels.min(),
                          test_predictions.min(), test_predictions.shape)
                    print('Train:', 'Acc, ROC_AUC, f1, matthews_corrcoef',
                          metrics.accuracy_score(train_labels, train_predictions.round()),
                          DTITool.dti_auroc(train_labels, train_predictions),
                          DTITool.micro_AUC_per_prot(train_labels, train_predictions, config.numDrugs),
                          DTITool.dti_f1_score(train_labels, train_predictions.round()),
                          metrics.matthews_corrcoef(train_labels, train_predictions.round()), file=f)
                    print('Test:', 'Acc, ROC_AUC, f1, matthews_corrcoef',
                          metrics.accuracy_score(test_labels, test_predictions.round()),
                          DTITool.dti_auroc(test_labels, test_predictions),
                          DTITool.micro_AUC_per_prot(test_labels, test_predictions, config.numDrugs),
                          DTITool.dti_f1_score(test_labels, test_predictions.round()),
                          metrics.matthews_corrcoef(test_labels, test_predictions.round()), file=f)
                    test_AUROC = DTITool.dti_auroc(test_labels, test_predictions)
                    if test_AUROC > best_AUROC:
                        best_AUROC = test_AUROC
                        model_filename = '../model/resultDTIModel/PPI_network_model_with_mol_features_fold_' + str(
                            fold) + '.model'
                        torch.save(model.state_dict(), model_filename)
            if epoch == 50:
            # if epoch == 1:
                drug_list_repeated = networkData.drugList.repeat(config.numProteins)
                protein_list_repeated = networkData.proteinList.reshape(1, -1).repeat(config.numDrugs, axis=0).reshape(
                    -1)
                pred_loader = DATA.DataLoader(trainDataset, config.batchSize, shuffle=False)
                labels, predictions = modelPrediction(model, device, pred_loader, round=False)
                zipped_list = list(zip(drug_list_repeated, protein_list_repeated, labels, predictions))
                pred_filename = '../model/resultDTIModel/PPI_network_model_with_mol_features_fold_' + str(
                    fold) + '_predictions.pkl'
                with open(file=pred_filename, mode='wb') as f:
                    pickle.dump(zipped_list, f, pickle.HIGHEST_PROTOCOL)
            sys.stdout.flush()
        results.append(ret)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # model_filename = '../models/PPI_network_' + config.arch + '_model_fold_' + str(fold) + '.model'
        # torch.save(model.state_dict(), model_filename)
    return
if __name__ == '__main__':
    # Add parser arguments
    # 建立解析对象
    parser = argparse.ArgumentParser()
    # 给parser实例添加属性  parser.add_argument()
    parser.add_argument("--numProteins", type=int, default=-1)
    parser.add_argument("--arch", type=str, default='GCNConv')
    parser.add_argument("--node_features", type=str, default='MolPred')
    parser.add_argument("--num_epochs", type=int, default=200)
    # parser.add_argument("--batchSize", type=int, default=13)
    parser.add_argument("--batchSize", type=int, default=30)
    parser.add_argument("--numFolds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--fold", type=int, default=3)
    # parser.add_argument("--drug_mode", type=str, default='trfm')
    # parser.add_argument("--include_mol_features", action='store_true')
    parser.add_argument("--herbTest", type=str, default='false')
    parser.add_argument("--mode", type=str, default='')
    parser.add_argument("--PPIMinScore", type=int, default=700)
    parser.add_argument("--includeMolecularFeatures", type=str, default='true')
    parser.add_argument("--drugMode", type=str, default='trfm')
    # 通过config = parser.parse_args()把刚才的属性从parser给config，后面直接通过config使用
    config = parser.parse_args()
    # Run classifier
    quickenedMissingTargetPredictor(config)
