## 构建模型类 ##
import numpy as np
from math import sqrt, log2
from scipy import stats
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import sys
import pickle
import sys
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as nn
import torch.utils.data as data
from modelPreTrain import *
class ModelFeatureNetwork(torch.nn.Module):
    def __init__(self, config, num_drugs, num_prots, num_features, conv_method, dropout=0.2):
        super(ModelFeatureNetwork, self).__init__()
        self.config = config
        self.num_drugs = num_drugs # 0
        self.num_prots = num_prots
        self.num_features = num_features
        # mask feature
        # GCN layers
        if 'GCNConv' in conv_method:
            # GCNConv参数：
            # in_channels：输入通道，比如节点分类中表示每个节点的特征数。
            # out_channels：输出通道，最后一层GCNConv的输出通道为节点类别数（节点分类）。
            # improved：如果为True表示自环增加，也就是原始邻接矩阵加上2I而不是I，默认为False。
            # cached：如果为True，GCNConv在第一次对邻接矩阵进行归一化时会进行缓存，以后将不再重复计算。
            # add_self_loops：如果为False不再强制添加自环，默认为True。
            # normalize：默认为True，表示对邻接矩阵进行归一化。
            # bias：默认添加偏置。
            # 第一层的参数的输入维度就是初始每个节点的特征维度，输出维度是200。
            self.conv1 = nn.GCNConv(200, 200, cached=True, improved=True)
            # 第二层的输入维度为200，输出维度为分类个数，因为我们需要对每个节点进行分类，最终加上softmax操作。
            self.conv2 = nn.GCNConv(200, 200, cached=True, improved=True)
            # torch.zeros函数是PyTorch中的一个创建张量（tensor）的函数，
            # 常用于初始化模型的权重矩阵、创建指定维度的全零张量等场景。
            weight1 = torch.zeros((200,200))
            weight2 = torch.zeros((200,200))
            for i in range(200):
                weight1[i,i] = 1
                weight2[i,i] = 1
            # torch.nn.Parameter是继承自torch.Tensor的子类，其主要作用是作为nn.Module中的可训练参数使用
            # nn.Parameter可以看作是一个类型转换函数，将一个不可训练的类型 Tensor 转换成可以训练的类型 parameter ，
            # 并将这个 parameter 绑定到这个module 里面nn.Parameter()添加的参数会被添加到Parameters列表中，
            # 会被送入优化器中随训练一起学习更新
            self.conv1.weight = torch.nn.Parameter(weight1)
            self.conv2.weight = torch.nn.Parameter(weight2)
            # self.conv3 = nn.GCNConv(1, 1, cached=False, add_self_loops=True)
        elif 'GENConv' in conv_method:
            conv1 = nn.GENConv(200,200, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm1 = torch.nn.LayerNorm(200, elementwise_affine=True)
            conv2 = nn.GENConv(200, 200, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm2 = torch.nn.LayerNorm(200, elementwise_affine=True)
            conv3 = nn.GENConv(200, 200, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm3 = torch.nn.LayerNorm(200, elementwise_affine=True)
            conv4 = nn.GENConv(200, 200, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm4 = torch.nn.LayerNorm(200, elementwise_affine=True)
            conv5 = nn.GENConv(200, 200, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm5 = torch.nn.LayerNorm(200, elementwise_affine=True)
            act = torch.nn.LeakyReLU(0.2, inplace=True)
            self.conv1 = nn.DeepGCNLayer(conv1, norm1, act, block='res', dropout=0.5)
            self.conv2 = nn.DeepGCNLayer(conv2, norm2, act, block='res', dropout=0.5)
            self.conv3 = nn.DeepGCNLayer(conv3, norm3, act, block='res', dropout=0.5)
            self.conv4 = nn.DeepGCNLayer(conv4, norm4, act, block='res', dropout=0.1)
            self.conv5 = nn.DeepGCNLayer(conv5, norm5, act, block='res', dropout=0.1)
        elif 'GATConv' in conv_method:
            self.conv1 = nn.GATConv(200, 200, heads=4, dropout=0.2, add_self_loops=False)
            self.conv2 = nn.GATConv(200*4, 200, heads=1, dropout=0.2, add_self_loops=False)
            # self.conv3 = nn.GATConv(8*2, 1, heads=1)
        elif 'APPNP' in conv_method:
            self.conv1 = nn.APPNP(K=50, alpha=0.15)
        else:
            print("No valid model selected.")
            sys.stdout.flush()
            raise ValueError
        state_dict_path = '../model/preTrainModel/pred_fold_' + str(config.fold) + '_model'
        # 调用 torch_networks.py 的 HPOPredNet
        self.HPO_model = PreTrainNetwork()
        state_dict = torch.load(state_dict_path)
        from collections import OrderedDict
        # OrderedDict  按照有序插入顺序存储的有序字典
        new_state_dict = OrderedDict() # 用来存放预训练模型参数
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        # 将预训练模型的参数传给HPO模型
        self.HPO_model.load_state_dict(new_state_dict)
        for param in self.HPO_model.parameters():
            # param.requires_grad = False的作用是: 屏蔽预训练模型的权重。只训练最后一层的全连接的权重
            param.requires_grad = False
        self.mol_protein_model = torch.nn.Sequential(
            torch.nn.Linear(8192, 256),
            torch.nn.Dropout(0.5),
            # nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 200),
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(50),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 1),
            # nn.Sigmoid()
        )
        self.mol_drug_model = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.Dropout(0.5),
            # nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 200),
            # nn.BatchNorm1d(50),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 1),
            # nn.Sigmoid()
        )
        # 设置蛋白质全连接层
        # in_features=400, # 输入的神经元个数(特征数)
        # out_features=200, # 输出神经元个数(特征数)
        # bias=True # 是否包含偏置
        self.protein_linear1 = torch.nn.Linear(400, 200)
        # 设置药物全连接层
        self.drug_linear1 = torch.nn.Linear(400, 200)
        self.drug_linear2 = torch.nn.Linear(200, 200)

        self.overall_linear1 = torch.nn.Linear(400, 200)
        self.overall_linear2 = torch.nn.Linear(200, 1)
        # self.overall_linear3 = torch.nn.Linear(16, 1)
        # ReLU 激活函数：非负数保留，负数归零
        self.relu = torch.nn.ReLU()
        self.activation = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(dropout)
        self.sim = torch.nn.CosineSimilarity(dim=1)
        self.eps = 1e-7
    # 外部数据会传入到forward()函数的PPI_data_object参数中作为张量x
    # forward()的参数是张量x并且将其传递到__init__方法定义的神经网络中进行操作计算。
    def forward(self, PPI_data_object):
        # DDI_feature = PPI_data_object.DDI_features
        PPI_x, PPI_edge_index, PPI_batch, edge_attr = PPI_data_object.x, PPI_data_object.edge_index, PPI_data_object.batch, PPI_data_object.edge_attr

        # drug_feature是一个数组，表示一个药物的表型特征向量
        drug_feature = PPI_data_object.drug_feature.view(-1, self.num_features)

        batch_size = drug_feature.size(0) # 药物的数量为1   batch_size=1

        drug_mol_feature = PPI_data_object.drug_mol_feature.view(batch_size, -1)

        PPI_x = self.HPO_model.model2(PPI_x)
        PPI_mol_x = self.mol_protein_model(PPI_data_object.protein_mol_feature)
        # torch.cat是将两个张量(tensor)拼接在一起（将蛋白质表型特征向量和结构特征向量拼接在一起）
        # activation(): 激活函数，将激活后的值赋值给PPI_x，作为下一层的输入
        PPI_x = self.activation(torch.cat([PPI_x, PPI_mol_x], dim=1))
        PPI_x = self.protein_linear1(PPI_x)
        PPI_x = PPI_x.view(-1,200)
        # PPI_x = self.dropout(PPI_x)
        # PPI_x = F.elu(self.linear3(PPI_x))
        drug_feature = self.HPO_model.model(drug_feature).view(batch_size, 1, -1)
        # drug_mol_feature = drug_feature.repeat(1,self.num_prots,1).view(batch_size*self.num_prots,-1)
        drug_mol_feature = self.mol_drug_model(drug_mol_feature).view(batch_size, 1, -1)
        drug_feature = self.activation(torch.cat([drug_mol_feature, drug_feature], dim=2))
        # activation(): 激活函数，将激活后的值赋值给drug_feature，作为下一层的输入
        drug_feature = self.activation(self.drug_linear1(drug_feature))
        drug_feature = self.drug_linear2(drug_feature)
        # tensor.repeat()函数，可以将张量看作一个整体，然后根据指定的形状进行重复填充，得到新的张量。
        drug_feature = drug_feature.repeat(1,self.num_prots,1).view(batch_size*self.num_prots,-1)
        # self.conv1 = nn.GCNConv(200, 200, cached=True, improved=True)
        # self.conv2 = nn.GCNConv(200, 200, cached=True, improved=True)
        PPI_x = self.conv1(PPI_x, PPI_edge_index) # PPI_edge_index: [(蛋白质1_index,蛋白质2_index), (蛋白质1_index,蛋白质3_index)...]
        PPI_x = self.conv2(PPI_x, PPI_edge_index)
        # PPI_x = self.conv3(PPI_x, PPI_edge_index)
        # PPI_x = self.conv4(PPI_x, PPI_edge_index)
        # PPI_x = self.conv5(PPI_x, PPI_edge_index)
        # tensor.unsqueeze 为tenor添加维度
        # .unsqueeze(-1)  # 负1表示 在最后一维上添加（即在最外层再添加一次[]）
        PPI_x = self.sim(drug_feature, PPI_x).unsqueeze(-1) # 一个药物和所有蛋白质的相似性组成的张量
        # view()函数是用于对Tensor(张量)进行形状变化的函数
        # 可以对view()中的一个参数设置为-1。 若是对目标张量的某一维度不明、待定、视情况而变、懒得计算等等，
        # 可以使用-1参数进行操作。函数会自动计算-1参数对应维度的值。
        cat_feature = PPI_x.view((-1, self.num_prots))
        # sigmoid 激活函数：将所有数限制在 0 ~ 1 范围内
        return self.sigmoid(cat_feature) # 一个药物和所有蛋白质的相似性组成的张量
# binary_cross_entropy(BCE): 二元交叉熵，用来评判一个二分类模型预测结果的好坏程度，测量目标和输出之间的二进制交叉熵的函数
# binary_cross_entropy_with_logits:用来评判多标签分类模型预测结果的好坏程度，测量目标和输出logits之间的二进制交叉熵的函数
def BCELossClassWeights(input, target, pos_weight):
    # input (n, d)
    # target (n, d)
    # class_weights (1, d)

    # clamp()函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    target = target.view(-1,1)
    weighted_bce = - pos_weight*target * torch.log(input) - 1*(1 - target) * torch.log(1 - input)
    final_reduced_over_batch = weighted_bce.sum(axis=0)
    return final_reduced_over_batch
# TODO 执行训练函数
'''
config: 参数列表
model: 定义好的图卷积模型 QuickTemplateNodeFeatureNet()
device: CPU
train_loader: 需要训练的数据，即DataListLoader(QuickProtFuncDTINetworkData().get())抛出的数据
optimizer: Adam参数优化器 torch.optim.Adam()
epoch: 当前训练批次
neg_to_pos_ratio: 权重（药物-蛋白质矩阵中0的和 / 药物-蛋白质矩阵中1的和）
train_mask: 蛋白质训练集特征数组，[0，1，1，1，1，1，0，1，1]，当前训练集中蛋白质对应的索引位置赋值为1
'''
def modelTrain(config, model, device, trainLoader, optimizer, epoch, neg_to_pos_ratio, train_mask):
    print('Training on {} samples...'.format(len(trainLoader.dataset))) # 长度等于self.num_drugs
    sys.stdout.flush()
    # 在训练开始之前写上model.trian()，在测试时写上model.eval()
    # model.train()的作用是启用 Batch Normalization 和 Dropout
    # model.train()是保证BN层(Batch Normalization)能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    # model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
    # 对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。
    model.train()
    return_loss = 0
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
    # 同时列出数据和数据下标，一般用在 for 循环当中。
    for batch_idx, data in enumerate(trainLoader): # 每次进来一个batch的数据（len(data)个药物的Data(图数据的构造器)），计算一次梯度，更新一次网络，默认每次进来50个药物的Data
        # TODO 清空过往梯度
        optimizer.zero_grad()
        # 将数据扔进模型开始训练
        output = model(data) # output: 50个药物和所有蛋白质的相似性组成的张量 tensor([0.7311, 0.3209, 0.6970...])
        # print('max/min:', output.max(), output.sigmoid().max(), output.min(), output.sigmoid().min())
        # graph_data.y: 即Data(图数据的构造器)中的y属性，对应一个药物和所有蛋白质的关系([1, 1, 0, 0, 1...])
        # 使用GPU训练的时候，需要将Module对象和Tensor类型的数据送入到device。通常会使用 to.(device)。但是需要注意的是：
        # 对于Tensor类型的数据，使用to.(device) 之后，需要接收返回值，返回值才是正确设置了device的Tensor。
        # 对于Module对象，只用调用to.(device) 就可以将模型设置为指定的device。不必接收返回值
        # y = torch.Tensor(np.array([graph_data.y.numpy() for graph_data in data])).float().to(output.device)
        y = torch.Tensor(np.array([data.y.numpy()])).view(len(output),-1).float().to(output.device)
        # my implementation of BCELoss
        # clamp()函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
        output = torch.clamp(input=output, min=1e-7, max=1 - 1e-7)
        # 权重
        pos_weight = neg_to_pos_ratio
        neg_weight = 1
        loss = BCELossClassWeights(input=output[:, train_mask==1].view(-1,1), target=y[:,train_mask==1].view(-1,1), pos_weight=pos_weight)
        # loss = loss/(config.num_drugs*config.num_proteins)
        loss = loss/(config.numDrugs*config.numProteins)
        return_loss += loss
        # TODO 反向传播，计算当前梯度
        loss.backward()
        # TODO 根据梯度更新网络参数
        optimizer.step()
        # if batch_idx % 10 == 0:
        if batch_idx % 1 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * output.size(0),
                                                                           len(trainLoader.dataset),
                                                                           100. * batch_idx / len(trainLoader),
                                                                           loss.item()))
            sys.stdout.flush()
    return return_loss
def modelPrediction(model, device, loader, round=False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset))) # 长度等于self.num_drugs
    # torch.no_grad()用于神经网络推理阶段，无需计算梯度。
    # 它实现了__enter__和__exit__方法，进入环境管理器时记录梯度并禁止梯度计算，退出环境管理器时还原
    with torch.no_grad():
        for data in loader:
            # data = data.to(device)
            output = model(data)#.sigmoid()
            # output.cpu(): 将数据output放到cpu上
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            # graph_data.y: 即Data(图数据的构造器)中的y属性，对应一个药物和所有蛋白质的关系([1, 1, 0, 0, 1...])
            # y = torch.Tensor(np.array([graph_data.y.numpy() for graph_data in data])) # 实际的标签
            y = torch.Tensor(np.array([data.y.numpy()])).view(len(output),-1) # 实际的标签
            total_labels = torch.cat((total_labels.view(-1,1), y.view(-1, 1).float().cpu()), 0)
    if round:
        # .round()返回一个新的张量，其中input的每个元素都舍入到最接近的整数。
        # .flatten()返回一个折叠成一维的数组，该函数只能适用于numpy对象。
        return total_labels.round().numpy().flatten(), total_preds.numpy().round().flatten()
    else:
        return total_labels.numpy().flatten(), total_preds.numpy().flatten()
# class DTIGraphDataset(torch_geometric.data.Dataset):
# import torch.utils.data as data2

class DTIGraphDataset(Dataset):
    def __init__(self, data_list):
        super(DTIGraphDataset, self).__init__()
        self.data_list = data_list
    def __getitem__(self, idx):
        return self.data_list[idx]
    def get(self, idx):
        return self.data_list[idx]
    def __len__(self):
        return len(self.data_list)
    def _download(self):
        pass
    def _process(self):
        pass





































