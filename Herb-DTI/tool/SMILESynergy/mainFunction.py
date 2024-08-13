## 构建Smiles转向量类
from pretrain_trfm import TrfmSeq2seq
from pretrain_rnn import RNNSeq2Seq
from build_vocab import WordVocab
from utils import split
import numpy as np
from tool import DDITool
import pickle
import torch
# 为药物生成向量
def generateDrugVector(drug_list, mode='trfm'):
    # 得到 CIDm 和 smiles式 的映射
    drug_SMILES_dict = DDITool.getDrugToSMILESDict(inputFileName = '../../data/structureData/drug/drug_smiles')
    # drug_list = sorted(list(drug_SMILES_dict.keys()))
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    # Adapted from https://github.com/DSPsleeporg/smiles-transformer
    pretrained_model_dir = '../../data/mateData/SMILESynergy/'
    # 拿到字符对应表（用来解析SMILE式）eg: 'C': 6, '(': 7, ')': 8, 'O': 9, '=': 10, '1': 11, 'N': 12,
    vocab = WordVocab.load_vocab(pretrained_model_dir + 'vocab.pkl')
    def getInputs(sm):
        seq_len = 1004 # formerly 220
        sm = sm.split()
        # 只能处理长度小于1002的SMILE式，如果长度超过1002则取前501个字符和后501个字符
        if len(sm) > 1002: #formerly 218
            print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:501] + sm[-501:]
        # 从字符对应表里查询出每个字符对应的数字
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        # 为每个SMILE式的字符列表添加特殊的token 头部（sos）把它当作每一句话的第一个token丢进模型中 和尾部（eos）
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        # 为不满1004位的SMILE式的字符列表 填补0 补满1004位
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    # 拿到所有SMILE式的 字符转数字列表和 全是1为特征的列表
    def getArray(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = getInputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    filename = '../../model/structureFeature/' + mode + 'DrugMolecularMapping.pkl'
    if mode=='trfm':
        # seq2seq进行时间序列预测
        # in_size = len(vocab) = 45    hidden_size=256  out_size=len(vocab)  n_layers=4(编码器和解码器层数)
        trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
        # 添加 map_location=torch.device('cpu') 指定cpu运行，否则会报错
        trfm.load_state_dict(torch.load(pretrained_model_dir+'trfm.pkl', map_location=torch.device('cpu')))
        trfm.eval()
        print('Building data...')
        x_split = [split(sm) for sm in [drug_SMILES_dict[drug] for drug in drug_list]]
        xid, xseg = getArray(x_split)
        # 为每个SMILE式生成其向量表示
        X = trfm.encode(torch.t(xid)) # torch.t()是一个类似于求矩阵的转置的函数
        print('Trfm size:', X.shape)
        return_dict = {drug_list[i]: np.array(X[i,:]) for i in range(len(drug_list))}
        with open(file=filename, mode='wb') as f:
            pickle.dump(return_dict, f, pickle.HIGHEST_PROTOCOL)
    elif mode=='rnn':
        rnn = RNNSeq2Seq(len(vocab), 256, len(vocab), 3)
        # torch.load_state_dict()函数就是用于将torch.load()的预训练参数权重加载到新的模型之中
        rnn.load_state_dict(torch.load(pretrained_model_dir+'seq2seq.pkl', map_location=torch.device('cpu')))
        rnn.eval()
        print('Building data...')
        x_split = [split(sm) for sm in [drug_SMILES_dict[drug] for drug in drug_list]]
        xid, _ = getArray(x_split)
        # torch.t()是一个类似于求矩阵的转置的函数
        # xid:   被切分成单个字符的字符列表
        X = rnn.encode(torch.t(xid))
        print('RNN size:', X.shape)
        # ('CIDm00000085', array([ 0.9996368 , -0.4192314 , -0.99621403, ...,  0.09350569,
        #                          -0.19105884, -0.21773046], dtype=float32))
        return_dict = {drug_list[i]: np.array(X[i,:]) for i in range(len(drug_list))}
        with open(file=filename, mode='wb') as f:
            # 将化合物ID 和 对应的向量 写入文件
            pickle.dump(return_dict, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("No valid mode selected for drug to SMILES encoding.")
        raise ValueError
if __name__ == '__main__':
    # filepath = '../../data/entityData/drug/drug_entity_list'
    filepath = '../../data/phenotypeData/drug/drug_list'

    drug_list=[]
    with open(filepath, "r") as f:
        for line in f.readlines():
           drug_list.append(line.strip())
    # for i in drug_list:
    #     print(i)
    # 使用 Transformer模型 生成化合物的嵌入（向量表示）
    generateDrugVector(drug_list, mode='trfm')
    # 使用 RNN模型 生成化合物的嵌入（向量表示）
    generateDrugVector(drug_list, mode='rnn')
    print('Successfully computed molecular drug embeddings. Aborting rest of script.')










































