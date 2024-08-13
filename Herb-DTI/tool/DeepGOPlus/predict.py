## 构建蛋白质转向量模型类 ##
import click as ck
import numpy as np
import pandas as pd
import pickle
from tensorflow.python.keras.models import load_model, Model
import time
from utils import Ontology
from aminoacids import to_onehot
from tqdm import tqdm
MAXLEN = 2000
@ck.command()
# @ck.option('--in-file', '-if', help='Input FASTA file', required=True)
@ck.option('--in-file', '-if', help='Input FASTA file')
@ck.option('--out-file', '-of', default='../../model/structureFeature/proteinSequenceMapping', help='Output result file')
@ck.option('--go-file', '-gf', default='../../data/mateData/DeepGOPlus/data/go.obo', help='Gene Ontology')
@ck.option('--model-file', '-mf', default='../../data/mateData/DeepGOPlus/data/model.h5', help='Tensorflow model file')
@ck.option('--terms-file', '-tf', default='../../data/mateData/DeepGOPlus/data/terms.pkl', help='List of predicted terms')
@ck.option('--annotations-file', '-tf', default='../../data/mateData/DeepGOPlus/data/train_data.pkl', help='Experimental annotations')
@ck.option('--chunk-size', '-cs', default=1000, help='Number of sequences to read at a time')
@ck.option('--threshold', '-t', default=0.0, help='Prediction threshold')
@ck.option('--batch-size', '-bs', default=128, help='Batch size for prediction model')
@ck.option('--alpha', '-a', default=0.5, help='Alpha weight parameter')

def main(in_file, out_file, go_file, model_file, terms_file, annotations_file,
         chunk_size, threshold, batch_size, alpha):
    in_file = '../../data/mateData/DeepGOPlus/PPI_graph_protein_seqs.fasta'
    # Load GO and read list of all terms
    go = Ontology(go_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    # Read known experimental annotations
    annotations = {}
    df = pd.read_pickle(annotations_file)
    for row in df.itertuples():
        annotations[row.proteins] = set(row.annotations)
    diamond_preds = {}
    mapping = {}
    with open('../../data/mateData/DeepGOPlus/data/diamond.csv', 'r') as f:
        for line in f:
            it = line.strip('\n').split(',')
            if it[0] not in mapping:
                mapping[it[0]] = {}
            mapping[it[0]][it[1]] = float(it[2])
    print("Building proteins ids...")
    for prot_id, sim_prots in mapping.items():
        annots = {}
        allgos = set()
        total_score = 0.0
        for p_id, score in sim_prots.items():
            allgos |= annotations[p_id]
            total_score += score
        allgos = list(sorted(allgos))
        sim = np.zeros(len(allgos), dtype=np.float32)
        for j, go_id in enumerate(allgos):
            s = 0.0
            for p_id, score in sim_prots.items():
                if go_id in annotations[p_id]:
                    s += score
            sim[j] = s / total_score
        for go_id, score in zip(allgos, sim):
            annots[go_id] = score
        diamond_preds[prot_id] = annots
    # Load CNN model
    model = load_model(model_file)
    truncated_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    encoding_dict = {}
    print("\nPredicting sequences...")
    print('Iterations:', sum(1 for _ in read_fasta(in_file, chunk_size)))
    for prot_ids, sequences in tqdm(read_fasta(in_file, chunk_size)):
        ids, data = get_data(sequences)
        preds = truncated_model.predict(data, batch_size=batch_size)
        for i in range(len(prot_ids)):
            encoding_dict[prot_ids[i]] = preds[i, :]
    print(len(encoding_dict.keys()))
    print('Done.')
    # filename = '../../model/structureFeature/proteinSequenceMapping'
    filename = out_file
    with open(file=filename+'.pkl', mode='wb') as f:
        pickle.dump(encoding_dict, f, pickle.HIGHEST_PROTOCOL)
    return
    # 结束返回只要蛋白质的向量表示
def read_fasta(filename, chunk_size):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    info.append(inf)
                    if len(info) == chunk_size:
                        yield (info, seqs)
                        seqs = list()
                        info = list()
                    seq = ''
                inf = line[1:].split()[0]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    yield (info, seqs)
def get_data(sequences):
    pred_seqs = []
    ids = []
    for i, seq in enumerate(sequences):
        if len(seq) > MAXLEN:
            st = 0
            while st < len(seq):
                pred_seqs.append(seq[st: st + MAXLEN])
                ids.append(i)
                st += MAXLEN - 128
        else:
            pred_seqs.append(seq)
            ids.append(i)
    n = len(pred_seqs)
    data = np.zeros((n, MAXLEN, 21), dtype=np.float32)
    
    for i in range(n):
        seq = pred_seqs[i]
        data[i, :, :] = to_onehot(seq)
    return ids, data
# if __name__ == '__main__':
#     main()
