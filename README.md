# Herb-DTI v1.0: 
### 基于异构网络和深度学习预测药物-靶标相互作用

## 环境版本要求
`Python 3.7` packages:
```
- pytorch 1.6+
- pytorch-geometric 1.6+
- numpy 1.19+
- scikit-learn 
- networkx
- gensim
- rdflib
- BioPython
- tqdm
- Groovy (Groovy: 2.4.10 JVM: 1.8.0_121)
- diamond & blastp
```

## 数据集

蛋白质统一表示格式（样例，与STRING数据库保持一致）：`9606.ENSP00000000412`
化合物同意表示格式（样例，与STITCH数据库保持一致）：`CIDm00000143`

根据以下链接下载数据并放到`data/databaseData/STRING/`文件夹下：
https://stringdb-static.org/download/protein.links.full.v11.0/9606.protein.links.full.v11.0.txt.gz
https://stringdb-static.org/download/protein.aliases.v11.0/9606.protein.aliases.v11.0.txt.gz
https://stringdb-static.org/download/protein.sequences.v11.0/9606.protein.sequences.v11.0.fa.gz

根据以下链接下载数据并放到`data/databaseData/STITCH/`文件夹下：
http://stitch.embl.de/download/protein_chemical.links.transfer.v5.0/9606.protein_chemical.links.transfer.v5.0.tsv.gz
http://stitch.embl.de/download/actions.v5.0/9606.actions.v5.0.tsv.gz
http://stitch.embl.de/download/chemical.aliases.v5.0.tsv.gz
http://stitch.embl.de/download/chemicals.v5.0.tsv.gz

DeepGOPlus和SMILESynergy根据官方文档说明将下载的数据放到`data/mateData`各自对应的文件夹当中
个人训练模型数据集中，蛋白质数据集请发在`data/entityData/target/GO_entity_list`文件中，化合物数据集请发在`data/entityData/drug/drug_entity_list`文件中

## 怎样运行

所有需要运行的脚本都在`tool/`文件夹或`main/`文件夹中提供，其中`modelMain.py`为模型主程序。
1.运行DeepGOPlus和SMILESynergy为蛋白质和化合物的结构数据生成向量
2.运行`tool/Herb2vec/mainFunction.py`脚本,为蛋白质和化合物的表型特征生成向量
3.运行`main/modelPreTrain.py`脚本,对模型进行预训练
4.运行`main/modelMain.py`脚本,开始训练模型
5.模型的训练预测结果及评级指标信息会存放在`model/resultDTIModel/`文件夹中。


