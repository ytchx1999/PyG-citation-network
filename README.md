# PyG-citation-network
GNN模型在引文网络数据集上的代码，包括Cora、Citeseer、Pubmed、ogbn-arxiv

### Cora、Citeseer、Pubmed

| 文件/文件夹      | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| full-citation.py | **批处理：** full-batch ；**图数据表示方法：** GS ；**模型：** GCN、GAT、SAGE、JKNet |
| mini-citation.py | **批处理：** mini-batch + 2-hop采样[10，10] ；**图数据表示方法：** GS ；**模型：** GraphSAGE |

### ogbn-arxiv

| 文件/文件夹  | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| full-ogbn.py | **批处理：** full-batch ；**图数据表示方法：** SpMM ；**模型：** GCN、GAT、SAGE、JKNet、GCN_res |
| mini-ogbn.py | **批处理：** mini-batch + 3-hop采样[15，10，5] ；**图数据表示方法：** SpMM ；**模型：** GraphSAGE |
