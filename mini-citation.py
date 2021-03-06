import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, JumpingKnowledge
from torch_geometric.data import NeighborSampler

"""
批处理：mini-batch
    + 采样方法：2-hop采样
    + 采样数量：[10，10]
图数据表示方法：GS
模型：GraphSAGE（minibatch）
数据集：Cora、Citeseer、Pubmed
"""

# 加载数据集
dataset = Planetoid(root='./cora/', name='Cora')
# dataset = Planetoid(root='./citeseer',name='Citeseer')
# dataset = Planetoid(root='./pubmed/', name='Pubmed')
print(dataset)
data = dataset[0]
print(data)

# 邻居采样
# 用于训练
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                               sizes=[10, 10], batch_size=16, shuffle=True,
                               num_workers=12)
# 用于推理
subgraph_loader = NeighborSampler(dataset[0].edge_index, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=12)


# 定义网络
# GraphSAGE
# SAGE
class SAGENet(nn.Module):
    def __init__(self, dataset, hidden=16, num_layers=2):
        super(SAGENet, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(dataset.num_node_features, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))

        for i in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.convs.append(SAGEConv(hidden, dataset.num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                # x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        x = F.log_softmax(x, dim=1)

        return x

    def inference(self, x_all):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    # x = self.bns[i](x)
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all


# 实例化模型
model = SAGENet(dataset=dataset, hidden=16, num_layers=2)
print(model)

# 转换为cpu或cuda格式
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = data.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)

# 定义损失函数和优化器
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# 定义训练函数(minibatch)
def train():
    model.train()

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


# 定义测试函数
@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


if __name__ == '__main__':
    for epoch in range(10):
        loss, acc = train()
        print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(epoch, loss, acc))

        train_acc, val_acc, test_acc = test()
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')
