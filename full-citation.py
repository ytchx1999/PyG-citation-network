import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, JumpingKnowledge

"""
批处理：full-batch
图数据表示方法：GS
模型：GCN、GAT、SAGE、JKNet
数据集：Cora、Citeseer、Pubmed
"""

# 加载数据集
dataset = Planetoid(root='./cora/', name='Cora')
# dataset = Planetoid(root='./citeseer',name='Citeseer')
# dataset = Planetoid(root='./pubmed/', name='Pubmed')
print(dataset)
data = dataset[0]
print(data)


# 定义网络
# GCN
class GCNNet(nn.Module):
    def __init__(self, dataset, hidden=16, num_layers=2):
        """
        :param dataset: 数据集
        :param hidden: 隐藏层维度，默认16
        :param num_layers: 模型层数，默认为2
        """
        super(GCNNet, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(dataset.num_node_features, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))

        for i in range(self.num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.convs.append(GCNConv(hidden, dataset.num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            # x = self.bns[i](x)  # 小数据集不norm反而效果更好
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x


# GAT
class GATNet(nn.Module):
    def __init__(self, dataset, hidden=8, num_layers=2):
        """
        :param dataset: 数据集
        :param hidden: 隐藏层维度，默认为8
        :param num_layers: 层数，默认2层
        """
        super(GATNet, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GATConv(dataset.num_node_features, hidden, heads=8))
        self.bns.append(nn.BatchNorm1d(8 * hidden))

        for i in range(self.num_layers - 2):
            self.convs.append(GATConv(8 * hidden, 8 * hidden))
            self.bns.append(nn.BatchNorm1d(8 * hidden))

        self.convs.append(GATConv(8 * hidden, dataset.num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x


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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x


# JKNet
class JKNet(nn.Module):
    def __init__(self, dataset, mode='max', hidden=16, num_layers=6):
        super(JKNet, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(dataset.num_node_features, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))

        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = nn.Linear(hidden, dataset.num_classes)
        elif mode == 'cat':
            self.fc = nn.Linear(num_layers * hidden, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        layer_out = []  # 保存每一层的结果
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # x = self.bns[i](x)  # 小数据集不norm反而效果更好
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            layer_out.append(x)

        h = self.jk(layer_out)  # JK层
        h = self.fc(h)
        h = F.log_softmax(h, dim=1)

        return h


# 实例化模型
model = GCNNet(dataset=dataset, hidden=16, num_layers=2)
# model = GATNet(dataset=dataset, hidden=8, num_layers=2)
# model = SAGENet(dataset=dataset, hidden=16, num_layers=2)
# model = JKNet(dataset=dataset, mode='max', hidden=16, num_layers=6)
print(model)

# 转换为cpu或cuda格式
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = data.to(device)
# data = dataset[0].to(device)
# print(data)

# 定义损失函数和优化器
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


# 定义训练函数(full-batch)
def train():
    model.train()

    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, pred = torch.max(out[data.train_mask], dim=1)
    correct = (pred == data.y[data.train_mask]).sum().item()
    acc = correct / data.train_mask.sum().item()

    return loss.item(), acc


# 定义测试函数
def test():
    model.eval()

    out = model(data)
    loss = criterion(out[data.test_mask], data.y[data.test_mask])

    _, pred = torch.max(out[data.test_mask], dim=1)
    correct = (pred == data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()

    return loss.item(), acc


# 程序入口
if __name__ == '__main__':
    for epoch in range(100):
        loss, acc = train()
        print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(epoch, loss, acc))

    test_loss, test_acc = test()
    print("test_loss: {:.4f} test_acc: {:.4f}".format(test_loss, test_acc))
