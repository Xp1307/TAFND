# RGCN on a heterogeneous post-comment graph (PyTorch Geometric)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

# ----- 1. 模拟数据定义 ----- #
num_posts = 3
num_comments = 5
hidden_dim = 64

data = {}

# 初始化节点特征（可以用ViT/Roberta提取的向量替换）
post_x = torch.randn(num_posts, hidden_dim)
comment_x = torch.randn(num_comments, hidden_dim)

# 拼接节点特征
x = torch.cat([post_x, comment_x], dim=0)  # [num_nodes, hidden_dim]

# 边类型映射
rel_map = {
    ('post', 'has_comment', 'comment'): 0,
    ('comment', 'replies_to', 'comment'): 1
}

# 构造边（post → comment）
edge_index_pc = torch.tensor([
    [0, 1, 2],        # post idx
    [0 + num_posts, 1 + num_posts, 2 + num_posts]   # comment idx
], dtype=torch.long)
edge_type_pc = torch.full((edge_index_pc.size(1),), rel_map[('post', 'has_comment', 'comment')], dtype=torch.long)

# 构造边（comment → comment）
edge_index_cc = torch.tensor([
    [0 + num_posts, 1 + num_posts, 2 + num_posts],
    [3 + num_posts, 3 + num_posts, 4 + num_posts]
], dtype=torch.long)
edge_type_cc = torch.full((edge_index_cc.size(1),), rel_map[('comment', 'replies_to', 'comment')], dtype=torch.long)

# 合并边
edge_index = torch.cat([edge_index_pc, edge_index_cc], dim=1)
edge_type = torch.cat([edge_type_pc, edge_type_cc], dim=0)

# 标签与掩码（只对帖子做分类）
y = torch.tensor([0, 1, 0] + [0] * num_comments)  # only first 3 have real labels
train_mask = torch.tensor([1, 0, 1] + [0] * num_comments, dtype=torch.bool)


# ----- 2. 模型定义 ----- #
class RGCNModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations)
        self.classifier = nn.Linear(out_channels, 2)  # Fake / Real

    def forward(self, x, edge_index, edge_type, train_mask, y):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type).relu()
        out = self.classifier(x[train_mask])
        loss = F.cross_entropy(out, y[train_mask])
        return loss, out


# ----- 3. 训练示例 ----- #
model = RGCNModel(hidden_dim, hidden_dim, num_relations=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    loss, out = model(x, edge_index, edge_type, train_mask, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
