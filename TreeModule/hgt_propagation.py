import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

class HgtGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, metadata, num_heads=8):
        '''
            数据形式(示例如下):
            -------
                HeteroDataBatch(
                post={
                    x=[64, 1536],
                    post_id=[64],
                    y=[64],
                    batch=[64],
                    text_feature=[64, 768],
                    image_feature=[64, 768],
                    ptr=[65],
                },
                comment={
                    x=[13635, 768],
                    comment_id=[64],
                    batch=[13635],
                    ptr=[65],
                },
                (comment, comments_on, post)={ edge_index=[2, 10118] },
                (comment, replies_to, comment)={ edge_index=[2, 3525] }
                )
        '''
        super().__init__()
        # 映射层, 将 post.x 从 1536 映射到 768, 目的是统一节点(nodes)的特征维度
        self.post_linear = nn.Linear(input_dim, hidden_dim)
        # 定义两层 HGTConv, 768->768
        self.conv1 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)
        self.conv2 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)

    def forward(self, data):
        """
            Parameters:
            ----------
                data : HeteroDataBatch
                    包含节点特征和边索引的异质图数据。
                    
            Returns:
            ----------
                torch.Tensor
                    更新后的 post 节点特征。
        """
        ## 获取节点特征和边索引, 后面的 self.conv1, self.conv2 会用到
        x_dict = data.x_dict                                    # 节点特征字典
        edge_index_dict = data.edge_index_dict                  # 边索引字典
        x_dict['post'] = self.post_linear(x_dict['post'])       # [64, 768]
        x_dict = self.conv1(x_dict, edge_index_dict)            # 第一层 HGTConv
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}      # 激活函数, 所有的节点都过了一遍非线性激活层
        x_dict = self.conv2(x_dict, edge_index_dict)            # 第二层 HGTConv
        return x_dict['post']                                   # 返回更新后的 post 节点特征

class HgtGNNPlus(nn.Module):
    def __init__(self, input_dim, hidden_dim, metadata, num_heads=8, dropout=0.2):
        super().__init__()

        # 统一 post 节点维度（假设只有 post 节点需要映射）
        self.post_linear = nn.Linear(input_dim, hidden_dim)

        # 可扩展：为其他节点类型也定义 Linear 层
        # self.comment_linear = nn.Linear(input_dim, hidden_dim)

        # GNN 层
        self.conv1 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        '''
            模块 | 功能
            LayerNorm | 每层归一化，防止梯度爆炸
            Dropout | 每层防止过拟合，提升鲁棒性
            Residual 残差连接 | 防止 GNN over-smoothing、提升训练稳定性
            模块结构清晰 | 每层结构 Linear → Norm → HGT → Residual，便于扩展
            可控 dropout 和 num_heads 参数 | 易于调参或写 Grid Search 脚本
        '''
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        ## --- 线性映射和归一化 --- ##
        x_dict['post'] = self.post_linear(x_dict['post'])
        x_dict['post'] = self.norm1(x_dict['post'])

        ## --- Layer 1 --- ##
        res1 = x_dict  # residual input
        x1 = self.conv1(x_dict, edge_index_dict)
        ## 对每个节点进行残差链接、dropout、激活函数、归一化
        x1 = {k: self.dropout(F.relu(self.norm1(x1[k] + res1[k]))) for k in x1} 
        
        ## --- Layer 2 --- ##
        res2 = x1
        x2 = self.conv2(x1, edge_index_dict)
        ## 对每个节点进行残差链接、dropout、激活函数、归一化
        x2 = {k: self.dropout(F.relu(self.norm2(x2[k] + res2[k]))) for k in x2}

        return x2['post']  # 返回增强后的 post 节点特征
