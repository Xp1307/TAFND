import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

class THgtGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, metadata, args, num_heads=8):
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
                        x=[13396, 768],
                        comment_id=[64],
                        batch=[13396],
                        ptr=[65],
                    },
                    (comment, comments_on, post)={
                        edge_time=[9942, 768],
                        edge_index=[2, 9942],
                    },
                    (comment, replies_to, comment)={
                        edge_time=[3454, 768],
                        edge_index=[2, 3454],
                    }
                )
        '''
        super().__init__()
        # 映射层, 将 post.x 从 1536 映射到 768, 目的是统一节点(nodes)的特征维度
        self.post_linear = nn.Linear(input_dim, hidden_dim)
        # 映射层, 将 comment.x 从 768 + time_feature_dim 映射到 768, 目的是统一节点(nodes)的特征维度
        self.comment_linear = nn.Linear(hidden_dim + args.time_feature_dim, hidden_dim)
        
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
        edge_time_dict = data.edge_time_dict                  # 边时间字典

        ## post 节点特征线性映射
        x_dict['post'] = self.post_linear(x_dict['post'])       # [64, 768]
        
        ## comment 节点特征线性映射, 原始特征 concat 上时间特征, 最后的效果会差一点
        comment_data = x_dict['comment']
        N = comment_data.size(0)  # 评论节点个数
        comment_data_time = torch.zeros(N, 768 + 64, device=comment_data.device)  # 初始化新特征
        for edge_index, edge_time in zip(edge_index_dict.values(), edge_time_dict.values()):
            src = edge_index[0]
            concat_feat  = torch.cat([comment_data[src], edge_time.to(comment_data.device)], dim=-1)
            comment_data_time[src] = concat_feat   
        x_dict['comment'] = comment_data_time                       # 让 评论数据添加时间特征
        x_dict['comment'] = self.comment_linear(x_dict['comment'])  # 过一遍线性层 832->768
        
        x_dict = self.conv1(x_dict, edge_index_dict)            # 第一层 HGTConv
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}      # 激活函数, 所有的节点都过了一遍非线性激活层
        x_dict = self.conv2(x_dict, edge_index_dict)            # 第二层 HGTConv
        return x_dict['post']                                   # 返回更新后的 post 节点特征
