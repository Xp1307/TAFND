import torch
import torch.nn as nn
# 因为主要运行文件是 train_weibo_threemodal_time.py, 所以主路径也就是它所在的文件夹下。也就可以用相对路径引入模型
from TreeModule.temporal_gnn_attention import HeteroTGAT
from TreeModule.hgt_propagation_time import THgtGNN
from TreeModule.hgt_propagation_time_aggr import THgtGNN_Aggr

class WeiboModel1(torch.nn.Module):
    '''

        Intro:
        ------------
            这个模型是三个模态的特征集合, 用于对图像, 文本, 图结构三种模态的联合训练 <br/>
            此外, 这里的这个模型还引入了时序特征(时间信息)

        相关参数:
        ------------
            text_features: 文本特征
            image_features: 图像特征
            enhanced_post_features: 图结构特征
    '''
    def __init__(self, args, metadata):
        super(WeiboModel1, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        
        # 使用异质时序图神经网络, 引入了时间概念, 时间特征作为了评论特征的补充        
        # 这里则是将时间信息融入到了聚合过程中
        self.gnn = THgtGNN_Aggr(1536, args.hidden_dim, metadata, args,
                            num_heads=args.num_heads_gnn).to(self.device)
        
        # 这个模块的作用是按顺序执行里面的模块
        self.mlp = nn.Sequential(
            nn.Linear(args.text_dim+args.image_dim+args.embed_dim, args.hidden_dim),     # 第一层全连接层, [768+768+768, 768]
            nn.ReLU(),                                                                   # 激活函数
            nn.Dropout(args.dropout_rate),                                               # Dropout 层, 不能去掉, 这个作用比较大
            nn.Linear(args.hidden_dim, 256),                                             # 第二层全连接层, [768, 2]
            nn.ReLU(),                                                                   # 激活函数
            nn.Linear(256, args.output_dim),                                             # 第二层全连接层, [768, 2]
        ).to(self.device)
        
    def forward(self, text_features, image_features, comment_graph):
        ## 把特征移动到 gpu 上
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        comment_graph = comment_graph.to(self.device_id)
        
        ## 将 comment_graph 过一遍 GNN, 得到评论增强后的 post 特征
        enhanced_post_features = self.gnn(comment_graph)
        
        ## 拼接三个模态的特征, 最后经过 MLP 进行分类, 并得到分类结果
        combined_features = torch.cat((text_features, image_features, enhanced_post_features), dim=1).to(self.device_id)    # 拼接特征
        output = self.mlp(combined_features)
        return output, combined_features
    

class FakedditModel1(torch.nn.Module):
    '''

        Intro:
        ------------
            这个模型是三个模态的特征集合, 用于对图像, 文本, 图结构三种模态的联合训练 <br/>
            此外, 这里的这个模型还引入了时序特征(时间信息)

        相关参数:
        ------------
            text_features: 文本特征
            image_features: 图像特征
            enhanced_post_features: 图结构特征
    '''
    def __init__(self, args, metadata):
        super(FakedditModel1, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        
        # 使用异质时序图神经网络, 引入了时间概念, 时间特征作为了评论特征的补充        
        # 这里则是将时间信息融入到了聚合过程中
        self.gnn = THgtGNN_Aggr(1536, args.hidden_dim, metadata, args,
                            num_heads=args.num_heads_gnn).to(self.device)
        
        # 这个模块的作用是按顺序执行里面的模块
        self.mlp = nn.Sequential(
            nn.Linear(args.text_dim+args.image_dim+args.embed_dim, args.hidden_dim),     # 第一层全连接层, [768+768+768, 768]
            nn.ReLU(),                                                                   # 激活函数
            nn.Dropout(args.dropout_rate),                                               # Dropout 层, 不能去掉, 这个作用比较大
            nn.Linear(args.hidden_dim, 256),                                             # 第二层全连接层, [768, 2]
            nn.ReLU(),                                                                   # 激活函数
            nn.Linear(256, args.output_dim),                                             # 第二层全连接层, [768, 2]
        ).to(self.device)
        
    def forward(self, text_features, image_features, comment_graph):
        ## 把特征移动到 gpu 上
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        comment_graph = comment_graph.to(self.device_id)
        
        ## 将 comment_graph 过一遍 GNN, 得到评论增强后的 post 特征
        enhanced_post_features = self.gnn(comment_graph)
        
        ## 拼接三个模态的特征, 最后经过 MLP 进行分类, 并得到分类结果
        combined_features = torch.cat((text_features, image_features, enhanced_post_features), dim=1).to(self.device_id)    # 拼接特征
        output = self.mlp(combined_features)
        return output, combined_features



class PhemeModel1(torch.nn.Module):
    '''

        Intro:
        ------------
            这个模型是三个模态的特征集合, 用于对图像, 文本, 图结构三种模态的联合训练 <br/>
            此外, 这里的这个模型还引入了时序特征(时间信息)

        相关参数:
        ------------
            text_features: 文本特征
            image_features: 图像特征
            enhanced_post_features: 图结构特征
    '''
    def __init__(self, args, metadata):
        super(PhemeModel1, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        
        # 使用异质时序图神经网络, 引入了时间概念, 时间特征作为了评论特征的补充 这里则是将时间信息融入到了聚合过程中
        self.gnn = THgtGNN_Aggr(1536, args.hidden_dim, metadata, args,
                            num_heads=args.num_heads_gnn).to(self.device)
        
        # 这个模块的作用是按顺序执行里面的模块
        self.mlp = nn.Sequential(
            nn.Linear(args.text_dim+args.image_dim+args.embed_dim, args.hidden_dim),     # 第一层全连接层, [768+768+768, 768]
            nn.ReLU(),                                                                   # 激活函数
            nn.Dropout(args.dropout_rate),                                               # Dropout 层, 不能去掉, 这个作用比较大
            nn.Linear(args.hidden_dim, 256),                                 # 第二层全连接层, [768, 2]
            nn.Linear(256, 128),                                             # 第二层全连接层, [768, 2]
            nn.Linear(128, args.output_dim),                                 # 第二层全连接层, [768, 2]
        ).to(self.device)
        
    def forward(self, text_features, image_features, comment_graph):
        ## 把特征移动到 gpu 上
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        comment_graph = comment_graph.to(self.device_id)
        
        ## 将 comment_graph 过一遍 GNN, 得到评论增强后的 post 特征
        enhanced_post_features = self.gnn(comment_graph)
        
        ## 拼接三个模态的特征, 最后经过 MLP 进行分类, 并得到分类结果
        combined_features = torch.cat((text_features, image_features, enhanced_post_features), dim=1).to(self.device_id)    # 拼接特征
        output = self.mlp(combined_features)
        return output