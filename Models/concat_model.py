import torch
import torch.nn as nn
from transformers import ViTModel, RobertaModel, BertModel
from TreeModule.hgt_propagation import HgtGNN, HgtGNNPlus

# 定义联合模型, 简单的模型结构 (单纯的 concat 两个特征)
# 这里的特征是要临时提取得到
class CombinedModel(torch.nn.Module):
    def __init__(self, vit_model, roberta_model):
        super(CombinedModel, self).__init__()
        self.vit_model = vit_model
        self.roberta_model = roberta_model
        self.fc = torch.nn.Linear(768 + 768, 2)

    def forward(self, image, text_inputs):
        vit_features = self.vit_model(image).last_hidden_state[:, 0, :]
        roberta_features = self.roberta_model(**text_inputs).last_hidden_state[:, 0, :]
        # vit_features.shape = torch.Size([64, 768]
        combined_features = torch.cat((vit_features, roberta_features), dim=1)
        output = self.fc(combined_features)
        return output

# 定义联合模型, 简单的模型结构 (单纯的 concat 两个特征, 然后过一个 MLP), 这里用的特征都已经是经过了预训练模型处理后的 embeddings
# 这里的 MLP 和上面的 FC 差距不大
# 在 pheme 数据集上用 seed=101
class CombinedModel2(torch.nn.Module):
    def __init__(self, args):
        super(CombinedModel2, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        self.linear_1 = nn.Linear(args.image_dim, args.embed_dim).to(self.device)  # 图像特征线性变换
        self.linear_2 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)   # 文本特征线性变换
        # 这个模块的作用是按顺序执行里面的模块
        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_dim, int(args.hidden_dim/2)),     # 第一层全连接层
            nn.ReLU(),                                              # 激活函数
            nn.Dropout(args.dropout_rate),                          # Dropout 层
            nn.Linear(int(args.hidden_dim/2), args.output_dim),     # 第二层全连接层
        ).to(self.device)
        
    def forward(self, text_features, image_features):
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        # image_features = self.linear_1(image_features)  # 图像特征线性变换
        # text_features = self.linear_2(text_features)    # 文本特征线性变换

        combined_features = torch.cat((text_features, image_features), dim=1).to(self.device_id)    # 拼接特征
        output = self.mlp(combined_features)
        return output

# 定义联合模型, 简单的模型结构 (单纯的 concat 两个特征, 然后过一个 MLP), 这里用的特征都已经是经过了预训练模型处理后的 embeddings
# 这里的 MLP 和上面的 FC 差距不大
# 在 pheme 数据集上用 seed=101
class CombinedModel2(torch.nn.Module):
    def __init__(self, args):
        super(CombinedModel2, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        self.linear_1 = nn.Linear(args.image_dim, args.embed_dim).to(self.device)  # 图像特征线性变换
        self.linear_2 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)   # 文本特征线性变换
        
        # 这个模块的作用是按顺序执行里面的模块
        self.mlp = nn.Sequential(
            nn.Linear(args.image_dim+args.text_dim, args.hidden_dim),     # 第一层全连接层
            nn.ReLU(),                                              # 激活函数
            nn.Dropout(args.dropout_rate),                          # Dropout 层
            nn.Linear(args.hidden_dim, args.output_dim),     # 第二层全连接层
        ).to(self.device)
        
        self.mlp_single = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),     # 第一层全连接层
            nn.ReLU(),                                              # 激活函数
            nn.Dropout(args.dropout_rate),                          # Dropout 层
            nn.Linear(args.hidden_dim, args.output_dim),     # 第二层全连接层
        ).to(self.device)

        # self.mlp_single2 = nn.Sequential(
        #     nn.Linear(args.hidden_dim, 600),     # 第一层全连接层
        #     nn.ReLU(),                                              # 激活函数
        #     nn.Dropout(args.dropout_rate),                          # Dropout 层
        #     nn.Linear(600, 300),     # 第二层全连接层
        #     nn.Linear(300, args.output_dim),     # 第二层全连接层
        # ).to(self.device)
        
    def forward(self, text_features, image_features):
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        # image_features = self.linear_1(image_features)  # 图像特征线性变换
        # text_features = self.linear_2(text_features)    # 文本特征线性变换

        combined_features = torch.cat((text_features, image_features), dim=1).to(self.device_id)    # 拼接特征
        output = self.mlp(combined_features)
        
        # output = self.mlp_single(image_features)
        return output

class fakeddit_CombinedModel2(torch.nn.Module):
    def __init__(self, args):
        super(fakeddit_CombinedModel2, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        self.linear_1 = nn.Linear(args.image_dim, args.embed_dim).to(self.device)  # 图像特征线性变换
        self.linear_2 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)   # 文本特征线性变换
        # 这个模块的作用是按顺序执行里面的模块
        self.mlp = nn.Sequential(
            nn.Linear(args.image_dim+args.text_dim, args.hidden_dim),     # 第一层全连接层
            nn.ReLU(),                                              # 激活函数
            nn.Dropout(args.dropout_rate),                          # Dropout 层
            nn.Linear(args.hidden_dim, args.output_dim),     # 第二层全连接层
        ).to(self.device)
        
        self.mlp_single = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),     # 第一层全连接层
            nn.ReLU(),                                              # 激活函数
            nn.Dropout(args.dropout_rate),                          # Dropout 层
            nn.Linear(args.hidden_dim, args.output_dim),     # 第二层全连接层
        ).to(self.device)
        
    def forward(self, text_features, image_features):
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        # image_features = self.linear_1(image_features)  # 图像特征线性变换
        # text_features = self.linear_2(text_features)    # 文本特征线性变换

        combined_features = torch.cat((text_features, image_features), dim=1).to(self.device_id)    # 拼接特征
        output = self.mlp(combined_features)

        # output = self.mlp_single(text_features)
        return output

# 定义联合模型, 简单的模型结构 (单纯的 concat 两个特征, 然后过一个 MLP), 这里用的特征都已经是经过了预训练模型处理后的 embeddings
# 这里的 MLP 和上面的 FC 差距不大
# 在 pheme 数据集上用 seed=101
class CombinedModel3(torch.nn.Module):
    '''

        Intro:
        ------------
            这个模型是三个模态的特征集合, 用于对图像, 文本, 图结构三种模态的联合训练 <br/>
            这个模型能达到 90% +/- 1.1% 的准确率 <br/>
            这个模型还能达到 91.86% 的准确率, 随机种子为: 114514, 因为设置了 dropout, 所以每次训练的结果会有差异: 在同一种子下 <br/>
            两次的运行结果还会不一样: <br/>
                Accuracy: 0.9186; F1 Score: 0.9190; Recall: 0.9186; Precision: 0.9201. <br/>
                Accuracy: 0.9051; F1 Score: 0.9054; Recall: 0.9051; Precision: 0.9060. <br/>

        相关参数:
        ------------
            text_features: 文本特征
            image_features: 图像特征
            enhanced_post_features: 图结构特征
    '''
    def __init__(self, args, metadata):
        super(CombinedModel3, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        self.linear_1 = nn.Linear(args.image_dim, args.embed_dim).to(self.device)  # 图像特征线性变换
        self.linear_2 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)   # 文本特征线性变换
        self.gnn = HgtGNN(1536, args.hidden_dim, metadata, 
                          num_heads=args.num_heads_gnn).to(self.device)  # 图结构特征线性变换
        # # 下面这个 HgtGNNPlus 是一个改进版的 HgtGNN, 主要是增加了 LayerNorm 和 Dropout 层
        # self.gnn = HgtGNNPlus(1536, args.hidden_dim, metadata, 
        #                     num_heads=args.num_heads_gnn, dropout=args.dropout_rate).to(self.device)
        
        # 这个模块的作用是按顺序执行里面的模块
        self.mlp = nn.Sequential(
            nn.Linear(args.text_dim+args.image_dim+args.embed_dim, args.hidden_dim),     # 第一层全连接层, [768+768+768, 768]
            nn.ReLU(),                                                                   # 激活函数
            nn.Dropout(args.dropout_rate),                                               # Dropout 层, 不能去掉, 这个作用比较大
            nn.Linear(args.hidden_dim, args.output_dim),                                 # 第二层全连接层, [768, 2]
        ).to(self.device)

        self.mlp_single = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),     # 第一层全连接层
            nn.ReLU(),                                              # 激活函数
            nn.Dropout(args.dropout_rate),                          # Dropout 层
            nn.Linear(args.hidden_dim, args.output_dim),     # 第二层全连接层
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
        
        # output = self.mlp_single(enhanced_post_features)
        return output, combined_features
    
# 定义联合模型, 简单的模型结构 (单纯的 concat 两个特征, 然后过一个 MLP), 这里用的特征都已经是经过了预训练模型处理后的 embeddings
# 这里的 MLP 和上面的 FC 差距不大
# 在 pheme 数据集上用 seed=101
class pheme_CombinedModel3(torch.nn.Module):
    '''

        Intro:
        ------------
            这个模型是三个模态的特征集合, 用于对图像, 文本, 图结构三种模态的联合训练 <br/>

        相关参数:
        ------------
            text_features: 文本特征
            image_features: 图像特征
            enhanced_post_features: 图结构特征
    '''
    def __init__(self, args, metadata):
        super(pheme_CombinedModel3, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        self.linear_1 = nn.Linear(args.image_dim, args.embed_dim).to(self.device)  # 图像特征线性变换
        self.linear_2 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)   # 文本特征线性变换
        self.linear_3 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)   # 文本特征线性变换
        self.gnn = HgtGNN(1536, args.hidden_dim, metadata, 
                          num_heads=args.num_heads_gnn).to(self.device)  # 图结构特征线性变换
        # # 下面这个 HgtGNNPlus 是一个改进版的 HgtGNN, 主要是增加了 LayerNorm 和 Dropout 层
        # self.gnn = HgtGNNPlus(1536, args.hidden_dim, metadata, 
        #                     num_heads=args.num_heads_gnn, dropout=args.dropout_rate).to(self.device)

        # # 门控融合层
        # self.gate_txt = nn.Sequential(
        #     nn.Linear(args.embed_dim, args.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(args.embed_dim, args.embed_dim),
        #     nn.Sigmoid()
        # )
        # self.gate_img = nn.Sequential(
        #     nn.Linear(args.embed_dim, args.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(args.embed_dim, args.embed_dim),
        #     nn.Sigmoid()
        # )
        
        # 这个模块的作用是按顺序执行里面的模块
        self.mlp = nn.Sequential(
            nn.Linear(args.text_dim+args.image_dim+args.embed_dim, args.hidden_dim),     # 第一层全连接层, [768+768+768, 768]
            nn.ReLU(),                                                                   # 激活函数
            nn.Dropout(args.dropout_rate),                                               # Dropout 层, 不能去掉, 这个作用比较大
            nn.Linear(args.hidden_dim, 256),                                 # 第二层全连接层, [768, 2]
            nn.Linear(256, 128),                                             # 第二层全连接层, [768, 2]
            nn.Linear(128, args.output_dim),                                 # 第二层全连接层, [768, 2]
            # nn.Linear(256, args.output_dim),                                 # 第二层全连接层, [768, 2]
        ).to(self.device)

    
    def forward(self, text_features, image_features, comment_graph):
        ## 把特征移动到 gpu 上
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        comment_graph = comment_graph.to(self.device_id)
        
        ## 将 comment_graph 过一遍 GNN, 得到评论增强后的 post 特征
        enhanced_post_features = self.gnn(comment_graph)
        
        # ## 额外增加的部分
        # gate_t = self.gate_txt(enhanced_post_features)  # 由主模态控制
        # gate_i = self.gate_img(enhanced_post_features)
        # text_features = gate_t * text_features
        # image_features = gate_i * image_features
                
        ## 拼接三个模态的特征, 最后经过 MLP 进行分类, 并得到分类结果
        combined_features = torch.cat((text_features, image_features, enhanced_post_features), dim=1).to(self.device_id)    # 拼接特征
        output = self.mlp(combined_features)
        
        return output
    
class fakeddit_CombinedModel3(torch.nn.Module):
    '''

        Intro:
        ------------
            这个模型是三个模态的特征集合, 用于对图像, 文本, 图结构三种模态的联合训练 <br/>

        相关参数:
        ------------
            text_features: 文本特征
            image_features: 图像特征
            enhanced_post_features: 图结构特征
    '''
    def __init__(self, args, metadata):
        super(fakeddit_CombinedModel3, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        self.linear_1 = nn.Linear(args.image_dim, args.embed_dim).to(self.device)  # 图像特征线性变换
        self.linear_2 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)   # 文本特征线性变换
        self.linear_3 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)   # 文本特征线性变换
        
        self.gnn = HgtGNN(1536, args.hidden_dim, metadata, 
                          num_heads=args.num_heads_gnn).to(self.device)  # 图结构特征线性变换
        # # 下面这个 HgtGNNPlus 是一个改进版的 HgtGNN, 主要是增加了 LayerNorm 和 Dropout 层
        # self.gnn = HgtGNNPlus(1536, args.hidden_dim, metadata, 
        #                     num_heads=args.num_heads_gnn, dropout=args.dropout_rate).to(self.device)

        
        # 这个模块的作用是按顺序执行里面的模块
        self.mlp = nn.Sequential(
            nn.Linear(args.text_dim+args.image_dim+args.embed_dim, args.hidden_dim),     # 第一层全连接层, [768+768+768, 768]
            nn.ReLU(),                                                                   # 激活函数
            nn.Dropout(args.dropout_rate),                                               # Dropout 层, 不能去掉, 这个作用比较大
            nn.Linear(args.hidden_dim, 256),                                 # 第二层全连接层, [768, 2]
            nn.Linear(256, 128),                                             # 第二层全连接层, [768, 2]
            nn.Linear(128, args.output_dim),                                 # 第二层全连接层, [768, 2]
        ).to(self.device)

        self.mlp_single = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),     # 第一层全连接层
            nn.ReLU(),                                              # 激活函数
            nn.Dropout(args.dropout_rate),                          # Dropout 层
            nn.Linear(args.hidden_dim, args.output_dim),     # 第二层全连接层
        ).to(self.device)
    
    def forward(self, text_features, image_features, comment_graph):
        ## 把特征移动到 gpu 上
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        comment_graph = comment_graph.to(self.device_id)
        
        ## 将 comment_graph 过一遍 GNN, 得到评论增强后的 post 特征
        enhanced_post_features = self.gnn(comment_graph)
                
        # ## 拼接三个模态的特征, 最后经过 MLP 进行分类, 并得到分类结果
        combined_features = torch.cat((text_features, image_features, enhanced_post_features), dim=1).to(self.device_id)    # 拼接特征
        output = self.mlp(combined_features)
        
        # output = self.mlp_single(enhanced_post_features)        
        return output, combined_features

# 读取预训练模型 
def load_pretrained_models(args):    
    pretrained_models_dir = args.pretrained_models_dir
    image_pretrained_model_name = args.image_pretrained_model_name
    text_pretrained_model_name = args.text_pretrained_model_name

    device_id = args.device_id
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")  # 构建 GPU 设备
    
    vit_model = ViTModel.from_pretrained(pretrained_models_dir + image_pretrained_model_name).to(device)
    if args.text_pretrained_model_name == 'roberta-base':
        roberta_model = RobertaModel.from_pretrained(pretrained_models_dir + text_pretrained_model_name).to(device)
    elif args.text_pretrained_model_name == 'chinese-roberta-wwm-ext':
        roberta_model = BertModel.from_pretrained(pretrained_models_dir + text_pretrained_model_name).to(device)
    else:
        roberta_model = RobertaModel.from_pretrained(pretrained_models_dir + text_pretrained_model_name).to(device)        
    
    # 冻结预训练模型参数
    for param in vit_model.parameters():
        param.requires_grad = False
    for param in roberta_model.parameters():
        param.requires_grad = False
    return vit_model, roberta_model