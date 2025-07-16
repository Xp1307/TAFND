import os
import csv
import sys
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import HeteroData
from datetime import datetime
# 设置只使用第 0 张 GPU（从 0 开始编号）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## 对时间进行编码
## 这个 TimeEncoding 模块是你实现 时序异质图建模 的重要组件，它能把原本无法输入模型的“时间戳”，
## 转换为有意义的向量表示，极大提升模型对动态传播过程的理解能力。
## 这个模块的设计灵感来自于 Transformer 中的时间位置编码（positional encoding），
'''
    完全适合用于构建 TGAT、TGN 或任何时间敏感的图神经网络结构中。
    这个模块本质是 TGAT（Temporal Graph Attention Network） 中时间编码的标准实现。它的优势在于：
    不需要额外学习参数（非学习式时间编码）
    直接融合到 message passing 中（如 src_feat + time_feat）
    在数据稀疏/低监督场景下表现稳定
    
    这个 TimeEncoding 模块是一个 正余弦基函数构造的非学习型时间嵌入器，用于将时间戳转换为模型可用的向量，
    在图神经网络中帮助模型感知和利用时间动态结构。

    下面是一个动机: 
        ✅ 使用 TimeEncoding 可以显著增强模型对“传播时间过程”的感知力，有助于发现“早期共识”、“后期对抗” 等有趣的动态结构模式。
    GPT可以做如下部署:
        如果你想可视化它在一条传播链条中的编码表现，我也可以帮你画出时间嵌入在 2D 空间中的分布图！要不要看看？
'''
#@ 这里的 TimeEncoding 的维度设置为 768 时, 似乎不太好, 要参考 TGAT 是如何处理的
#@ 建议设置为 64 (备选为32), 可以做一个 32 和 64 的对比实验
class TimeEncoding(nn.Module):
    def __init__(self, out_channels, dropout=0.0):  ## 这里的 dropout 先设置为 0.0, 以调通整个的 pipeline, 后续再设置为0.1。
        super(TimeEncoding, self).__init__()
        self.out_channels = out_channels
        self.dropout = nn.Dropout(p=dropout)

        # 使用 log space 构建频率因子
        inv_freq = 1.0 / (10000 ** (torch.arange(0.0, out_channels, 2.0) / out_channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, time_tensor):
        """
            time_tensor: shape [E], float, 已归一化到 [0, 1] 区间
            return: shape [E, out_channels]
        """
        if len(time_tensor.shape) == 1:
            time_tensor = time_tensor.unsqueeze(1)  # [E, 1]

        sinusoid_inp = time_tensor * self.inv_freq  # [E, out_channels // 2]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return self.dropout(pos_emb)

def load_model():
    pretrained_models_dir = '/data3/xupin/1_FakeNewsOnLLM/UNnamePro/PretrainedModels/'
    text_pretrained_model_name = "chinese-roberta-wwm-ext"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测是否有 GPU

    # 加载模型和 tokenizer
    text_pretrained_model = AutoModel.from_pretrained(pretrained_models_dir + text_pretrained_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_models_dir + text_pretrained_model_name)

    # 冻结参数
    for param in text_pretrained_model.parameters():
        param.requires_grad = False

    return text_pretrained_model, tokenizer, device

# def feature_extraction(text, text_pretrained_model, tokenizer, device):
#     # 将输入文本处理为张量
#     text_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
#     with torch.no_grad():
#         # 在 GPU 上进行推理
#         text_feature = text_pretrained_model(**text_inputs).last_hidden_state[:, 0, :]  # 提取 [CLS] token 的表示
#         text_feature = text_feature.squeeze(0)
#     return text_feature

def feature_extraction(text, text_pretrained_model, tokenizer, device, pooling_type='cls'):
    # 将输入文本处理为张量
    text_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = text_pretrained_model(**text_inputs).last_hidden_state  # (1, seq_len, 768)
        if pooling_type == 'cls':
            text_feature = outputs[:, 0, :]  # (1, 768)
        elif pooling_type == 'mean':
            text_feature = outputs.mean(dim=1)  # (1, 768)
        elif pooling_type == 'max':
            text_feature = outputs.max(dim=1).values  # (1, 768)
        text_feature = text_feature.squeeze(0)  # (768)
        
    return text_feature



from datetime import datetime, timezone, timedelta

def parse_to_standard_time(raw_time) -> str:
    """
    将各种格式的时间（Unix秒数 or 微博格式字符串）统一转为 'YYYY-mm-dd HH:MM:SS' 格式字符串（北京时间）。
    
    参数:
        raw_time: 可以是 int/float（Unix秒），也可以是 str（微博格式字符串）

    返回:
        str: 转换后的 'YYYY-mm-dd HH:MM:SS' 字符串
    """
    beijing_tz = timezone(timedelta(hours=8))

    # case 1: Unix timestamp（如 1353034033）
    if isinstance(raw_time, (int, float, torch.Tensor)):
        ts = float(raw_time)
        dt = datetime.fromtimestamp(ts, tz=beijing_tz)
    # case 2: 微博字符串（如 "Sun Jul 01 17:29:06 +0800 2012"）
    elif isinstance(raw_time, str):
        raw_time = raw_time.strip()
        # 判断是否是纯数字字符串
        if raw_time.isdigit():
            ts = float(raw_time)
            dt = datetime.fromtimestamp(ts, tz=beijing_tz)
        else:
            try:
                dt = datetime.strptime(raw_time, "%a %b %d %H:%M:%S %z %Y")  # 微博格式自带时区
                dt = dt.astimezone(beijing_tz)
            except Exception as e:
                raise ValueError(f"未知时间格式: {raw_time}, 无法解析: {e}")
    else:
        raise TypeError(f"不支持的时间类型: {type(raw_time)}")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# 构造多个 HeteroData 子图（每个帖子及其评论回复）
def build_post_comment_graph(feature_dict, post_id, file_name,
                             src_nodes_id, dst_nodes_id, created_at_list,
                             rumor_file_names, non_rumor_file_names, save_dir, device, base_time, time_feat_dim):
    data = HeteroData()

    ## 1. 设置 post 节点（特征、标签、batch）
    data['post'].x = torch.rand(768)  #@ 随机初始化 post 的特征, 后续记得弥补, 补上
    data['post'].post_id = post_id
    # 标签
    if file_name in rumor_file_names:               # 如果是 fake_news
        data['post'].y = 1
    elif file_name in non_rumor_file_names:         # 如果是 real_news
        data['post'].y = 0
    data['post'].batch = torch.tensor([0], dtype=torch.int)  # 单个 post 的 batch_idx 为 0（拼接后由 HeteroData 自动赋值）
    
    ## 2. comment 节点特征和 id
    data['comment'].x = torch.stack(list(feature_dict.values()), dim=0)
    data['comment'].comment_id = list(feature_dict.keys())

    ## 3. 构建边及其时间
    cp_src_node_list = []; cp_dst_node_list = []; cp_edge_time_list = []    # comment->post
    cc_src_node_list = []; cc_dst_node_list = []; cc_edge_time_list = []    # comment->comment
    
    ## 逐行筛选边, src_nodes_id 全是关于 comment 的 id
    for src_node, dst_node, time_str in zip(src_nodes_id, dst_nodes_id, created_at_list):
        ## 获得 int 形式的时间戳, 数据集中的时间格式为 "2012-11-17 00:53:25" (for example)
        try:
            ts = int(datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M:%S").timestamp())
        except Exception as e:
            ## 这里是因为时间格式错误, 不符合时间戳
            print(f"[时间解析错误] {time_str} → {e}")
            print(file_name)
            return #@ 直接结束这个函数, 不保存这个文件
        
        # 表示 comment -> post 的边
        if dst_node == post_id:
            cp_src_node_list.append(src_nodes_id.index(src_node))
            cp_dst_node_list.append(0)
            cp_edge_time_list.append(ts)    # 添加时间
        # 表示 comment -> comment 的边
        else:
            cc_src_node_list.append(src_nodes_id.index(src_node))
            cc_dst_node_list.append(src_nodes_id.index(dst_node))
            cc_edge_time_list.append(ts)    # 添加时间

    time_encoder = TimeEncoding(out_channels=time_feat_dim).to(device)
    
    try:
        #@ min_time = base_time, 因为帖子有发布时间, 所以最早的时间, 也就是 min_time 应该是帖子的发布时间
        #@ 以帖子的时间作为 anchor。
        #@ 如果帖子没有发布时间, 那么就取评论的最早时间 edge_time.min()
        min_time = base_time
        min_time = int(datetime.strptime(min_time.strip(), "%Y-%m-%d %H:%M:%S").timestamp())
        min_time = torch.tensor(min_time, dtype=torch.float).to(device)     
           
        ## comment -> post | time
        if len(cp_edge_time_list)>0:    # 表示有这样的边
            edge_time = torch.tensor(cp_edge_time_list, dtype=torch.float).to(device)
            ## 归一化时间戳到 [0, 1] 区间, 是一个标准化的 min-max 归一化
            ## 这里的时间①是避免时间值太大, 造成模型很难收敛(次要原因)
            ## ②是强调相对时间, 忽略绝对时间, 模型应该更关注：“这个评论比最早的评论晚了多久？”
            ## 而不是：“这个评论是2013年还是2017年？”
            if edge_time.max() - min_time == 0:
                edge_time = (edge_time - min_time) / (edge_time.max() - min_time + 1e-6)  # 归一化, 并且防止分母为0
            else:
                edge_time = (edge_time - min_time) / (edge_time.max() - min_time)
            edge_time = time_encoder(edge_time)
            data['comment', 'comments_on', 'post'].edge_time = edge_time
        else:
            edge_time = torch.tensor(cp_edge_time_list, dtype=torch.float).to(device)
            data['comment', 'comments_on', 'post'].edge_time = edge_time

        ## comment -> comment | time
        if len(cc_edge_time_list)>0:    # 表示有这样的边
            edge_time = torch.tensor(cc_edge_time_list, dtype=torch.float).to(device)
            if edge_time.max() - min_time == 0:
                edge_time = (edge_time - min_time) / (edge_time.max() - min_time + 1e-6)  # 归一化, 并且防止分母为0
            else:
                edge_time = (edge_time - min_time) / (edge_time.max() - min_time)
            edge_time = time_encoder(edge_time)
            data['comment', 'replies_to', 'comment'].edge_time = edge_time
        else:
            edge_time = torch.tensor(cc_edge_time_list, dtype=torch.float).to(device)
            data['comment', 'replies_to', 'comment'].edge_time = edge_time
    except Exception as e:
        ## 这里似乎是因为csv数据为空产生的错误
        print(f"[时间编码错误] {e}")
        print(file_name)
        sys.exit(1)
  
    ## comment -> post (comments)       
    data['comment', 'comments_on', 'post'].edge_index = torch.tensor([cp_src_node_list, cp_dst_node_list], dtype=torch.int)
    ## comment -> comment (replies)
    data['comment', 'replies_to', 'comment'].edge_index = torch.tensor([cc_src_node_list, cc_dst_node_list], dtype=torch.int)


    torch.save(data, os.path.join(save_dir, file_name.split('.')[0] + '.pt'))

if __name__ == '__main__':
    #@ 记得修改下面这行路径, 来选择想要使用的数据
    folder_path = '/data3/xupin/1_FakeNewsOnLLM/0_Datasets/2022_MFAN_PHEME_Weibo/dataset/weibo/weibocontentwithreactions_relationv6_time/'
    file_names = os.listdir(folder_path)
    text_pretrained_model, tokenizer, device = load_model()
    
    folder_path_1 = '/data3/xupin/1_FakeNewsOnLLM/0_Datasets/2022_MFAN_PHEME_Weibo/dataset/weibo/weibocontentwithreactions_relation/rumor'
    rumor_file_names = os.listdir(folder_path_1)

    folder_path_2 = '/data3/xupin/1_FakeNewsOnLLM/0_Datasets/2022_MFAN_PHEME_Weibo/dataset/weibo/weibocontentwithreactions_relation/non_rumor'
    non_rumor_file_names = os.listdir(folder_path_2)
    
    pooling_type = 'max'       ## 默认为 cls, 还可选为 'mean' 或 'max'
    #@ 记得修改下面这行路径, 来选择想要保存数据的位置
    time_feat_dim = 64      ## 这个表示时间编码的维度, 建议为 32, 64d; 768d 维度太大了, 会引入太多的噪声
    save_dir = '/data3/xupin/1_FakeNewsOnLLM/UNnamePro/DataModule/dataset/weibo/weibocontentwithreactions_relationv6_time_embed_{}d_{}/'.format(time_feat_dim, pooling_type)

    print('pooling_type:', pooling_type)
    print('time_feat_dim:', time_feat_dim)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file_name in tqdm(file_names, desc='comment feature extracting, time string encoding and graph constructing ...', ncols=125):
        ## 如果已经存在文件, 则跳过
        if os.path.exists(os.path.join(save_dir, file_name.split('.')[0] + '.pt')):
            continue
        feature_dict = {}
        post_id = file_name.split('_')[1]
        src_nodes_id = []; dst_nodes_id = []; created_at_list = []
        
        #@ 获得帖子的发布时间
        with open('/data3/xupin/1_FakeNewsOnLLM/0_Datasets/2022_MFAN_PHEME_Weibo/dataset/weibo/'+\
                  'weibocontentwithimage/original-microblog/'+file_name.split('.')[0]+'.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            # print(json.dumps(data, indent=4, ensure_ascii=False))
            base_time = data['time']
            base_time = parse_to_standard_time(base_time)  # 转换为标准时间格式
            
        ## 按行读取 csv 文件, 每执行完整个流程, 就会在 weibocontentwithreactions_relationv6_time_embed 目录下生成一个 .pt 文件
        with open(folder_path+file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            head = next(reader) # 跳过表头
            
            # 按行读取评论数据
            for row in reader:
                src_nodes_id.append(row[0]); dst_nodes_id.append(row[2]); created_at_list.append(row[3])
                text = row[1]
                # feature_dict[row[0]] = feature_extraction(text, text_pretrained_model, tokenizer, device)
                feature_dict[row[0]] = feature_extraction(text, text_pretrained_model, tokenizer, device, pooling_type)
            build_post_comment_graph(feature_dict, post_id, file_name, 
                                     src_nodes_id, dst_nodes_id, created_at_list,
                                     rumor_file_names, non_rumor_file_names, save_dir, device, base_time, time_feat_dim)