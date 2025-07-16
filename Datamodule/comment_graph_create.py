import os
import csv
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import HeteroData, DataLoader
from torch_geometric.nn import HGTConv

# 设置只使用第 0 张 GPU（从 0 开始编号）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 读取特征提取模型、分词器
def load_model(text_pretrained_model_name):
    pretrained_models_dir = '/data3/xupin/1_FakeNewsOnLLM/UNnamePro/PretrainedModels/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测是否有 GPU

    # 加载模型和 tokenizer
    text_pretrained_model = AutoModel.from_pretrained(pretrained_models_dir + text_pretrained_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_models_dir + text_pretrained_model_name)

    # 冻结参数
    for param in text_pretrained_model.parameters():
        param.requires_grad = False

    return text_pretrained_model, tokenizer, device

def feature_extraction(text, text_pretrained_model, tokenizer, device):
    # 将输入文本处理为张量
    text_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        # 在 GPU 上进行推理
        text_feature = text_pretrained_model(**text_inputs).last_hidden_state[:, 0, :]  # 提取 [CLS] token 的表示
        text_feature = text_feature.squeeze(0)
    return text_feature

# 构造多个 HeteroData 子图（每个帖子及其评论回复）
def build_post_comment_graph(feature_dict, post_id, file_name,
                             src_nodes_id, dst_nodes_id,
                             rumor_file_names, non_rumor_file_names, save_dir):
    data = HeteroData()

    # 构建 post 节点特征
    data['post'].x = torch.rand(768)  #@ 随机初始化 post 的特征, 后续记得弥补, 补上
    data['post'].post_id = post_id
    # 标签
    if file_name in rumor_file_names:               # 如果是 fake_news
        data['post'].y = 1
    elif file_name in non_rumor_file_names:         # 如果是 real_news
        data['post'].y = 0
    data['post'].batch = torch.tensor([0], dtype=torch.int)  # 单个 post 的 batch_idx 为 0（拼接后由 DataLoader 自动赋值）
    
    # 构建 comment 节点特征
    data['comment'].x = torch.stack(list(feature_dict.values()), dim=0)
    data['comment'].comment_id = list(feature_dict.keys())

    # 构建 comment -> post
    cp_src_node_list = []; cp_dst_node_list = []
    cc_src_node_list = []; cc_dst_node_list = []
    
    # 逐行筛选边, src_nodes_id 全是关于 comment 的 id
    for src_node, dst_node in zip(src_nodes_id, dst_nodes_id):
        # 表示 comment -> post 的边
        if dst_node == post_id:
            cp_src_node_list.append(src_nodes_id.index(src_node))
            cp_dst_node_list.append(0)
        # 表示 comment -> comment 的边
        else:
            cc_src_node_list.append(src_nodes_id.index(src_node))
            cc_dst_node_list.append(src_nodes_id.index(dst_node))
            
     # comment -> post (comments)       
    data['comment', 'comments_on', 'post'].edge_index = torch.tensor([cp_src_node_list, cp_dst_node_list], dtype=torch.int)
    # comment -> comment (replies)
    data['comment', 'replies_to', 'comment'].edge_index = torch.tensor([cc_src_node_list, cc_dst_node_list], dtype=torch.int)

    torch.save(data, os.path.join(save_dir, file_name.split('.')[0] + '.pt'))

if __name__ == '__main__':
    folder_path = '/data3/xupin/1_FakeNewsOnLLM/0_Datasets/2022_MFAN_PHEME_Weibo/dataset/weibo/weibocontentwithreactions_relationv4/'
    file_names = os.listdir(folder_path)
    
    ## 读取预训练模型
    text_pretrained_model_name = "chinese-roberta-wwm-ext"
    text_pretrained_model, tokenizer, device = load_model(text_pretrained_model_name)
    
    folder_path_1 = '/data3/xupin/1_FakeNewsOnLLM/0_Datasets/2022_MFAN_PHEME_Weibo/dataset/weibo/weibocontentwithreactions_relation/rumor'
    rumor_file_names = os.listdir(folder_path_1)

    folder_path_2 = '/data3/xupin/1_FakeNewsOnLLM/0_Datasets/2022_MFAN_PHEME_Weibo/dataset/weibo/weibocontentwithreactions_relation/non_rumor'
    non_rumor_file_names = os.listdir(folder_path_2)
    save_dir = '/data3/xupin/1_FakeNewsOnLLM/0_Datasets/2022_MFAN_PHEME_Weibo/dataset/weibo/weibocontentwithreactions_relationv4_embed/'

    for file_name in tqdm(file_names, desc='comment feature extracting and graph constructing ...', ncols=150):
        feature_dict = {}
        post_id = file_name.split('_')[1]
        src_nodes_id = []
        dst_nodes_id = []
        with open(folder_path+file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            head = next(reader) # 跳过表头
            
            # 按行读取评论数据
            for row in reader:
                src_nodes_id.append(row[0]); dst_nodes_id.append(row[2]); text = row[1]
                feature_dict[row[0]] = feature_extraction(text, text_pretrained_model, tokenizer, device)
            build_post_comment_graph(feature_dict, post_id, file_name, 
                                     src_nodes_id, dst_nodes_id,
                                     rumor_file_names, non_rumor_file_names, save_dir)