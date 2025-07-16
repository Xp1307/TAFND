import os
import csv
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import HeteroData, DataLoader
from torch_geometric.nn import HGTConv

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def load_model(text_pretrained_model_name):
    pretrained_models_dir = '/data3/xupin/1_FakeNewsOnLLM/UNnamePro/PretrainedModels/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_pretrained_model = AutoModel.from_pretrained(pretrained_models_dir + text_pretrained_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_models_dir + text_pretrained_model_name)

    for param in text_pretrained_model.parameters():
        param.requires_grad = False

    return text_pretrained_model, tokenizer, device

def feature_extraction(text, text_pretrained_model, tokenizer, device):
    text_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        text_feature = text_pretrained_model(**text_inputs).last_hidden_state[:, 0, :]
        text_feature = text_feature.squeeze(0)
    return text_feature

def build_post_comment_graph(feature_dict, post_id, file_name,
                             src_nodes_id, dst_nodes_id,
                             rumor_file_names, non_rumor_file_names, save_dir):
    data = HeteroData()

    data['post'].x = torch.rand(768)
    data['post'].post_id = post_id
    if file_name in rumor_file_names:               # fake_news
        data['post'].y = 1
    elif file_name in non_rumor_file_names:         # real_news
        data['post'].y = 0
    data['post'].batch = torch.tensor([0], dtype=torch.int)
    
    data['comment'].x = torch.stack(list(feature_dict.values()), dim=0)
    data['comment'].comment_id = list(feature_dict.keys())

    cp_src_node_list = []; cp_dst_node_list = []
    cc_src_node_list = []; cc_dst_node_list = []
    
    for src_node, dst_node in zip(src_nodes_id, dst_nodes_id):
        # comment -> post
        if dst_node == post_id:
            cp_src_node_list.append(src_nodes_id.index(src_node))
            cp_dst_node_list.append(0)
        # comment -> comment
        else:
            cc_src_node_list.append(src_nodes_id.index(src_node))
            cc_dst_node_list.append(src_nodes_id.index(dst_node))
            
    # comment -> post (comments)       
    data['comment', 'comments_on', 'post'].edge_index = torch.tensor([cp_src_node_list, cp_dst_node_list], dtype=torch.int)
    # comment -> comment (replies)
    data['comment', 'replies_to', 'comment'].edge_index = torch.tensor([cc_src_node_list, cc_dst_node_list], dtype=torch.int)

    torch.save(data, os.path.join(save_dir, file_name.split('.')[0] + '.pt'))

if __name__ == '__main__':
    folder_path = 'Datamodule/dataset/weibo/weibocontentwithreactions_relationv6/'
    file_names = os.listdir(folder_path)
    
    text_pretrained_model_name = "chinese-roberta-wwm-ext"
    text_pretrained_model, tokenizer, device = load_model(text_pretrained_model_name)
    
    folder_path_1 = 'Datamodule/dataset/weibo/weibocontentwithreactions_relation/rumor'
    rumor_file_names = os.listdir(folder_path_1)

    folder_path_2 = 'Datamodule/dataset/weibo/weibocontentwithreactions_relation/non_rumor'
    non_rumor_file_names = os.listdir(folder_path_2)
    save_dir = 'Datamodule/dataset/weibo/weibocontentwithreactions_relationv6_embed/'

    for file_name in tqdm(file_names, desc='comment feature extracting and graph constructing ...', ncols=150):
        feature_dict = {}
        post_id = file_name.split('_')[1]
        src_nodes_id = []
        dst_nodes_id = []
        with open(folder_path+file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            head = next(reader)
            
            for row in reader:
                src_nodes_id.append(row[0]); dst_nodes_id.append(row[2]); text = row[1]
                feature_dict[row[0]] = feature_extraction(text, text_pretrained_model, tokenizer, device)
            build_post_comment_graph(feature_dict, post_id, file_name, 
                                     src_nodes_id, dst_nodes_id,
                                     rumor_file_names, non_rumor_file_names, save_dir)