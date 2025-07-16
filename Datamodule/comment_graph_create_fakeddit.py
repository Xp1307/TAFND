import os
import csv
import json
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import HeteroData

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_model(text_pretrained_model_name):
    pretrained_models_dir = 'PretrainedModels/'
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
        if dst_node == post_id:
            cp_src_node_list.append(src_nodes_id.index(src_node))
            cp_dst_node_list.append(0)
        else:
            cc_src_node_list.append(src_nodes_id.index(src_node))
            cc_dst_node_list.append(src_nodes_id.index(dst_node))
            
    # comment -> post (comments)       
    data['comment', 'comments_on', 'post'].edge_index = torch.tensor([cp_src_node_list, cp_dst_node_list], dtype=torch.int)
    # comment -> comment (replies)
    data['comment', 'replies_to', 'comment'].edge_index = torch.tensor([cc_src_node_list, cc_dst_node_list], dtype=torch.int)

    torch.save(data, os.path.join(save_dir, file_name.split('.')[0] + '.pt'))


def comment_embedding_generator(args, version=7):
    folder_path = 'DataModule/dataset/fakeddit/fakedditcontentwithreactions_relationv{}/'.format(version)
    file_names = os.listdir(folder_path)
    
    failed_file_name_list = []
    text_pretrained_model_name = args.text_pretrained_model_name
    text_pretrained_model, tokenizer, device = load_model(text_pretrained_model_name)
    
    rumor_file_names = []                                  
    non_rumor_file_names = []        
    for dataset_split in ['train', 'val', 'test']:
        temp_dir = 'DataModule/dataset/fakeddit/data_json'
        with open(os.path.join(temp_dir, dataset_split+'.json'), 'r', encoding='utf-8') as f:
            data_json = json.load(f)
            for key, value in data_json.items():
                if value['tweet_label']==0:  
                    non_rumor_file_names.append(key+'.csv')  # real_news      
                elif value['tweet_label']==1:
                    rumor_file_names.append(key+'.csv')         
    
    save_dir = 'DataModule/data/fakeddit_comment_tree_graph/'

    for file_name in tqdm(file_names, desc='comment feature extracting and graph constructing ...', ncols=150):
        feature_dict = {}
        post_id = file_name.split('.')[0]
        src_nodes_id = []
        dst_nodes_id = []
        with open(folder_path+file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            for row in reader:
                src_nodes_id.append(row[0]); dst_nodes_id.append(row[2]); text = row[1]
                feature_dict[row[0]] = feature_extraction(text, text_pretrained_model, tokenizer, device)

            try:
                build_post_comment_graph(feature_dict, post_id, file_name, 
                                        src_nodes_id, dst_nodes_id,
                                        rumor_file_names, non_rumor_file_names, save_dir)
            except:
                print(file_name)
                failed_file_name_list.append(file_name)
    print('failed file names:', failed_file_name_list)    