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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TimeEncoding(nn.Module):
    def __init__(self, out_channels, dropout=0.0):
        super(TimeEncoding, self).__init__()
        self.out_channels = out_channels
        self.dropout = nn.Dropout(p=dropout)

        inv_freq = 1.0 / (10000 ** (torch.arange(0.0, out_channels, 2.0) / out_channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, time_tensor):
        if len(time_tensor.shape) == 1:
            time_tensor = time_tensor.unsqueeze(1)  # [E, 1]

        sinusoid_inp = time_tensor * self.inv_freq  # [E, out_channels // 2]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return self.dropout(pos_emb)

def load_model(text_pretrained_model_name):
    pretrained_models_dir = 'PretrainedModels/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_pretrained_model = AutoModel.from_pretrained(pretrained_models_dir + text_pretrained_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_models_dir + text_pretrained_model_name)

    for param in text_pretrained_model.parameters():
        param.requires_grad = False

    return text_pretrained_model, tokenizer, device

def feature_extraction(text, text_pretrained_model, tokenizer, device, pooling_type='cls'):
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
    return:
        str: 'YYYY-mm-dd HH:MM:SS'
    """
    beijing_tz = timezone(timedelta(hours=8))

    # case 1: Unix timestamp（1353034033）
    if isinstance(raw_time, (int, float, torch.Tensor)):
        ts = float(raw_time)
        dt = datetime.fromtimestamp(ts, tz=beijing_tz)
    # case 2: UTC-TIME | 2018-04-15 03:55:13
    elif isinstance(raw_time, str):
        raw_time = raw_time.strip()
        
        if raw_time.isdigit():
            ts = float(raw_time)
            dt = datetime.fromtimestamp(ts, tz=beijing_tz)
        else:
            try:
                dt = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
                dt = dt.astimezone(beijing_tz)
            except Exception as e:
                raise ValueError(f"unknown time format: {raw_time}, can't parse: {e}")
    else:
        raise TypeError(f"Unsuported time type: {type(raw_time)}")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def build_post_comment_graph(feature_dict, post_id, file_name,
                             src_nodes_id, dst_nodes_id, created_at_list,
                             rumor_file_names, non_rumor_file_names, save_dir, device, base_time, time_feat_dim):
    data = HeteroData()
    data['post'].x = torch.rand(768)
    data['post'].post_id = post_id

    if file_name in rumor_file_names:               
        data['post'].y = 1
    elif file_name in non_rumor_file_names:        
        data['post'].y = 0
        
    data['post'].batch = torch.tensor([0], dtype=torch.int)
    
    data['comment'].x = torch.stack(list(feature_dict.values()), dim=0)
    data['comment'].comment_id = list(feature_dict.keys())

    cp_src_node_list = []; cp_dst_node_list = []; cp_edge_time_list = []    # comment->post
    cc_src_node_list = []; cc_dst_node_list = []; cc_edge_time_list = []    # comment->comment

    for src_node, dst_node, time_str in zip(src_nodes_id, dst_nodes_id, created_at_list):
        try:
            ts = int(datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M:%S").timestamp())
        except Exception as e:
            print(f"[error!] {time_str} → {e}")
            print(file_name)
            return
        
        # comment -> post
        if dst_node == post_id:
            cp_src_node_list.append(src_nodes_id.index(src_node))
            cp_dst_node_list.append(0)
            cp_edge_time_list.append(ts)
        # comment -> comment
        else:
            cc_src_node_list.append(src_nodes_id.index(src_node))
            cc_dst_node_list.append(src_nodes_id.index(dst_node))
            cc_edge_time_list.append(ts)

    time_encoder = TimeEncoding(out_channels=time_feat_dim).to(device)
    
    try:
        min_time = base_time
        min_time = int(datetime.strptime(min_time.strip(), "%Y-%m-%d %H:%M:%S").timestamp())
        min_time = torch.tensor(min_time, dtype=torch.float).to(device)     
           
        if len(cp_edge_time_list)>0:
            edge_time = torch.tensor(cp_edge_time_list, dtype=torch.float).to(device)
            if edge_time.max() - min_time == 0:
                edge_time = (edge_time - min_time) / (edge_time.max() - min_time + 1e-6)
            else:
                edge_time = (edge_time - min_time) / (edge_time.max() - min_time)
            edge_time = time_encoder(edge_time)
            data['comment', 'comments_on', 'post'].edge_time = edge_time
        else:
            edge_time = torch.tensor(cp_edge_time_list, dtype=torch.float).to(device)
            data['comment', 'comments_on', 'post'].edge_time = edge_time

        if len(cc_edge_time_list)>0:
            edge_time = torch.tensor(cc_edge_time_list, dtype=torch.float).to(device)
            if edge_time.max() - min_time == 0:
                edge_time = (edge_time - min_time) / (edge_time.max() - min_time + 1e-6)
            else:
                edge_time = (edge_time - min_time) / (edge_time.max() - min_time)
            edge_time = time_encoder(edge_time)
            data['comment', 'replies_to', 'comment'].edge_time = edge_time
        else:
            edge_time = torch.tensor(cc_edge_time_list, dtype=torch.float).to(device)
            data['comment', 'replies_to', 'comment'].edge_time = edge_time
    except Exception as e:
        print(f"[时间编码错误] {e}")
        print(file_name)
        sys.exit(1)
  
    ## comment -> post (comments)       
    data['comment', 'comments_on', 'post'].edge_index = torch.tensor([cp_src_node_list, cp_dst_node_list], dtype=torch.int)
    ## comment -> comment (replies)
    data['comment', 'replies_to', 'comment'].edge_index = torch.tensor([cc_src_node_list, cc_dst_node_list], dtype=torch.int)


    torch.save(data, os.path.join(save_dir, file_name.split('.')[0] + '.pt'))

def comment_time_embedding_generator(args, version=4):
    folder_path = 'DataModule/dataset/fakeddit/fakedditcontentwithreactions_relationv{}/'.format(version)
    file_names = os.listdir(folder_path)
    
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
    
    pooling_type = 'mean'
    time_feat_dim = args.time_feature_dim
    save_dir = 'DataModule/dataset/fakeddit/fakedditcontentwithreactions_relationv{}_time_embed_{}d_{}/'.format(version, time_feat_dim, pooling_type)
    print('pooling_type:', pooling_type)
    print('time_feat_dim:', time_feat_dim)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dict_data = {}
    for dataset_split in ['train', 'val', 'test']:
        temp_dir = 'DataModule/dataset/fakeddit/data_json'
        with open(os.path.join(temp_dir, dataset_split+'.json'), 'r', encoding='utf-8') as f:
            data_json = json.load(f)
            dict_data.update(data_json)

    for file_name in tqdm(file_names, desc='comment feature extracting, time string encoding and graph constructing ...', ncols=125):
        if os.path.exists(os.path.join(save_dir, file_name.split('.')[0] + '.pt')):
            continue
        feature_dict = {}
        post_id = file_name.split('.')[0]
        src_nodes_id = []; dst_nodes_id = []; created_at_list = []
        
        base_time = dict_data[file_name.split('.')[0]]['created_utc']
        base_time = parse_to_standard_time(base_time)
            
        with open(folder_path+file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            head = next(reader)
            
            for row in reader:
                src_nodes_id.append(row[0]); dst_nodes_id.append(row[2]); created_at_list.append(row[3])
                text = row[1]
                
                feature_dict[row[0]] = feature_extraction(text, text_pretrained_model, tokenizer, device, pooling_type)
            build_post_comment_graph(feature_dict, post_id, file_name, 
                                     src_nodes_id, dst_nodes_id, created_at_list,
                                     rumor_file_names, non_rumor_file_names, save_dir, device, base_time, time_feat_dim)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate comment time embedding for fakeddit dataset")
    parser.add_argument('--text_pretrained_model_name', type=str, default='roberta-base', help='Pretrained model name for text feature extraction')
    parser.add_argument('--time_feature_dim', type=int, default=64, help='Dimension of time feature embedding')
    args = parser.parse_args()
    
    print(json.dumps(vars(args), indent=4, ensure_ascii=False))
    comment_time_embedding_generator(args, version=4)