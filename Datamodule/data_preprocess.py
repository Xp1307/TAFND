import os
import re
import json
import torch
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from PIL import Image

from tqdm import tqdm

def get_embeddings(args):
    pretrained_models_dir = args.pretrained_models_dir
    image_pretrained_model_name = args.image_pretrained_model_name
    text_pretrained_model_name = args.text_pretrained_model_name

    image_pretrained_model = AutoModel.from_pretrained(pretrained_models_dir + image_pretrained_model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_models_dir + image_pretrained_model_name)    
    
    text_pretrained_model = AutoModel.from_pretrained(pretrained_models_dir + text_pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_models_dir + text_pretrained_model_name)
        
    for param in image_pretrained_model.parameters():
        param.requires_grad = False
    for param in text_pretrained_model.parameters():
        param.requires_grad = False    

    for dataset_split in ['train', 'val', 'test']:
        result_list = []
        dataset_dir = args.dataset_dir.format(args.dataset_name, dataset_split)
        
        if args.roberta_feature_choice =='cls':
            if os.path.exists('DataModule/data/{}_{}.pt'.format(args.dataset_name, dataset_split)):
                continue
        elif os.path.exists('DataModule/data/{}_{}_{}.pt'.format(args.dataset_name, dataset_split, args.roberta_feature_choice)):
            continue
        
        with open(dataset_dir, 'r', encoding='utf-8') as file:
            raw_dataset = json.load(file)
        for _, value in tqdm(raw_dataset.items(), 
                             desc='Extracting embeddings of text & image from {}/{} ...'.format(args.dataset_name, dataset_split),
                             ncols=125):
            text = value['tweet_content']
            image_path = value['tweet_image_local']
            image = Image.open(image_path).convert('RGB')  # (224, 224, 3)
            label = value['tweet_label']
            
            image_inputs = feature_extractor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = image_pretrained_model(**image_inputs).last_hidden_state  # (1, seq_len, 768)
                if args.roberta_feature_choice == 'cls':
                    image_feature = outputs[:, 0, :]  # (1, 768)
                elif args.roberta_feature_choice == 'mean':
                    image_feature = outputs.mean(dim=1)  # (1, 768)
                elif args.roberta_feature_choice == 'max':
                    image_feature = outputs.max(dim=1).values  # (1, 768)
                image_feature = image_feature.squeeze(0)  # (768)
            
            text_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = text_pretrained_model(**text_inputs).last_hidden_state  # (1, seq_len, 768)
                if args.roberta_feature_choice == 'cls':
                    text_feature = outputs[:, 0, :]  # (1, 768)
                elif args.roberta_feature_choice == 'mean':
                    text_feature = outputs.mean(dim=1)  # (1, 768)
                elif args.roberta_feature_choice == 'max':
                    text_feature = outputs.max(dim=1).values  # (1, 768)
                text_feature = text_feature.squeeze(0)  # (768)
                
            result_list.append((text_feature, image_feature, label))        # (768), (768), (1)
            
        if args.roberta_feature_choice == 'cls':
            torch.save(result_list, 'DataModule/data/{}_{}.pt'.format(args.dataset_name, dataset_split))
        else:
            torch.save(result_list, 'DataModule/data/{}_{}_{}.pt'.format(args.dataset_name, dataset_split, args.roberta_feature_choice))

def get_comment_graph(args):
    for dataset_split in ['train', 'val', 'test']:
        if os.path.exists(args.dataset_graph_embeddings_dir.format(args.dataset_name, dataset_split)):
            continue
        else:
            file_dir = args.dataset_dir.format(args.dataset_name, dataset_split)
            raw_graph_embeddings_dir = args.dataset_raw_graph_embeddings_dir.format(args.dataset_name)
            
            text_image_dir = args.dataset_embeddings_dir.format(args.dataset_name, dataset_split)
            text_image_feature = torch.load(text_image_dir, 
                                            weights_only=False)
            
            file_names = os.listdir(raw_graph_embeddings_dir)
            with open(file_dir, 'r', encoding='utf-8') as file:
                raw_dataset = json.load(file)
            graph_pt_list = []
            for data_tupe, key in zip(text_image_feature, raw_dataset.keys()):
                text_feature, image_feature, _ = data_tupe
                file_name = next(filter(lambda x: key in x, file_names), None)
                graph_pt = torch.load(os.path.join(raw_graph_embeddings_dir, file_name), weights_only=False)
                graph_pt['post'].text_feature = text_feature.unsqueeze(0)
                graph_pt['post'].image_feature = image_feature.unsqueeze(0)
                graph_pt['post'].x = torch.cat([graph_pt['post'].text_feature, 
                                                graph_pt['post'].image_feature], dim=1)
                graph_pt_list.append(graph_pt)
            torch.save(graph_pt_list, args.dataset_graph_embeddings_dir.format(args.dataset_name, dataset_split))

def get_comment_graph_time(args, version='v6'):
    for dataset_split in ['train', 'val', 'test']:
        save_dir = args.dataset_graph_embeddings_dir.format(args.dataset_name, dataset_split)
        
        if args.roberta_feature_choice == 'cls':
            save_dir = save_dir.replace('.pt', '_time_'+str(args.time_feature_dim)+'d.pt')
        else:
            save_dir = save_dir.replace('.pt', '_time_'+str(args.time_feature_dim)+'d_{}.pt'.format(args.roberta_feature_choice))

        if os.path.exists(save_dir):
            continue
        else:
            file_dir = args.dataset_dir.format(args.dataset_name, dataset_split)
            version = version
            raw_graph_embeddings_dir = args.dataset_raw_graph_embeddings_dir.format(args.dataset_name)

            raw_graph_embeddings_dir = raw_graph_embeddings_dir.replace('_graph', '_graph_time_'+version+'_'+str(args.time_feature_dim)+'d_{}'.format(args.roberta_feature_choice)) 
            text_image_feature = torch.load(args.dataset_embeddings_dir.format(args.dataset_name, dataset_split, args.roberta_feature_choice), 
                                            weights_only=False)
            
            file_names = os.listdir(raw_graph_embeddings_dir)
            with open(file_dir, 'r', encoding='utf-8') as file:
                raw_dataset = json.load(file)
            graph_pt_list = []
            for data_tupe, key in zip(text_image_feature, raw_dataset.keys()):
                text_feature, image_feature, _ = data_tupe
                file_name = next(filter(lambda x: key in x, file_names), None)
                graph_pt = torch.load(os.path.join(raw_graph_embeddings_dir, file_name), weights_only=False)
                graph_pt['post'].text_feature = text_feature.unsqueeze(0)
                graph_pt['post'].image_feature = image_feature.unsqueeze(0)
                graph_pt['post'].x = torch.cat([graph_pt['post'].text_feature, 
                                                graph_pt['post'].image_feature], dim=1)
                graph_pt_list.append(graph_pt)
            torch.save(graph_pt_list, save_dir)