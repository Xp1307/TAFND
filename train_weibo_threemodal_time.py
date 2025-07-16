from DataModule.load_data import GetDataset_embeddings
from DataModule.load_graph import GetDataset_comment_graph_time
from DataModule.data_preprocess import get_embeddings

from DataModule.data_preprocess import get_comment_graph_time
from Utils.args import weibo_args

from Models.temporal_model import WeiboModel1
from Models.trainer import trainer_threemodal

import json
import torch
import numpy as np
import random

def set_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__=='__main__':
    args = weibo_args()
    args.device_id = 0
    print(json.dumps(vars(args), indent=4))
    set_seed(args)
    
    device_id = args.device_id
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

    args.time_feature_dim = 64
    args.roberta_feature_choice = 'cls'
    
    get_embeddings(args)
    get_comment_graph_time(args)
    train_dataloader, test_dataloader = GetDataset_embeddings(args)                      
    train_graph_dataloader, test_graph_dataloader = GetDataset_comment_graph_time(args)
    
    for batch in train_graph_dataloader:
        metadata = batch.metadata()
        print(batch)
        break
    
    times = 10
    acc_list, f1_list, recall_list, precision_list = [], [], [], []
    f1_neg_list, recall_neg_list, precision_neg_list = [], [], []
    for run_time in range(0, times):
        args.seed = run_time + 1
        print('seed: ', args.seed)
        set_seed(args)
        model = WeiboModel1(args, metadata).to(device)
        print(model)
        
        trainer_modal = trainer_threemodal()
        acc, f1, recall, precision, f1_neg, recall_neg, precision_neg, features_list, all_labels = trainer_modal.train(model, train_dataloader, test_dataloader, 
                                        train_graph_dataloader, test_graph_dataloader, args)

        acc_list.append(acc)
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_neg_list.append(f1_neg)
        recall_neg_list.append(recall_neg)
        precision_neg_list.append(precision_neg)
        
        # features_array = np.concatenate(features_list, axis=0)
        # labels_array = np.concatenate(all_labels, axis=0)

        # np.savez("tsne_features_labels.npz", features=features_array, labels=labels_array)

    print('='*100)
    for data, metric in zip([acc_list, precision_list, recall_list, f1_list, precision_neg_list, recall_neg_list, f1_neg_list], 
                            ['accuracy', 'precision_fake', 'recall_fake', 'f1_fake', 'precision_real', 'recall_real', 'f1_real']):
        
        avg = np.mean(data) * 100
        std = np.std(data, ddof=1) * 100
        
        print("{} 平均值：{:.2f} ; {} 标准差：{:.2f}".format(metric, avg, metric, std))
    print('='*100)

    print('\n\n\n\n')
    print('*'*150)
    print('Test Acc list', acc_list)
    print('Fake Precision list', precision_list)
    print('Fake Recall list', recall_list)
    print('Fake F1 list', f1_list)
    print('Real Precision list', precision_neg_list)
    print('Real Recall list', recall_neg_list)
    print('Real F1 list', f1_neg_list)
    print('*'*150)