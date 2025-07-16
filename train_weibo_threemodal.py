from VisualModule.tool import calculate_dataset
from DataModule.load_data import GetDataset_embeddings
from DataModule.load_graph import GetDataset_comment_graph

from DataModule.data_preprocess import get_embeddings
from Utils.args import weibo_args, pheme_args
from Models.concat_model import CombinedModel2, CombinedModel3
from Models.trainer import trainer, trainer_threemodal

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
    # 让 CuDNN 计算可复现（但可能会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__=='__main__':
    args = weibo_args()                         ## weibo 数据集相关的参数
    args.time_feature_dim = 64
    args.roberta_feature_choice = 'cls'
    print(json.dumps(vars(args), indent=4))

    
    device_id = args.device_id
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")  # 构建 GPU 设备

    acc_list, fake_precision_list, fake_recall_list, fake_f1_list = [], [], [], []
    real_f1_list, real_recall_list, real_precision_list = [], [], []
    times = 10
    
    for run_time in range(times):
        print('='*150)
        print('run at {}-th time'.format(run_time+1))
        args.seed = run_time + 1
        print('seed: ', args.seed)
        set_seed(args)                              ## 设置随机种子
        ## 加载 text_features, image_features 
        train_dataloader, test_dataloader = GetDataset_embeddings(args)
        ## 加载 comment_tree graph 形式的数据集
        train_graph_dataloader, test_graph_dataloader = GetDataset_comment_graph(args)
        ## 获取 hetero_graph 的元数据
        for batch in train_graph_dataloader:
            metadata = batch.metadata()
            break

        ## 创建模型
        model = CombinedModel3(args, metadata).to(device)
        print(model)
        
        ## 训练模型
        trainer_threemodal_ins = trainer_threemodal()
        acc, f1, recall, precision, f1_neg, recall_neg, precision_neg, features_list, all_labels = trainer_threemodal_ins.train(model, train_dataloader, test_dataloader, 
                                    train_graph_dataloader, test_graph_dataloader, args)   # 训练模型
        
        acc_list.append(acc)
        fake_precision_list.append(precision)
        fake_recall_list.append(recall)
        fake_f1_list.append(f1)
        real_precision_list.append(precision_neg)
        real_recall_list.append(recall_neg)
        real_f1_list.append(f1_neg)

        # # 合并 batch 级别的数据为一个完整数组
        # features_array = np.concatenate(features_list, axis=0)
        # labels_array = np.concatenate(all_labels, axis=0)

        # # 保存为压缩文件（包含两个键：features 和 labels）
        # np.savez("tsne_features_labels_wt.npz", features=features_array, labels=labels_array)
        
    print('='*150)
    print('*'*150)
    print('Test Acc           : {:.2f}% ± {:.2f}%'.format(np.mean(acc_list) * 100, np.std(acc_list) * 100))
    print('Fake Precision     : {:.2f}% ± {:.2f}%'.format(np.mean(fake_precision_list) * 100, np.std(fake_precision_list) * 100))
    print('Fake Recall        : {:.2f}% ± {:.2f}%'.format(np.mean(fake_recall_list) * 100, np.std(fake_recall_list) * 100))
    print('Fake F1            : {:.2f}% ± {:.2f}%'.format(np.mean(fake_f1_list) * 100, np.std(fake_f1_list) * 100))
    print('Real Precision     : {:.2f}% ± {:.2f}%'.format(np.mean(real_precision_list) * 100, np.std(real_precision_list) * 100))
    print('Real Recall        : {:.2f}% ± {:.2f}%'.format(np.mean(real_recall_list) * 100, np.std(real_recall_list) * 100))
    print('Real F1            : {:.2f}% ± {:.2f}%'.format(np.mean(real_f1_list) * 100, np.std(real_f1_list) * 100))
    print('*'*150)

    print('\n\n\n\n')
    print('*'*150)
    print('Test Acc list', acc_list)
    print('Fake Precision list', fake_precision_list)
    print('Fake Recall list', fake_recall_list)
    print('Fake F1 list', fake_f1_list)
    print('Real Precision list', real_precision_list)
    print('Real Recall list', real_recall_list)
    print('Real F1 list', real_f1_list)
    print('*'*150)