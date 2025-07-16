from VisualModule.tool import calculate_dataset
from DataModule.load_data import GetDataset_embeddings
from DataModule.load_graph import GetDataset_comment_graph_time
from DataModule.data_preprocess import get_embeddings

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from DataModule.data_preprocess import get_comment_graph_time
from Utils.args import weibo_args, pheme_args
from Models.concat_model import CombinedModel2, CombinedModel3
from Models.temporal_model import WeiboModel1
from Models.trainer import trainer, trainer_threemodal

import json
import torch
import numpy as np
import random
import statistics

"""
    这个相比于 train_weibo_threemodal_time.py 文件, 主要是引入了时间信息概念
    建模 “先评论-后评论” 的评论时间传播过程, 提升对谣言演化的理解。
"""

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
    args.device_id = 0
    print(json.dumps(vars(args), indent=4))
    set_seed(args)                              ## 设置随机种子
    
    device_id = args.device_id
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")  # 构建 GPU 设备

    ## 更新图中的 post 节点信息, 并创建 train, val, test_graph.pt, 保存时序子图列表
    args.time_feature_dim = 64                #@ 这里的表示时间特征的维度, 这里设置为 768 效果会很差, 默认为 64
    args.roberta_feature_choice = 'cls'
    
    ## 加载 text_features, image_features 
    get_embeddings(args)                     # 提取文本和图像的特征, 这个特征提取非常的重要, 直接影响性能
    
    ## 构建时序图
    get_comment_graph_time(args)              # 构建时序图, 保存为 train, val, test_graph.pt
    
    train_dataloader, test_dataloader = GetDataset_embeddings(args)                     #得到 text 和 image 的 features                        
    ## 加载 comment_tree graph 形式的数据集
    train_graph_dataloader, test_graph_dataloader = GetDataset_comment_graph_time(args) #时序图, dataloader 形式的图, 在 xxx_graph.pt 基础上

    ## 获取 hetero_graph 的元数据
    for batch in train_graph_dataloader:
        metadata = batch.metadata()
        print(batch)
        break
    
    times = 10
    acc_list, f1_list, recall_list, precision_list = [], [], [], []
    f1_neg_list, recall_neg_list, precision_neg_list = [], [], []  # 用于存储负类的指标
    for run_time in range(0, times):
        ## 创建模型, 这里要换成 temporal models
        args.seed = run_time + 1
        print('seed: ', args.seed)
        set_seed(args)
        model = WeiboModel1(args, metadata).to(device)
        print(model)
        
        ## 训练模型
        trainer_modal = trainer_threemodal()
        acc, f1, recall, precision, f1_neg, recall_neg, precision_neg, features_list, all_labels = trainer_modal.train(model, train_dataloader, test_dataloader, 
                                        train_graph_dataloader, test_graph_dataloader, args)
        if all(v != 0 for v in [acc, f1, recall, precision, f1_neg, recall_neg, precision_neg]):
            acc_list.append(acc)
            f1_list.append(f1)
            recall_list.append(recall)
            precision_list.append(precision)
            f1_neg_list.append(f1_neg)
            recall_neg_list.append(recall_neg)
            precision_neg_list.append(precision_neg)
        # acc_list.append(acc)
        # f1_list.append(f1)
        # recall_list.append(recall)
        # precision_list.append(precision)
        # f1_neg_list.append(f1_neg)
        # recall_neg_list.append(recall_neg)
        # precision_neg_list.append(precision_neg)
        
        # # 合并 batch 级别的数据为一个完整数组
        # features_array = np.concatenate(features_list, axis=0)
        # labels_array = np.concatenate(all_labels, axis=0)

        # # 保存为压缩文件（包含两个键：features 和 labels）
        # np.savez("tsne_features_labels.npz", features=features_array, labels=labels_array)

    print('='*100)
    for data, metric in zip([acc_list, precision_list, recall_list, f1_list, precision_neg_list, recall_neg_list, f1_neg_list], 
                            ['accuracy', 'precision_fake', 'recall_fake', 'f1_fake', 'precision_real', 'recall_real', 'f1_real']):
        
        avg = np.mean(data) * 100            # 转为百分制
        std = np.std(data, ddof=1) * 100     # 样本标准差，ddof=1 和 statistics.stdev 保持一致
        
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