import json
import torch
import numpy as np
import random
import statistics

from DataModule.data_preprocess import get_embeddings
from DataModule.data_preprocess import get_comment_graph
from DataModule.data_preprocess import get_comment_graph_time

from DataModule.comment_graph_create_fakeddit import comment_embedding_generator
from DataModule.comment_graph_create_fakeddit_time import comment_time_embedding_generator
from DataModule.load_data import GetDataset_embeddings
from DataModule.load_graph import GetDataset_comment_graph
from DataModule.load_graph import GetDataset_comment_graph_time

from Utils.args import fakeddit_args
from Models.trainer import trainer, trainer_threemodal

from Models.concat_model import fakeddit_CombinedModel3
from Models.temporal_model import FakedditModel1

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
    args = fakeddit_args()                                     ## 数据集相关的参数
    args.time_feature_dim = 64               
    args.roberta_feature_choice = 'cls'
    print(json.dumps(vars(args), indent=4))
    
    # 创建 gpu 设备
    device_id = args.device_id
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")  # 构建 GPU 设备

    ## 得到文本、图像特征
    get_embeddings(args)                                                # 提取文本和图像的特征, 这个特征提取非常的重要, 直接影响性能; 使用 roberta, vit 来的
    train_dataloader, test_dataloader = GetDataset_embeddings(args)     # 从保存的特征文件中加载特征数据
    
    ## 得到异质图特征
    #@ 注意! 这里也用上了 text and image 的特征, 因为需要作为 graph 的 post 的初始节点特征，所以需要和上面的特征统一。
    # comment_embedding_generator(args, version=4)                                         # step1. 生成 comment_tree_graph 的特征文件, 文件保存在fakeddit_comment_tree_graph下, 执行完一次可以跳过
    # get_comment_graph(args)                                                              # step2. 分别生成 pheme_train(val, test)_graph.pt                                                        
    # train_graph_dataloader, test_graph_dataloader = GetDataset_comment_graph(args)       # step3. 加载数据
    
    # comment_time_embedding_generator(args, version=4)                                               # step1. 生成 comment_tree_graph_time 的特征文件, 执行完一次可以跳过  
    get_comment_graph_time(args, version='v4')                                                         # step2. 分别生成 pheme_train(val, test)_graph_time.pt                                                        
    train_graph_dataloader, test_graph_dataloader = GetDataset_comment_graph_time(args)             # step3. 加载数据    
    
    
    ## 获取 hetero_graph 的元数据
    for batch in train_graph_dataloader:
        print(batch)
        metadata = batch.metadata()
        break
    
    times = 10
    acc_list, f1_list, recall_list, precision_list = [], [], [], []
    f1_neg_list, recall_neg_list, precision_neg_list = [], [], []  # 用于存储负类的指标
    
    for run_time in range(0, times):
        print('='*150)
        print('run at {}-th time'.format(run_time+1))
        args.seed = run_time + 1
        print('seed: ', args.seed)
        set_seed(args)                              ## 设置随机种子

        ## 创建模型, 这里要换成 temporal models
        model = FakedditModel1(args, metadata).to(device)
        print(model)
        
        ## 训练模型
        trainer_modal = trainer_threemodal()
        acc, f1, recall, precision, f1_neg, recall_neg, precision_neg, features_list, all_labels = trainer_modal.train(model, train_dataloader, test_dataloader, 
                                    train_graph_dataloader, test_graph_dataloader, args)   # 训练模型

        acc_list.append(acc)
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_neg_list.append(f1_neg)
        recall_neg_list.append(recall_neg)
        precision_neg_list.append(precision_neg)

    print('=' * 100)
    for data, metric in zip([acc_list, precision_list, recall_list, f1_list, precision_neg_list, recall_neg_list, f1_neg_list], 
                            ['accuracy', 'precision_fake', 'recall_fake', 'f1_fake', 'precision_real', 'recall_real', 'f1_real']):
        data = np.array(data)
        avg = np.mean(data) * 100       # 转为百分制
        std = np.std(data, ddof=1) * 100  # 样本标准差
        print(f"{metric} 平均值：{avg:.2f} ; {metric} 标准差：{std:.2f}")
    print('=' * 100)
    
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