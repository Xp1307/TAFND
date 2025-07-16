import json
import torch
import numpy as np
import random

from DataModule.data_preprocess import get_embeddings
from DataModule.load_data import GetDataset_embeddings

from DataModule.data_preprocess import get_comment_graph
from DataModule.load_graph import GetDataset_comment_graph
from DataModule.comment_graph_create_fakeddit import comment_embedding_generator

from Utils.args import fakeddit_args
from Models.trainer import trainer, trainer_threemodal
from Models.concat_model import fakeddit_CombinedModel3

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
    print(json.dumps(vars(args), indent=4))
    # set_seed(args)
    
    # 创建 gpu 设备
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
        
        ## 得到文本、图像特征
        get_embeddings(args)                                                # 提取文本和图像的特征, 这个特征提取非常的重要, 直接影响性能
        train_dataloader, test_dataloader = GetDataset_embeddings(args)     # 从保存的特征文件中加载特征数据
        
        
        ## 得到异质图特征
        #@ 注意! 这里也用上了 text and image 的特征, 因为需要作为 graph 的 post 的初始节点特征，所以需要和上面的特征统一。
        # comment_embedding_generator(args, version=4)                                  # step1. 生成 comment_tree_graph 的特征文件, 文件保存在fakeddit_comment_tree_graph下
        get_comment_graph(args)                                                         # step2. 分别生成 pheme_train(val, test)_graph.pt                                                        
        train_graph_dataloader, test_graph_dataloader = GetDataset_comment_graph(args)  # step3. 加载数据
        
        ## 获取 hetero_graph 的元数据
        for batch in train_graph_dataloader:
            print(batch)
            metadata = batch.metadata()
            break
        
        model = fakeddit_CombinedModel3(args, metadata).to(device)
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