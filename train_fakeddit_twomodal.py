import json
import torch
import numpy as np
import random
from transformers import ViTModel, RobertaModel, BertModel

from VisualModule.tool import calculate_dataset
from DataModule.load_data import GetDataset, GetDataset_embeddings, GetDataset_all_token_embeddings
from DataModule.load_data import GetDataset_embeddings_v2

from DataModule.data_preprocess import get_embeddings, get_all_token_embeddings
from DataModule.data_preprocess import get_embeddings_pheme_v2

from Utils.args import fakeddit_args
from Models.trainer import trainer
from Models.concat_model import fakeddit_CombinedModel2
from Models.attention_model import CombinedModelAttention1, CombinedModelAttention12, CombinedModelAttention2

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
    set_seed(args)
    
    # 创建 gpu 设备
    device_id = args.device_id
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")  # 构建 GPU 设备

    acc_list, fake_precision_list, fake_recall_list, fake_f1_list = [], [], [], []
    real_f1_list, real_recall_list, real_precision_list = [], [], []
    repeat_times = 10
    for run_time in range(repeat_times):
        print('='*150)
        print('run at {}-th time'.format(run_time+1))
        ## 只是用 text 和 image 的特征
        get_embeddings(args)                                                # 提取文本和图像的特征, 这个特征提取非常的重要, 直接影响性能
        train_dataloader, test_dataloader = GetDataset_embeddings(args)     # 从保存的特征文件中加载特征数据
        model = fakeddit_CombinedModel2(args).to(device)
        
        ## 训练模型
        trainer_ins = trainer()
        acc, fake_f1, fake_recall, fake_precision, real_f1, real_recall, real_precision = trainer_ins.train(model, train_dataloader, test_dataloader, args) 
        acc_list.append(acc)
        fake_precision_list.append(fake_precision)
        fake_recall_list.append(fake_recall)
        fake_f1_list.append(fake_f1)
        real_precision_list.append(real_precision)
        real_recall_list.append(real_recall)
        real_f1_list.append(real_f1)
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