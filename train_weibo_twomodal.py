from VisualModule.tool import calculate_dataset
from DataModule.load_data import GetDataset, GetDataset_embeddings
from DataModule.data_preprocess import get_embeddings
from Utils.args import weibo_args, pheme_args
from Models.concat_model import CombinedModel, CombinedModel2, load_pretrained_models
from Models.trainer import trainer
from transformers import ViTModel, RobertaModel, BertModel
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
    args = weibo_args()     ## weibo 数据集相关的参数
    # get_embeddings(args)
    print(json.dumps(vars(args), indent=4))
    
    # torch.Size([64, 1, 768]) torch.Size([64, 1, 768]) torch.Size([64]) | 每个 batch 的数据形式
    # train_dataloader, test_dataloader = GetDataset_embeddings(args)
    

    set_seed(args)
    print(torch.rand(1).item())             # 测试随机种子
    
    device_id = args.device_id
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")  # 构建 GPU 设备

    # 创建模型
    # train_dataloader, test_dataloader = GetDataset(args) # 加载数据集
    # vit_model, roberta_model = load_pretrained_models(args) # 加载 Vit 和 Roberta 模型
    # model = CombinedModel(vit_model, roberta_model).to(device) # 初始化联合模型
    
    acc_list, fake_precision_list, fake_recall_list, fake_f1_list = [], [], [], []
    real_f1_list, real_recall_list, real_precision_list = [], [], []
    times = 10
    for run_time in range(times):
        print('='*150)
        print('run at {}-th time'.format(run_time+1))
        train_dataloader, test_dataloader = GetDataset_embeddings(args)
        model = CombinedModel2(args).to(device)
        ## 训练模型
        trainer_ins = trainer()
        acc, fake_f1, fake_recall, fake_precision, real_f1, real_recall, real_precision = trainer_ins.train(model, train_dataloader, test_dataloader, args) # 训练模型
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