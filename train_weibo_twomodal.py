from DataModule.load_data import GetDataset_embeddings
from Utils.args import weibo_args
from Models.concat_model import CombinedModel2
from Models.trainer import trainer
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
    print(json.dumps(vars(args), indent=4))
    
    set_seed(args)
    print(torch.rand(1).item())
    
    device_id = args.device_id
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

    acc_list, fake_precision_list, fake_recall_list, fake_f1_list = [], [], [], []
    real_f1_list, real_recall_list, real_precision_list = [], [], []
    times = 10
    for run_time in range(times):
        print('='*150)
        print('run at {}-th time'.format(run_time+1))
        train_dataloader, test_dataloader = GetDataset_embeddings(args)
        model = CombinedModel2(args).to(device)
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