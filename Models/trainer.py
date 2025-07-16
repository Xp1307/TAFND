import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from . import draw_figure       # 使用相对路径导入 draw_figure 函数

class trainer:
    '''
        Intro:
        -----------
            这个训练器是针对两种模态模型的训练器, 主要是针对图像和文本两种模态的联合训练
        Args:
        -----------
            model: 训练的模型
            train_dataloader: 训练集数据加载器
            test_dataloader: 测试集数据加载器
            args: 参数设置
    '''
    def __init__(self):
        pass
    
    # 训练模型
    def train(self, model, train_dataloader, test_dataloader, args):    
        #@ 训练模型
        num_epochs = args.num_epochs; train_losses = []
        ## 定义损失函数和优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
        ## 设置调度器管理学习率(用的余弦退火来逐步降低学习率)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
        ## 设置设备
        device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        for epoch in range(num_epochs):
            model.train()
            epoch_pbar = tqdm(total=len(train_dataloader), 
                            desc='Epoch {}/{} | Processing...'.format(epoch+1, num_epochs),
                            ncols=100)
            for images, text_inputs, labels in train_dataloader:
                # embeddings 形式的数据才需要这行代码
                images=images.to(device); text_inputs=text_inputs.to(device); labels=labels.to(device)
                optimizer.zero_grad()
                outputs = model(images, text_inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_pbar.update(1)
            epoch_pbar.set_postfix({'Loss': loss.item()})
            train_losses.append(loss.item())
        print("训练完成")
        acc, f1, recall, precision, real_f1, real_recall, real_precision = self.test(model, test_dataloader)           # 测试模型并计算指标
        draw_figure.loss_curve(train_losses)        # 绘制损失曲线
        return acc, f1, recall, precision, real_f1, real_recall, real_precision

    # 测试模型并计算指标
    def test(self, model, test_dataloader):
        model.eval()
        all_labels = []
        all_preds = []

        #@ 测试模型
        with torch.no_grad():
            for images, text_inputs, labels in test_dataloader:
                outputs = model(images, text_inputs)
                _, preds = torch.max(outputs, 1)          # torch.max 返回：(value, index)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        
        print('='*100)
        res = classification_report(all_labels, all_preds, digits=3, output_dict=True)
        for k, v in res.items():
            print(k, v)
        print("result:{:.4f}".format(res['accuracy']))
        print('='*100)    
        real_precision = res['0']['precision']
        real_recall = res['0']['recall']
        real_f1 = res['0']['f1-score']

        return acc, f1, recall, precision, real_f1, real_recall, real_precision

class trainer_threemodal:
    '''
        Intro:
        -----------
            这个训练器是针对三模态模型的训练器, 主要是针对图像, 文本, 图结构三种模态的联合训练
        相关参数:
        -----------
            model: 训练的模型
            train_dataloader: 训练集数据加载器
            test_dataloader: 测试集数据加载器
            train_graph_dataloader: 训练集图结构数据加载器
            test_graph_dataloader: 测试集图结构数据加载器
            args: 参数设置
    '''
    def __init__(self):
        pass
    
    # 训练模型
    def train(self, model, train_dataloader, test_dataloader, 
                train_graph_dataloader, test_graph_dataloader, args):    
        #@ 训练模型
        num_epochs = args.num_epochs; train_losses = []
        ## 定义损失函数和优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
        ## 设置调度器管理学习率(用的余弦退火来逐步降低学习率)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
        ## 设置设备
        device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        
        model.train()
        for epoch in range(num_epochs):
            epoch_pbar = tqdm(total=len(train_dataloader), 
                            desc='Epoch {}/{} | Processing...'.format(epoch+1, num_epochs),
                            ncols=100)
            for text_and_image, comment_graph in zip(train_dataloader, train_graph_dataloader):
                # embeddings 形式的数据才需要这行代码
                images, text_inputs, labels = text_and_image
                images=images.to(device); text_inputs=text_inputs.to(device); labels=labels.to(device)
                comment_graph = comment_graph.to(device)
                
                optimizer.zero_grad()
                outputs, _ = model(images, text_inputs, comment_graph)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_pbar.update(1)
            epoch_pbar.set_postfix({'Loss': loss.item()})
            train_losses.append(loss.item())
        print("训练完成")
        # 测试模型并计算指标
        acc, f1, recall, precision, f1_neg, recall_neg, precision_neg, features_list, all_labels = self.test(model, test_dataloader, test_graph_dataloader, device) 
        # 绘制损失曲线         
        draw_figure.loss_curve(train_losses)                               
        return acc, f1, recall, precision, f1_neg, recall_neg, precision_neg, features_list, all_labels

    # 测试模型并计算指标
    def test(self, model, test_dataloader, test_graph_dataloader, device):
        model.eval()
        all_labels = []
        all_preds = []
        tsne_e_labels = []

        features_list = []
        #@ 测试模型
        with torch.no_grad():
            for text_and_image, comment_graph in zip(test_dataloader, test_graph_dataloader):
                images, text_inputs, labels = text_and_image
                outputs, features = model(images, text_inputs, comment_graph)
                
                features_list.append(features.cpu().numpy())
                tsne_e_labels.append(labels.detach().cpu().numpy().reshape(-1))

                # print(images.shape, text_inputs.shape, labels.shape)
                # images=images.to(device); text_inputs=text_inputs.to(device); labels=labels.to(device)
                # comment_graph = comment_graph.to(device)
                
                _, preds = torch.max(outputs, 1)          # torch.max 返回：(value, index)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        
        print('='*100)
        res = classification_report(all_labels, all_preds, digits=3, output_dict=True)
        f1_neg = res["0"]["f1-score"]
        recall_neg = res["0"]["recall"]
        precision_neg = res["0"]["precision"]
        
        for k, v in res.items():
            print(k, v)
        print("result:{:.4f}".format(res['accuracy']))
        print('='*100)    
        return acc, f1, recall, precision, f1_neg, recall_neg, precision_neg, features_list, tsne_e_labels
    
    
class pheme_trainer_threemodal:
    '''
        Intro:
        -----------
            这个训练器是针对三模态模型的训练器, 主要是针对图像, 文本, 图结构三种模态的联合训练
        相关参数:
        -----------
            model: 训练的模型
            train_dataloader: 训练集数据加载器
            test_dataloader: 测试集数据加载器
            train_graph_dataloader: 训练集图结构数据加载器
            test_graph_dataloader: 测试集图结构数据加载器
            args: 参数设置
    '''
    def __init__(self):
        pass
    
    # 训练模型
    def train(self, model, train_dataloader, test_dataloader, 
                train_graph_dataloader, test_graph_dataloader, args):    
        #@ 训练模型
        num_epochs = args.num_epochs; train_losses = []
        ## 定义损失函数和优化器
        # weight = torch.tensor([1.0, 2.0]).to('cuda:{}'.format(args.device_id))  # 设置类别权重, 这里是为了处理不平衡数据集
        # criterion = torch.nn.CrossEntropyLoss(weight=weight)  # 设置类别权重, 这里是为了处理不平衡数据集

        criterion = torch.nn.CrossEntropyLoss()  # 设置类别权重, 这里是为了处理不平衡数据集 
        # 小模型 or 少数据	1e-3 ~ 1e-4（正则强一点）
        # 大模型（如 ViT / RoBERTa）	1e-5 ~ 5e-5（较弱，避免抑制过头）
        # 不想加正则	weight_decay=0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
        
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
        
        ## 设置设备
        device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        
        model.train()
        for epoch in range(num_epochs):
            epoch_pbar = tqdm(total=len(train_dataloader), 
                            desc='Epoch {}/{} | Processing...'.format(epoch+1, num_epochs),
                            ncols=100)
            for text_and_image, comment_graph in zip(train_dataloader, train_graph_dataloader):
                # embeddings 形式的数据才需要这行代码
                images, text_inputs, labels = text_and_image
                images=images.to(device); text_inputs=text_inputs.to(device); labels=labels.to(device)
                comment_graph = comment_graph.to(device)
                
                optimizer.zero_grad()
                outputs = model(images, text_inputs, comment_graph)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_pbar.update(1)
            epoch_pbar.set_postfix({'Loss': loss.item()})
            train_losses.append(loss.item())
        print("训练完成")
        # 测试模型并计算指标
        acc, f1, recall, precision = self.test(model, test_dataloader, test_graph_dataloader, device) 
        # 绘制损失曲线         
        draw_figure.loss_curve(train_losses)                               
        return acc, f1, recall, precision

    # 测试模型并计算指标
    def test(self, model, test_dataloader, test_graph_dataloader, device):
        model.eval()
        all_labels = []
        all_preds = []

        #@ 测试模型
        with torch.no_grad():
            for text_and_image, comment_graph in zip(test_dataloader, test_graph_dataloader):
                images, text_inputs, labels = text_and_image
                outputs = model(images, text_inputs, comment_graph)
                # print(images.shape, text_inputs.shape, labels.shape)
                # images=images.to(device); text_inputs=text_inputs.to(device); labels=labels.to(device)
                # comment_graph = comment_graph.to(device)
                
                _, preds = torch.max(outputs, 1)          # torch.max 返回：(value, index)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        # f1 = f1_score(all_labels, all_preds, average='weighted')
        # recall = recall_score(all_labels, all_preds, average='weighted')
        # precision = precision_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        
        print('='*100)
        res = classification_report(all_labels, all_preds, digits=3, output_dict=True)
        for k, v in res.items():
            print(k, v)
        print("result:{:.4f}".format(res['accuracy']))
        print('='*100)    
        return acc, f1, recall, precision