import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from . import draw_figure

class trainer:
    def __init__(self):
        pass
    
    def train(self, model, train_dataloader, test_dataloader, args):    
        num_epochs = args.num_epochs; train_losses = []
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
        device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        for epoch in range(num_epochs):
            model.train()
            epoch_pbar = tqdm(total=len(train_dataloader), 
                            desc='Epoch {}/{} | Processing...'.format(epoch+1, num_epochs),
                            ncols=100)
            for images, text_inputs, labels in train_dataloader:
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
        print("done!")
        acc, f1, recall, precision, real_f1, real_recall, real_precision = self.test(model, test_dataloader)
        draw_figure.loss_curve(train_losses)
        return acc, f1, recall, precision, real_f1, real_recall, real_precision

    def test(self, model, test_dataloader):
        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, text_inputs, labels in test_dataloader:
                outputs = model(images, text_inputs)
                _, preds = torch.max(outputs, 1)
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
    def __init__(self):
        pass
    
    def train(self, model, train_dataloader, test_dataloader, 
                train_graph_dataloader, test_graph_dataloader, args):    
   
        num_epochs = args.num_epochs; train_losses = []
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

        device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        
        model.train()
        for epoch in range(num_epochs):
            epoch_pbar = tqdm(total=len(train_dataloader), 
                            desc='Epoch {}/{} | Processing...'.format(epoch+1, num_epochs),
                            ncols=100)
            for text_and_image, comment_graph in zip(train_dataloader, train_graph_dataloader):

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
        print("done!")
        acc, f1, recall, precision, f1_neg, recall_neg, precision_neg, features_list, all_labels = self.test(model, test_dataloader, test_graph_dataloader, device)       
        draw_figure.loss_curve(train_losses)                               
        return acc, f1, recall, precision, f1_neg, recall_neg, precision_neg, features_list, all_labels

    def test(self, model, test_dataloader, test_graph_dataloader, device):
        model.eval()
        all_labels = []
        all_preds = []
        tsne_e_labels = []

        features_list = []
        with torch.no_grad():
            for text_and_image, comment_graph in zip(test_dataloader, test_graph_dataloader):
                images, text_inputs, labels = text_and_image
                outputs, features = model(images, text_inputs, comment_graph)
                
                features_list.append(features.cpu().numpy())
                tsne_e_labels.append(labels.detach().cpu().numpy().reshape(-1))
                
                _, preds = torch.max(outputs, 1)
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