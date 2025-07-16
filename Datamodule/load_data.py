import torch
from torch.utils.data import DataLoader

def GetDataset_embeddings(args):
    
    dataset_name = args.dataset_name
    dataset_embeddings_dir = args.dataset_embeddings_dir
    batch_size = args.batch_size
    
    if args.roberta_feature_choice == 'cls':
        train_dataset_dir = dataset_embeddings_dir.format(dataset_name, 'train', args.roberta_feature_choice)
        train_dataset_dir = train_dataset_dir.replace('_{}.pt'.format(args.roberta_feature_choice), '.pt')
        test_dataset_dir = dataset_embeddings_dir.format(dataset_name, 'test', args.roberta_feature_choice)
        test_dataset_dir = test_dataset_dir.replace('_{}.pt'.format(args.roberta_feature_choice), '.pt')
    else:
        train_dataset_dir = dataset_embeddings_dir.format(dataset_name, 'train', args.roberta_feature_choice)
        test_dataset_dir = dataset_embeddings_dir.format(dataset_name, 'test', args.roberta_feature_choice)
        
    train_dataset = torch.load(train_dataset_dir, weights_only=False)
    test_datset = torch.load(test_dataset_dir, weights_only=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_datset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader