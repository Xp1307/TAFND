import os
import json
import torch
from torch_geometric.loader import DataLoader 

def init_post_nodes():
    pass

def GetDataset_comment_graph(args):
    '''
        获得 comment tree graph 形式的数据集, 即原始数据被相应的预训练模型转化为了固定维度的 embeddings
        这里的 comment tree graph 是指评论树的图结构, 是一个异质图
        其中 post 节点表示微博的原始内容, comment 节点表示评论的内容,
        [comment, comment_on, post] 表示对 post 的comment, [comment, reply, comment] 表示评论之间的边
        
        Parameters:
        -------
            args: argparse.Namespace
                参数集合
        
        Returns:
        -------
            train_dataloader: torch.utils.data.DataLoader
                训练数据集加载器
            test_dataloader: torch.utils.data.DataLoader
                测试数据集加载器
        
        数据形式(示例如下):
        -------
            HeteroDataBatch(
            post={
                x=[64, 1536],
                post_id=[64],
                y=[64],
                batch=[64],
                text_feature=[64, 768],
                image_feature=[64, 768],
                ptr=[65],
            },
            comment={
                x=[13635, 768],
                comment_id=[64],
                batch=[13635],
                ptr=[65],
            },
            (comment, comments_on, post)={ edge_index=[2, 10118] },
            (comment, replies_to, comment)={ edge_index=[2, 3525] }
            )
    '''
    ## 相关参数
    dataset_name = args.dataset_name
    dataset_graph_embeddings_dir = args.dataset_graph_embeddings_dir
    batch_size = args.batch_size

    train_dataset_dir = dataset_graph_embeddings_dir.format(dataset_name, 'train')
    test_dataset_dir = dataset_graph_embeddings_dir.format(dataset_name, 'test')
    train_dataset = torch.load(train_dataset_dir, weights_only=False)
    test_datset = torch.load(test_dataset_dir, weights_only=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_datset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def GetDataset_comment_graphv2(args):
    ## 相关参数
    dataset_name = args.dataset_name
    dataset_graph_embeddings_dir = args.dataset_graph_embeddings_dir
    dataset_graph_embeddings_dir = '/data3/xupin/1_FakeNewsOnLLM/UNnamePro/DataModule/data/{}_{}_graph_wolink.pt'

    batch_size = args.batch_size

    train_dataset_dir = dataset_graph_embeddings_dir.format(dataset_name, 'train')
    test_dataset_dir = dataset_graph_embeddings_dir.format(dataset_name, 'test')
    train_dataset = torch.load(train_dataset_dir, weights_only=False)
    test_datset = torch.load(test_dataset_dir, weights_only=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_datset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def GetDataset_comment_graph_time(args):
    '''
        获得 comment tree graph 形式的数据集, 即原始数据被相应的预训练模型转化为了固定维度的 embeddings
        这里的 comment tree graph 是指评论树的图结构, 是一个异质图
        其中 post 节点表示微博的原始内容, comment 节点表示评论的内容,
        [comment, comment_on, post] 表示对 post 的comment, [comment, reply, comment] 表示评论之间的边
        **区别于 GetDataset_comment_graph() 的地方在于, 这里的 graph 节点有时间戳信息**
        
        Parameters:
        -------
            args: argparse.Namespace
                参数集合
        
        Returns:
        -------
            train_dataloader: torch.utils.data.DataLoader
                训练数据集加载器
            test_dataloader: torch.utils.data.DataLoader
                测试数据集加载器
        
        数据形式(示例如下):
        -------
            HeteroDataBatch(
                post={
                    x=[64, 1536],
                    post_id=[64],
                    y=[64],
                    batch=[64],
                    text_feature=[64, 768],
                    image_feature=[64, 768],
                    ptr=[65],
                },
                comment={
                    x=[13396, 768],
                    comment_id=[64],
                    batch=[13396],
                    ptr=[65],
                },
                (comment, comments_on, post)={
                    edge_time=[9942, 64],
                    edge_index=[2, 9942],
                },
                (comment, replies_to, comment)={
                    edge_time=[3454, 64],
                    edge_index=[2, 3454],
                }
            )
    '''
    ## 相关参数
    dataset_name = args.dataset_name
    dataset_graph_embeddings_dir = args.dataset_graph_embeddings_dir
    batch_size = args.batch_size

    train_dataset_dir = dataset_graph_embeddings_dir.format(dataset_name, 'train')
    test_dataset_dir = dataset_graph_embeddings_dir.format(dataset_name, 'test')
    
    ## 之前的实验都是 'cls' 做的实验, 并且都是无标识的
    if args.roberta_feature_choice == 'cls':
        train_dataset_dir = train_dataset_dir.replace('.pt', '_time_'+str(args.time_feature_dim)+'d.pt')
        test_dataset_dir = test_dataset_dir.replace('.pt', '_time_'+str(args.time_feature_dim)+'d.pt')
    else:
        train_dataset_dir = train_dataset_dir.replace('.pt', '_time_'+str(args.time_feature_dim)+'d_{}.pt'.format(args.roberta_feature_choice))     ## 获得带有时间的 .pt 数据
        test_dataset_dir = test_dataset_dir.replace('.pt', '_time_'+str(args.time_feature_dim)+'d_{}.pt'.format(args.roberta_feature_choice))       ## 获得带有时间的 .pt 数据
        
    train_dataset = torch.load(train_dataset_dir, weights_only=False)
    test_datset = torch.load(test_dataset_dir, weights_only=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_datset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader