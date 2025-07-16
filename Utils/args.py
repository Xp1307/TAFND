import argparse

def weibo_args():
    parser = argparse.ArgumentParser(description="Arguments...")

    parser.add_argument('--seed', type=int, default=114514, help="Random Seed.")

    parser.add_argument('--dataset_name', type=str, default='weibo', help="Name of the dataset.")
    parser.add_argument('--dataset_dir', type=str, 
                        default='Datamodule/dataset/{}/data_json/{}.json', 
                        help="Root directory of datasets.")
    parser.add_argument('--dataset_embeddings_dir', type=str, 
                        default='DataModule/data/{}_{}_{}.pt', 
                        help="Root directory of datasets' texts and images in embeddings format.")

    parser.add_argument('--dataset_graph_embeddings_dir', type=str, 
                        default='DataModule/data/{}_{}_graph.pt', 
                        help="Root directory of datasets' graph in embeddings format.")
    parser.add_argument('--dataset_raw_graph_embeddings_dir', type=str, 
                        default='DataModule/data/{}_comment_tree_graph', 
                        help="Root directory of datasets' graph in embeddings format without text and image features.")

    parser.add_argument('--time_feature_dim', type=int, 
                        default=64, 
                        help="Dimension of time feature.")

    parser.add_argument('--device_id', type=int, default=0, help="Number of GPU device.")
    parser.add_argument('--num_classes', type=int, default=2, help="Class number of dataset.")

    parser.add_argument('--roberta_feature_choice', type=str, 
                        choices=["cls", "mean", "max"], default="cls", 
                        help='Available roberta feature extraction choices: "cls", "mean", or "max".')

    parser.add_argument('--pretrained_models_dir', type=str, 
                        default='PretrainedModels/', 
                        help="Root directory of pretrained models.")
    parser.add_argument('--text_pretrained_model_name', 
                        type=str, default='chinese-roberta-wwm-ext', 
                        help="Name of text pretrained model.")
    parser.add_argument('--image_pretrained_model_name', 
                        type=str, default='vit-base-patch16-224-in21k', 
                        help="Name of image pretrained model.")

    parser.add_argument('--image_dim', type=int, default=768, help="Dimension of image feature.")
    parser.add_argument('--text_dim', type=int, default=768, help="Dimension of text feature.")
    parser.add_argument('--embed_dim', type=int, default=768, help="Dimension of image feature.")
    parser.add_argument('--hidden_dim', type=int, default=768, help="Dimension of hidden layer.")
    parser.add_argument('--output_dim', type=int, default=2, help="Dimension of ouput layer.")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Drop out rate of model training.")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of multi head attention.")
    parser.add_argument('--num_heads_gnn', type=int, default=8, help="Number of graph attention.")

    parser.add_argument('--num_epochs', type=int, default=30, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--lr', type=float, default=2e-3, help="Learning rate.")
    return parser.parse_args()

def fakeddit_args():
    parser = argparse.ArgumentParser(description="Arguments...")

    parser.add_argument('--seed', type=int, default=114514, help="Random Seed.")

    parser.add_argument('--dataset_name', type=str, default='fakeddit', help="Name of the dataset.")
    parser.add_argument('--dataset_dir', type=str, 
                        default='DataModule/dataset/{}/data_json/{}.json', 
                        help="Root directory of datasets.")
    parser.add_argument('--dataset_embeddings_dir', type=str, 
                        default='DataModule/data/{}_{}.pt', 
                        help="Root directory of datasets' texts and images in embeddings format.")

    parser.add_argument('--dataset_graph_embeddings_dir', type=str, 
                        default='DataModule/data/{}_{}_graph.pt', 
                        help="Root directory of datasets' graph in embeddings format.")
    parser.add_argument('--dataset_raw_graph_embeddings_dir', type=str, 
                        default='DataModule/data/{}_comment_tree_graph', 
                        help="Root directory of datasets' graph in embeddings format without text and image features.")

    parser.add_argument('--time_feature_dim', type=int, 
                        default=64, 
                        help="Dimension of time feature.")

    parser.add_argument('--device_id', type=int, default=0, help="Number of GPU device.")
    parser.add_argument('--num_classes', type=int, default=2, help="Class number of dataset.")

    parser.add_argument('--roberta_feature_choice', type=str, 
                        choices=["cls", "mean", "max"], default="cls", 
                        help='Available roberta feature extraction choices: "cls", "mean", or "max".')

    parser.add_argument('--pretrained_models_dir', type=str, 
                        default='PretrainedModels/', 
                        help="Root directory of pretrained models.")
    parser.add_argument('--text_pretrained_model_name', 
                        type=str, default='roberta-base', 
                        help="Name of text pretrained model.")
    parser.add_argument('--image_pretrained_model_name', 
                        type=str, default='vit-base-patch16-224-in21k', 
                        help="Name of image pretrained model.")

    parser.add_argument('--image_dim', type=int, default=768, help="Dimension of image feature.")
    parser.add_argument('--text_dim', type=int, default=768, help="Dimension of text feature.")
    parser.add_argument('--embed_dim', type=int, default=768, help="Dimension of image feature.")
    parser.add_argument('--hidden_dim', type=int, default=768, help="Dimension of hidden layer.")
    parser.add_argument('--output_dim', type=int, default=2, help="Dimension of ouput layer.")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Drop out rate of model training.")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of multi head attention.")
    parser.add_argument('--num_heads_gnn', type=int, default=8, help="Number of graph attention.")

    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    return parser.parse_args()