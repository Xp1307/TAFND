import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

class HgtGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, metadata, num_heads=8):
        '''
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
        super().__init__()

        self.post_linear = nn.Linear(input_dim, hidden_dim)

        self.conv1 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)
        self.conv2 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)

    def forward(self, data):
        x_dict = data.x_dict                                    
        edge_index_dict = data.edge_index_dict                 
        x_dict['post'] = self.post_linear(x_dict['post'])      
        x_dict = self.conv1(x_dict, edge_index_dict)            
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}      
        x_dict = self.conv2(x_dict, edge_index_dict)           
        return x_dict['post']                                  
    
class HgtGNNPlus(nn.Module):
    def __init__(self, input_dim, hidden_dim, metadata, num_heads=8, dropout=0.2):
        super().__init__()

        self.post_linear = nn.Linear(input_dim, hidden_dim)

        self.conv1 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        x_dict['post'] = self.post_linear(x_dict['post'])
        x_dict['post'] = self.norm1(x_dict['post'])

        res1 = x_dict
        x1 = self.conv1(x_dict, edge_index_dict)
        x1 = {k: self.dropout(F.relu(self.norm1(x1[k] + res1[k]))) for k in x1} 
        
        res2 = x1
        x2 = self.conv2(x1, edge_index_dict)
        x2 = {k: self.dropout(F.relu(self.norm2(x2[k] + res2[k]))) for k in x2}

        return x2['post']
