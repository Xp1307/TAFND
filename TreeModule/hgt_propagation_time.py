import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

class THgtGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, metadata, args, num_heads=8):
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
                        x=[13396, 768],
                        comment_id=[64],
                        batch=[13396],
                        ptr=[65],
                    },
                    (comment, comments_on, post)={
                        edge_time=[9942, 768],
                        edge_index=[2, 9942],
                    },
                    (comment, replies_to, comment)={
                        edge_time=[3454, 768],
                        edge_index=[2, 3454],
                    }
                )
        '''
        super().__init__()

        self.post_linear = nn.Linear(input_dim, hidden_dim)
        self.comment_linear = nn.Linear(hidden_dim + args.time_feature_dim, hidden_dim)
        
        self.conv1 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)
        self.conv2 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)

    def forward(self, data):

        x_dict = data.x_dict                                   
        edge_index_dict = data.edge_index_dict                
        edge_time_dict = data.edge_time_dict


        x_dict['post'] = self.post_linear(x_dict['post'])       # [64, 768]
        

        comment_data = x_dict['comment']
        N = comment_data.size(0)
        comment_data_time = torch.zeros(N, 768 + 64, device=comment_data.device)
        for edge_index, edge_time in zip(edge_index_dict.values(), edge_time_dict.values()):
            src = edge_index[0]
            concat_feat  = torch.cat([comment_data[src], edge_time.to(comment_data.device)], dim=-1)
            comment_data_time[src] = concat_feat   
        x_dict['comment'] = comment_data_time
        x_dict['comment'] = self.comment_linear(x_dict['comment'])
        
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict['post']
