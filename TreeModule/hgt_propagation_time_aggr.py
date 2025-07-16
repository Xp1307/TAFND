import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_scatter import scatter_add


class THgtGNN_Aggr(nn.Module):
    def __init__(self, input_dim, hidden_dim, metadata, args, num_heads=8):
        super().__init__()

        self.post_linear = nn.Linear(input_dim, hidden_dim)
        self.comment_linear = nn.Linear(hidden_dim, hidden_dim)

        self.time_weight_layers = nn.ModuleDict({
            'comment__replies_to__comment': nn.Linear(args.time_feature_dim, 1),
            'comment__comments_on__post': nn.Linear(args.time_feature_dim, 1),
        })

        self.conv1 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)
        self.conv2 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_time_dict = data.edge_time_dict

        device = x_dict['comment'].device
        comment_feat = x_dict['comment']        # [N_comment, 768]
        post_feat = x_dict['post']              # [N_post, 1536]

        N_comment = comment_feat.size(0)
        N_post = post_feat.size(0)

        new_comment_feat = torch.zeros_like(comment_feat)             # [N_comment, 768]
        new_post_feat = torch.zeros(N_post, comment_feat.size(1), device=device)  # [N_post, 768]

        for (src_type, rel_type, dst_type), edge_index in edge_index_dict.items():
            if src_type != 'comment':
                continue
            rel_key = f'{src_type}__{rel_type}__{dst_type}'
            if rel_key not in self.time_weight_layers:
                continue
            if (src_type, rel_type, dst_type) not in edge_time_dict:
                continue

            edge_time = edge_time_dict[(src_type, rel_type, dst_type)]  # [E, time_dim]
            src, dst = edge_index

            time_weight_layer = self.time_weight_layers[rel_key]
            time_weight = torch.sigmoid(time_weight_layer(edge_time))  # [E, 1]
            weighted_msg = comment_feat[src] * time_weight             # [E, 768]

            if dst_type == 'comment':
                new_comment_feat = new_comment_feat.index_add(0, dst, weighted_msg)
            elif dst_type == 'post':
                new_post_feat = new_post_feat.index_add(0, dst, weighted_msg)

        x_dict['comment'] = comment_feat + new_comment_feat
        x_dict['post'] = self.post_linear(post_feat) + new_post_feat

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)

        return x_dict['post']
