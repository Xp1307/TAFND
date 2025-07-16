import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# 时间感知的 attention 层，融合时间向量影响 attention 权重
class TimeAwareAttentionGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, time_dim=64, heads=1, dropout=0.0):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.time_proj = nn.Linear(time_dim, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr):
        H, F = self.heads, self.out_channels
        x = self.lin(x).view(-1, H, F)                  # [N, H, F]
        time_emb = self.time_proj(edge_attr).view(-1, H, F)  # [E, H, F]
        return self.propagate(edge_index, x=x, time_emb=time_emb)

    def message(self, x_i, x_j, time_emb):
        x_j_time = x_j + time_emb  # 源节点 + 时间
        att_input = torch.cat([x_i, x_j_time], dim=-1)  # [E, H, 2F]
        alpha = (att_input * self.att).sum(dim=-1)      # [E, H]
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index=None)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j_time * alpha.unsqueeze(-1)            # [E, H, F]


class HeteroTGATLayer(nn.Module):
    def __init__(self, in_channels_dict, out_channels, relations, args):
        super().__init__()
        self.relations = relations
        self.out_channels = out_channels

        self.gats = nn.ModuleDict({
            rel: TimeAwareAttentionGATConv(
                in_channels=in_channels_dict['comment'],
                out_channels=out_channels,
                time_dim=args.time_feature_dim,
                heads=1,
                dropout=0.0
            ) for rel in relations
        })

        self.out_proj = nn.ModuleDict({
            'post': nn.Linear(out_channels, out_channels),
            'comment': nn.Linear(out_channels, out_channels)
        })

    def forward(self, x_dict, edge_index_dict, edge_time_dict):
        outputs = {
            'post': torch.zeros(x_dict['post'].size(0), self.out_channels, device=x_dict['post'].device),
            'comment': torch.zeros(x_dict['comment'].size(0), self.out_channels, device=x_dict['comment'].device)
        }

        for rel in self.relations:
            rel_tuple = ('comment', rel, 'post') if rel == 'comments_on' else ('comment', rel, 'comment')
            edge_index = edge_index_dict[rel_tuple]
            edge_time = edge_time_dict[rel_tuple]

            if edge_index.size(1) == 0:
                continue

            # src, dst = edge_index
            # src_feat = x_dict['comment'][src]
            
            msg_out = self.gats[rel](x_dict['comment'], edge_index, edge_time)
            outputs[rel_tuple[2]] += msg_out

        return {k: self.out_proj[k](v) for k, v in outputs.items()}


class HeteroTGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, metadata, args, num_layers=2, num_classes=2):
        super().__init__()
        self.post_linear = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            HeteroTGATLayer(
                in_channels_dict={'comment': hidden_dim, 'post': hidden_dim},
                out_channels=hidden_dim,
                relations=[rel for (_, rel, _) in metadata[1]],
                args=args
            ) for _ in range(num_layers)
        ])

    def forward(self, data):
        x_dict = data.x_dict
        x_dict['post'] = self.post_linear(x_dict['post'])

        for layer in self.layers:
            x_dict = layer(
                x_dict=x_dict,
                edge_index_dict=data.edge_index_dict,
                edge_time_dict=data.edge_time_dict
            )
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        return x_dict['post']
