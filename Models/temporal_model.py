import torch
import torch.nn as nn
from TreeModule.hgt_propagation_time_aggr import THgtGNN_Aggr

class WeiboModel1(torch.nn.Module):
    def __init__(self, args, metadata):
        super(WeiboModel1, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        
        self.gnn = THgtGNN_Aggr(1536, args.hidden_dim, metadata, args,
                            num_heads=args.num_heads_gnn).to(self.device)
        
        self.mlp = nn.Sequential(
            nn.Linear(args.text_dim+args.image_dim+args.embed_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, args.output_dim),
        ).to(self.device)
        
    def forward(self, text_features, image_features, comment_graph):
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        comment_graph = comment_graph.to(self.device_id)

        enhanced_post_features = self.gnn(comment_graph)
        
        combined_features = torch.cat((text_features, image_features, enhanced_post_features), dim=1).to(self.device_id)    # 拼接特征
        output = self.mlp(combined_features)
        return output, combined_features
    

class FakedditModel1(torch.nn.Module):
    def __init__(self, args, metadata):
        super(FakedditModel1, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        
        self.gnn = THgtGNN_Aggr(1536, args.hidden_dim, metadata, args,
                            num_heads=args.num_heads_gnn).to(self.device)
        
        self.mlp = nn.Sequential(
            nn.Linear(args.text_dim+args.image_dim+args.embed_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, args.output_dim),
        ).to(self.device)
        
    def forward(self, text_features, image_features, comment_graph):
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        comment_graph = comment_graph.to(self.device_id)
        
        enhanced_post_features = self.gnn(comment_graph)
        combined_features = torch.cat((text_features, image_features, enhanced_post_features), dim=1).to(self.device_id)    # 拼接特征
        output = self.mlp(combined_features)
        return output, combined_features