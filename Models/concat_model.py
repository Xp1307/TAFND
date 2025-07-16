import torch
import torch.nn as nn
from transformers import ViTModel, RobertaModel, BertModel
from TreeModule.hgt_propagation import HgtGNN, HgtGNNPlus

class CombinedModel(torch.nn.Module):
    def __init__(self, vit_model, roberta_model):
        super(CombinedModel, self).__init__()
        self.vit_model = vit_model
        self.roberta_model = roberta_model
        self.fc = torch.nn.Linear(768 + 768, 2)

    def forward(self, image, text_inputs):
        vit_features = self.vit_model(image).last_hidden_state[:, 0, :]
        roberta_features = self.roberta_model(**text_inputs).last_hidden_state[:, 0, :]
        # vit_features.shape = torch.Size([64, 768]
        combined_features = torch.cat((vit_features, roberta_features), dim=1)
        output = self.fc(combined_features)
        return output

class CombinedModel2(torch.nn.Module):
    def __init__(self, args):
        super(CombinedModel2, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        self.linear_1 = nn.Linear(args.image_dim, args.embed_dim).to(self.device)
        self.linear_2 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_dim, int(args.hidden_dim/2)),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate), 
            nn.Linear(int(args.hidden_dim/2), args.output_dim),
        ).to(self.device)
        
    def forward(self, text_features, image_features):
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)

        combined_features = torch.cat((text_features, image_features), dim=1).to(self.device_id)
        output = self.mlp(combined_features)
        return output

class CombinedModel2(torch.nn.Module):
    def __init__(self, args):
        super(CombinedModel2, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        self.linear_1 = nn.Linear(args.image_dim, args.embed_dim).to(self.device)
        self.linear_2 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)
        
        self.mlp = nn.Sequential(
            nn.Linear(args.image_dim+args.text_dim, args.hidden_dim),
            nn.ReLU(),                                              
            nn.Dropout(args.dropout_rate),                         
            nn.Linear(args.hidden_dim, args.output_dim),     
        ).to(self.device)
        
        self.mlp_single = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),                                             
            nn.Dropout(args.dropout_rate),                          
            nn.Linear(args.hidden_dim, args.output_dim),
        ).to(self.device)
        
    def forward(self, text_features, image_features):
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)

        combined_features = torch.cat((text_features, image_features), dim=1).to(self.device_id)
        output = self.mlp(combined_features)
        
        # output = self.mlp_single(image_features)
        return output

class fakeddit_CombinedModel2(torch.nn.Module):
    def __init__(self, args):
        super(fakeddit_CombinedModel2, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        self.linear_1 = nn.Linear(args.image_dim, args.embed_dim).to(self.device)
        self.linear_2 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(args.image_dim+args.text_dim, args.hidden_dim),
            nn.ReLU(),                                              
            nn.Dropout(args.dropout_rate),                         
            nn.Linear(args.hidden_dim, args.output_dim),    
        ).to(self.device)
        
        self.mlp_single = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),    
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),                      
            nn.Linear(args.hidden_dim, args.output_dim),    
        ).to(self.device)
        
    def forward(self, text_features, image_features):
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)

        combined_features = torch.cat((text_features, image_features), dim=1).to(self.device_id)
        output = self.mlp(combined_features)

        return output

class CombinedModel3(torch.nn.Module):
    def __init__(self, args, metadata):
        super(CombinedModel3, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        self.linear_1 = nn.Linear(args.image_dim, args.embed_dim).to(self.device)
        self.linear_2 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)
        self.gnn = HgtGNN(1536, args.hidden_dim, metadata, 
                          num_heads=args.num_heads_gnn).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(args.text_dim+args.image_dim+args.embed_dim, args.hidden_dim),
            nn.ReLU(),                                                                  
            nn.Dropout(args.dropout_rate),                                             
            nn.Linear(args.hidden_dim, args.output_dim),                                
        ).to(self.device)

        self.mlp_single = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),   
            nn.ReLU(),                                          
            nn.Dropout(args.dropout_rate),                        
            nn.Linear(args.hidden_dim, args.output_dim),    
        ).to(self.device)
        
    def forward(self, text_features, image_features, comment_graph):
  
        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        comment_graph = comment_graph.to(self.device_id)
        

        enhanced_post_features = self.gnn(comment_graph)
        

        combined_features = torch.cat((text_features, image_features, enhanced_post_features), dim=1).to(self.device_id)
        output = self.mlp(combined_features)
        
        return output, combined_features
    
class fakeddit_CombinedModel3(torch.nn.Module):
    def __init__(self, args, metadata):
        super(fakeddit_CombinedModel3, self).__init__()
        self.device_id = args.device_id
        self.device = torch.device("cuda:{}".format(self.device_id) if torch.cuda.is_available() else "cpu")
        self.linear_1 = nn.Linear(args.image_dim, args.embed_dim).to(self.device)
        self.linear_2 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)
        self.linear_3 = nn.Linear(args.text_dim, args.embed_dim).to(self.device)
        
        self.gnn = HgtGNN(1536, args.hidden_dim, metadata, 
                          num_heads=args.num_heads_gnn).to(self.device)
        
        # 这个模块的作用是按顺序执行里面的模块
        self.mlp = nn.Sequential(
            nn.Linear(args.text_dim+args.image_dim+args.embed_dim, args.hidden_dim),     
            nn.ReLU(),                                                                   
            nn.Dropout(args.dropout_rate),                                               
            nn.Linear(args.hidden_dim, 256),
            nn.Linear(256, 128), 
            nn.Linear(128, args.output_dim),
        ).to(self.device)

        self.mlp_single = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),     
            nn.ReLU(),                                              
            nn.Dropout(args.dropout_rate),                         
            nn.Linear(args.hidden_dim, args.output_dim),    
        ).to(self.device)
    
    def forward(self, text_features, image_features, comment_graph):

        image_features = image_features.to(self.device_id)
        text_features = text_features.to(self.device_id)
        comment_graph = comment_graph.to(self.device_id)
        
        enhanced_post_features = self.gnn(comment_graph)
                
        combined_features = torch.cat((text_features, image_features, enhanced_post_features), dim=1).to(self.device_id)    # 拼接特征
        output = self.mlp(combined_features)
               
        return output, combined_features

def load_pretrained_models(args):    
    pretrained_models_dir = args.pretrained_models_dir
    image_pretrained_model_name = args.image_pretrained_model_name
    text_pretrained_model_name = args.text_pretrained_model_name

    device_id = args.device_id
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
    
    vit_model = ViTModel.from_pretrained(pretrained_models_dir + image_pretrained_model_name).to(device)
    if args.text_pretrained_model_name == 'roberta-base':
        roberta_model = RobertaModel.from_pretrained(pretrained_models_dir + text_pretrained_model_name).to(device)
    elif args.text_pretrained_model_name == 'chinese-roberta-wwm-ext':
        roberta_model = BertModel.from_pretrained(pretrained_models_dir + text_pretrained_model_name).to(device)
    else:
        roberta_model = RobertaModel.from_pretrained(pretrained_models_dir + text_pretrained_model_name).to(device)        
    
    for param in vit_model.parameters():
        param.requires_grad = False
    for param in roberta_model.parameters():
        param.requires_grad = False
    return vit_model, roberta_model