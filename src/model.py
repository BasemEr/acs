import torch
from torch_geometric.data import Data
from torch_geometric.nn import NNConv
from torch_geometric.nn import aggr
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ModelWrapper(torch.nn.Module):
    def __init__(self, task_type, model_type, node_feature_size, edge_feature_size, 
                 num_props, mlp_multiplier, conv_layers, dropout_rate, all_aggregation):
        super(ModelWrapper, self).__init__()
        
        self.model_type = model_type
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.num_props = num_props
        self.mlp_multiplier = mlp_multiplier
        self.conv_layers = conv_layers
        self.dropout_rate = dropout_rate
        self.all_aggregation = all_aggregation
        self.task_type = task_type

        self.model = self.build_model()

    def build_model(self):
        model_list = nn.ModuleList()

        if self.model_type == "stl":
            for _ in range(self.num_props):
                model_list.append(STLModel(self.task_type, self.mlp_multiplier, self.node_feature_size, self.edge_feature_size, self.conv_layers, self.all_aggregation, num_props=1))  

        if self.model_type in ['acs', 'mtl', 'mtl-glc']:
            graph_model = GNN(self.node_feature_size, self.edge_feature_size, self.conv_layers, self.all_aggregation, self.num_props)
            model_list.append(graph_model)
            for _ in range(self.num_props):
                model_list.append(MLPModel(self.task_type, self.mlp_multiplier, self.all_aggregation, self.dropout_rate))

        return model_list


    def forward(self, batch: Data):
        if self.model_type == "stl":
            x = []
            for model in self.model:
                x.append(model(batch))
            return torch.cat(x, dim=1)
        
        else:
            x = self.model[0](batch) # Graph forward propagation
            x_mlp = []
            for model in self.model[1:]:
                x_mlp.append(model(x)) # STL MLP forward propagation
            return torch.cat(x_mlp, dim=1)


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class STLModel(torch.nn.Module):
    def __init__(self,
                 task_type,
                 mlp_multiplier: int,
                 node_feature_size: int,
                 edge_feature_size: int,
                 conv_layers: int,
                 all_aggregation: bool,
                 num_props: int) -> None:

        super().__init__()

        self.task_type = task_type
        self.mlp_multiplier = mlp_multiplier
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.num_props = num_props
        self.conv_layers = conv_layers
        self.all_aggregation = all_aggregation

        if self.all_aggregation:
            self.aggr_multip = 6
        else:
            self.aggr_multip = 2

        MAX_ATOMIC_NUM = 100
        atom_emb_size = 4
        self.atomic_num_emb = torch.nn.Embedding(MAX_ATOMIC_NUM, atom_emb_size)

        num_as_is_features = node_feature_size - 1
        in_ch = atom_emb_size + num_as_is_features

        channel_config = self.conv_layers * [256]

        conv_list = []
        for out_ch in channel_config:
            edge_inter_dim = 8
            edge_transform = nn.Sequential(
                nn.Linear(edge_feature_size, edge_inter_dim),
                nn.ReLU(),
                nn.Linear(edge_inter_dim, in_ch*out_ch),
                LambdaModule(lambda x: (1 / (in_ch*out_ch)) * x)
            )
            conv = NNConv(in_ch, out_ch, nn=edge_transform)
            conv_list.append(conv)
            in_ch = out_ch
        self.convs = torch.nn.ModuleList(conv_list)

        self.aggr_mean = aggr.MeanAggregation()
        self.aggr_sum = aggr.SumAggregation()
        self.aggr_max = aggr.MaxAggregation()
        self.aggr_min = aggr.MinAggregation()
        self.aggr_softmax = aggr.SoftmaxAggregation()
        self.aggr_power_mean = aggr.PowerMeanAggregation(p=2)

        self.fc1 = nn.Linear(self.aggr_multip*256, int(256*mlp_multiplier)) 
        self.fc2 = nn.Linear(int(256*mlp_multiplier), int(128*mlp_multiplier))
        self.fc3 = nn.Linear(int(128*mlp_multiplier), int(64*mlp_multiplier))
        self.fc4 = nn.Linear(int(64*mlp_multiplier), int(32*mlp_multiplier))
        self.fc5 = nn.Linear(int(32*mlp_multiplier), int(16*mlp_multiplier))
        self.fc6 = nn.Linear(int(16*mlp_multiplier), self.num_props)
    
    def forward(self, batch: Data):
        atomic_nums = batch.x[:, 1]
        atomic_num_embs = self.atomic_num_emb(atomic_nums)

        node_features_as_is = batch.x[:, 1:].float()

        x = torch.cat((atomic_num_embs, node_features_as_is), dim=1)

        for conv in self.convs:
            edge_attr = batch.edge_attr.float()
            x = checkpoint(conv, x, batch.edge_index, edge_attr)
            x = torch.relu(x)

        xmean = self.aggr_mean(x, batch.batch)
        xsum = self.aggr_sum(x, batch.batch)
        xmax = self.aggr_max(x, batch.batch)
        xmin = self.aggr_min(x, batch.batch)
        xsoftmax = self.aggr_softmax(x, batch.batch)
        xpowermean = self.aggr_power_mean(x, batch.batch)

        if self.all_aggregation:
            x = torch.cat([xmean, xsum, xmax, xmin, xsoftmax, xpowermean], dim=-1)
        else:
            x = torch.cat([xmean, xsum], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        if self.task_type == "classification":
            x = torch.sigmoid(self.fc6(x))
        else:
            x = self.fc6(x)

        return x

class GNN(torch.nn.Module):
    def __init__(self,
                 node_feature_size: int,
                 edge_feature_size: int,
                 conv_layers: int,
                 all_aggregation: bool,
                 num_props: int) -> None:

        super().__init__()

        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.num_props = num_props
        self.conv_layers = conv_layers
        self.all_aggregation = all_aggregation


        MAX_ATOMIC_NUM = 100
        atom_emb_size = 4
        self.atomic_num_emb = torch.nn.Embedding(MAX_ATOMIC_NUM, atom_emb_size)

        num_as_is_features = node_feature_size - 1
        in_ch = atom_emb_size + num_as_is_features

        channel_config = self.conv_layers * [256]

        conv_list = []
        for out_ch in channel_config:
            edge_inter_dim = 8
            edge_transform = nn.Sequential(
                nn.Linear(edge_feature_size, edge_inter_dim),
                nn.ReLU(),
                nn.Linear(edge_inter_dim, in_ch*out_ch),
                LambdaModule(lambda x: (1 / (in_ch*out_ch)) * x)
            )
            conv = NNConv(in_ch, out_ch, nn=edge_transform)
            conv_list.append(conv)
            in_ch = out_ch
        self.convs = torch.nn.ModuleList(conv_list)

        self.aggr_mean = aggr.MeanAggregation()
        self.aggr_sum = aggr.SumAggregation()
        self.aggr_max = aggr.MaxAggregation()
        self.aggr_min = aggr.MinAggregation()
        self.aggr_softmax = aggr.SoftmaxAggregation()
        self.aggr_power_mean = aggr.PowerMeanAggregation(p=2)

    
    def forward(self, batch: Data):
        atomic_nums = batch.x[:, 1]
        atomic_num_embs = self.atomic_num_emb(atomic_nums)

        node_features_as_is = batch.x[:, 1:].float()

        x = torch.cat((atomic_num_embs, node_features_as_is), dim=1)

        for conv in self.convs:
            edge_attr = batch.edge_attr.float()
            x = conv(x, batch.edge_index, edge_attr)
            x = torch.relu(x)

        xmean = self.aggr_mean(x, batch.batch)
        xsum = self.aggr_sum(x, batch.batch)
        xmax = self.aggr_max(x, batch.batch)
        xmin = self.aggr_min(x, batch.batch)
        xsoftmax = self.aggr_softmax(x, batch.batch)
        xpowermean = self.aggr_power_mean(x, batch.batch)

        if self.all_aggregation:
            x = torch.cat([xmean, xsum, xmax, xmin, xsoftmax, xpowermean], dim=-1)
        else:
            x = torch.cat([xmean, xsum], dim=-1)

        return x


class MLPModel(torch.nn.Module):
    def __init__(self, task_type, mlp_multiplier: int, all_aggregation, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.task_type = task_type

        if all_aggregation:
            aggr_multip = 6
        else:
            aggr_multip = 2

        self.fc1 = nn.Linear(aggr_multip*256, int(256*mlp_multiplier)) # 2*256 because of the concatenation of xm and xs in graph model
        self.fc2 = nn.Linear(int(256*mlp_multiplier), int(128*mlp_multiplier))
        self.fc3 = nn.Linear(int(128*mlp_multiplier), int(64*mlp_multiplier))
        self.fc4 = nn.Linear(int(64*mlp_multiplier), int(32*mlp_multiplier))
        self.fc5 = nn.Linear(int(32*mlp_multiplier), int(16*mlp_multiplier))
        self.fc6 = nn.Linear(int(16*mlp_multiplier), 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        if self.task_type == "classification":
            x = torch.sigmoid(self.fc6(x))
        else:
            x = self.fc6(x)
        
        return x

