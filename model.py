import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import sympy
import scipy
import numpy as np
from dgl.nn.pytorch import GATConv, GraphConv, HeteroGraphConv

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import sympy
import scipy.special
from torch.nn import init

def min_max_normalize(tensor):
    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    range_val = max_val - min_val
    range_val = torch.clamp(range_val, min=1e-8)  # 防止除零错误
    return (tensor - min_val) / range_val


class PolyConv(nn.Module):
    def __init__(self, in_feats, out_feats, theta, activation=F.leaky_relu, lin=False, bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias) 
        self.lin = lin

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k] * feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for j in range(d+1):
            inv_coeff.append(float(coeff[d-j]))
        thetas.append(inv_coeff)
    return thetas

class BWGNNModule(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, graph, d, mode, batch=False):
        super(BWGNNModule, self).__init__()
        self.g = graph
        self.mode = mode
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList()
        for theta in self.thetas:
            self.conv.append(PolyConv(in_feats, h_feats, theta, lin=False))
        
        self.linear1 = nn.Linear(in_feats, h_feats).to(graph.device)
        self.linear2 = nn.Linear(h_feats, h_feats).to(graph.device)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats).to(graph.device)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, feat):
        h = self.linear1(feat)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.act(h)
        h = self.dropout(h)
        
        h_final = torch.zeros(feat.size(0), 0).to(feat.device)
        for conv in self.conv:
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], dim=-1)
        combined_feat = self.linear3(h_final)
        combined_feat = self.act(combined_feat)
        combined_feat = self.dropout(combined_feat)

        return combined_feat

class BWGNN_Hetero(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d, mode, batch=False):
        super(BWGNN_Hetero, self).__init__()
        self.g = graph
        self.mode = mode
        self.thetas = calculate_theta2(d=d)
        self.h_feats = h_feats
        self.linear1_dict = nn.ModuleDict()
        self.linear2_dict = nn.ModuleDict()
        # print(self.g.canonical_etypes, self.g.etypes)
        # 为每种特征类型创建独立的线性层
        for relation in self.g.etypes:
            self.linear1_dict[relation] = nn.Linear(in_feats, h_feats)
            self.linear2_dict[relation] = nn.Linear(h_feats, h_feats)
        self.conv = [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]
        self.linear1 = nn.Linear(in_feats, h_feats).to(graph.device)
        self.linear2 = nn.Linear(h_feats, h_feats).to(graph.device)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats).to(graph.device)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        # print(self.thetas)
        # for param in self.parameters():
        #     print(type(param), param.size())

    def forward(self, in_feat):
        h_dict = {}
        # 遍历 in_feat 字典的所有键值对
        for feat_type, feat in in_feat.items():
            # 对每种类型的数据进行处理
            h = self.linear1_dict[feat_type](feat)
            h = self.act(h)
            h = self.dropout(h)
            h = self.linear2_dict[feat_type](h)
            h = self.act(h)
            h = self.dropout(h)
            # 将处理后的结果存入字典
            h_dict[feat_type] = h
        h_all = []
        
        for relation in self.g.etypes:
            h_final = torch.zeros([len(in_feat[relation]), 0]).to(self.g.device)
            for conv in self.conv:
                h0 = conv(self.g[relation], h_dict[relation])
                # print(h_final.shape, h0.shape)
                h_final = torch.cat([h_final, h0], -1)

            h = self.linear3(h_final)
            h_all.append(h)

        h_all = torch.stack(h_all).sum(0)
        h_all = self.act(h_all)
        h_all = self.dropout(h_all)
        # h_all = self.linear4(h_all)
        return h_all

class EdgePartitioner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EdgePartitioner, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 使用 Tanh 将输出范围限制在 [-1, 1]
        )

    def forward(self, edge_feat):
        # Classify edge, output in the range [-1, 1]
        edge_pred = self.classifier(edge_feat)
        return edge_pred.squeeze()


class GraphPartitionModule(nn.Module):
    def __init__(self, in_feats, h_feats, graph, pos_quantile, neg_quantile, d, mode, num_heads=8):
        super(GraphPartitionModule, self).__init__()
        self.mode = mode
        self.pos_quantile = pos_quantile
        self.neg_quantile = neg_quantile
        self.thetas = calculate_theta2(d=d)  # 获取多项式卷积的系数
        self.convs = nn.ModuleList([PolyConv(in_feats, h_feats, theta) for theta in self.thetas]).to(graph.device)
        self.act = nn.ReLU()
        # 为每种边类型创建独立的线性层
        self.linear1_dict = nn.ModuleDict()
        self.linear2_dict = nn.ModuleDict()
        self.linear3_dict = nn.ModuleDict()

        for etype in graph.etypes:
            self.linear1_dict[etype] = nn.Linear(in_feats, h_feats)
            self.linear2_dict[etype] = nn.Linear(h_feats, h_feats)
            self.linear3_dict[etype] = nn.Linear(h_feats * len(self.convs), h_feats)
        
        self.linear1 = nn.Linear(in_feats, h_feats).to(graph.device)
        self.linear2 = nn.Linear(h_feats, h_feats).to(graph.device)
        self.linear3 = nn.Linear(h_feats * len(self.convs), h_feats).to(graph.device)
        self.dropout = nn.Dropout(0.5).to(graph.device)

    def forward(self, g, feat, edge_preds):
        output_list_all = []
        for etype, edge_pred in edge_preds.items():
            graph = g[etype]
            # 计算正边和负边的阈值
            neg_edge_preds = edge_pred[edge_pred <= 0]  # 筛选出负值
            pos_edge_preds = edge_pred[edge_pred > 0]  # 筛选出正值

            # 对正边和负边分别使用不同的分位数计算阈值
            neg_threshold_value = torch.quantile(neg_edge_preds, self.neg_quantile) if len(neg_edge_preds) > 0 else torch.tensor(0.0).to(edge_pred.device)
            pos_threshold_value = torch.quantile(pos_edge_preds, self.pos_quantile) if len(pos_edge_preds) > 0 else torch.tensor(0.0).to(edge_pred.device)

            # 根据阈值筛选边
            keep_edges_neg = (edge_pred < neg_threshold_value)
            keep_edges_pos = (edge_pred > pos_threshold_value)

            keep_edges_unknown = (edge_pred <= pos_threshold_value) & (edge_pred >= neg_threshold_value)
            subgraph_neg = dgl.edge_subgraph(graph, keep_edges_neg, relabel_nodes=False)
            subgraph_pos = dgl.edge_subgraph(graph, keep_edges_pos, relabel_nodes=False)
            subgraph_unknown = dgl.edge_subgraph(graph, keep_edges_unknown, relabel_nodes=False)
            # 使用独立的线性层处理特征
            h = self.linear1_dict[etype](feat)
            h = self.act(h)
            h = self.dropout(h)
            h = self.linear2_dict[etype](h)
            h = self.act(h)
            h = self.dropout(h)
            output_list = []
            # 使用独立的卷积层处理正、负子图
            h_pos = self.convs[len(self.convs) - 1](subgraph_pos, h)
            h_neg = self.convs[0](subgraph_neg, h)

            # 遍历独立的卷积层，处理未知子图
            for i in range(1, len(self.convs) - 1):
                h_layer = self.convs[i](subgraph_unknown, h)
                output_list.append(h_layer)

            # 将列表中的特征进行拼接
            h_unknown = torch.cat(output_list, dim=-1)
            # 节点特征融合
            h_final = torch.cat([h_pos, h_unknown, h_neg], dim=-1)
            combined_feat = self.linear3_dict[etype](h_final)
            combined_feat = self.act(combined_feat)
            combined_feat = self.dropout(combined_feat)
            output_list_all.append(combined_feat)
        final_out = torch.stack(output_list_all).sum(0)

        return final_out
    

class NodeReconstructionModule(nn.Module):
    def __init__(self, in_feats, h_feats, graph, d, num_layers=2):
        super(NodeReconstructionModule, self).__init__()

        self.num_layers = num_layers
        # 编码器：为每个边类型定义不同的卷积层
        self.convs = nn.ModuleDict()
        for feat_type in graph.etypes:
            # 假设图有多个边类型，例如 'homo', 'u', 't', 's'
            self.convs[feat_type] = GraphConv(in_feats, h_feats, activation=F.relu, allow_zero_in_degree=True)

        # 使用 HeteroGraphConv 来组合不同边类型的卷积层
        self.hetero_conv = HeteroGraphConv(self.convs)

        # 解码器：一个更复杂的 MLP 解码器
        self.reconstruction_layer = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, in_feats)
        )

    def forward(self, graph, feat):
        # 假设 feat 是一个字典，其中包含每个边类型对应的节点特征
        h = {etype: feat for etype in graph.etypes}  # 构建字典，包含每种边类型的特征
        
#       # 对每种边类型，分别进行图卷积
        for etype in graph.etypes:
            # 创建对应边类型的子图
            subgraph = dgl.edge_type_subgraph(graph, [etype])
            h[etype] = self.convs[etype](subgraph, h[etype])  # 针对每个边类型使用对应的卷积层
        
        # 解码器部分：通过 MLP 重建节点特征
        # 动态地将所有边类型对应的特征相加
        combined_feat = sum(h.values())
        reconstructed_feat = self.reconstruction_layer(combined_feat)
        return reconstructed_feat, h  # 返回重建的特征和节点隐式表示

class NeighborTypeAwareGraphAttention(nn.Module):
    def __init__(self, in_feats, h_feats, graph, pos_quantile, neg_quantile, mode, num_heads=8, num_layers=2):
        super(NeighborTypeAwareGraphAttention, self).__init__()
        self.mode = mode
        self.pos_quantile = pos_quantile
        self.neg_quantile = neg_quantile
        self.num_heads = num_heads
        self.num_layers = num_layers  # 设置GAT的层数
        # 用于存储每种边类型的GAT层和线性层
        self.gat_dict = nn.ModuleDict()
        self.gat_pos_dict = nn.ModuleDict()
        self.gat_neg_dict = nn.ModuleDict()
        self.gat_unknown_dict = nn.ModuleDict()
        self.output_linear_dict = nn.ModuleDict()
        self.linear1_dict = nn.ModuleDict()
        self.linear2_dict = nn.ModuleDict()
        self.linear3_dict = nn.ModuleDict()
        self.interaction_weights_dict = nn.ParameterDict()
        # 为每种边类型创建独立的模块
        for etype in graph.etypes:
            self.gat_dict[etype] = GATConv(in_feats, h_feats, self.num_heads, allow_zero_in_degree=True)
            self.gat_pos_dict[etype] = GATConv(in_feats, h_feats, self.num_heads, allow_zero_in_degree=True)
            self.gat_neg_dict[etype] = GATConv(in_feats, h_feats, self.num_heads, allow_zero_in_degree=True)
            self.gat_unknown_dict[etype] = GATConv(in_feats, h_feats, self.num_heads, allow_zero_in_degree=True)
            self.output_linear_dict[etype] = nn.Linear(h_feats * self.num_heads, h_feats)
            self.linear1_dict[etype] = nn.Linear(in_feats, h_feats)
            self.linear2_dict[etype] = nn.Linear(in_feats, h_feats)
            self.linear3_dict[etype] = nn.Linear(in_feats, h_feats)
            self.interaction_weights_dict[etype] = nn.Parameter(torch.ones(3))

        self.gat = GATConv(in_feats, h_feats, num_heads, allow_zero_in_degree=True)
        self.gat_pos = GATConv(in_feats, h_feats, num_heads, allow_zero_in_degree=True)
        self.gat_neg = GATConv(in_feats, h_feats, num_heads, allow_zero_in_degree=True)
        self.gat_unknown = GATConv(in_feats, h_feats, num_heads, allow_zero_in_degree=True)

        # 用于融合不同头输出的线性层
        self.output_linear = nn.Linear(h_feats * num_heads, h_feats)
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(in_feats, h_feats)
        self.linear3 = nn.Linear(in_feats, h_feats)
        self.interaction_weights = nn.Parameter(torch.ones(3))  # 三种类型的权重
        

        # 激活函数
        self.act = nn.ReLU()

        # Dropout层
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, feat, edge_preds, mode='full'):
        feat = F.normalize(feat, p=2, dim=-1)  # 特征归一化
        output_list = []
        for etype, edge_pred in edge_preds.items():
            # 计算正边和负边的阈值
            neg_edge_preds = edge_pred[edge_pred <= 0]  # 筛选出负值
            pos_edge_preds = edge_pred[edge_pred > 0]  # 筛选出正值

            # 对正边和负边分别使用不同的分位数计算阈值
            neg_threshold_value = torch.quantile(neg_edge_preds, self.neg_quantile) if len(neg_edge_preds) > 0 else torch.tensor(0.0).to(edge_pred.device)
            pos_threshold_value = torch.quantile(pos_edge_preds, self.pos_quantile) if len(pos_edge_preds) > 0 else torch.tensor(0.0).to(edge_pred.device)
            # 根据阈值筛选边
            keep_edges_neg = (edge_pred < neg_threshold_value)
            keep_edges_pos = (edge_pred > pos_threshold_value)
            keep_edges_unknown = (edge_pred <= pos_threshold_value) & (edge_pred >= neg_threshold_value)

            # 创建三个子图：同配、异配、未知邻居
            subgraph_neg = dgl.edge_subgraph(graph[etype], keep_edges_neg, relabel_nodes=False)
            subgraph_pos = dgl.edge_subgraph(graph[etype], keep_edges_pos, relabel_nodes=False)
            subgraph_unknown = dgl.edge_subgraph(graph[etype], keep_edges_unknown, relabel_nodes=False)
            # 计算总和
            edge_pred = -edge_pred
            # 获取边的预测值的相反数作为边权重

            # 按边数归一化
            edge_weights_neg = F.softmax(edge_pred[keep_edges_neg], dim=0)
            edge_weights_pos = F.softmax(edge_pred[keep_edges_pos], dim=0)
            edge_weights_unknown = F.softmax(edge_pred[keep_edges_unknown], dim=0)
            # 将权重赋值给子图
            graph.edges[etype].data['weight'] = F.softmax(edge_pred)
            subgraph_neg.edata['weight'] = edge_weights_neg
            subgraph_pos.edata['weight'] = edge_weights_pos
            subgraph_unknown.edata['weight'] = edge_weights_unknown
            # 全局特征计算
            global_out = self.gat_dict[etype](graph[etype], feat, edge_weight=graph.edges[etype].data['weight']).flatten(1)

            # 子图特征计算
            same_out = self.gat_neg_dict[etype](subgraph_neg, feat, edge_weight=subgraph_neg.edata['weight']).flatten(1)
            diff_out = self.gat_pos_dict[etype](subgraph_pos, feat, edge_weight=subgraph_pos.edata['weight']).flatten(1)
            unknown_out = self.gat_unknown_dict[etype](subgraph_unknown, feat, edge_weight=subgraph_unknown.edata['weight']).flatten(1)

            # 特征交互：通过线性层融合
            combined_out = torch.stack([same_out, diff_out, unknown_out], dim=1)  # (batch_size, 3, feature_dim)
            interaction_weights = torch.softmax(self.interaction_weights_dict[etype], dim=0)  # Learnable weights
            shared_out = torch.sum(combined_out * interaction_weights.unsqueeze(-1), dim=1)

            # 融合全局特征
            feat_with_neighbors = shared_out + global_out
            
            # 通过线性层进一步处理
            out = self.output_linear_dict[etype](feat_with_neighbors)
            out = self.act(out)
            out = self.dropout(out)
            output_list.append(out)
        final_out = torch.stack(output_list).sum(0)

        return final_out

class CombinedModel(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, num_classes, graph, d, pos_quantile=0.5, neg_quantile=0.5, num_heads=8, mode='full'):
        super(CombinedModel, self).__init__()
        self.graph = graph
        self.mode = mode
        self.edge_classifiers = nn.ModuleDict()
        # 为每种边类型创建一个边分类器
        for etype in graph.etypes:
            self.edge_classifiers[etype] = EdgePartitioner(in_feats, h_feats).to(graph.device)

        self.bwgnn_module = BWGNN_Hetero(h_feats, h_feats, out_feats, graph, d, mode).to(graph.device)
        self.node_reconstruction_module = NodeReconstructionModule(in_feats, h_feats, graph, d).to(graph.device)
        self.neighbor_aware_module = NeighborTypeAwareGraphAttention(in_feats, h_feats, graph, pos_quantile, neg_quantile, mode, num_heads, 1).to(graph.device)
        self.graph_partition_module = GraphPartitionModule(in_feats, h_feats, graph, pos_quantile, neg_quantile, d, mode, num_heads=num_heads).to(graph.device)
        
        fusion_nums = 3

        # 构造特征融合模块
        self.feat_fusion = nn.Sequential(
            nn.Linear(h_feats * fusion_nums, h_feats),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_feats, num_classes),
        ).to(graph.device)
        # 构造特征融合模块
        self.linear = nn.Linear(h_feats, num_classes).to(graph.device)


    def forward(self, graph, feat):
        # 初始化一个空字典，用于存储每种边类型的边分类结果
        edge_pred = {}
        # 遍历图中所有的边类型
        for etype in graph.etypes:
            # 检查当前边类型是否已经存在边特征
            if 'edge_feat' not in graph.edges[etype].data:
                # 获取当前边类型的源节点和目标节点索引
                src_nodes, dst_nodes = graph.edges(etype=etype)
                # 根据源节点和目标节点索引获取对应的节点特征
                h_u = feat[src_nodes]
                h_v = feat[dst_nodes]
                # 计算源节点和目标节点特征的差值的绝对值
                h_diff = abs(h_u - h_v)
                # 将计算得到的边特征存储到图中当前边类型的数据中
                graph.edges[etype].data['edge_feat'] = h_diff
            # 获取当前边类型的边特征
            edge_feat = graph.edges[etype].data['edge_feat']
            # 使用边分类器对当前边类型的边特征进行分类
            e_pred = self.edge_classifiers[etype](edge_feat)
            # 将当前边类型的边分类结果存入字典
            edge_pred[etype] = e_pred
        
        reconstructed_feat, node_embeddings = self.node_reconstruction_module(graph, feat)
        # 归一化
        normalized_feat = min_max_normalize(feat)
        normalized_reconstructed_feat = min_max_normalize(reconstructed_feat)
        # 计算归一化后的均方误差损失
        reconstruction_loss = F.mse_loss(normalized_reconstructed_feat, normalized_feat)

        # 初始化一个列表来存储各个模块的输出
        outputs = []

        bwgnn_out = self.bwgnn_module(node_embeddings)
        outputs.append(bwgnn_out)
        neighbor_out = self.neighbor_aware_module(self.graph, feat, edge_pred)
        outputs.append(neighbor_out)
        partition_out = self.graph_partition_module(self.graph, feat, edge_pred)
        outputs.append(partition_out)

        # 拼接所有模块的输出
        combined_out = torch.cat(outputs, dim=1)
        
        # 通过解码器重建节点特征
        output= self.feat_fusion(combined_out)
        
        return output, edge_pred, reconstruction_loss
    
    def compute_loss(self, output, edge_preds, labels, edge_labels, node_mask, edge_masks, reconstruction_loss):
        # 获取被掩蔽的输出和标签
        masked_output = output[node_mask]
        masked_labels = labels[node_mask]
        # 节点分类损失计算
        num_classes = masked_labels.max().item() + 1
        class_counts = torch.bincount(masked_labels)
        total_count = masked_labels.size(0)
        class_weights = total_count / (len(class_counts) * class_counts.float())
        class_weights = class_weights.to(masked_output.device)  # 确保权重在正确的设备上
        node_loss = F.cross_entropy(masked_output, masked_labels, weight=class_weights)

        def calculate_edge_loss(edge_pred, edge_mask, edge_label):
            masked_edge_pred = edge_pred[edge_mask]
            masked_edge_labels = edge_label[edge_mask]
            # 处理边分类标签，将 -1 映射到 0
            edge_labels_adjusted = (masked_edge_labels + 1).long()  # -1 -> 0, 1 -> 1
            # 计算每个类别的样本数量
            edge_class_counts = torch.bincount(edge_labels_adjusted)
            edge_total_count = masked_edge_labels.size(0)
            edge_class_weights = edge_total_count / (len(edge_class_counts) * edge_class_counts.float())
            edge_class_weights = edge_class_weights.to(masked_edge_pred.device)  # 确保权重在正确的设备上
            # 使用加权均方误差损失
            def weighted_mse_loss(preds, targets, weights):
                mse_loss = (preds - targets) ** 2
                weighted_loss = weights * mse_loss
                return weighted_loss.mean()
            edge_weights = edge_class_weights[edge_labels_adjusted]
            e_loss = weighted_mse_loss(masked_edge_pred, masked_edge_labels.float(), edge_weights)
            return e_loss

        edge_losses = []
        if isinstance(edge_masks, torch.Tensor):
            for key1, edge_pred in edge_preds.items():
                edge_mask = edge_masks   
                edge_label = edge_labels
                e_loss = calculate_edge_loss(edge_pred, edge_mask, edge_label)
                edge_losses.append(e_loss)
        else:
            for (key1, edge_pred), (key2, edge_mask) in zip(edge_preds.items(), edge_masks.items()):
                edge_label = edge_labels[key2]
                e_loss = calculate_edge_loss(edge_pred, edge_mask, edge_label)
                edge_losses.append(e_loss)

        edge_losses_tensor = torch.stack(edge_losses)
        edge_loss = torch.sum(edge_losses_tensor)
        # 计算损失加权系数
        node_loss_weight = node_loss.detach() / (node_loss.detach() + edge_loss.detach() + reconstruction_loss.detach() + 1e-8)
        edge_loss_weight = edge_loss.detach() / (node_loss.detach() + edge_loss.detach() + reconstruction_loss.detach() + 1e-8)
        reconstruction_loss_weight = reconstruction_loss.detach() / (node_loss.detach() + edge_loss.detach() + reconstruction_loss.detach() + 1e-8)
        total_loss = node_loss * node_loss_weight + edge_loss * edge_loss_weight + reconstruction_loss * reconstruction_loss_weight
        return total_loss
    

