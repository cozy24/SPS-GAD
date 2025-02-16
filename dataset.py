import pickle
import dgl
import torch
from dgl.data import DGLDataset, FraudYelpDataset, FraudAmazonDataset, CoraGraphDataset, RedditDataset
from dgl.data.utils import load_graphs, save_graphs
import os
from sklearn.model_selection import train_test_split
import numpy as np
from pygod.utils import load_data as pygod_load_data
import pandas as pd
import numpy as np

class YelpDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='yelp')

    def process(self):
        # 从文件中加载图数据
        dataset = FraudYelpDataset()
        # dataset = FraudAmazonDataset()
        self.graph = dataset[0]

        # 将图转换为同构图
        self.graph = dgl.to_homogeneous(self.graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])

        # 生成训练、验证和测试掩码
        num_nodes = self.graph.num_nodes()
        num_edges = self.graph.num_edges()
        # 生成边标签
        src, dst = self.graph.edges()
        node_labels = self.graph.ndata['label']
        edge_labels = (2 * (node_labels[src] == node_labels[dst]) - 1).long()
        self.graph.edata['label'] = edge_labels
        # 打印边标签
        # print("Edge labels:", edge_labels)  

        train_ratio = 0.4
        val_ratio = 0.2
        test_ratio = 0.4

        index = list(range(len(self.graph.ndata['label'])))
        labels = self.graph.ndata['label']
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index].cpu().detach().numpy(),  # 移动到CPU并转换为NumPy数组
                                                            stratify=labels[index].cpu().detach().numpy(),  # 移动到CPU并转换为NumPy数组
                                                            train_size=train_ratio,
                                                            random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest,
                                                            stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # train_mask[:int(train_ratio * num_nodes)] = 1
        # val_mask[int(train_ratio * num_nodes):int((train_ratio + val_ratio) * num_nodes)] = 1
        # test_mask[int((train_ratio + val_ratio) * num_nodes):] = 1

        train_mask[idx_train] = 1
        val_mask[idx_valid] = 1
        test_mask[idx_test] = 1

        
        # 生成边的掩码
        num_edges = self.graph.num_edges()
        edge_train_mask = torch.zeros(num_edges, dtype=torch.bool)
        edge_val_mask = torch.zeros(num_edges, dtype=torch.bool)
        edge_test_mask = torch.zeros(num_edges, dtype=torch.bool)

        # 根据节点掩码生成边掩码
        edge_train_mask = train_mask[src] & train_mask[dst]
        edge_val_mask = val_mask[src] & val_mask[dst]
        edge_test_mask = test_mask[src] & test_mask[dst]

        # # 根据节点掩码生成边掩码
        # for i in range(num_edges):
        #     src_node = src[i]
        #     dst_node = dst[i]
        #     if train_mask[src_node] and train_mask[dst_node]:
        #         edge_train_mask[i] = 1
        #     elif val_mask[src_node] and val_mask[dst_node]:
        #         edge_val_mask[i] = 1
        #     elif test_mask[src_node] and test_mask[dst_node]:
        #         edge_test_mask[i] = 1

        self.graph.edata['train_mask'] = edge_train_mask
        self.graph.edata['val_mask'] = edge_val_mask
        self.graph.edata['test_mask'] = edge_test_mask

        # print("Node train mask:", train_mask)
        # print("Node val mask:", val_mask)
        # print("Node test mask:", test_mask)

        # print("Edge train mask:", edge_train_mask)
        # print("Edge val mask:", edge_val_mask)
        # print("Edge test mask:", edge_test_mask)


        self.graph.ndata['label'] = self.graph.ndata['label'].long().squeeze(-1)
        self.graph.ndata['feature'] = self.graph.ndata['feature'].float()

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1
    
class Dataset:
    def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None):
        self.name = name
        graph = None
        if name == 'tfinance':
            graph, label_dict = load_graphs('dataset/tfinance')
            graph = graph[0]
            # 检查标签数据的形状
            print(f"Original label shape: {graph.ndata['label'].shape}")

            if len(graph.ndata['label'].shape) > 1 and graph.ndata['label'].shape[1] > 1:
                # 标签是独热编码形式，使用argmax转换为单一类别
                graph.ndata['label'] = graph.ndata['label'].argmax(dim=1)
            else:
                # 标签已经是单一类别形式
                graph.ndata['label'] = graph.ndata['label'].view(-1)

            # 检查转换后的标签形状
            print(f"Processed label shape: {graph.ndata['label'].shape}")

            if anomaly_std:
                graph, label_dict = load_graphs('dataset/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = graph.ndata['label'][:,1].nonzero().squeeze(1)
                feat = (feat-np.average(feat,0)) / np.std(feat,0)
                feat[anomaly_id] = anomaly_std * feat[anomaly_id]
                graph.ndata['feature'] = torch.tensor(feat)
                graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_alpha:
                graph, label_dict = load_graphs('dataset/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
                normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
                label = graph.ndata['label'].argmax(1)
                diff = anomaly_alpha * len(label) - len(anomaly_id)
                import random
                new_id = random.sample(normal_id, int(diff))
                # new_id = random.sample(anomaly_id, int(diff))
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    # aid = random.choice(normal_id)
                    feat[idx] = feat[aid]
                    label[idx] = 1  # 0
            # 1. 获取图中边的总数
            num_edges = graph.num_edges()

            # 2. 设定删除比例，比如删除 20% 的边
            delete_ratio = 0.8
            num_delete = int(num_edges * delete_ratio)

            # 3. 随机生成待删除的边的索引
            edge_indices = torch.randperm(num_edges)[:num_delete]

            # 4. 直接在原图上删除这些边（in-place 操作）
            graph.remove_edges(edge_indices)

        elif name == 'tsocial':
            graph, label_dict = load_graphs('dataset/tsocial')
            graph = graph[0]

        elif name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)    
            #     import scipy.io as scio
            #     yelp_path = 'dataset/YelpChi.mat'
            #     yelp = scio.loadmat(yelp_path)
            #     # print(yelp)
            #     homo = yelp['homo']
            #     homo = homo+homo.transpose()
            #     homo = homo.tocoo()
            #     feats = yelp['features'].todense()
            #     features = torch.from_numpy(feats)
            #     lbs = yelp['label'][0]
            #     labels = torch.from_numpy(lbs)
            #     rur = yelp['net_rur']
            #     rur = rur+rur.transpose()
            #     rur = rur.tocoo()
            #     rtr = yelp['net_rtr']
            #     rtr = rtr+rtr.transpose()
            #     rtr = rtr.tocoo()
            #     rsr = yelp['net_rsr']
            #     rsr = rsr+rsr.transpose()
            #     rsr = rsr.tocoo()
                
            #     yelp_graph_structure = {
            #         # ('r','homo','r'):(torch.tensor(src), torch.tensor(dst)),
            #         ('r','homo','r'):(torch.tensor(homo.row), torch.tensor(homo.col)),
            #         ('r','u','r'):(torch.tensor(rur.row), torch.tensor(rur.col)),
            #         ('r','t','r'):(torch.tensor(rtr.row), torch.tensor(rtr.col)),
            #         ('r','s','r'):(torch.tensor(rsr.row), torch.tensor(rsr.col))
            #     }
            #     yelp_graph = dgl.heterograph(yelp_graph_structure)
            #     # 给每个边类型添加自环
            #     yelp_graph = dgl.add_self_loop(yelp_graph, etype='homo')  # 添加'homo'类型的自环
            #     # yelp_graph = dgl.add_self_loop(yelp_graph, etype='u')     # 添加'u'类型的自环
            #     # yelp_graph = dgl.add_self_loop(yelp_graph, etype='t')     # 添加't'类型的自环
            #     # yelp_graph = dgl.add_self_loop(yelp_graph, etype='s')     # 添加's'类型的自环
            #     yelp_graph.nodes['r'].data['feature'] = features
            #     yelp_graph.nodes['r'].data['label'] = labels
            #     graph = yelp_graph

        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
            # else:
                # for relation in graph.etypes:
                #     graph = dgl.add_self_loop(graph, etype=relation)     
            # else:
            #     import scipy.io as scio
            #     amazon_path = 'dataset/Amazon.mat'
            #     amazon = scio.loadmat(amazon_path)
            #     # print(amazon)
            #     feats = amazon['features'].todense()
            #     features = torch.from_numpy(feats)
            #     lbs = amazon['label'][0]
            #     labels = torch.from_numpy(lbs)
            #     homo = amazon['homo']
            #     homo = homo+homo.transpose()
            #     homo = homo.tocoo()
            #     upu = amazon['net_upu']
            #     upu = upu+upu.transpose()
            #     upu = upu.tocoo()
            #     usu = amazon['net_usu']
            #     usu = usu+usu.transpose()
            #     usu = usu.tocoo()
            #     uvu = amazon['net_uvu']
            #     uvu = uvu+uvu.transpose()
            #     uvu = uvu.tocoo()
                          
            #     amazon_graph_structure = {
            #         ('r','homo','r'):(torch.tensor(homo.row), torch.tensor(homo.col)),
            #         ('r','u','r'):(torch.tensor(upu.row), torch.tensor(upu.col)),
            #         ('r','t','r'):(torch.tensor(usu.row), torch.tensor(usu.col)),
            #         ('r','s','r'):(torch.tensor(uvu.row), torch.tensor(uvu.col))
            #     }
            #     amazon_graph = dgl.heterograph(amazon_graph_structure)
            #     amazon_graph.nodes['r'].data['feature'] = features
            #     amazon_graph.nodes['r'].data['label'] = labels
            #     graph = amazon_graph


        elif name == 'cora':
            dataset = CoraGraphDataset()
            graph = dataset[0]
            graph.ndata['feature'] = graph.ndata['feat'].float()

        elif name == 'reddit':
            data = pygod_load_data(name)
            graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
            graph.ndata['feature'] = data.x
            graph.ndata['label'] = data.y.type(torch.LongTensor)

        elif name == 'tolokers':
            from dgl.data import TolokersDataset
            dataset = TolokersDataset()
            graph = dataset[0]
            graph.ndata['feature'] = graph.ndata['feat']
        elif name == 'minesweeper':
            from dgl.data import MinesweeperDataset
            dataset = MinesweeperDataset()
            graph = dataset[0]
            num_classes = dataset.num_classes
            graph.ndata['feature'] = graph.ndata['feat']
            # data = np.load('dataset/minesweeper.npz')
            # # 查看文件中包含的数组
            # print(data.files)  # 输出文件中所有的数组名
            # # 提取节点特征
            # x = data['node_features']  # 节点特征矩阵
            # # 提取边列表
            # edge_index = data['edges'].T  # 边列表，通常是 (2, num_edges) 的形状
            # # 提取标签
            # y = data['node_labels']  # 节点标签
            # # 假设 edge_index 的形状为 (2, num_edges)，第一行是源节点，第二行是目标节点
            # src, dst = edge_index
            # # 创建 DGL 图
            # g = dgl.graph((src, dst))
            # # 将节点特征和标签附加到图上
            # g.ndata['feature'] = torch.tensor(x, dtype=torch.float32)  # 节点特征
            # g.ndata['label'] = torch.tensor(y, dtype=torch.long)  # 节点标签
            # graph = g

        elif name == 'questions':
            # 加载 .npz 文件
            data = np.load('dataset/questions.npz')
            # 查看文件中包含的数组
            print(data.files)  # 输出文件中所有的数组名
            # 提取节点特征
            x = data['node_features']  # 节点特征矩阵
            # 提取边列表
            edge_index = data['edges'].T  # 边列表，通常是 (2, num_edges) 的形状
            # 提取标签
            y = data['node_labels']  # 节点标签
            # 假设 edge_index 的形状为 (2, num_edges)，第一行是源节点，第二行是目标节点
            src, dst = edge_index
            # 创建 DGL 图
            g = dgl.graph((src, dst))
            # 将节点特征和标签附加到图上
            g.ndata['feature'] = torch.tensor(x, dtype=torch.float32)  # 节点特征
            g.ndata['label'] = torch.tensor(y, dtype=torch.long)  # 节点标签
            graph = g
        

        elif name == 'elliptic':
            # 加载节点特征文件
            # features_df = pd.read_csv('dataset/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
            # # 加载边列表文件
            # edges_df = pd.read_csv('dataset/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
            # # 加载节点标签文件
            # labels_df = pd.read_csv('dataset/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
            # # 过滤出有标签的节点，去除 'unknown' 的标签
            # labeled_nodes_df = labels_df[labels_df['class'] != 'unknown']
            # #提取有标签节点的ID列表
            # labeled_node_ids = set(labeled_nodes_df['txId'].values)

            # # 过滤节点特征，只保留有标签的节点
            # labeled_features_df = features_df[features_df[0].isin(labeled_node_ids)]

            # # 过滤边列表，去除包含无标签节点的边
            # filtered_edges_df = edges_df[edges_df['txId1'].isin(labeled_node_ids) & edges_df['txId2'].isin(labeled_node_ids)]
            # # 重新映射节点ID，确保它们是连续的
            # node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(labeled_node_ids)}
            # mapped_edges = filtered_edges_df.replace(node_id_mapping)
            # mapped_features = labeled_features_df[0].map(node_id_mapping)
            # # 提取有标签节点的特征和边
            # labeled_node_features = torch.tensor(labeled_features_df.iloc[:, 1:].values, dtype=torch.float32)
            # edge_list = torch.tensor(mapped_edges.values.T, dtype=torch.int64)  # 转置使之成为 (2, num_edges)

            # # 创建 DGL 图，使用映射后的边
            # graph = dgl.graph((edge_list[0], edge_list[1]))

            # # 将节点特征添加到图中
            # graph.ndata['feature'] = labeled_node_features

            # # 将节点标签添加到图中
            # node_labels = torch.tensor(labeled_nodes_df['class'].map({'1': 1, '2': 0}).values, dtype=torch.long)
            # graph.ndata['label'] = node_labels

            # print(graph)
            graphs, _ = dgl.load_graphs("dataset/elliptic_graph.dgl")
            graph = graphs[0]  # 读取第一个图
            
            # graph.ndata['feature'] = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=True), dtype=torch.float)
        elif name == 'genius':
            graphs, _ = dgl.load_graphs("dataset/genius.dgl")
            graph = graphs[0]
        elif name == 'weibo':
            # 加载张量
            data = torch.load('dataset/weibo.pt')
            src, dst = data.edge_index

            # 1. 手动将图转化为无向图
            # 添加反向边 (dst, src) 来使其成为无向图
            graph = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])))

            # 将节点特征从 PyG 转移到 DGL
            graph.ndata['feature'] = data.x

            # 将节点标签从 PyG 转移到 DGL
            graph.ndata['label'] = data.y
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph
