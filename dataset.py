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
    
class Dataset:
    def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None):
        self.name = name
        graph = None
        if name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)    

        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)

        elif name == 'tolokers':
            from dgl.data import TolokersDataset
            dataset = TolokersDataset()
            graph = dataset[0]
            graph.ndata['feature'] = graph.ndata['feat']

        elif name == 'elliptic':
            graphs, _ = dgl.load_graphs("datasets/elliptic_graph.dgl")
            graph = graphs[0]  # 读取第一个图

        elif name == 'weibo':
            # 加载张量
            data = torch.load('datasets/weibo.pt')
            src, dst = data.edge_index
            graph = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])))
            graph.ndata['feature'] = data.x
            graph.ndata['label'] = data.y
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph
