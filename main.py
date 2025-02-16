# -*- coding: utf-8 -*-
import argparse
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, accuracy_score
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from dataset import Dataset  # 假设Dataset类用于加载数据和构建图
from utils import *
import warnings
warnings.filterwarnings('ignore')
import time

# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

# def get_best_f1_edge(labels, probs):
#     best_f1, best_thre = 0, 0
#     for thres in np.linspace(0.05, 0.95, 19):
#         preds = np.zeros_like(labels)
#         preds = (probs >= thres).astype(int)
#         mf1 = f1_score(labels, preds, average='macro')
#         if mf1 > best_f1:
#             best_f1 = mf1
#             best_thre = thres
#     return best_f1, best_thre

def train(model, graph, args, device, quantile):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=50)

    features = graph.ndata['feature'].to(device)
    labels = graph.ndata['label'].to(device)
    train_mask = graph.ndata['train_mask'].to(device).bool()
    graph = graph.to(device)

    best_f1, final_trec, final_tpre, final_tmf1, final_tauc, final_gmean = 0., 0., 0., 0., 0., 0.
    train_losses, val_losses = [], []
    epochs_without_improvement = 0
    patience = args.patience  # Early Stopping 的耐心值
    best_model_state = None  # 保存最佳模型的状态
    time_start = time.time()

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()

        output, edge_pred, reconstruction_loss, _ = model(graph, features)
        total_loss = 0

        # 处理边标签和掩码
        if isinstance(graph.edata['label'], dict):
            edge_labels = {etype: label.to(device) for etype, label in graph.edata['label'].items()}
        else:
            edge_labels = graph.edata['label'].to(device)

        if isinstance(graph.edata['edge_train_mask'], dict):
            edge_train_mask = {etype: mask.to(device).bool() for etype, mask in graph.edata['edge_train_mask'].items()}
        else:
            edge_train_mask = graph.edata['edge_train_mask'].to(device).bool()

        loss = model.compute_loss(output, edge_pred, labels, edge_labels, train_mask, edge_train_mask, reconstruction_loss)
        total_loss += loss
        train_losses.append(loss.item())

        total_loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            output, edge_pred, reconstruction_loss, combined_out = model(graph, features)
            if isinstance(graph.edata['label'], dict):
                edge_labels = {etype: label.to(device) for etype, label in graph.edata['label'].items()}
            else:
                edge_labels = graph.edata['label'].to(device)

            if isinstance(graph.edata['edge_test_mask'], dict):
                edge_test_mask = {etype: mask.to(device).bool() for etype, mask in graph.edata['edge_test_mask'].items()}
            else:
                edge_test_mask = graph.edata['edge_train_mask'].to(device).bool()

            val_loss = model.compute_loss(output, edge_pred, labels, edge_labels, test_mask, edge_test_mask, reconstruction_loss)
            val_losses.append(val_loss.item())

            probs = F.softmax(output, dim=1)
            f1, thres = get_best_f1(labels[val_mask].cpu(), probs[val_mask].cpu().detach().numpy())
            pred = torch.zeros_like(labels, device=device)
            pred[probs[:, 1] > thres] = 1

            labels_np = labels[test_mask].cpu().detach().numpy()
            preds_np = pred[test_mask].cpu().detach().numpy()
            probs_np = probs[test_mask][:, 1].cpu().detach().numpy()

            trec = recall_score(labels_np, preds_np)
            tpre = precision_score(labels_np, preds_np)
            tmf1 = f1_score(labels_np, preds_np, average='macro')
            tauc = roc_auc_score(labels_np, np.nan_to_num(probs_np, nan=0.0, posinf=0.0, neginf=0.0))
            gmean = compute_gmean(labels_np, preds_np)

            # 更新最佳指标
            if best_f1 < f1:
                best_f1 = f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
                final_gmean = gmean
                epochs_without_improvement = 0
                best_model_state = model.state_dict()  # 保存最佳模型状态
            else:
                epochs_without_improvement += 1

            # Early Stopping 检查
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} as validation F1 did not improve for {patience} epochs.")
                break  # 停止训练

            if (epoch) % 10 == 0:
                print(f"Test Loss {val_loss.item()}")
                print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(epoch, val_loss.item(), f1, best_f1))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')

    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} G-Mean {:.2f}'.format(
        final_trec * 100, final_tpre * 100, final_tmf1 * 100, final_tauc * 100, final_gmean * 100))
    # # 获取测试集的节点原始特征、最后的节点表征和标签
    test_nodes = graph.ndata['feature'][test_mask].cpu().numpy()  # 测试集节点的原始特征
    final_embeddings = combined_out[test_mask].cpu().detach().numpy()  # 最后的节点表征
    test_labels = graph.ndata['label'][test_mask].cpu().numpy()  # 测试集节点的标签

    # 创建 tsne 文件夹（如果不存在）
    tsne_folder = 'tsne'
    if not os.path.exists(tsne_folder):
        os.makedirs(tsne_folder)
    # 构建文件名
    test_nodes_filename = os.path.join(tsne_folder, f"{args.dataset}_test_nodes_features.npy")
    final_embeddings_filename = os.path.join(tsne_folder, f"{args.dataset}_final_node_embeddings.npy")
    test_labels_filename = os.path.join(tsne_folder, f"{args.dataset}_test_nodes_labels.npy")
    # 保存数据
    np.save(test_nodes_filename, test_nodes)
    np.save(final_embeddings_filename, final_embeddings)
    np.save(test_labels_filename, test_labels)


    # # 加载最佳模型状态
    # if best_model_state is not None:
    #     model.load_state_dict(best_model_state)
    #     print("Loaded best model state.")

    return final_trec, final_tpre, final_tmf1, final_tauc, final_gmean

def StartTrain(in_feats, h_feats, out_feats, num_classes, graph, d, pos_quantile, neg_quantile):

    print('train/test samples: ', train_mask.sum().item(), test_mask.sum().item())
    labels = graph.ndata['label'].to(device)
    weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    edge_weight = 0
    print("Starting training")
    # 遍历模型参数并打印每个参数的维度
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, Parameter shape: {param.shape}")
    if args.type == 'full':
        from model import CombinedModel
    elif args.type == 'without_attention':
        from model_without_attention import CombinedModel
    elif args.type == 'without_partition':
        from model_without_partition import CombinedModel
    elif args.type == 'without_reconstruction':
        from model_without_reconstruction import CombinedModel
    model = CombinedModel(in_feats, h_feats, out_feats, num_classes, graph, d=d, pos_quantile=pos_quantile, neg_quantile=neg_quantile, num_heads=args.num_heads, mode=args.mode)
    model = model.to(device)
    return model, weight, edge_weight

def compute_gmean(labels_np, preds_np):
    from sklearn.metrics import confusion_matrix
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np).ravel()
    # 计算 Sensitivity (召回率)
    if (tp + fn == 0):
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)
    # 计算 Specificity (特异度)
    if (tn + fp == 0):
        specificity = 0
    else:
        specificity = tn / (tn + fp)
    # 计算 G-Mean
    gmean = np.sqrt(sensitivity * specificity)
    return gmean

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--homo', type=int, default=1)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--run', type=int, default=10)
    parser.add_argument('--train_ratio', type=float, default=0.4)
    parser.add_argument('--pos_quantile', type=float, default=0.3)
    parser.add_argument('--neg_quantile', type=float, default=0.3)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--mode', type=str, default='homo')
    parser.add_argument('--type', type=str, default='full')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Choose a specific CUDA device number, like 0, 1, etc.',
                        # 根据 torch.cuda.device_count() 生成合法的整数范围选项
                        choices=list(range(torch.cuda.device_count())))

    # 解析命令行参数
    args = parser.parse_args()

    # 根据传入的参数设置设备，将整数转换为对应的 cuda 设备标识格式
    device = torch.device(f'cuda:{args.gpu}')
    # dataset = YelpDataset(args.dataset)
    dataset_name = args.dataset
    homo = args.homo
    graph = Dataset(dataset_name, homo).graph

    print(f"Done loading data from cached files.")
    # print(graph)

    # 生成训练、验证和测试掩码
    num_nodes = graph.num_nodes()
    node_labels = graph.ndata['label']

    # 循环遍历图的所有边类型
    for etype in graph.etypes:
        # 获取当前边类型的源节点和目标节点
        src, dst = graph.edges(etype=etype)
        # 生成边标签：根据源节点和目标节点的标签是否相同来生成
        def generate_edge_labels(src, dst, node_labels):
            edge_labels = (node_labels[src] != node_labels[dst]).long()
            return 2 * edge_labels - 1
        # 为当前边类型生成边标签
        edge_labels = generate_edge_labels(src, dst, node_labels)
        # 将生成的边标签添加到当前边类型的数据中
        graph.edges[etype].data['label'] = edge_labels

    train_ratio = 0.4
    val_ratio = 0.2
    test_ratio = 0.4

    if args.dataset == 'amazon':
        index = list(range(3305, len(graph.ndata['label'])))
    else:
        index = list(range(len(graph.ndata['label'])))
    labels = graph.ndata['label']
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index].cpu().detach().numpy(),  # 移动到CPU并转换为NumPy数组
                                                        stratify=labels[index].cpu().detach().numpy(),  # 移动到CPU并转换为NumPy数组
                                                        train_size=train_ratio,
                                                        random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest,
                                                        stratify=y_rest,
                                                        test_size=0.67,
                                                        random_state=2, shuffle=True)

    # 节点掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    # 获取节点的掩码
    train_mask = graph.ndata['train_mask'].bool()
    val_mask = graph.ndata['val_mask'].bool()
    test_mask = graph.ndata['test_mask'].bool()

    # 定义一个函数来生成边掩码
    def generate_edge_masks(src, dst, train_mask, val_mask, test_mask):
        edge_train_mask = train_mask[src] & train_mask[dst]
        edge_val_mask = (val_mask[src] & (train_mask[dst] | val_mask[dst])) | (val_mask[dst] & (train_mask[src] | val_mask[src]))
        edge_test_mask = (test_mask[src] & (train_mask[dst] | val_mask[dst] | test_mask[dst])) | (test_mask[dst] & (train_mask[src] | val_mask[src] | test_mask[src]))
        return edge_train_mask, edge_val_mask, edge_test_mask

    # 循环遍历图的所有边类型
    for etype in graph.etypes:
        # 获取当前边类型的源节点和目标节点
        src, dst = graph.edges(etype=etype)
        # 生成当前边类型的训练、验证和测试掩码
        edge_train_mask, edge_val_mask, edge_test_mask = generate_edge_masks(src, dst, train_mask, val_mask, test_mask)
        # 将生成的掩码赋值给图中当前边类型的数据
        graph.edges[etype].data['edge_train_mask'] = edge_train_mask
        graph.edges[etype].data['edge_val_mask'] = edge_val_mask
        graph.edges[etype].data['edge_test_mask'] = edge_test_mask

    # 将节点特征和标签转换为正确的数据类型
    graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
    graph.ndata['feature'] = graph.ndata['feature'].float()

    # 打印训练集和测试集的节点数量、边数量及其标签
    train_mask = graph.ndata['train_mask'].bool()
    val_mask = graph.ndata['val_mask'].bool()
    test_mask = graph.ndata['test_mask'].bool()

    # 输出节点掩码的信息
    print("Train nodes:", train_mask.sum().item())
    print("Val nodes:", val_mask.sum().item())
    print("Test nodes:", test_mask.sum().item())

    # 输出节点标签和边标签
    print("Node labels:", graph.ndata['label'].unique())

    # 输出图中所有的节点数据（ndata）和边数据（edata）键
    print(f"Graph ndata keys: {graph.ndata.keys()}")
    print(f"Graph edata keys: {graph.edata.keys()}")

    graph = graph.to(device)
    in_feats = graph.ndata['feature'].shape[1]
    h_feats = args.hid_dim
    out_feats = args.hid_dim
    d = args.order
    pos_quantile = args.pos_quantile
    neg_quantile = args.neg_quantile
    num_classes = len(torch.unique(graph.ndata['label']))

    

    final_recs, final_pres, final_mf1s, final_aucs, final_gmeans = [], [], [], [], []
    for tt in range(args.run):
        print(f"Running {tt+1}/{args.run}...") 
        model, weight, edge_weight = StartTrain(in_feats, h_feats, out_feats, num_classes, graph, d, pos_quantile, neg_quantile)
        rec, pre, mf1, auc, gmean = train(model, graph, args, device, edge_weight)
        final_recs.append(rec)
        final_pres.append(pre)
        final_mf1s.append(mf1)
        final_aucs.append(auc)
        final_gmeans.append(gmean)
    final_recs = np.array(final_recs)
    final_pres = np.array(final_pres)
    final_mf1s = np.array(final_mf1s)
    final_aucs = np.array(final_aucs)
    final_gmes = np.array(final_gmeans)

    print('-' * 60)  # 输出一条分隔线
    print('Test Results:'.center(60))  # 输出标题并居中
    print('-' * 60)
    print('Rec-mean: {:.2f}, Rre-std: {:.2f}'.format(100 * np.mean(final_recs), 100 * np.std(final_recs)))
    print('Pre-mean: {:.2f}, Pre-std: {:.2f}'.format(100 * np.mean(final_pres), 100 * np.std(final_pres)))
    print('MF1-mean: {:.2f}, MF1-std: {:.2f}'.format(100 * np.mean(final_mf1s), 100 * np.std(final_mf1s)))
    print('AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_aucs), 100 * np.std(final_aucs)))
    print('GMe-mean: {:.2f}, GMe-std: {:.2f}'.format(100 * np.mean(final_gmes), 100 * np.std(final_gmes)))
    print('-' * 60)  # 输出结束分隔线
